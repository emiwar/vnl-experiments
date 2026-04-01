"""Utilities for loading pre-trained network checkpoints.

Provides config-parsing helpers and a high-level
``load_network_from_checkpoint`` function used by both benchmarking
and distillation scripts.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jp
from flax import nnx

from vnl_playground.tasks.modular_rodent.imitation_v4 import ModularImitation_v4, default_config
from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state
from vnl_experiments.networks.nervenet_style_v3 import NerveNetNetwork_v3
from vnl_experiments.networks.mlp_modular import MLPModularNetwork
from vnl_experiments.networks.recurrent_modular import RecurrentModularNetwork


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def parse_env_config(env_params: dict):
    """Reconstruct env ConfigDict from the string-valued JSON dict.

    Starts from default_config() and overrides with values from env_params,
    using the local XML paths (not cluster paths from the checkpoint).
    """
    cfg = default_config()

    for field, conv in [
        ("clip_length", int),
        ("ctrl_dt", float),
        ("energy_cost", float),
        ("impratio", float),
        ("iterations", int),
        ("ls_iterations", int),
        ("max_target_distance", float),
        ("min_torso_z", float),
        ("mocap_hz", int),
        ("naconmax", int),
        ("njmax", int),
        ("noslip_iterations", int),
        ("rescale_factor", float),
        ("sim_dt", float),
        ("tolerance", float),
    ]:
        if field in env_params:
            setattr(cfg, field, conv(env_params[field]))

    for field in ["clip_set", "cone", "mujoco_impl", "qvel_init", "solver"]:
        if field in env_params:
            setattr(cfg, field, env_params[field])

    for field in ["include_vel", "torque_actuators"]:
        if field in env_params:
            setattr(cfg, field, env_params[field] == "True")

    if env_params.get("keep_clips_idx", "None") not in ("None", None):
        cfg.keep_clips_idx = env_params["keep_clips_idx"]

    if "start_frame_range" in env_params:
        cfg.start_frame_range = [int(x) for x in env_params["start_frame_range"]]

    if "reward_terms" in env_params:
        for k, v in env_params["reward_terms"].items():
            cfg.reward_terms[k] = float(v)

    # Always use local paths — cluster paths from the checkpoint are invalid here
    default = default_config()
    cfg.reference_data_path = default.reference_data_path
    cfg.walker_xml_path = default.walker_xml_path
    cfg.arena_xml_path = default.arena_xml_path

    return cfg


def parse_net_params(net_params: dict) -> dict:
    """Convert string-valued net_params to proper Python types, skipping network_class."""
    result = {}
    for k, v in net_params.items():
        if k == "network_class":
            continue
        if isinstance(v, list):
            result[k] = [int(x) for x in v]
        elif v == "True":
            result[k] = True
        elif v == "False":
            result[k] = False
        elif v == "None":
            result[k] = None
        else:
            try:
                result[k] = int(v)
            except (ValueError, TypeError):
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    result[k] = v
    return result


# ---------------------------------------------------------------------------
# Network construction and parameter counting
# ---------------------------------------------------------------------------

def build_network(net_params: dict, env: ModularImitation_v4, rngs: nnx.Rngs):
    """Instantiate the correct network class from config net_params."""
    network_class_str = net_params.get("network_class", "")
    kwargs = parse_net_params(net_params)

    obs_sizes = {
        k: jp.squeeze(jax.tree.reduce(jp.add, o))
        for k, o in env.non_flattened_observation_size.items()
    }

    if "NerveNetNetwork_v3" in network_class_str:
        return NerveNetNetwork_v3(obs_sizes, env.action_size, rngs=rngs, **kwargs)

    if "RecurrentModularNetwork" in network_class_str:
        return RecurrentModularNetwork(obs_sizes, env.action_size, rngs=rngs, **kwargs)

    if "MLPModularNetwork" in network_class_str:
        sample_state = jax.jit(env.reset)(jax.random.key(0))
        reward_keys = list(sample_state.reward.keys())
        flat_obs_size = int(jax.tree.reduce(jp.add, env.observation_size))
        action_sizes = {k: int(v) for k, v in env.action_size.items()}
        return MLPModularNetwork(
            obs_size=flat_obs_size,
            action_sizes=action_sizes,
            reward_keys=reward_keys,
            rngs=rngs,
            **kwargs,
        )

    raise ValueError(f"Unknown network class: {network_class_str!r}")


def _count_params(module) -> int:
    return sum(jax.tree.leaves(
        jax.tree.map(lambda x: x.size, nnx.state(module, nnx.Param))
    ))


def get_param_counts(nets, network_class_str: str) -> tuple[int, int]:
    """Return (n_actor_params, n_critic_params)."""
    total = _count_params(nets)
    if "NerveNetNetwork_v3" in network_class_str:
        n_critic = _count_params(nets.critic)
    elif "MLPModularNetwork" in network_class_str:
        n_critic = _count_params(nets.critic_encoder) + _count_params(nets.critic_heads)
    else:
        raise ValueError(f"Unknown network class: {network_class_str!r}")
    return total - n_critic, n_critic


# ---------------------------------------------------------------------------
# High-level checkpoint loader
# ---------------------------------------------------------------------------

def load_network_from_checkpoint(
    ckpt_dir: "str | Path",
    env: ModularImitation_v4,
    *,
    rngs: Optional[nnx.Rngs] = None,
    seed: int = 0,
):
    """Load network weights from a checkpoint directory.

    Reads config.json, instantiates the correct network class, locates the
    latest ``step_*`` subdirectory, and restores the saved weights in-place.

    Args:
        ckpt_dir: Path to the checkpoint directory (contains config.json and
            one or more ``step_*/`` subdirectories).
        env: Already-constructed environment whose architecture matches the
            checkpoint (same observation/action spaces).
        rngs: NNX rngs used when building the network. Defaults to
            ``nnx.Rngs(seed)``.
        seed: Used when ``rngs`` is None and for ``new_training_state``.

    Returns:
        network: Loaded network with restored weights.
    """
    ckpt_dir = Path(ckpt_dir)
    if rngs is None:
        rngs = nnx.Rngs(seed)

    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)

    net_params = cfg["net_params"]

    # Find the latest step checkpoint directory
    step_dirs = sorted(ckpt_dir.glob("step_*/"))
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directory found in {ckpt_dir}")
    step_dir = step_dirs[-1]

    nets = build_network(net_params, env, rngs=rngs)

    # new_training_state initializes the optimizer and network states so that
    # load_checkpoint has a correctly-shaped template to restore into.
    with open(step_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    ppo_cfg = meta.get("config")
    ppo_cfg = ppo_cfg.ppo if ppo_cfg is not None else None

    training_state = new_training_state(
        env,
        nets,
        n_envs=1,
        seed=seed,
        learning_rate=ppo_cfg.learning_rate if ppo_cfg else 1e-4,
        gradient_clipping=ppo_cfg.gradient_clipping if ppo_cfg else 1.0,
        weight_decay=ppo_cfg.weight_decay if ppo_cfg else None,
    )
    load_checkpoint(str(step_dir), training_state.networks, training_state.optimizer)

    # training_state.networks was updated in-place by load_checkpoint
    return training_state.networks
