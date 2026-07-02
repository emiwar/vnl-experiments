"""Shared network reconstruction + checkpoint loading for the delays eval scripts.

Both ``eval_videos.py`` (video rendering) and ``eval_runs.py`` (batch
metrics) need to rebuild a trained network from its ``config.json`` and restore the
latest checkpoint. That logic lives here so neither script imports the other.

Public API:
    * ``_parse_net_params(raw)``         — JSON-string → typed net-param dict.
    * ``build_delay_network(...)``       — reconstruct ``RodentEncDecDelays``.
    * ``build_forward_model_network(...)``— reconstruct ``RodentForwardModel``.
    * ``build_network(net_params, env, rngs)`` — dispatch on ``network_class``.
    * ``load_network(ckpt_dir, net_params, env, seed)`` — build + restore latest
      step, returning ``(networks, step)`` or ``None`` when unavailable.

The builders are env-agnostic: they only query ``env.non_flattened_observation_size``
and ``env.action_size``, so the same builders serve both the ``Imitation`` and
``AbsoluteImitation`` environments.
"""

import pickle
import warnings
from pathlib import Path

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state
from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Concat, Sequential
from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.utils import Filter, Flattener, Map
from nnx_ppo.networks.variational import VariationalBottleneck

from vnl_experiments.delays.efference_copy import EfferenceCopy
from vnl_experiments.delays.forward_model import ForwardModel


# ---------------------------------------------------------------------------
# Net-param parsing
# ---------------------------------------------------------------------------

def _parse_net_params(raw: dict) -> dict:
    """Convert JSON string values to proper Python types (matching checkpoint_utils)."""
    result = {}
    for k, v in raw.items():
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
# Network construction
# ---------------------------------------------------------------------------

def build_delay_network(net_params: dict, env, rngs: nnx.Rngs):
    """Reconstruct the enc-dec delay network from saved net_params."""
    p = _parse_net_params({k: v for k, v in net_params.items()
                           if k != "network_class"})

    obs_size = env.non_flattened_observation_size
    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0]))
    action_size = env.action_size

    delay_k = int(p.get("delay_k", 0))
    efference_length = int(p.get("efference_length", delay_k))

    enc_hidden = list(p.get("enc_hidden_sizes", [512] * 4))
    dec_hidden = list(p.get("dec_hidden_sizes", [512] * 4))
    critic_hidden = list(p.get("critic_hidden_sizes", [1024] * 2))
    latent_size = int(p.get("latent_size", 16))
    kl_weight = float(p.get("kl_weight", 0.01))
    latent_min_std = float(p.get("latent_min_std", 0.01))
    entropy_weight = float(p.get("entropy_weight", 1e-2))
    min_std = float(p.get("min_std", 1e-1))
    std_scale = float(p.get("std_scale", 1.0))
    normalize_obs = bool(p.get("normalize_obs", True))
    initializer_scale = float(p.get("initializer_scale", 1.0))

    activation_name = str(p.get("activation", "swish"))
    activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[activation_name]

    kernel_init = nnx.initializers.variance_scaling(
        initializer_scale, "fan_in", "uniform"
    )

    enc_sizes = [task_obs_size] + enc_hidden + [latent_size * 2]
    decoder_in = latent_size + proprio_size + efference_length * action_size
    dec_sizes = [decoder_in] + dec_hidden + [action_size * 2]
    critic_sizes = [task_obs_size + proprio_size] + critic_hidden + [1]

    encoder_branch = Sequential([
        Flattener(),
        *make_mlp_layers(enc_sizes, rngs, activation,
                         activation_last_layer=False, kernel_init=kernel_init),
        VariationalBottleneck(latent_size, rngs, kl_weight, latent_min_std),
    ])

    proprio_branch_layers = [Flattener()]
    if delay_k > 0:
        proprio_branch_layers.append(Delay(jp.zeros(proprio_size), k_steps=delay_k))
    proprio_branch = Sequential(proprio_branch_layers)

    decoder = Sequential([
        *make_mlp_layers(dec_sizes, rngs, activation,
                         activation_last_layer=False, kernel_init=kernel_init),
        NormalTanhSampler(rngs, entropy_weight=entropy_weight,
                          min_std=min_std, std_scale=std_scale),
    ])

    actor = Sequential([
        Concat(task_obs=encoder_branch, proprioception=proprio_branch),
        EfferenceCopy(inner=decoder, sample_action=jp.zeros(action_size),
                      queue_length=efference_length),
    ])

    critic = Sequential([
        Flattener(),
        *make_mlp_layers(critic_sizes, rngs, activation,
                         activation_last_layer=False, kernel_init=kernel_init),
    ])

    adapter = PPOAdapter(action=actor, value=critic)

    pre = Flattener(preserve_levels=2)
    lift = Filter({
        "task_obs": ("state", "task_obs"),
        "proprioception": ("state", "proprioception"),
    })

    if normalize_obs:
        normalizer_shape = {
            "state": {
                "task_obs": task_obs_size,
                "proprioception": proprio_size,
            }
        }
        return Sequential([pre, Normalizer(normalizer_shape), lift, adapter])
    return Sequential([pre, lift, adapter])


def build_forward_model_network(net_params: dict, env, rngs: nnx.Rngs):
    """Reconstruct the explicit-forward-model network (mirrors the train script)."""
    p = _parse_net_params({k: v for k, v in net_params.items() if k != "network_class"})

    obs_size = env.non_flattened_observation_size
    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0]))
    action_size = env.action_size

    delay_k = int(p.get("delay_k", 0))
    efference_length = int(p.get("efference_length", delay_k))
    fm_loss_weight = float(p.get("fm_loss_weight", 1.0))

    enc_hidden = list(p.get("enc_hidden_sizes", [512] * 4))
    dec_hidden = list(p.get("dec_hidden_sizes", [512] * 4))
    fm_hidden = list(p.get("fm_hidden_sizes", [512] * 4))
    critic_hidden = list(p.get("critic_hidden_sizes", [1024] * 2))
    latent_size = int(p.get("latent_size", 32))
    kl_weight = float(p.get("kl_weight", 0.001))
    latent_min_std = float(p.get("latent_min_std", 0.01))
    entropy_weight = float(p.get("entropy_weight", 1e-2))
    min_std = float(p.get("min_std", 1e-1))
    std_scale = float(p.get("std_scale", 1.0))
    normalize_obs = bool(p.get("normalize_obs", True))
    activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
        str(p.get("activation", "swish"))
    ]

    enc_sizes = [task_obs_size] + enc_hidden + [latent_size * 2]
    decoder_in = latent_size + proprio_size
    dec_sizes = [decoder_in] + dec_hidden + [action_size * 2]
    predictor_sizes = (
        [proprio_size + efference_length * action_size] + fm_hidden + [proprio_size]
    )
    critic_sizes = [task_obs_size + proprio_size] + critic_hidden + [1]

    encoder_branch = Sequential([
        Flattener(),
        *make_mlp_layers(enc_sizes, rngs, activation, activation_last_layer=False),
        VariationalBottleneck(latent_size, rngs, kl_weight, latent_min_std),
    ])
    decoder = Sequential([
        *make_mlp_layers(dec_sizes, rngs, activation, activation_last_layer=False),
        NormalTanhSampler(rngs, entropy_weight=entropy_weight,
                          min_std=min_std, std_scale=std_scale),
    ])
    predictor = make_mlp(predictor_sizes, rngs, activation, activation_last_layer=False)

    actor = Sequential([
        Map(task_obs=encoder_branch, proprioception=Flattener()),
        EfferenceCopy(
            inner=ForwardModel(
                decoder=decoder, predictor=predictor, proprio_size=proprio_size,
                delay_steps=delay_k, loss_weight=fm_loss_weight,
            ),
            sample_action=jp.zeros(action_size),
            queue_length=efference_length,
            inject_key="efference",
        ),
    ])
    critic = Sequential([
        Flattener(),
        *make_mlp_layers(critic_sizes, rngs, activation, activation_last_layer=False),
    ])
    adapter = PPOAdapter(action=actor, value=critic)

    pre = Flattener(preserve_levels=2)
    lift = Filter({
        "task_obs": ("state", "task_obs"),
        "proprioception": ("state", "proprioception"),
    })
    if normalize_obs:
        normalizer_shape = {"state": {
            "task_obs": task_obs_size, "proprioception": proprio_size}}
        return Sequential([pre, Normalizer(normalizer_shape), lift, adapter])
    return Sequential([pre, lift, adapter])


def build_network(net_params: dict, env, rngs: nnx.Rngs):
    """Dispatch on ``network_class``. Returns None for unknown classes."""
    network_class = str(net_params.get("network_class", ""))
    if "RodentEncDecDelays" in network_class:
        return build_delay_network(net_params, env, rngs)
    if "RodentForwardModel" in network_class:
        return build_forward_model_network(net_params, env, rngs)
    warnings.warn(
        f"Unknown network_class {network_class!r}; add a builder for it. Skipping."
    )
    return None


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_network(ckpt_dir: Path, net_params: dict, env, seed: int):
    """Build the network, restore the latest step, return (nets, step) or None."""
    ckpt_dir = Path(ckpt_dir)
    step_dirs = sorted(ckpt_dir.glob("step_*/"))
    if not step_dirs:
        warnings.warn(f"No step_* directory in {ckpt_dir}; skipping.")
        return None
    step_dir = step_dirs[-1]

    nets = build_network(net_params, env, nnx.Rngs(seed))
    if nets is None:
        return None

    with open(step_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    ppo_cfg = meta.get("config")
    ppo_cfg = ppo_cfg.ppo if ppo_cfg is not None else None
    ts = new_training_state(
        env, nets, n_envs=1, seed=seed,
        learning_rate=ppo_cfg.learning_rate if ppo_cfg else 1e-4,
        gradient_clipping=ppo_cfg.gradient_clipping if ppo_cfg else 1.0,
        weight_decay=ppo_cfg.weight_decay if ppo_cfg else None,
    )
    ckpt = load_checkpoint(str(step_dir), ts.networks, ts.optimizer)
    return ts.networks, int(ckpt["step"])
