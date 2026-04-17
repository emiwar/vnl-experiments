#!/usr/bin/env python3
"""Benchmark ALL checkpoints in each run folder to produce training curves.

For each run folder, evaluates every step_* checkpoint on both train and eval
clip splits, collecting aggregate metrics only (no per-clip breakdown).

Output is a plot-friendly JSON where each metric is a parallel array indexed
by step, making it easy to plot training speed and convergence:

    steps[i]  →  train.mean_episode_reward[i]
              →  eval.mean_episode_reward[i]
              →  ...

Usage (from vnl-experiments root):
    # Use the hardcoded RUN_FOLDERS list below:
    ../.venv/bin/python vnl_experiments/tools/benchmark_all_checkpoints.py

    # Or pass run folders as CLI arguments:
    ../.venv/bin/python vnl_experiments/tools/benchmark_all_checkpoints.py \\
        checkpoints/Recurrent-20260331-114919 \\
        checkpoints/MLPModular-20260328-004410
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jp
from flax import nnx

from vnl_playground.tasks.modular_rodent.imitation_v4 import ModularImitation_v4, default_config
from vnl_playground.tasks.reference_clips import ReferenceClips
from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state
from vnl_experiments.tools.checkpoint_utils import (
    parse_env_config,
    parse_net_params,
    build_network,
    get_param_counts,
)

# ---------------------------------------------------------------------------
# Run folders — add more here as training completes
# Paths are relative to the repo root (vnl-experiments/).
# ---------------------------------------------------------------------------

RUN_FOLDERS = [
    "checkpoints/Recurrent-20260416-163835",
    "checkpoints/MLPModular-20260318-113105",
    "checkpoints/Imitation_detached_critic_v4-20260319-155439",
]

OUTPUT_FILE = "benchmark_training_curves.json"

# Repo root: vnl-experiments/vnl_experiments/tools/ → up 3 levels
REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Per-clip evaluation loop (identical to benchmark_checkpoints.py)
# ---------------------------------------------------------------------------

def eval_all_clips(
    env: ModularImitation_v4,
    networks,
    n_clips: int,
    clip_length: int,
    key: jax.Array,
    flatten_obs: bool,
) -> dict:
    """Run all clips once in parallel, returning per-clip metric arrays of shape [n_clips]."""
    keys = jax.random.split(key, n_clips)
    clip_ids = jp.arange(n_clips)

    env_states = jax.vmap(
        lambda k, c: env.reset(k, clip_idx=c, start_frame=0)
    )(keys, clip_ids)
    net_states = networks.initialize_state(n_clips)

    def step(env, networks, carry):
        env_state, net_state, cuml_reward, cuml_hand_err, cuml_foot_err, cuml_root_err, cuml_appendages_err, lifespan = carry

        if flatten_obs:
            obs = jax.vmap(lambda o: jax.flatten_util.ravel_pytree(o)[0])(env_state.obs)
        else:
            obs = env_state.obs

        next_net_state, net_output = networks(net_state, obs)
        next_env_state = jax.vmap(env.step)(env_state, net_output.actions)

        next_env_state = next_env_state.replace(
            done=jp.logical_or(next_env_state.done, env_state.done).astype(float)
        )

        already_done = env_state.done.astype(bool)

        step_reward = jax.tree.reduce(jp.add, next_env_state.reward)
        cuml_reward = cuml_reward + jp.where(already_done, 0.0, step_reward)

        m = next_env_state.metrics
        hand_err = m["hand_L"]["pos_err"]
        foot_err = m["foot_L"]["pos_err"]
        root_err = m["root"]["pos_err"]
        appendages_err = m["average_appendage_err"]
        cuml_hand_err = cuml_hand_err + jp.where(already_done, 0.0, hand_err)
        cuml_foot_err = cuml_foot_err + jp.where(already_done, 0.0, foot_err)
        cuml_root_err = cuml_root_err + jp.where(already_done, 0.0, root_err)
        cuml_appendages_err = cuml_appendages_err + jp.where(already_done, 0.0, appendages_err)

        lifespan = lifespan + jp.where(next_env_state.done.astype(bool), 0.0, 1.0)

        return (
            next_env_state,
            next_net_state,
            cuml_reward,
            cuml_hand_err,
            cuml_foot_err,
            cuml_root_err,
            cuml_appendages_err,
            lifespan,
        )

    step_fn = functools.partial(step, env)
    step_scan = nnx.scan(
        step_fn,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=nnx.Carry,
        length=clip_length,
    )

    init_carry = (
        env_states,
        net_states,
        jp.zeros(n_clips),
        jp.zeros(n_clips),
        jp.zeros(n_clips),
        jp.zeros(n_clips),
        jp.zeros(n_clips),
        jp.zeros(n_clips),
    )
    _, _, cuml_reward, cuml_hand_err, cuml_foot_err, cuml_root_err, cuml_appendages_err, lifespan = step_scan(
        networks, init_carry
    )

    safe_lifespan = jp.maximum(lifespan, 1.0)
    return {
        "episode_reward": cuml_reward,
        "lifespan": lifespan,
        "hand_L_pos_err_mean_m": cuml_hand_err / safe_lifespan,
        "foot_L_pos_err_mean_m": cuml_foot_err / safe_lifespan,
        "root_pos_err_mean_m": cuml_root_err / safe_lifespan,
        "appendages_pos_err_mean_m": cuml_appendages_err / safe_lifespan,
    }


# ---------------------------------------------------------------------------
# Aggregate evaluation (no per-clip or by-label breakdown)
# ---------------------------------------------------------------------------

def evaluate_split_aggregate(
    env, networks, clips, clip_length: int, key: jax.Array, flatten_obs: bool
) -> dict:
    """Evaluate on all clips in one split and return aggregate scalars only."""
    n_clips = clips.qpos.shape[0]
    eval_jit = nnx.jit(eval_all_clips, static_argnums=(0, 2, 3, 5))
    networks.eval()
    arrays = eval_jit(env, networks, n_clips, clip_length, key, flatten_obs)
    networks.train()

    arrays = jax.tree.map(lambda x: x.tolist(), arrays)
    n = n_clips
    return {
        "n_clips": n,
        "mean_episode_reward":       sum(arrays["episode_reward"]) / n,
        "mean_lifespan":             sum(arrays["lifespan"]) / n,
        "mean_hand_L_pos_err_m":     sum(arrays["hand_L_pos_err_mean_m"]) / n,
        "mean_foot_L_pos_err_m":     sum(arrays["foot_L_pos_err_mean_m"]) / n,
        "mean_root_pos_err_m":       sum(arrays["root_pos_err_mean_m"]) / n,
        "mean_appendages_pos_err_m": sum(arrays["appendages_pos_err_mean_m"]) / n,
    }


# ---------------------------------------------------------------------------
# Per-run benchmarking (sweeps all step checkpoints)
# ---------------------------------------------------------------------------

def _append_scalar_dict(target: dict, source: dict) -> None:
    """Append each scalar from source into the corresponding list in target."""
    for k, v in source.items():
        target.setdefault(k, []).append(v)


def benchmark_run(run_dir_rel: str) -> dict:
    """Evaluate all step checkpoints in one run folder.

    Builds the environment and clip split once, then loads each step checkpoint
    in order and evaluates it on both train and eval splits.

    Returns a dict with parallel arrays keyed by metric name, plus a 'steps'
    array, suitable for direct JSON serialisation and plotting.
    """
    ckpt_dir = REPO_ROOT / run_dir_rel
    name = ckpt_dir.name
    print(f"\n=== {name} ===")

    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)
    env_params = cfg["env_params"]
    net_params = cfg["net_params"]
    network_class_str = net_params.get("network_class", "")
    clip_length = int(env_params.get("clip_length", 250))

    step_dirs = sorted(ckpt_dir.glob("step_*/"))
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directory in {ckpt_dir}")
    print(f"  Found {len(step_dirs)} checkpoints")

    # Build env + clip split once for the whole run
    env_cfg = parse_env_config(env_params)
    all_clips = ReferenceClips(
        env_cfg.reference_data_path,
        env_cfg.clip_length,
        env_cfg.keep_clips_idx,
    )
    train_clips, test_clips = all_clips.split()  # seed=0, ratio=0.8 — matches training

    print("  Initialising environments...")
    train_env = ModularImitation_v4(env_cfg, clips=train_clips)
    eval_env = ModularImitation_v4(env_cfg, clips=test_clips)

    flatten_obs = "MLPModularNetwork" in network_class_str

    steps = []
    train_arrays: dict = {}
    eval_arrays: dict = {}

    for i, step_dir in enumerate(step_dirs, 1):
        print(f"  [{i}/{len(step_dirs)}] {step_dir.name}")

        print("    Building network...")
        nets = build_network(net_params, train_env, rngs=nnx.Rngs(0))

        with open(step_dir / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        ppo_cfg = meta["config"].ppo if meta.get("config") is not None else None
        training_state = new_training_state(
            train_env,
            nets,
            n_envs=1,
            seed=0,
            learning_rate=ppo_cfg.learning_rate if ppo_cfg else 1e-4,
            gradient_clipping=ppo_cfg.gradient_clipping if ppo_cfg else 1.0,
            weight_decay=ppo_cfg.weight_decay if ppo_cfg else None,
        )
        ckpt = load_checkpoint(str(step_dir), training_state.networks, training_state.optimizer)
        nets = ckpt["training_state"].networks
        step = int(ckpt["step"])
        steps.append(step)

        key = jax.random.key(42)
        key, subkey = jax.random.split(key)
        train_agg = evaluate_split_aggregate(train_env, nets, train_clips, clip_length, subkey, flatten_obs)
        key, subkey = jax.random.split(key)
        eval_agg = evaluate_split_aggregate(eval_env, nets, test_clips, clip_length, subkey, flatten_obs)

        _append_scalar_dict(train_arrays, train_agg)
        _append_scalar_dict(eval_arrays, eval_agg)

        print(
            f"    step={step:,}  "
            f"train_reward={train_agg['mean_episode_reward']:.2f}  "
            f"eval_reward={eval_agg['mean_episode_reward']:.2f}"
        )

    n_actor, n_critic = get_param_counts(nets, network_class_str)

    return {
        "checkpoint_dir": str(ckpt_dir),
        "n_actor_params": n_actor,
        "n_critic_params": n_critic,
        "steps": steps,
        "train": train_arrays,
        "eval": eval_arrays,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    run_folders = sys.argv[1:] if len(sys.argv) > 1 else RUN_FOLDERS

    results = {}
    for run_dir in run_folders:
        name = Path(run_dir).name
        results[name] = benchmark_run(run_dir)

    out_path = REPO_ROOT / OUTPUT_FILE
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
