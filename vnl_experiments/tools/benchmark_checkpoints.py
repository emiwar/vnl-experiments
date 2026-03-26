#!/usr/bin/env python3
"""Benchmark checkpoints on train and eval clip splits.

For each checkpoint, evaluates on:
  - Train clips (same split used during training: seed=0, ratio=0.8)
  - Eval/test clips (the held-out 20%)

Runs each clip exactly once in parallel via vmap, with no action noise.
Results are saved to benchmark_results.json in the repo root.

Usage (from vnl-experiments root):
    ../.venv/bin/python vnl_experiments/tools/benchmark_checkpoints.py
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json
import pickle
from collections import defaultdict
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
    _count_params,
    get_param_counts,
)

# ---------------------------------------------------------------------------
# Checkpoints — add more here as training completes
# Paths are relative to the repo root (vnl-experiments/).
# ---------------------------------------------------------------------------

CHECKPOINTS = [
    "downloaded_checkpoints/MLPModular-20260320-054435",
    "downloaded_checkpoints/Imitation_detached_critic_v4-20260319-155439",
    "downloaded_checkpoints/Imitation_detached_critic_v4-20260319-171039",
    "downloaded_checkpoints/Imitation_detached_critic_v4-20260320-054423",
    "downloaded_checkpoints/MLPModular-20260319-153658",
]

OUTPUT_FILE = "benchmark_results.json"

# Repo root: vnl-experiments/vnl_experiments/tools/ → up 3 levels
REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Per-clip evaluation loop
# ---------------------------------------------------------------------------

def eval_all_clips(
    env: ModularImitation_v4,
    networks,
    n_clips: int,
    clip_length: int,
    key: jax.Array,
    flatten_obs: bool,
) -> dict:
    """Run all clips once in parallel, returning per-clip metric arrays of shape [n_clips].

    Args:
        env: Environment constructed with the appropriate clip subset.
        networks: Loaded network in eval mode (no action noise).
        n_clips: Number of clips (determines vmap width and scan length).
        clip_length: Max steps per clip (= clip_length from config).
        key: PRNG key.
        flatten_obs: If True, flatten dict obs to a flat array before passing to
            the network (required for MLPModularNetwork).
    """
    keys = jax.random.split(key, n_clips)
    clip_ids = jp.arange(n_clips)

    # Reset each environment deterministically at clip start (frame 0)
    env_states = jax.vmap(
        lambda k, c: env.reset(k, clip_idx=c, start_frame=0)
    )(keys, clip_ids)
    net_states = networks.initialize_state(n_clips)

    def step(env, networks, carry):
        env_state, net_state, cuml_reward, cuml_hand_err, cuml_foot_err, cuml_root_err, lifespan = carry

        # Prepare obs (flatten for MLP, keep dict for NerveNet)
        if flatten_obs:
            obs = jax.vmap(lambda o: jax.flatten_util.ravel_pytree(o)[0])(env_state.obs)
        else:
            obs = env_state.obs

        next_net_state, net_output = networks(net_state, obs)
        next_env_state = jax.vmap(env.step)(env_state, net_output.actions)

        # Propagate done: once terminated, stays terminated (no reset)
        next_env_state = next_env_state.replace(
            done=jp.logical_or(next_env_state.done, env_state.done).astype(float)
        )

        # Gate accumulation on whether env was already done *before* this step
        already_done = env_state.done.astype(bool)  # [n_clips]

        # Total reward across all modules
        step_reward = jax.tree.reduce(jp.add, next_env_state.reward)  # [n_clips]
        cuml_reward = cuml_reward + jp.where(already_done, 0.0, step_reward)

        # Position errors (meters) from env extra_metrics
        m = next_env_state.metrics
        hand_err = m["hand_L"]["pos_err"]   # [n_clips]
        foot_err = m["foot_L"]["pos_err"]   # [n_clips]
        root_err = m["root"]["pos_err"]     # [n_clips]
        cuml_hand_err = cuml_hand_err + jp.where(already_done, 0.0, hand_err)
        cuml_foot_err = cuml_foot_err + jp.where(already_done, 0.0, foot_err)
        cuml_root_err = cuml_root_err + jp.where(already_done, 0.0, root_err)

        # Lifespan: count non-terminated steps
        lifespan = lifespan + jp.where(next_env_state.done.astype(bool), 0.0, 1.0)

        return (
            next_env_state,
            next_net_state,
            cuml_reward,
            cuml_hand_err,
            cuml_foot_err,
            cuml_root_err,
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
        jp.zeros(n_clips),  # cuml_reward
        jp.zeros(n_clips),  # cuml_hand_err
        jp.zeros(n_clips),  # cuml_foot_err
        jp.zeros(n_clips),  # cuml_root_err
        jp.zeros(n_clips),  # lifespan
    )
    _, _, cuml_reward, cuml_hand_err, cuml_foot_err, cuml_root_err, lifespan = step_scan(
        networks, init_carry
    )

    safe_lifespan = jp.maximum(lifespan, 1.0)
    return {
        "episode_reward": cuml_reward,
        "lifespan": lifespan,
        "hand_L_pos_err_mean_m": cuml_hand_err / safe_lifespan,
        "foot_L_pos_err_mean_m": cuml_foot_err / safe_lifespan,
        "root_pos_err_mean_m": cuml_root_err / safe_lifespan,
    }


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def _aggregate(clips_list: list[dict]) -> dict:
    n = len(clips_list)
    return {
        "n_clips": n,
        "mean_episode_reward": sum(c["episode_reward"] for c in clips_list) / n,
        "mean_lifespan": sum(c["lifespan"] for c in clips_list) / n,
        "mean_hand_L_pos_err_m": sum(c["hand_L_pos_err_mean_m"] for c in clips_list) / n,
        "mean_foot_L_pos_err_m": sum(c["foot_L_pos_err_mean_m"] for c in clips_list) / n,
        "mean_root_pos_err_m": sum(c["root_pos_err_mean_m"] for c in clips_list) / n,
    }


def evaluate_split(env, networks, clips, clip_length: int, key: jax.Array, flatten_obs: bool) -> dict:
    """Evaluate on all clips in one split. Returns aggregate + by_label + per_clip."""
    n_clips = clips.qpos.shape[0]
    print(f"    Running {n_clips} clips in parallel...")

    eval_jit = nnx.jit(eval_all_clips, static_argnums=(0, 2, 3, 5))
    networks.eval()
    arrays = eval_jit(env, networks, n_clips, clip_length, key, flatten_obs)
    networks.train()

    # Move to Python scalars
    arrays = jax.tree.map(lambda x: x.tolist(), arrays)

    per_clip = []
    for i in range(n_clips):
        label = str(clips.clip_names[i]) if clips.clip_names is not None else f"clip_{i}"
        per_clip.append({
            "clip_idx": i,
            "label": label,
            "episode_reward": arrays["episode_reward"][i],
            "lifespan": arrays["lifespan"][i],
            "hand_L_pos_err_mean_m": arrays["hand_L_pos_err_mean_m"][i],
            "foot_L_pos_err_mean_m": arrays["foot_L_pos_err_mean_m"][i],
            "root_pos_err_mean_m": arrays["root_pos_err_mean_m"][i],
        })

    by_label = defaultdict(list)
    for c in per_clip:
        by_label[c["label"]].append(c)

    return {
        "aggregate": _aggregate(per_clip),
        "by_label": {label: _aggregate(clips) for label, clips in sorted(by_label.items())},
        "per_clip": per_clip,
    }


# ---------------------------------------------------------------------------
# Per-checkpoint benchmarking
# ---------------------------------------------------------------------------

def benchmark_checkpoint(ckpt_dir_rel: str) -> dict:
    ckpt_dir = REPO_ROOT / ckpt_dir_rel
    name = ckpt_dir.name
    print(f"\n=== {name} ===")

    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)
    env_params = cfg["env_params"]
    net_params = cfg["net_params"]
    network_class_str = net_params.get("network_class", "")
    clip_length = int(env_params.get("clip_length", 250))

    # Auto-detect step checkpoint directory (use the last/highest step)
    step_dirs = sorted(ckpt_dir.glob("step_*/"))
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directory in {ckpt_dir}")
    step_dir = step_dirs[-1]
    print(f"  Step dir: {step_dir.name}")

    # Build env config and reproduce the train/eval split
    env_cfg = parse_env_config(env_params)
    all_clips = ReferenceClips(
        env_cfg.reference_data_path,
        env_cfg.clip_length,
        env_cfg.keep_clips_idx,
    )
    train_clips, test_clips = all_clips.split()  # seed=0, ratio=0.8 — matches training

    print("  Initialising environment...")
    train_env = ModularImitation_v4(env_cfg, clips=train_clips)

    print("  Building network...")
    nets = build_network(net_params, train_env, rngs=nnx.Rngs(0))

    print(f"  Loading weights from {step_dir.name}...")
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
    print(f"  Loaded step {ckpt['step']}")

    n_actor, n_critic = get_param_counts(nets, network_class_str)
    print(f"  Actor params: {n_actor:,}   Critic params: {n_critic:,}")

    # MLPModularNetwork expects flat observations
    flatten_obs = "MLPModularNetwork" in network_class_str

    key = jax.random.key(42)

    print("  Evaluating train split...")
    key, subkey = jax.random.split(key)
    train_results = evaluate_split(train_env, nets, train_clips, clip_length, subkey, flatten_obs)

    print("  Evaluating eval split...")
    eval_env = ModularImitation_v4(env_cfg, clips=test_clips)
    key, subkey = jax.random.split(key)
    eval_results = evaluate_split(eval_env, nets, test_clips, clip_length, subkey, flatten_obs)

    print(
        f"  Train reward: {train_results['aggregate']['mean_episode_reward']:.2f}  "
        f"lifespan: {train_results['aggregate']['mean_lifespan']:.1f}"
    )
    print(
        f"  Eval  reward: {eval_results['aggregate']['mean_episode_reward']:.2f}  "
        f"lifespan: {eval_results['aggregate']['mean_lifespan']:.1f}"
    )

    return {
        "checkpoint_dir": str(ckpt_dir),
        "step": int(ckpt["step"]),
        "n_actor_params": n_actor,
        "n_critic_params": n_critic,
        "train": train_results,
        "eval": eval_results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    results = {}
    for ckpt_dir in CHECKPOINTS:
        name = Path(ckpt_dir).name
        results[name] = benchmark_checkpoint(ckpt_dir)

    out_path = REPO_ROOT / OUTPUT_FILE
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
