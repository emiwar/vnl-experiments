#!/usr/bin/env python3
"""Checkpoint eval diagnostic.

Loads a checkpoint and runs eval_rollout to verify the weights loaded
correctly. Compare the printed metrics against WandB logs for the same step.

Usage:
    .venv/bin/python vnl_experiments/tools/eval_checkpoint.py \
        --checkpoint downloaded_checkpoints/my_run/step_0050012160 \
        --hidden_size 32 \
        [--n_envs 64] [--episode_length 500]
"""

import argparse
import os
import pickle

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state
from nnx_ppo.algorithms import rollout
from vnl_playground.tasks.modular_rodent.imitation import ModularImitation, default_config
from vnl_experiments.networks.nervenet_style import NerveNetNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval a VNL checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_envs", type=int, default=64)
    parser.add_argument("--episode_length", type=int, default=500)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Environment — match training config exactly
    # -----------------------------------------------------------------------
    config = default_config()
    config.naconmax = 64 * 1024
    config.njmax = 1024
    config.torque_actuators = True
    config.reward_terms["limb_pos_exp_scale"] = 0.015
    config.reward_terms["joint_exp_scale"] = 0.1
    config.solver = "newton"
    config.iterations = 50
    config.ls_iterations = 50
    config.sim_dt = 0.002

    print("Initialising environment…")
    env = ModularImitation(config)

    # -----------------------------------------------------------------------
    # Network
    # -----------------------------------------------------------------------
    obs_sizes = {
        k: jp.squeeze(jax.tree.reduce(jp.add, o))
        for k, o in env.non_flattened_observation_size.items()
    }
    network = NerveNetNetwork(
        obs_sizes, env.action_size, args.hidden_size, rngs=nnx.Rngs(0)
    )

    # -----------------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(os.path.join(args.checkpoint, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    ppo_cfg = meta["config"].ppo if meta.get("config") is not None else None

    training_state = new_training_state(
        env, network, n_envs=1, seed=0,
        learning_rate=ppo_cfg.learning_rate if ppo_cfg else 1e-4,
        gradient_clipping=ppo_cfg.gradient_clipping if ppo_cfg else 1.0,
        weight_decay=ppo_cfg.weight_decay if ppo_cfg else None,
    )
    ckpt = load_checkpoint(args.checkpoint, training_state.networks, training_state.optimizer)
    network = ckpt["training_state"].networks
    print(f"  Loaded step {ckpt['step']}")
    print(f"  Normalizer counter: {float(network.normalizer.counter.get_value()):.0f}")

    # -----------------------------------------------------------------------
    # Eval rollout — same as PPO training does it
    # -----------------------------------------------------------------------
    print(f"\nRunning eval rollout ({args.n_envs} envs, {args.episode_length} steps)…")
    eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3, 5))

    network.eval()
    metrics = eval_rollout_jit(
        env,
        network,
        args.n_envs,
        args.episode_length,
        jax.random.key(0),
        (0, 25, 50, 75, 100),
    )
    network.train()

    print("\n--- Eval metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {float(v):.4f}")


if __name__ == "__main__":
    main()