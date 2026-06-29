"""Train PPO on the rodent imitation task with the CBGTC network.

Run as::

    python -m vnl_experiments.cbgtc_net.train_rodent_cbgtc

Uses the :class:`AbsoluteImitation` env (absolute, egocentric targets) so the
reference target does not leak instantaneous proprioceptive state through the
CBGTC graph's intrinsic one-step recurrence. The network is built by
:func:`vnl_experiments.cbgtc_net.cbgtc_net.build_cbgtc_net`: proprioception
enters at the spinal cord, the reference target (``task_obs``) at the thalamus,
and a privileged flat-MLP critic sees the full obs.
"""

import os


os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import dataclasses
import json
from datetime import datetime

import jax
import wandb
from flax import nnx
from ml_collections import config_dict

from vnl_playground.tasks.reference_clips import ReferenceClips

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn
from nnx_ppo.algorithms.config import (
    EvalConfig,
    PPOConfig,
    TrainConfig,
    VideoConfig,
)
from nnx_ppo.algorithms.types import LoggingLevel

from vnl_experiments.cbgtc_net.cbgtc_net import build_cbgtc_net
from vnl_experiments.envs.absolute_imitation import AbsoluteImitation, default_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="nnx-ppo-rodent-cbgtc")
    p.add_argument("--exp-name-suffix", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed

    env_config = default_config()
    env_config.solver = "newton"
    env_config.reward_terms["bodies_pos"]["weight"] = 0.0
    env_config.reward_terms["joints_vel"]["weight"] = 0.0
    env_config.mujoco_impl = "warp"
    env_config.naconmax = 32 * 4096
    env_config.njmax = 256
    env_config.ctrl_dt = 0.01
    env_config.body_target_frame = "reference_root"

    net_config = config_dict.create(
        base_size=256,
        critic_hidden_sizes=[1024] * 2,
        activation="swish",
        entropy_weight=1e-2,
        min_std=1e-1,
        std_scale=1.0,
    )

    config = TrainConfig(
        ppo=PPOConfig(
            n_envs=4096,
            rollout_length=20,
            total_steps=600_000_000,
            discounting_factor=0.95,
            normalize_advantages=True,
            learning_rate=1e-4,
            n_epochs=4,
            n_minibatches=8,
            gradient_clipping=1.0,
            weight_decay=None,
            logging_level=LoggingLevel.BASIC | LoggingLevel.THROUGHPUT,
            logging_percentiles=(0, 25, 50, 75, 100),
        ),
        eval=EvalConfig(
            enabled=True,
            every_steps=10_000_000,
            n_envs=1024,
            max_episode_length=1000,
            logging_percentiles=None,
        ),
        video=VideoConfig(
            enabled=True,
            every_steps=50_000_000,
            episode_length=1000,
            render_kwargs={
                "height": 480,
                "width": 640,
                "camera": "close_profile-rodent",
                "add_labels": True,
            },
        ),
        seed=seed,
        checkpoint_every_steps=50_000_000,
    )

    clips = ReferenceClips(
        env_config.reference_data_path,
        env_config.clip_length,
        env_config.keep_clips_idx,
    )
    train_clips, test_clips = clips.split()
    train_env = AbsoluteImitation(env_config, clips=train_clips)
    eval_env = AbsoluteImitation(env_config, clips=test_clips)

    eval_env = train_env
    obs_size = train_env.non_flattened_observation_size
    action_size = train_env.action_size

    rngs = nnx.Rngs(seed)
    activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
        net_config.activation
    ]

    nets = build_cbgtc_net(
        obs_size,
        action_size,
        net_config.base_size,
        list(net_config.critic_hidden_sizes),
        activation,
        net_config.entropy_weight,
        rngs,
        min_std=net_config.min_std,
        std_scale=net_config.std_scale,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{args.exp_name_suffix}" if args.exp_name_suffix else ""
    exp_name = f"CBGTC-{timestamp}{suffix}"

    ckpt_dir = f"checkpoints/{exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as _f:
        json.dump(
            {
                "env_params": env_config.to_dict(),
                "net_params": {
                    **net_config.to_dict(),
                    "network_class": "CBGTC",
                },
            },
            _f,
            indent=2,
            default=str,
        )

    wandb.init(
        project=args.wandb_project,
        config={
            "env": "AbsoluteImitation",
            "seed": seed,
            "config": dataclasses.asdict(config),
            "net_params": net_config.to_dict(),
            "env_params": env_config.to_dict(),
        },
        name=exp_name,
        tags=("CBGTC", "warp", "PopulationGraph", "TrainEvalSplit"),
        notes="Cortico-basal ganglia-thalamo-cortical network",
    )
    result = ppo.train_ppo(
        train_env,
        nets,
        config,
        log_fn=wandb.log,
        video_fn=wandb_video_fn(fps=50),
        checkpoint_fn=make_checkpoint_fn(ckpt_dir, config),
        eval_env=eval_env,
    )

    print(
        f"Training complete: {result.total_steps} steps, "
        f"{result.total_iterations} iterations"
    )
    if result.eval_history:
        print(
            "Final eval reward: "
            f"{result.eval_history[-1].get('eval/episode_reward/mean', 'N/A')}"
        )


if __name__ == "__main__":
    main()
