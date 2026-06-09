"""Train PPO on the rodent imitation task with actor-side proprioception
delay + efference copy.

Run as::

    python -m vnl_experiments.delays.train_rodent_delays --delay 5

Network shape mirrors ``vnl_experiments/non-modular/enc_dec.py`` (encoder
with variational bottleneck on ``imitation_target``, decoder on the latent
+ proprioception), but only the **proprioception** stream is delayed; the
``imitation_target`` is treated as an external reference and stays
un-delayed. The critic sees the full un-delayed dict obs (privileged).
Efference copy feeds the most recent ``efference_length`` actions into the
decoder input. ``--delay 0`` and ``--efference 0`` reproduce the baseline.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import dataclasses
import json
from datetime import datetime

import jax
import jax.numpy as jp
import wandb
from flax import nnx
from ml_collections import config_dict

from vnl_playground.tasks.rodent.imitation import Imitation, default_config

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
from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Concat, Sequential
from nnx_ppo.networks.utils import Filter, Flattener
from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.variational import VariationalBottleneck

from vnl_experiments.delays.efference_copy import EfferenceCopy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delay", type=int, default=5,
                   help="Proprioception delay in steps. 0 = no delay.")
    p.add_argument("--efference", type=int, default=None,
                   help="Efference-copy queue length. Defaults to --delay.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="nnx-ppo-rodent-delays")
    p.add_argument("--exp-name-suffix", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    efference_length = args.efference if args.efference is not None else args.delay
    seed = args.seed

    env_config = default_config()
    env_config.solver = "newton"
    env_config.reward_terms["bodies_pos"]["weight"] = 0.0
    env_config.reward_terms["joints_vel"]["weight"] = 0.0
    env_config.mujoco_impl = "warp"
    env_config.naconmax = 32 * 4096
    env_config.njmax = 256
    env_config.ctrl_dt = 0.01

    net_config = config_dict.create(
        enc_hidden_sizes=[512] * 4,
        dec_hidden_sizes=[512] * 4,
        critic_hidden_sizes=[1024] * 2,
        activation="swish",
        entropy_weight=1e-2,
        min_std=1e-1,
        std_scale=1.0,
        normalize_obs=True,
        initializer_scale=1.0,
        kl_weight=0.001,
        latent_min_std=0.01,
        latent_size=32,
        latent_ar1_weight=None,
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
            logging_level=LoggingLevel.BASIC,
            logging_percentiles=(0, 25, 50, 75, 100),
        ),
        eval=EvalConfig(
            enabled=True,
            every_steps=10_000_000,
            n_envs=1024,
            max_episode_length=1000,
            logging_percentiles=None,#(0, 25, 50, 75, 100),
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

    train_env = Imitation(env_config)
    eval_env = train_env
    obs_size = train_env.non_flattened_observation_size
    action_size = train_env.action_size

    rngs = nnx.Rngs(seed)
    activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
        net_config.activation
    ]

    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0]))

    enc_sizes = (
        [task_obs_size]
        + list(net_config.enc_hidden_sizes)
        + [net_config.latent_size * 2]
    )
    decoder_in = net_config.latent_size + proprio_size + efference_length * action_size
    dec_sizes = (
        [decoder_in] + list(net_config.dec_hidden_sizes) + [action_size * 2]
    )
    critic_sizes = (
        [task_obs_size + proprio_size]
        + list(net_config.critic_hidden_sizes)
        + [1]
    )

    encoder_branch = Sequential(
        [
            Flattener(),
            *make_mlp_layers(
                enc_sizes, rngs, activation, activation_last_layer=False,
            ),
            VariationalBottleneck(
                net_config.latent_size,
                rngs,
                net_config.kl_weight,
                net_config.latent_min_std,
            ),
        ]
    )

    proprio_branch_layers = [Flattener()]
    if args.delay > 0:
        proprio_branch_layers.append(
            Delay(jp.zeros(proprio_size), k_steps=args.delay)
        )
    proprio_branch = Sequential(proprio_branch_layers)

    decoder = Sequential(
        [
            *make_mlp_layers(
                dec_sizes, rngs, activation, activation_last_layer=False,
            ),
            NormalTanhSampler(
                rngs,
                entropy_weight=net_config.entropy_weight,
                min_std=net_config.min_std,
                std_scale=net_config.std_scale,
            ),
        ]
    )

    actor = Sequential(
        [
            Concat(
                task_obs=encoder_branch,
                proprioception=proprio_branch,
            ),
            EfferenceCopy(
                inner=decoder,
                sample_action=jp.zeros(action_size),
                queue_length=efference_length,
            ),
        ]
    )

    critic = Sequential(
        [
            Flattener(),
            *make_mlp_layers(
                critic_sizes, rngs, activation, activation_last_layer=False,
            ),
        ]
    )

    adapter = PPOAdapter(action=actor, value=critic)

    # The env wraps obs under a top-level "state" key: {state: {task_obs, proprioception}}.
    # Pre-flatten each inner leaf to 1D (preserve_levels=2 keeps state.<key>),
    # normalise per inner key, then lift to a flat {task_obs, proprioception}
    # dict so downstream Concat / Flattener see the simpler structure.
    pre = Flattener(preserve_levels=2)
    lift = Filter({
        "task_obs": ("state", "task_obs"),
        "proprioception": ("state", "proprioception"),
    })
    if net_config.normalize_obs:
        normalizer_shape = {
            "state": {
                "task_obs": task_obs_size,
                "proprioception": proprio_size,
            }
        }
        nets = Sequential([pre, Normalizer(normalizer_shape), lift, adapter])
    else:
        nets = Sequential([pre, lift, adapter])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{args.exp_name_suffix}" if args.exp_name_suffix else ""
    exp_name = (
        f"RodentEncDec_delay{args.delay}_eff{efference_length}"
        f"-{timestamp}{suffix}"
    )

    ckpt_dir = f"checkpoints/{exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as _f:
        json.dump({
            "env_params": env_config.to_dict(),
            "net_params": {
                **net_config.to_dict(),
                "delay_k": args.delay,
                "efference_length": efference_length,
                "network_class": "RodentEncDecDelays",
            },
        }, _f, indent=2, default=str)

    wandb.init(
        project=args.wandb_project,
        config={
            "env": "StandardImitation",
            "delay_k": args.delay,
            "efference_length": efference_length,
            "seed": seed,
            "config": dataclasses.asdict(config),
            "net_params": net_config.to_dict(),
            "env_params": env_config.to_dict(),
        },
        name=exp_name,
        tags=("MLP", "warp", "EncDec",
              f"delay{args.delay}", f"eff{efference_length}"),
        notes="More long delay runs.",
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
            f"{result.eval_history[-1].get('episode_reward_mean', 'N/A')}"
        )


if __name__ == "__main__":
    main()
