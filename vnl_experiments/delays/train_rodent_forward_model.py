
"""Train PPO on the rodent imitation task with an explicit forward model.

Run as::

    python -m vnl_experiments.delays.train_rodent_forward_model --delay 5

Like ``train_rodent_delays`` the encoder applies a variational bottleneck on the
``imitation_target`` and only the **proprioception** stream is affected by the
delay; the ``imitation_target`` stays un-delayed and the critic sees the full
un-delayed dict obs (privileged). The difference: instead of feeding the decoder
the delayed proprioception + efference copy directly, a learned ``ForwardModel``
predicts the *current* proprioception from the delayed proprioception + action
buffer, and the decoder consumes only ``[latent, predicted proprioception]``. A
self-supervised L2 loss (logged as ``fm_pred_mse``) trains the prediction toward
the true current proprioception. ``--delay 0`` keeps the forward model active as
an identity model (capacity diagnostic / baseline).
"""

import os


os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import dataclasses
import gc
import json
from datetime import datetime

import jax
import jax.numpy as jp
import wandb
from flax import nnx
from ml_collections import config_dict

from vnl_playground.tasks.rodent import consts
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
from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Sequential
from nnx_ppo.networks.utils import Filter, Flattener, Map
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.variational import VariationalBottleneck

from vnl_experiments.delays import evaluation
from vnl_experiments.delays.efference_copy import EfferenceCopy
from vnl_experiments.delays.forward_model import ForwardModel
from vnl_experiments.envs.absolute_imitation import AbsoluteImitation, default_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delay", type=int, default=5,
                   help="Proprioception delay in steps. 0 = identity model.")
    p.add_argument("--efference", type=int, default=None,
                   help="Efference-copy queue length. Defaults to --delay.")
    p.add_argument("--fm-loss-weight", type=float, default=1.0,
                   help="Weight on the self-supervised forward-model L2 loss.")
    p.add_argument("--detach-prediction", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Detach the prediction before the decoder (forward-model "
                        "behavior). Use --no-detach-prediction to train the "
                        "predictor with policy gradients (architecture ablation).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="nnx-ppo-rodent-delays")
    p.add_argument("--exp-name-suffix", default="")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Override the configured total_steps (smoke testing).")
    p.add_argument("--n-envs", type=int, default=None,
                   help="Override the configured n_envs (smoke testing).")
    p.add_argument("--no-video", dest="video", action="store_false",
                   help="Disable video recording (smoke testing).")
    p.add_argument("--no-final-eval", dest="final_eval", action="store_false",
                   help="Skip the end-of-training evaluation.")
    p.add_argument("--eval-limit-clips", type=int, default=None,
                   help="Cap clips per split in the final eval (memory knob).")
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
    # NOTE: body_target_frame is consumed by the *environment* (AbsoluteImitation),
    # not the network. It must live on env_config to take effect; setting it on
    # net_config is inert (only logged to WandB net_params, never read). See the
    # "reference-representation bug" note in analysis/README.md.
    env_config.body_target_frame = "current_root"
    env_config.torque_actuators = False
    env_config.walker_xml_path = consts.RODENT_NO_TAIL_COLLISION_XML
    
    net_config = config_dict.create(
        enc_hidden_sizes=[512] * 4,
        dec_hidden_sizes=[512] * 4,
        fm_hidden_sizes=[512] * 4,
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
            logging_level=(
                LoggingLevel.BASIC
                | LoggingLevel.THROUGHPUT
                | LoggingLevel.ENV_METRICS | LoggingLevel.NETWORK_METRICS
            ),
            logging_percentiles=(0, 25, 50, 75, 100),
        ),
        eval=EvalConfig(
            enabled=True,
            every_steps=10_000_000,
            n_envs=1024,
            max_episode_length=1000,
            logging_level=LoggingLevel.BASIC | LoggingLevel.ENV_METRICS | LoggingLevel.NETWORK_METRICS,
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
    if args.n_envs is not None:
        config = dataclasses.replace(
            config, ppo=dataclasses.replace(config.ppo, n_envs=args.n_envs))
    if not args.video:
        config = dataclasses.replace(
            config, video=dataclasses.replace(config.video, enabled=False))


    clips = ReferenceClips(env_config.reference_data_path,
                       env_config.clip_length,
                       env_config.keep_clips_idx)
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

    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0]))

    enc_sizes = (
        [task_obs_size]
        + list(net_config.enc_hidden_sizes)
        + [net_config.latent_size * 2]
    )
    # Decoder sees [latent, predicted proprioception] only — the forward model
    # has already consumed the efference copy.
    decoder_in = net_config.latent_size + proprio_size
    dec_sizes = (
        [decoder_in] + list(net_config.dec_hidden_sizes) + [action_size * 2]
    )
    # Predictor: [delayed proprioception, action buffer] -> current proprioception.
    predictor_sizes = (
        [proprio_size + efference_length * action_size]
        + list(net_config.fm_hidden_sizes)
        + [proprio_size]
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

    predictor = make_mlp(
        predictor_sizes, rngs, activation, activation_last_layer=False
    )

    actor = Sequential(
        [
            Map(
                task_obs=encoder_branch,
                proprioception=Flattener(),
            ),
            EfferenceCopy(
                inner=ForwardModel(
                    decoder=decoder,
                    predictor=predictor,
                    proprio_size=proprio_size,
                    delay_steps=args.delay,
                    loss_weight=args.fm_loss_weight,
                    detach_prediction=args.detach_prediction,
                ),
                sample_action=jp.zeros(action_size),
                queue_length=efference_length,
                inject_key="efference",
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
    # dict so downstream Map / Flattener see the simpler structure.
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
    detach_token = "" if args.detach_prediction else "_nodetach"
    exp_name = (
        f"RodentForwardModel_delay{args.delay}_eff{efference_length}"
        f"{detach_token}-{timestamp}{suffix}"
    )

    ckpt_dir = f"checkpoints/{exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    # One source of truth: config.json (for offline reconstruction by
    # eval_runs.py) and the end-of-training eval's metadata read the same dict.
    net_params = {
        **net_config.to_dict(),
        "delay_k": args.delay,
        "efference_length": efference_length,
        "fm_loss_weight": args.fm_loss_weight,
        "detach_prediction": args.detach_prediction,
        "network_class": "RodentForwardModel",
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as _f:
        json.dump({
            "env_params": env_config.to_dict(),
            "net_params": net_params,
        }, _f, indent=2, default=str)

    wandb.init(
        project=args.wandb_project,
        config={
            "env": "AbsoluteImitation",
            "delay_k": args.delay,
            "efference_length": efference_length,
            "fm_loss_weight": args.fm_loss_weight,
            "detach_prediction": args.detach_prediction,
            "seed": seed,
            "config": dataclasses.asdict(config),
            "net_params": net_config.to_dict(),
            "env_params": env_config.to_dict(),
        },
        name=exp_name,
        tags=("MLP", "warp", "ForwardModel", "TrainEvalSplit",
              f"delay{args.delay}", f"eff{efference_length}",
              *(() if args.detach_prediction else ("nodetach",))),
        notes="No-detach forward model with new XML.",
    )
    ckpt_fn = make_checkpoint_fn(ckpt_dir, config)
    result = ppo.train_ppo(
        train_env,
        nets,
        config,
        log_fn=wandb.log,
        video_fn=wandb_video_fn(fps=50),
        checkpoint_fn=ckpt_fn,
        eval_env=eval_env,
        **({} if args.total_steps is None else {"total_steps": args.total_steps}),
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

    # train_ppo only checkpoints on the checkpoint_every_steps grid and never
    # after the loop exits, so save the final weights explicitly. Without this
    # the end-of-training eval would measure a network that is not on disk and
    # that eval_runs.py could never reproduce.
    total_steps = result.total_steps
    ckpt_fn(result.training_state, total_steps)
    print(f"Saved final checkpoint at step {total_steps}")

    if args.final_eval:
        # Release the training state (4096 env states + optimizer moments)
        # before the eval allocates. `nets` is the same object as
        # result.training_state.networks, so the trained weights survive.
        del result
        jax.clear_caches()
        gc.collect()

        evaluation.run_final_eval(
            nets, AbsoluteImitation, env_config,
            ckpt_dir=ckpt_dir,
            wandb_id=wandb.run.id, wandb_name=exp_name, step=total_steps,
            net_params=net_params,
            train_env=train_env, train_clips=train_clips, test_clips=test_clips,
            seed=seed, limit_clips=args.eval_limit_clips,
            summary_fn=wandb.run.summary.update,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
