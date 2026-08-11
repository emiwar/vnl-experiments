"""Train PPO on a dm_control_suite env with actor-only delay + efference copy.

Run as::

    python -m vnl_experiments.delays.train_delays --env CartpoleBalance --delay 5

Logs to wandb in the style of ``nnx-ppo/examples/wandb_logging.py``. ``--network``
selects the model: ``delayed_mlp`` (the asymmetric actor-only-delayed MLP from
this folder) or ``forward_model`` (an explicit predictor of the current obs from
the delayed obs + action buffer; the whole flat obs is treated as the
"proprioception", with no encoder/latent). PPO hyperparameters come from MuJoCo
Playground's default brax_ppo_config for the chosen env.
"""

import argparse
import dataclasses
from datetime import datetime

import jax
import mujoco_playground
import mujoco_playground.config.dm_control_suite_params
import wandb
from flax import nnx

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.algorithms.config import (
    EvalConfig,
    PPOConfig,
    TrainConfig,
    VideoConfig,
)
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.wrappers import reward_scaling_wrapper, episode_wrapper

from vnl_experiments.delays import (
    make_delayed_mlp_actor_critic,
    make_forward_model_actor_critic,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="CartpoleBalance",
                   help="dm_control_suite env name (e.g. CartpoleBalance).")
    p.add_argument("--network", choices=["delayed_mlp", "forward_model"],
                   default="delayed_mlp",
                   help="Network model: 'delayed_mlp' (delayed obs + efference "
                        "copy) or 'forward_model' (explicit predictor of the "
                        "current obs from delayed obs + action buffer).")
    p.add_argument("--delay", type=int, default=5,
                   help="Actor observation delay in steps. 0 = no delay.")
    p.add_argument("--efference", type=int, default=None,
                   help="Efference-copy queue length. Defaults to --delay.")
    p.add_argument("--fm-loss-weight", type=float, default=1.0,
                   help="Weight on the self-supervised forward-model L2 loss "
                        "(--network forward_model only).")
    p.add_argument("--detach-prediction", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Detach the prediction before the decoder (forward-model "
                        "behavior). Use --no-detach-prediction to train the "
                        "predictor with policy gradients (architecture ablation). "
                        "--network forward_model only.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--wandb-project", default="nnx-ppo-delays")
    p.add_argument("--notes", default="",
                   help="Free-text notes attached to the WandB run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    efference_length = args.efference if args.efference is not None else args.delay

    env = mujoco_playground.registry.load(args.env)

    ppo_params = (
        mujoco_playground.config.dm_control_suite_params.brax_ppo_config(args.env)
    )
    ppo_params.num_evals = 100

    config = TrainConfig(
        ppo=PPOConfig(
            n_envs=8192,#ppo_params.num_envs*4,
            rollout_length=ppo_params.unroll_length,
            total_steps=ppo_params.num_timesteps*8, #Delays might be slower to train
            gae_lambda=0.95,
            discounting_factor=0.99,#ppo_params.discounting,
            clip_range=0.3,
            normalize_advantages=True,
            n_epochs=4,#ppo_params.num_updates_per_batch,
            learning_rate=1e-4,#ppo_params.learning_rate,
            n_minibatches=8,#ppo_params.num_minibatches,
            critic_loss_weight=1.0,#0.5
            logging_level=LoggingLevel.BASIC | LoggingLevel.THROUGHPUT | LoggingLevel.ROLLOUT_STATS | LoggingLevel.CRITIC_EXTRA | LoggingLevel.ACTOR_EXTRA,
            logging_percentiles=None,
        ),
        eval=EvalConfig(
            enabled=True,
            every_steps=ppo_params.num_timesteps // ppo_params.num_evals,
            n_envs=256,
            max_episode_length=ppo_params.episode_length,
            logging_percentiles=None,
        ),
        video=VideoConfig(
            enabled=True,
            every_steps=ppo_params.num_timesteps // 10,
            episode_length=ppo_params.episode_length,
            render_kwargs={"height": 480, "width": 640},
        ),
        seed=args.seed,
    )
    
    train_env = episode_wrapper.EpisodeWrapper(env, 1000)
    train_env = reward_scaling_wrapper.RewardScalingWrapper(
        train_env, ppo_params.reward_scaling
    )
    eval_env = env
    rngs = nnx.Rngs(args.seed)
    forward = args.network == "forward_model"
    if forward:
        nets = make_forward_model_actor_critic(
            obs_size=train_env.observation_size,
            action_size=train_env.action_size,
            actor_hidden_sizes=[256] * 4,
            critic_hidden_sizes=[256] * 5,
            delay_k=args.delay,
            efference_length=efference_length,
            rngs=rngs,
            fm_loss_weight=args.fm_loss_weight,
            detach_prediction=args.detach_prediction,
            activation=nnx.swish,
            normalize_obs=ppo_params.normalize_observations,
            entropy_weight=ppo_params.entropy_cost,
            min_std=1e-3,
            std_scale=1.0,
        )
    else:
        nets = make_delayed_mlp_actor_critic(
            obs_size=train_env.observation_size,
            action_size=train_env.action_size,
            actor_hidden_sizes=[256] * 4,
            critic_hidden_sizes=[256] * 5,
            delay_k=args.delay,
            efference_length=efference_length,
            rngs=rngs,
            activation=nnx.swish,
            normalize_obs=ppo_params.normalize_observations,
            entropy_weight=ppo_params.entropy_cost,
            min_std=1e-3,
            std_scale=1.0,
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_token = "FM" if forward else "MLP"
    detach_token = "_nodetach" if (forward and not args.detach_prediction) else ""
    exp_name = (
        f"{args.env}_{model_token}_delay{args.delay}_eff{efference_length}"
        f"{detach_token}-{timestamp}"
    )

    model_tag = "ForwardModel" if forward else "DelayedMLP"
    tags = (
        args.env, model_tag, f"delay{args.delay}", f"eff{efference_length}",
        *(("nodetach",) if (forward and not args.detach_prediction) else ()),
    )

    # Count actor parameters (aids later analyses). nets is Sequential([...,
    # adapter]) when normalize_obs else the PPOAdapter itself; the adapter
    # exposes the action branch as `.action`.
    adapter = nets if hasattr(nets, "action") else nets[-1]
    n_actor_params = int(sum(jax.tree.leaves(
        jax.tree.map(lambda x: x.size, nnx.state(adapter.action, nnx.Param))
    )))

    wandb_config = {
        "env": args.env,
        "network": args.network,
        "delay_k": args.delay,
        "efference_length": efference_length,
        "n_actor_params": n_actor_params,
        "seed": args.seed,
        "config": dataclasses.asdict(config),
    }
    if forward:
        wandb_config["detach_prediction"] = args.detach_prediction
        wandb_config["fm_loss_weight"] = args.fm_loss_weight

    wandb.init(
        project=args.wandb_project,
        config=wandb_config,
        name=exp_name,
        tags=tags,
        notes=args.notes,
    )

    result = ppo.train_ppo(
        train_env,
        nets,
        config,
        log_fn=wandb.log,
        video_fn=wandb_video_fn(fps=int(round(1 / eval_env.dt))),
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
