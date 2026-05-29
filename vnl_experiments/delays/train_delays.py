"""Train PPO on a dm_control_suite env with actor-only delay + efference copy.

Run as::

    python -m vnl_experiments.delays.train_delays --env CartpoleBalance --delay 5

Logs to wandb in the style of ``nnx-ppo/examples/wandb_logging.py``. The
network is the asymmetric actor-only-delayed MLP from this folder; PPO
hyperparameters come from MuJoCo Playground's default brax_ppo_config for
the chosen env.
"""

import argparse
import dataclasses
from datetime import datetime

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

from vnl_experiments.delays import make_delayed_mlp_actor_critic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="CartpoleBalance",
                   help="dm_control_suite env name (e.g. CartpoleBalance).")
    p.add_argument("--delay", type=int, default=5,
                   help="Actor observation delay in steps. 0 = no delay.")
    p.add_argument("--efference", type=int, default=None,
                   help="Efference-copy queue length. Defaults to --delay.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--wandb-project", default="nnx-ppo-delays")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    efference_length = args.efference if args.efference is not None else args.delay

    env = mujoco_playground.registry.load(args.env)
    env = episode_wrapper.EpisodeWrapper(env, 1000)
    ppo_params = (
        mujoco_playground.config.dm_control_suite_params.brax_ppo_config(args.env)
    )
    ppo_params.num_evals = 100

    config = TrainConfig(
        ppo=PPOConfig(
            n_envs=8192,#ppo_params.num_envs*4,
            rollout_length=ppo_params.unroll_length,
            total_steps=ppo_params.num_timesteps,
            gae_lambda=0.95,
            discounting_factor=0.99,#ppo_params.discounting,
            clip_range=0.3,
            normalize_advantages=True,
            n_epochs=4,#ppo_params.num_updates_per_batch,
            learning_rate=1e-4,#ppo_params.learning_rate,
            n_minibatches=8,#ppo_params.num_minibatches,
            critic_loss_weight=1.0,#0.5
            logging_level=LoggingLevel.BASIC | LoggingLevel.TRAIN_ROLLOUT_STATS | LoggingLevel.CRITIC_EXTRA | LoggingLevel.ACTOR_EXTRA,
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
            render_kwargs={"height": 240, "width": 320},
        ),
        seed=args.seed,
    )

    train_env = reward_scaling_wrapper.RewardScalingWrapper(
        env, ppo_params.reward_scaling
    )
    eval_env = env
    rngs = nnx.Rngs(args.seed)
    nets = make_delayed_mlp_actor_critic(
        obs_size=train_env.observation_size,
        action_size=train_env.action_size,
        actor_hidden_sizes=[64] * 4,
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
    exp_name = f"{args.env}_delay{args.delay}_eff{efference_length}-{timestamp}"
    wandb.init(
        project=args.wandb_project,
        config={
            "env": args.env,
            "delay_k": args.delay,
            "efference_length": efference_length,
            "seed": args.seed,
            "config": dataclasses.asdict(config),
        },
        name=exp_name,
        tags=(args.env, f"delay{args.delay}", f"eff{efference_length}"),
        notes="Trying yet different training params."
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
            f"{result.eval_history[-1].get('episode_reward_mean', 'N/A')}"
        )


if __name__ == "__main__":
    main()
