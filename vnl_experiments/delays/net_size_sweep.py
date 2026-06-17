"""Train PPO on a dm_control_suite env with actor-only delay + efference copy.

Run as::

    python -m vnl_experiments.delays.train_delays --env CartpoleBalance --delay 5

Logs to wandb in the style of ``nnx-ppo/examples/wandb_logging.py``. The
network is the asymmetric actor-only-delayed MLP from this folder; PPO
hyperparameters come from MuJoCo Playground's default brax_ppo_config for
the chosen env.
"""

import argparse

import mujoco_playground
import mujoco_playground.config.dm_control_suite_params
import wandb
import jax
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

DELAYS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NET_SIZES = [
    [16]*2,
    [32]*2,
    [32]*4,
    [64]*4,
    [128]*4,
    [256]*4,
    [512]*4,
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="CartpoleSwingup",
                   help="dm_control_suite env name (e.g. CartpoleSwingup).")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    
    ppo_params = (
        mujoco_playground.config.dm_control_suite_params.brax_ppo_config(args.env)
    )
    env = mujoco_playground.registry.load(args.env)
    
    
    ppo_params.num_evals = 10

    config = TrainConfig(
        ppo=PPOConfig(
            n_envs=8192,#ppo_params.num_envs*4,
            rollout_length=ppo_params.unroll_length,
            total_steps=2*ppo_params.num_timesteps,
            gae_lambda=0.95,
            discounting_factor=0.99,#ppo_params.discounting,
            clip_range=0.3,
            normalize_advantages=True,
            n_epochs=4,#ppo_params.num_updates_per_batch,
            learning_rate=1e-4,#ppo_params.learning_rate,
            n_minibatches=8,#ppo_params.num_minibatches,
            critic_loss_weight=1.0,#0.5
            logging_level=LoggingLevel.NONE,
            logging_percentiles=None,
        ),
        eval=EvalConfig(
            enabled=True,
            every_steps=ppo_params.num_timesteps // ppo_params.num_evals,
            n_envs=512,
            max_episode_length=ppo_params.episode_length,
            logging_percentiles=None,
        ),
        video=VideoConfig(enabled=False),
        seed=args.seed,
    )

    train_env = episode_wrapper.EpisodeWrapper(env, ppo_params.episode_length)
    train_env = reward_scaling_wrapper.RewardScalingWrapper(
        train_env, ppo_params.reward_scaling
    )
    eval_env = env
    print(args.env)
    print(config)

    print("delay,net_size,n_actor_params,reward_mean,reward_std")
    for seed_offset in range(3):
        for delay in DELAYS:
            for net_size in NET_SIZES:            
                seed = args.seed + seed_offset
                config.seed = seed
                rngs = nnx.Rngs(seed)
                nets = make_delayed_mlp_actor_critic(
                    obs_size=train_env.observation_size,
                    action_size=train_env.action_size,
                    actor_hidden_sizes=net_size,
                    critic_hidden_sizes=[256] * 5,
                    delay_k=delay,
                    efference_length=delay,
                    rngs=rngs,
                    activation=nnx.swish,
                    normalize_obs=ppo_params.normalize_observations,
                    entropy_weight=ppo_params.entropy_cost,
                    min_std=1e-3,
                    std_scale=1.0,
                )

                result = ppo.train_ppo(
                    train_env,
                    nets,
                    config,
                    eval_env=eval_env,
                )
                final_eval = result.eval_history[-1]
                reward_mean  = final_eval["eval/episode_reward/mean"]
                reward_std  = final_eval["eval/episode_reward/std"]
                n_actor_params = sum(jax.tree.leaves(
                    jax.tree.map(lambda x: x.size, nnx.state(nets[-1].action, nnx.Param))
                ))
                print(delay,str(net_size),n_actor_params,reward_mean,reward_std,sep=",")
if __name__ == "__main__":
    main()
