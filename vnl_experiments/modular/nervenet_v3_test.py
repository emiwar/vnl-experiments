"""Rodent imitation learning with modular architecture."""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from datetime import datetime
import json

import dataclasses
import jax
import jax.numpy as jp
from flax import nnx
import wandb
from ml_collections import config_dict

from vnl_playground.tasks.modular_rodent.imitation_v4 import ModularImitation_v4, default_config
from vnl_playground.tasks.reference_clips import ReferenceClips

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig, VideoConfig
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn

from vnl_experiments.networks.nervenet_style_v3 import NerveNetNetwork_v3

SEED = 40
env_config = default_config()
env_config.naconmax = 64*1024
env_config.njmax = 1024
env_config.torque_actuators = True
env_config.reward_terms["root_pos_scale"] = 0.05
env_config.reward_terms["limb_pos_exp_scale"] = 0.02
env_config.reward_terms["joint_exp_scale"] = 0.2
env_config.solver = "newton"
env_config.iterations = 50
env_config.ls_iterations = 50
env_config.sim_dt = 0.002
env_config.energy_cost = -0.04

net_config = config_dict.create(
    hidden_size=16,
    root_size=32,
    critic_scale=1.0,
    entropy_weight=1e-2,
    min_std=1e-1,
    motor_scale=1.0,
    normalize_obs=True,
    combine_likelihoods=True,
    detached_critic=True,
    detached_critic_hidden_sizes=[512, 512],
    activation="swish",
    reveal_targets="all",
)

config = TrainConfig(
    ppo=PPOConfig(
        n_envs=1024,
        rollout_length=20,
        total_steps=500_000_000,
        discounting_factor=0.95,
        normalize_advantages=True,
        combine_advantages=True,
        learning_rate=1e-4,
        n_epochs=4,
        n_minibatches=4,
        critic_loss_weight=0.05,
        gradient_clipping=1.0,
        weight_decay=None,
        logging_level=LoggingLevel.LOSSES | LoggingLevel.TRAIN_ROLLOUT_STATS | LoggingLevel.TRAINING_ENV_METRICS | LoggingLevel.CRITIC_EXTRA,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    eval=EvalConfig(
        enabled=True,
        every_steps=5_000_000,
        n_envs=512,
        max_episode_length=500,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    video=VideoConfig(
        enabled=True,
        every_steps=10_000_000,
        episode_length=2000,
        render_kwargs={
            "height": 480,
            "width": 640,
            "camera": "close_profile",
            "add_labels": True,
            "termination_extra_frames": 20,
        },
    ),
    seed=SEED,
    checkpoint_every_steps=50_000_000,
)

clips = ReferenceClips(env_config.reference_data_path,
                       env_config.clip_length,
                       env_config.keep_clips_idx)
train_clips, test_clips = clips.split()
train_env = ModularImitation_v4(env_config, clips=train_clips)
eval_env = ModularImitation_v4(env_config, clips=test_clips)

# Setup network
rngs = nnx.Rngs(SEED)

if net_config.reveal_targets == "all":
    obs_sizes = {k: jp.squeeze(jax.tree.reduce(jp.add, o)) for k, o in train_env.non_flattened_observation_size.items()}
else:
    obs_sizes = {k: jp.squeeze(jax.tree.reduce(jp.add, o["proprioception"])) for k, o in train_env.non_flattened_observation_size.items() if k != "root"}
    if net_config.reveal_targets == "root_only":
        obs_sizes["root"] = jp.squeeze(jax.tree.reduce(jp.add, train_env.non_flattened_observation_size["root"]))
    elif net_config.reveal_targets == "joystick_only":
        obs_sizes["root"] = 3
nets = NerveNetNetwork_v3(
    obs_sizes, train_env.action_size, rngs=rngs, **net_config
)

# Initialize wandb
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

exp_name = f"Imitation_detached_critic_v4-{timestamp}"
net_config["network_class"] = str(type(nets))
combined_config = {
        "env": str(type(train_env)),
        "SEED": SEED,
        "config": dataclasses.asdict(config),
        "net_params": net_config.to_dict(),
        "env_params": env_config.to_dict(),
    }
wandb.init(
    project="nnx-ppo-modular-rodent-imitation",
    config=combined_config,
    name=exp_name,
    tags=("NerveNet", "warp", "Modular", "train_test_split"),
    notes="Train-test split.",
)

checkpoint_dir = f"checkpoints/{exp_name}/"
os.makedirs(checkpoint_dir, exist_ok=True)
with open(f"{checkpoint_dir}config.json", "w") as f:
    json.dump(jax.tree.map(str, combined_config), f)

# Train with wandb callbacks
result = ppo.train_ppo(
    train_env,
    nets,
    config,
    log_fn=wandb.log,
    video_fn=wandb_video_fn(fps=50),
    checkpoint_fn=make_checkpoint_fn(checkpoint_dir, config),
    eval_env=eval_env,
)

print(
    f"Training complete: {result.total_steps} steps, {result.total_iterations} iterations"
)
print(f"Final eval reward: {result.eval_history[-1].get('episode_reward_mean', 'N/A')}")
