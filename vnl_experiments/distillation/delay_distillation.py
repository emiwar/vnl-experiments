"""Distillation with observation delay: teacher (clean) → student (k-step delayed obs).

Compares NerveNetNetwork_v3 and MLPModularNetwork under different delay values
using distillation from a pre-trained teacher checkpoint.

The obs delay wrapper stores a circular buffer in the student's carry state.
During rollout and loss replay the delay is handled identically to an LSTM's
hidden state — carried across steps, reset on episode end, replayed over the
full time sequence per minibatch. One physics simulation is used throughout.

Top-level switches
------------------
NETWORK : "nervenet" or "mlp"
DELAY   : int >= 0; 0 = no-delay baseline, student is unwrapped
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import dataclasses
import json
import pickle
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import orbax.checkpoint as ocp
import wandb
from flax import nnx
from ml_collections import config_dict

from vnl_playground.tasks.modular_rodent.imitation_v4 import ModularImitation_v4, default_config
from vnl_playground.tasks.reference_clips import ReferenceClips

from nnx_ppo.algorithms.distillation import train_distillation
from nnx_ppo.algorithms.config import (
    DistillationConfig,
    DistillationTrainConfig,
    EvalConfig,
    VideoConfig,
)
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.algorithms.checkpointing import _split_net_state
from nnx_ppo.algorithms.types import LoggingLevel

from vnl_experiments.networks.nervenet_style_v3 import NerveNetNetwork_v3
from vnl_experiments.networks.mlp_modular import MLPModularNetwork
from vnl_experiments.wrappers.obs_delay import DelayedObsNetwork
from vnl_experiments.tools.checkpoint_utils import load_network_from_checkpoint

# ---------------------------------------------------------------------------
# Top-level switches — edit these to change the experiment
# ---------------------------------------------------------------------------

NETWORK = "nervenet"   # "nervenet" or "mlp"
DELAY   = 2            # observation delay in steps; 0 = no delay (baseline)

SEED = 40

TEACHER_CHECKPOINT = "checkpoints/MLPModular-20260507-042541"

# ---------------------------------------------------------------------------
# Shared student config — params common to both network types
# ---------------------------------------------------------------------------

student_config = config_dict.create(
    entropy_weight=1e-2,
    min_std=1e-1,
    normalize_obs=True,
    activation="swish",
    reveal_targets="all",
)

if NETWORK == "nervenet":
    student_config.hidden_size = 256
    student_config.root_size = 256
    student_config.critic_scale = 1.0
    student_config.motor_scale = 1.0
    student_config.combine_likelihoods = True
    student_config.detached_critic = True
    student_config.detached_critic_hidden_sizes = [512, 512]
elif NETWORK == "mlp":
    student_config.actor_hidden_sizes = [1024, 1024]
    student_config.critic_hidden_sizes = [512, 512]
else:
    raise ValueError(f"Unknown NETWORK: {NETWORK!r}")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent

env_config = default_config()
env_config.naconmax = 64 * 2048
env_config.njmax = 1024
env_config.torque_actuators = True
env_config.reward_terms["root_pos_scale"] = 0.05
env_config.reward_terms["limb_pos_exp_scale"] = 0.02
env_config.reward_terms["joint_exp_scale"] = 0.2
env_config.solver = "newton"
env_config.iterations = 50
env_config.ls_iterations = 50
env_config.sim_dt = 0.001
env_config.ctrl_dt = 0.002
env_config.energy_cost = -0.04

clips = ReferenceClips(
    env_config.reference_data_path,
    env_config.clip_length,
    env_config.keep_clips_idx,
)
train_clips, test_clips = clips.split()
train_env = ModularImitation_v4(env_config, clips=train_clips)
eval_env  = ModularImitation_v4(env_config, clips=test_clips)

# ---------------------------------------------------------------------------
# Teacher — load from checkpoint
# ---------------------------------------------------------------------------

teacher = load_network_from_checkpoint(
    REPO_ROOT / TEACHER_CHECKPOINT,
    train_env,
    seed=SEED,
)

# ---------------------------------------------------------------------------
# Student — fresh network, optionally wrapped with obs delay
# ---------------------------------------------------------------------------

if student_config.reveal_targets == "all":
    obs_sizes = {
        k: jp.squeeze(jax.tree.reduce(jp.add, o))
        for k, o in train_env.non_flattened_observation_size.items()
    }
else:
    obs_sizes = {
        k: jp.squeeze(jax.tree.reduce(jp.add, o["proprioception"]))
        for k, o in train_env.non_flattened_observation_size.items() if k != "root"
    }
    if student_config.reveal_targets == "root_only":
        obs_sizes["root"] = jp.squeeze(
            jax.tree.reduce(jp.add, train_env.non_flattened_observation_size["root"])
        )
    elif student_config.reveal_targets == "joystick_only":
        obs_sizes["root"] = 3

if NETWORK == "nervenet":
    base_student = NerveNetNetwork_v3(
        obs_sizes, train_env.action_size, rngs=nnx.Rngs(SEED + 1), **student_config
    )
elif NETWORK == "mlp":
    base_student = MLPModularNetwork(
        obs_sizes, train_env.action_size, rngs=nnx.Rngs(SEED + 1), **student_config
    )

if DELAY > 0:
    sample_obs = jax.jit(train_env.reset)(jax.random.key(0)).obs
    student = DelayedObsNetwork(base_student, delay_k=DELAY, sample_obs=sample_obs)
else:
    student = base_student

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

config = DistillationTrainConfig(
    distillation=DistillationConfig(
        n_envs=2048,
        rollout_length=20,
        total_steps=100_000_000,
        learning_rate=1e-4,
        n_epochs=4,
        n_minibatches=8,
        gradient_clipping=1.0,
        weight_decay=None,
        logging_level=(
            LoggingLevel.LOSSES
            | LoggingLevel.TRAIN_ROLLOUT_STATS
            | LoggingLevel.TRAINING_ENV_METRICS
        ),
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    eval=EvalConfig(
        enabled=True,
        every_steps=2_500_000,
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

# ---------------------------------------------------------------------------
# Experiment name, WandB, checkpoint directory
# ---------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
exp_name  = f"DelayDistillation_{NETWORK}_delay{DELAY}-{timestamp}"

student_config["network_class"] = str(type(student))
combined_config = {
    "env": str(type(train_env)),
    "seed": SEED,
    "network": NETWORK,
    "delay": DELAY,
    "teacher_checkpoint": str(TEACHER_CHECKPOINT),
    "teacher_class": str(type(teacher)),
    "config": dataclasses.asdict(config),
    "net_params": student_config.to_dict(),
    "env_params": env_config.to_dict(),
}

wandb.init(
    project="nnx-ppo-modular-rodent-imitation",
    config=combined_config,
    name=exp_name,
    tags=("DelayDistillation", NETWORK, f"delay{DELAY}", "warp", "Modular"),
    notes=f"Distillation with {DELAY}-step obs delay. Network: {NETWORK}.",
)

checkpoint_dir = REPO_ROOT / f"checkpoints/{exp_name}/"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
with open(checkpoint_dir / "config.json", "w") as f:
    json.dump(jax.tree.map(str, combined_config), f)

# ---------------------------------------------------------------------------
# Checkpoint callback (reused from train.py)
# ---------------------------------------------------------------------------

def make_distillation_checkpoint_fn(directory: Path):
    abs_directory = directory.resolve()

    def checkpoint_fn(distillation_state, step: int) -> None:
        step_dir = abs_directory / f"step_{step:010d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        student_net = distillation_state.student
        non_key_state, rng_key_state, _ = _split_net_state(student_net)
        _, opt_state = nnx.split(distillation_state.optimizer)

        checkpointer = ocp.StandardCheckpointer()
        try:
            checkpointer.save(str(step_dir / "networks"), non_key_state)
            checkpointer.save(str(step_dir / "optimizer"), opt_state)
        finally:
            checkpointer.close()

        metadata = {
            "networks_rng_key_state": rng_key_state,
            "network_states": distillation_state.student_states,
            "env_states": distillation_state.env_states,
            "rng_key": distillation_state.rng_key,
            "steps_taken": distillation_state.steps_taken,
            "step": step,
            "config": None,
        }
        with open(step_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    return checkpoint_fn

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

result = train_distillation(
    train_env,
    teacher,
    student,
    config,
    log_fn=wandb.log,
    video_fn=wandb_video_fn(fps=50),
    checkpoint_fn=make_distillation_checkpoint_fn(checkpoint_dir),
    eval_env=eval_env,
)

print(
    f"Training complete: {result.total_steps} steps, "
    f"{result.total_iterations} iterations"
)
print(f"Final eval reward: {result.eval_history[-1].get('episode_reward_mean', 'N/A')}")
