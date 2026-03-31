"""Distillation training script: NerveNetNetwork_v3 teacher → NerveNetNetwork_v3 student.

Usage (from vnl-experiments root):
    ../.venv/bin/python vnl_experiments/distillation/train.py

To use a different teacher checkpoint, change TEACHER_CHECKPOINT below.
To use a smaller student, reduce the hidden_size / root_size in STUDENT_CONFIG.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import dataclasses
import json
import os
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
from vnl_experiments.networks.recurrent_modular import RecurrentModularNetwork
from vnl_experiments.networks.rnn_modular import RNNModularNetwork
from vnl_experiments.networks.mlp_modular import MLPModularNetwork
from vnl_experiments.tools.checkpoint_utils import load_network_from_checkpoint

# ---------------------------------------------------------------------------
# Configuration — edit these to change the experiment
# ---------------------------------------------------------------------------

SEED = 40

# Path to the teacher checkpoint (relative to vnl-experiments root).
TEACHER_CHECKPOINT = (
    "checkpoints/Imitation_detached_critic_v4-20260319-155439"
)

# Architecture for the student. Can differ from the teacher's config.
# Defaults to the same architecture used by the teacher checkpoint above.
STUDENT_CONFIG = config_dict.create(
    hidden_size=256,
    root_size=1024,
    #actor_hidden_sizes = [512, 512],
    #critic_hidden_sizes = [512, 512],
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

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Repo root: vnl_experiments/distillation/ → up 3 levels
REPO_ROOT = Path(__file__).parent.parent.parent

env_config = default_config()
env_config.naconmax = 64 * 1024
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

clips = ReferenceClips(
    env_config.reference_data_path,
    env_config.clip_length,
    env_config.keep_clips_idx,
)
train_clips, test_clips = clips.split()
train_env = ModularImitation_v4(env_config, clips=train_clips)
eval_env = ModularImitation_v4(env_config, clips=test_clips)

# ---------------------------------------------------------------------------
# Teacher — load from checkpoint
# ---------------------------------------------------------------------------

teacher = load_network_from_checkpoint(
    REPO_ROOT / TEACHER_CHECKPOINT,
    train_env,
    seed=SEED,
)

# ---------------------------------------------------------------------------
# Student — fresh network, same architecture by default
# ---------------------------------------------------------------------------
if STUDENT_CONFIG["reveal_targets"] == "all":
    obs_sizes = {
        k: jp.squeeze(jax.tree.reduce(jp.add, o))
        for k, o in train_env.non_flattened_observation_size.items()
    }
else:
    obs_sizes = {
        k: jp.squeeze(jax.tree.reduce(jp.add, o["proprioception"]))
        for k, o in train_env.non_flattened_observation_size.items() if k != "root"
    }
    if STUDENT_CONFIG["reveal_targets"] == "root_only":
        obs_sizes["root"] = jp.squeeze(jax.tree.reduce(jp.add, train_env.non_flattened_observation_size["root"]))


student = RNNModularNetwork(
    obs_sizes,
    train_env.action_size,
    rngs=nnx.Rngs(SEED + 1),
    **STUDENT_CONFIG,
)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

config = DistillationTrainConfig(
    distillation=DistillationConfig(
        n_envs=1024,
        rollout_length=20,
        total_steps=1_000_000_000,
        learning_rate=1e-4,
        n_epochs=4,
        n_minibatches=4,
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
        n_envs=256,
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
# Experiment name, WandB, and checkpoint directory
# ---------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
exp_name = f"Distillation-{timestamp}"

STUDENT_CONFIG["network_class"] = str(type(student))
combined_config = {
    "env": str(type(train_env)),
    "SEED": SEED,
    "teacher_checkpoint": str(TEACHER_CHECKPOINT),
    "teacher_class": str(type(teacher)),
    "config": dataclasses.asdict(config),
    "net_params": STUDENT_CONFIG.to_dict(),
    "env_params": env_config.to_dict(),
}

wandb.init(
    project="nnx-ppo-modular-rodent-imitation",
    config=combined_config,
    name=exp_name,
    tags=("Distillation", "NerveNet", "warp", "Modular", "masked_inputs"),
    notes="Distillation with new version of modular RNN.",
)

checkpoint_dir = REPO_ROOT / f"checkpoints/{exp_name}/"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
with open(checkpoint_dir / "config.json", "w") as f:
    json.dump(jax.tree.map(str, combined_config), f)


# ---------------------------------------------------------------------------
# Checkpoint callback for DistillationState
# ---------------------------------------------------------------------------

def make_distillation_checkpoint_fn(directory: Path):
    """Save the student's weights and distillation state to disk.

    Saves the same file layout as make_checkpoint_fn so that
    load_network_from_checkpoint can restore the student later.
    """
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
            "config": None,  # DistillationTrainConfig not yet supported by load_checkpoint
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
print(
    f"Final eval reward: {result.eval_history[-1].get('episode_reward_mean', 'N/A')}"
)
