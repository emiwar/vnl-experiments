"""JAX profiler script for the NerveNet training pipeline.

Runs a few warm-up PPO steps (to trigger JIT compilation), then profiles
subsequent steps with jax.profiler.trace().

Warp kernels appear as custom_call ops in the trace — timing is accurate
but labels are opaque. View results with TensorBoard or Perfetto.

Usage:
    python profile_nervenet.py

View trace:
    tensorboard --logdir /tmp/jax_profile
    # or drag the .json.gz from LOG_DIR into https://ui.perfetto.dev
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Disable XLA CUDA command buffers (CUDA graphs). XLA normally compiles JIT
# functions as CUDA graphs for efficiency, but jax.profiler.trace() is
# incompatible with CUDA graph parameter updates and causes CUDA_ERROR_LAUNCH_FAILED.
# Regular kernel launches are slower but work correctly with the profiler.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
).strip()

# --- Profiling config ---
N_WARMUP = 3       # iterations before profiling (triggers JIT compilation)
N_PROFILE = 2      # iterations to capture in the trace
LOG_DIR = "/tmp/jax_profile"
N_ENVS = 512      # match training; reduce (e.g. 32) for faster warm-up

import jax
import jax.numpy as jp
from flax import nnx
from ml_collections import config_dict

from vnl_playground.tasks.modular_rodent.imitation_v2 import ModularImitation_v2, default_config
from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import PPOConfig
from vnl_experiments.networks.nervenet_style import NerveNetNetwork

# --- Same config as independent_nervenet.py ---
SEED = 40
env_config = default_config()
env_config.naconmax = 4096
env_config.njmax = 1024
env_config.torque_actuators = True
env_config.reward_terms["root_pos_scale"] = 0.05
env_config.reward_terms["limb_pos_exp_scale"] = 0.015
env_config.reward_terms["joint_exp_scale"] = 0.1
env_config.solver = "newton"
env_config.iterations = 50
env_config.ls_iterations = 50
env_config.sim_dt = 0.002

ppo_config = PPOConfig(
    n_envs=N_ENVS,
    rollout_length=20,
    total_steps=500_000_000,
    discounting_factor=0.95,
    normalize_advantages=True,
    combine_advantages=True,
    learning_rate=1e-4,
    n_epochs=4,
    n_minibatches=8,
    gradient_clipping=1.0,
    weight_decay=None,
    logging_level=LoggingLevel.LOSSES,
)

net_config = config_dict.create(
    hidden_size=512,
    entropy_weight=1e-2,
    min_std=1e-1,
    motor_scale=1.0,
    normalize_obs=True,
    activation="tanh",
)

# --- Setup ---
print("Setting up environment and network...")
env = ModularImitation_v2(env_config)

rngs = nnx.Rngs(SEED)
obs_sizes = {k: jp.squeeze(jax.tree.reduce(jp.add, o)) for k, o in env.non_flattened_observation_size.items()}
nets = NerveNetNetwork(obs_sizes, env.action_size, rngs=rngs, **net_config)

print(f"Initializing training state with {N_ENVS} envs...")
training_state = ppo.new_training_state(
    env,
    nets,
    ppo_config.n_envs,
    SEED,
    ppo_config.learning_rate,
    ppo_config.gradient_clipping,
    ppo_config.weight_decay,
)

# JIT-compile ppo_step with same static_argnums as train_ppo()
ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 6, 7, 8, 9, 10, 11, 12))

def run_step(state):
    return ppo_step_jit(
        env,
        state,
        ppo_config.n_envs,
        ppo_config.rollout_length,
        ppo_config.gae_lambda,
        ppo_config.discounting_factor,
        ppo_config.clip_range,
        ppo_config.normalize_advantages,
        ppo_config.combine_advantages,
        ppo_config.n_epochs,
        ppo_config.n_minibatches,
        ppo_config.logging_level,
        ppo_config.logging_percentiles,
    )

# --- Warm-up (triggers JIT compilation) ---
print(f"Running {N_WARMUP} warm-up iterations (JIT compilation may take several minutes)...")
for i in range(N_WARMUP):
    print(f"  Warm-up step {i + 1}/{N_WARMUP}")
    training_state, metrics = run_step(training_state)

# Flush all pending computation before starting the trace
print("Flushing computation before profiling...")
jax.block_until_ready(training_state)

# --- Profile ---
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Profiling {N_PROFILE} iterations. Trace will be written to: {LOG_DIR}")
with jax.profiler.trace(LOG_DIR):
    for i in range(N_PROFILE):
        print(f"  Profile step {i + 1}/{N_PROFILE}")
        training_state, metrics = run_step(training_state)
    jax.block_until_ready(training_state)

print(f"\nDone. View trace with:")
print(f"  tensorboard --logdir {LOG_DIR}")
print(f"  (then open http://localhost:6006 and go to the Profile tab)")
print(f"  or drag the .json.gz file from {LOG_DIR} into https://ui.perfetto.dev")
