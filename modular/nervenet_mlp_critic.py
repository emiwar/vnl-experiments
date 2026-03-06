"""NerveNet actor + MLP multi-critic hybrid for modular rodent imitation.

The actor uses the full NerveNet message-passing (identical to independent_nervenet.py).
The critic uses a shared MLP encoder over the full concatenated normalized obs,
followed by per-module heads (same pattern as mlp_multi_critic.py).

This tests whether the NerveNet actor learns better with a stronger, centralized critic.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from datetime import datetime
import dataclasses
from typing import Optional
from collections.abc import Mapping

import jax
import jax.numpy as jp
from flax import nnx
import wandb
from ml_collections import config_dict

from vnl_playground.tasks.modular_rodent.imitation import ModularImitation, default_config

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.types import LoggingLevel, Transition
from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig, VideoConfig
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.factories import make_mlp
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.containers import Flattener


class NerveNetMLPCriticNetwork(PPONetwork, nnx.Module):
    """NerveNet actor + shared MLP encoder + per-module critic heads.

    Actor: identical message-passing to NerveNetNetwork (input → afferents → efferents → motor).
    Critic: shared encoder over the full concatenated *normalized* obs → per-module value heads.
            Same structure as MLPMultiCriticNetwork in mlp_multi_critic.py.

    The per-module Normalizer serves both paths — the critic encoder receives the
    already-normalized per-module obs concatenated into a single flat vector.

    Args:
        obs_sizes: Dict of {module_name: flat obs size}.
        action_sizes: Dict of {module_name: action_dim}.
        hidden_size: NerveNet hidden dim (all actor layers share this).
        critic_hidden_sizes: Hidden sizes for the shared critic MLP encoder.
            Last element is the feature dim fed into each critic head.
        reward_keys: Module names for critic heads (must match env reward dict keys).
        rngs: NNX RNG streams.
        entropy_weight, min_std, motor_scale: Actor sampler / output params.
        normalize_obs: Whether to use online per-module obs normalization.
        activation: Activation for all hidden layers.
        initializer_scale: Scale for variance_scaling kernel init (critic; NerveNet
            layers keep default Glorot init).
    """

    def __init__(
        self,
        obs_sizes: Mapping[str, int],
        action_sizes: Mapping[str, int],
        hidden_size: int,
        critic_hidden_sizes: list[int],
        reward_keys,
        rngs: nnx.Rngs,
        entropy_weight: float = 1e-2,
        min_std: float = 1e-1,
        motor_scale: float = 1.0,
        normalize_obs: bool = True,
        activation=nnx.swish,
        initializer_scale: float = 1.0,
    ):
        if isinstance(activation, str):
            activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[activation]

        all_modules = list(obs_sizes.keys())

        # --- NerveNet actor (identical to NerveNetNetwork, minus per-module critics) ---
        self.input_layers = nnx.Dict({k: Dense(os, hidden_size, rngs, activation) for k, os in obs_sizes.items()})
        self.afferents = nnx.Dict({k: Dense(hidden_size, hidden_size, rngs, activation) for k in all_modules})
        self.efferents = nnx.Dict({k: Dense(hidden_size, hidden_size, rngs, activation) for k in all_modules})
        self.motor_layers = nnx.Dict({k: Dense(hidden_size, 2 * a, rngs) for k, a in action_sizes.items()})
        self.action_samplers = nnx.Dict({
            k: NormalTanhSampler(rngs, entropy_weight, min_std)
            for k in action_sizes.keys()
        })
        self.motor_scale = motor_scale
        self.normalizer: Optional[Normalizer] = Normalizer(obs_sizes) if normalize_obs else None

        # --- MLP critic (same pattern as mlp_multi_critic.py) ---
        total_obs_size = sum(obs_sizes.values())
        kernel_init = nnx.initializers.variance_scaling(initializer_scale, "fan_in", "uniform")
        self.critic_encoder = make_mlp(
            [total_obs_size] + critic_hidden_sizes, rngs, activation,
            activation_last_layer=True, kernel_init=kernel_init,
        )
        critic_feature_size = critic_hidden_sizes[-1]
        self.critic_heads = nnx.Dict({
            k: Dense(critic_feature_size, 1, rngs, kernel_init=kernel_init)
            for k in reward_keys
        })

    def __call__(self, network_state, obs, raw_action=None):
        flattener = Flattener()

        # Flatten nested per-module obs
        x = {k: flattener((), xx).output for k, xx in obs.items()}

        # Per-module normalization (shared by actor and critic paths)
        if self.normalizer is not None:
            x = self.normalizer((), x).output
        x_norm = x  # save before NerveNet input layers

        # --- NerveNet actor path ---
        x = {k: self.input_layers[k]((), xx).output for k, xx in x.items()}

        # Afferents (leaf → parent)
        x["arm_L"] += self.afferents["hand_L"]((), x["hand_L"]).output
        x["arm_R"] += self.afferents["hand_R"]((), x["hand_R"]).output
        x["leg_L"] += self.afferents["foot_L"]((), x["foot_L"]).output
        x["leg_R"] += self.afferents["foot_R"]((), x["foot_R"]).output
        x["root"] += (
            self.afferents["arm_L"]((), x["arm_L"]).output
            + self.afferents["arm_R"]((), x["arm_R"]).output
            + self.afferents["leg_L"]((), x["leg_L"]).output
            + self.afferents["leg_R"]((), x["leg_R"]).output
            + self.afferents["torso"]((), x["torso"]).output
            + self.afferents["head"]((), x["head"]).output
        )

        # Efferents (root → leaf)
        x["head"] += self.efferents["head"]((), x["root"]).output
        x["torso"] += self.efferents["torso"]((), x["root"]).output
        x["arm_L"] += self.efferents["arm_L"]((), x["root"]).output
        x["hand_L"] += self.efferents["hand_L"]((), x["arm_L"]).output
        x["arm_R"] += self.efferents["arm_R"]((), x["root"]).output
        x["hand_R"] += self.efferents["hand_R"]((), x["arm_R"]).output
        x["leg_L"] += self.efferents["leg_L"]((), x["root"]).output
        x["foot_L"] += self.efferents["foot_L"]((), x["leg_L"]).output
        x["leg_R"] += self.efferents["leg_R"]((), x["root"]).output
        x["foot_R"] += self.efferents["foot_R"]((), x["leg_R"]).output

        # Motor layers → actions
        motor = {k: ml((), x[k]).output * self.motor_scale for k, ml in self.motor_layers.items()}
        actions, new_raw_actions, loglikelihoods = {}, {}, {}
        regularization_loss = jp.array(0.0)
        metrics = {}
        for k in self.action_samplers.keys():
            out = self.action_samplers[k]((), motor[k], raw_action[k] if raw_action is not None else None)
            actions[k], new_raw_actions[k], loglikelihoods[k] = out.output
            regularization_loss += out.regularization_loss
            metrics[k] = out.metrics
        loglikelihoods = sum(loglikelihoods.values())  # scalar joint log-prob

        # --- MLP critic path ---
        # Concatenate normalized per-module obs in insertion order
        critic_input = jp.concatenate([x_norm[k] for k in self.input_layers.keys()], axis=-1)
        critic_features = self.critic_encoder((), critic_input).output
        value_estimates = {
            k: jp.squeeze(self.critic_heads[k]((), critic_features).output, axis=-1)
            for k in self.critic_heads
        }

        return (), PPONetworkOutput(
            actions=actions,
            raw_actions=new_raw_actions,
            loglikelihoods=loglikelihoods,
            regularization_loss=regularization_loss,
            value_estimates=value_estimates,
            metrics=metrics,
        )

    def initialize_state(self, batch_size) -> tuple:
        return ()

    def update_statistics(self, last_rollout: Transition, total_steps) -> None:
        if self.normalizer is not None:
            flattener = Flattener()
            last_rollout = last_rollout.replace(
                obs={k: jax.vmap(lambda o: flattener((), o).output)(o)
                     for k, o in last_rollout.obs.items()}
            )
            self.normalizer.update_statistics(last_rollout, total_steps)


# ---------------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------------

SEED = 40
env_config = default_config()
env_config.naconmax = 64 * 1024
env_config.njmax = 1024
env_config.torque_actuators = True
env_config.reward_terms["limb_pos_exp_scale"] = 0.015
env_config.reward_terms["joint_exp_scale"] = 0.1
env_config.solver = "newton"
env_config.iterations = 50
env_config.ls_iterations = 50
env_config.sim_dt = 0.002

net_config = config_dict.create(
    hidden_size=32,
    critic_hidden_sizes=[1024] * 2,
    entropy_weight=1e-2,
    min_std=1e-1,
    motor_scale=1.0,
    normalize_obs=True,
    activation="swish",
    initializer_scale=1.0,
)

config = TrainConfig(
    ppo=PPOConfig(
        n_envs=2048,
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
        logging_level=LoggingLevel.LOSSES | LoggingLevel.TRAIN_ROLLOUT_STATS | LoggingLevel.TRAINING_ENV_METRICS,
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
        },
    ),
    seed=SEED,
    checkpoint_every_steps=50_000_000,
)

base_env = ModularImitation(env_config)
train_env = base_env
eval_env = train_env

obs_sizes = {
    k: int(jp.squeeze(jax.tree.reduce(jp.add, o)))
    for k, o in base_env.non_flattened_observation_size.items()
}
action_sizes = {k: int(v) for k, v in base_env.action_size.items()}

_sample_state = jax.jit(base_env.reset)(jax.random.key(0))
reward_keys = list(_sample_state.reward.keys())
del _sample_state

rngs = nnx.Rngs(SEED)
nets = NerveNetMLPCriticNetwork(
    obs_sizes=obs_sizes,
    action_sizes=action_sizes,
    reward_keys=reward_keys,
    rngs=rngs,
    **net_config,
)

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"NerveNetMLPCritic-{timestamp}"
wandb.init(
    project="nnx-ppo-modular-rodent-imitation",
    config={
        "env": "ModularImitation",
        "SEED": SEED,
        "config": dataclasses.asdict(config),
        "net_params": net_config.to_dict(),
        "env_params": env_config.to_dict(),
        "obs_sizes": obs_sizes,
        "action_sizes": action_sizes,
        "reward_keys": reward_keys,
    },
    name=exp_name,
    tags=("NerveNet", "MLPCritic", "warp", "Modular"),
    notes="NerveNet actor + MLP critic (encoder + heads).",
)

result = ppo.train_ppo(
    train_env,
    nets,
    config,
    log_fn=wandb.log,
    video_fn=wandb_video_fn(fps=50),
    eval_env=eval_env,
)

print(f"Training complete: {result.total_steps} steps, {result.total_iterations} iterations")
print(f"Final eval reward: {result.eval_history[-1].get('episode_reward_mean', 'N/A')}")
