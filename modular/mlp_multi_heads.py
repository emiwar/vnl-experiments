"""MLP actor + critic with per-module heads for modular rodent imitation.

Both actor and critic use a shared encoder over the full flat observation,
followed by per-module heads. The actor produces per-module action distributions;
the critic produces per-module value estimates for multi-objective PPO.
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
import jax.flatten_util
import jax.numpy as jp
from flax import nnx
import wandb
from ml_collections import config_dict

from vnl_playground.tasks.modular_rodent.imitation import ModularImitation, default_config

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.types import LoggingLevel, RLEnv, EnvState, Transition
from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig, VideoConfig
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.factories import make_mlp
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.containers import Flattener


# ---------------------------------------------------------------------------
# Wrapper: same as mlp_multi_critic — flat obs, flat action, reward dict intact
# ---------------------------------------------------------------------------


class FlatObsMultiRewardWrapper(RLEnv):
    """Flattens obs and unwraps flat action → dict action; keeps reward as dict."""

    def __init__(self, base_env: RLEnv) -> None:
        self.base_env = base_env
        null_action = self.base_env.null_action()
        _, self.unwrap_action = jax.flatten_util.ravel_pytree(null_action)

    def reset(self, rng) -> EnvState:
        return self._flatten_obs(self.base_env.reset(rng))

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        return self._flatten_obs(self.base_env.step(state, self.unwrap_action(action)))

    def _flatten_obs(self, state: EnvState) -> EnvState:
        new_obs, _ = jax.flatten_util.ravel_pytree(state.obs)
        return state.replace(obs=new_obs)

    @property
    def observation_size(self) -> jax.Array:
        return jax.tree.reduce(jp.add, self.base_env.observation_size)

    @property
    def action_size(self) -> jax.Array:
        return jax.tree.reduce(jp.add, self.base_env.action_size)

    def render(self, trajectory, **kwargs):
        return self.base_env.render(trajectory, **kwargs)


# ---------------------------------------------------------------------------
# Network: shared encoder → per-module heads, for both actor and critic
# ---------------------------------------------------------------------------


class MLPModularNetwork(PPONetwork, nnx.Module):
    """Shared-encoder MLP with per-module actor and critic heads.

    Both actor and critic follow the same pattern:
      shared encoder (full flat obs → features) → per-module linear heads.

    Actor heads output 2*action_size[k] (mean + log-std) per action module.
    Critic heads output 1 value estimate per reward module.
    Loglikelihoods are summed across modules to a scalar for combine_advantages.

    Args:
        obs_size: Total flat observation size.
        action_sizes: Dict of {module_name: action_dim}.
        reward_keys: Iterable of reward/module names for critic heads.
        actor_hidden_sizes: Hidden layer sizes for the shared actor encoder.
            The last element is the feature dim fed into each actor head.
        critic_hidden_sizes: Hidden layer sizes for the shared critic encoder.
            The last element is the feature dim fed into each critic head.
        rngs: NNX RNG streams.
        activation: Activation function for hidden layers.
        normalize_obs: Whether to online-normalize observations.
        entropy_weight, min_std, std_scale: Action sampler parameters.
        initializer_scale: Scale for variance_scaling kernel init.
    """

    def __init__(
        self,
        obs_size: int,
        action_sizes: Mapping[str, int],
        reward_keys,
        actor_hidden_sizes: list[int],
        critic_hidden_sizes: list[int],
        rngs: nnx.Rngs,
        activation=nnx.swish,
        normalize_obs: bool = True,
        entropy_weight: float = 1e-2,
        min_std: float = 1e-1,
        std_scale: float = 1.0,
        initializer_scale: float = 1.0,
    ):
        if isinstance(activation, str):
            activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[activation]

        kernel_init = nnx.initializers.variance_scaling(initializer_scale, "fan_in", "uniform")

        # Actor: shared encoder + per-module heads
        self.actor_encoder = make_mlp(
            [obs_size] + actor_hidden_sizes, rngs, activation,
            activation_last_layer=True, kernel_init=kernel_init,
        )
        actor_feature_size = actor_hidden_sizes[-1]
        self.actor_heads = nnx.Dict({
            k: Dense(actor_feature_size, 2 * action_dim, rngs, kernel_init=kernel_init)
            for k, action_dim in action_sizes.items()
        })
        self.action_samplers = nnx.Dict({
            k: NormalTanhSampler(rngs, entropy_weight, min_std, std_scale)
            for k in action_sizes.keys()
        })

        # Critic: shared encoder + per-module heads (same pattern as mlp_multi_critic)
        self.critic_encoder = make_mlp(
            [obs_size] + critic_hidden_sizes, rngs, activation,
            activation_last_layer=True, kernel_init=kernel_init,
        )
        critic_feature_size = critic_hidden_sizes[-1]
        self.critic_heads = nnx.Dict({
            k: Dense(critic_feature_size, 1, rngs, kernel_init=kernel_init)
            for k in reward_keys
        })

        self.preprocessor: Optional[Normalizer] = Normalizer(obs_size) if normalize_obs else None

    def __call__(
        self,
        network_state: dict,
        obs: jax.Array,
        raw_action: Optional[Mapping[str, jax.Array]] = None,
    ) -> tuple[dict, PPONetworkOutput]:
        regularization_loss = jp.array(0.0)

        # Obs normalization
        if self.preprocessor is not None:
            prep_out = self.preprocessor(network_state["preprocessor"], obs)
            obs_proc = prep_out.output
            network_state = {**network_state, "preprocessor": prep_out.next_state}
        else:
            obs_proc = obs

        # Shared actor encoder
        enc_out = self.actor_encoder(network_state["actor_encoder"], obs_proc)
        network_state = {**network_state, "actor_encoder": enc_out.next_state}
        actor_features = enc_out.output

        # Per-module actor heads + samplers
        actions = {}
        new_raw_actions = {}
        loglikelihoods = {}
        new_head_states = {}
        new_sampler_states = {}
        for k, head in self.actor_heads.items():
            head_out = head(network_state["actor_heads"][k], actor_features)
            new_head_states[k] = head_out.next_state
            raw_a = raw_action[k] if raw_action is not None else None
            sampler_out = self.action_samplers[k](
                network_state["action_samplers"][k], head_out.output, raw_a
            )
            actions[k], new_raw_actions[k], loglikelihoods[k] = sampler_out.output
            new_sampler_states[k] = sampler_out.next_state
            regularization_loss = regularization_loss + sampler_out.regularization_loss
        network_state = {
            **network_state,
            "actor_heads": new_head_states,
            "action_samplers": new_sampler_states,
        }

        # Sum loglikelihoods → scalar joint log-prob
        loglikelihood_scalar = sum(loglikelihoods.values())

        # Shared critic encoder
        enc_out = self.critic_encoder(network_state["critic_encoder"], obs_proc)
        network_state = {**network_state, "critic_encoder": enc_out.next_state}
        critic_features = enc_out.output

        # Per-module critic heads
        value_estimates = {}
        new_critic_head_states = {}
        for k, head in self.critic_heads.items():
            head_out = head(network_state["critic_heads"][k], critic_features)
            value_estimates[k] = jp.squeeze(head_out.output, axis=-1)
            new_critic_head_states[k] = head_out.next_state
        network_state = {**network_state, "critic_heads": new_critic_head_states}

        return network_state, PPONetworkOutput(
            actions=actions,
            raw_actions=new_raw_actions,
            loglikelihoods=loglikelihood_scalar,
            regularization_loss=regularization_loss,
            value_estimates=value_estimates,
            metrics={},
        )

    def initialize_state(self, batch_size: int) -> dict:
        state = {
            "actor_encoder": self.actor_encoder.initialize_state(batch_size),
            "actor_heads": {k: h.initialize_state(batch_size) for k, h in self.actor_heads.items()},
            "action_samplers": {k: s.initialize_state(batch_size) for k, s in self.action_samplers.items()},
            "critic_encoder": self.critic_encoder.initialize_state(batch_size),
            "critic_heads": {k: h.initialize_state(batch_size) for k, h in self.critic_heads.items()},
        }
        if self.preprocessor is not None:
            state["preprocessor"] = self.preprocessor.initialize_state(batch_size)
        return state

    def update_statistics(self, last_rollout: Transition, total_steps) -> None:
        if self.preprocessor is not None:
            self.preprocessor.update_statistics(last_rollout, total_steps)


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
env_config.sim_dt = 0.001

net_config = config_dict.create(
    actor_hidden_sizes=[1024] * 2,
    critic_hidden_sizes=[1024] * 2,
    activation="swish",
    entropy_weight=1e-2,
    min_std=1e-1,
    std_scale=1.0,
    normalize_obs=True,
    initializer_scale=1.0,
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
        n_minibatches=1,
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
train_env = FlatObsMultiRewardWrapper(base_env)
eval_env = train_env

# Determine action sizes and reward keys from a sample reset
_sample_state = jax.jit(base_env.reset)(jax.random.key(0))
reward_keys = list(_sample_state.reward.keys())
del _sample_state
action_sizes = {k: int(v) for k, v in base_env.action_size.items()}

rngs = nnx.Rngs(SEED)
nets = MLPModularNetwork(
    obs_size=int(train_env.observation_size),
    action_sizes=action_sizes,
    reward_keys=reward_keys,
    rngs=rngs,
    **net_config,
)

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"MLPModular-{timestamp}"
wandb.init(
    project="nnx-ppo-modular-rodent-imitation",
    config={
        "env": "ModularImitation",
        "SEED": SEED,
        "config": dataclasses.asdict(config),
        "net_params": net_config.to_dict(),
        "env_params": env_config.to_dict(),
        "action_sizes": action_sizes,
        "reward_keys": reward_keys,
    },
    name=exp_name,
    tags=("MLP", "Modular", "MultiHead", "warp"),
    notes="Encoders + per-module heads for both actor and critic.",
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