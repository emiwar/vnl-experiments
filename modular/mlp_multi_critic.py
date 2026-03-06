"""MLP actor + multi-objective critic for modular rodent imitation.

Tests whether multi-objective PPO works independently of the NerveNet architecture,
by using the same MLP actor as dense_mlp.py with per-module critics.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from datetime import datetime
import dataclasses
from typing import Optional

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
from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput, ModuleState
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.containers import Sequential


# ---------------------------------------------------------------------------
# Wrapper: flatten obs + unwrap actions, keep rewards as per-module dict
# ---------------------------------------------------------------------------

class FlatObsMultiRewardWrapper(RLEnv):
    """Like dense_mlp's CustomFlattenWrapper, but keeps reward as a dict.

    - obs: flattened to a 1-D array
    - action: flat array unwrapped back to the env's dict on step
    - reward: unchanged (dict keyed by module name)
    """

    def __init__(self, base_env: RLEnv) -> None:
        self.base_env = base_env
        null_action = self.base_env.null_action()
        _, self.unwrap_action = jax.flatten_util.ravel_pytree(null_action)

    def reset(self, rng) -> EnvState:
        state = self.base_env.reset(rng)
        return self._flatten_obs(state)

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        dict_action = self.unwrap_action(action)
        new_state = self.base_env.step(state, dict_action)
        return self._flatten_obs(new_state)

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
# Network: MLP actor + shared encoder + per-module critic heads
# ---------------------------------------------------------------------------

class MLPMultiCriticNetwork(PPONetwork, nnx.Module):
    """MLP actor (identical to dense_mlp) with per-module value critics.

    The actor is unchanged from dense_mlp.py. The critic uses a shared MLP
    encoder followed by one linear head per reward module, so each module gets
    its own value estimate for the multi-objective PPO update.

    Args:
        obs_size: Flat observation size.
        action_size: Flat action size.
        reward_keys: Iterable of reward/module names (matching env reward dict keys).
        actor_hidden_sizes: Hidden layer sizes for the actor MLP.
        critic_hidden_sizes: Hidden layer sizes for the shared critic encoder.
            The last element is the feature dimension fed into each head.
        rngs: NNX RNG streams.
        activation: Activation function (or name) for all hidden layers.
        normalize_obs: Whether to online-normalize observations.
        entropy_weight, min_std, std_scale: Action sampler parameters.
        initializer_scale: Scale for variance_scaling kernel init (actor + critic).
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
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

        # Actor: same architecture as make_mlp_actor_critic
        actor_sizes = [obs_size] + actor_hidden_sizes + [action_size * 2]
        self.actor = make_mlp(actor_sizes, rngs, activation, activation_last_layer=False,
                              kernel_init=kernel_init)
        self.action_sampler = NormalTanhSampler(rngs, entropy_weight, min_std, std_scale)

        # Critic: shared encoder + one linear head per reward key
        enc_sizes = [obs_size] + critic_hidden_sizes
        self.critic_encoder = make_mlp(enc_sizes, rngs, activation, activation_last_layer=True,
                                       kernel_init=kernel_init)
        critic_feature_size = critic_hidden_sizes[-1]
        self.critic_heads = nnx.Dict({
            k: Dense(critic_feature_size, 1, rngs, kernel_init=kernel_init)
            for k in reward_keys
        })

        # Obs normalizer
        self.preprocessor: Optional[Normalizer] = Normalizer(obs_size) if normalize_obs else None

    def __call__(
        self,
        network_state: dict,
        obs: jax.Array,
        raw_action: Optional[jax.Array] = None,
    ) -> tuple[dict, PPONetworkOutput]:
        regularization_loss = jp.array(0.0)

        # Obs normalization
        if self.preprocessor is not None:
            prep_out = self.preprocessor(network_state["preprocessor"], obs)
            obs_proc = prep_out.output
            network_state = {**network_state, "preprocessor": prep_out.next_state}
        else:
            obs_proc = obs

        # Actor
        actor_out = self.actor(network_state["actor"], obs_proc)
        network_state = {**network_state, "actor": actor_out.next_state}

        # Action sampler
        sampler_out = self.action_sampler(network_state["action_sampler"], actor_out.output, raw_action)
        action, new_raw_action, loglikelihood = sampler_out.output
        network_state = {**network_state, "action_sampler": sampler_out.next_state}
        regularization_loss = regularization_loss + sampler_out.regularization_loss

        # Shared critic encoder
        enc_out = self.critic_encoder(network_state["critic_encoder"], obs_proc)
        network_state = {**network_state, "critic_encoder": enc_out.next_state}
        features = enc_out.output

        # Per-module critic heads
        value_estimates = {}
        new_head_states = {}
        for k, head in self.critic_heads.items():
            head_out = head(network_state["critic_heads"][k], features)
            value_estimates[k] = jp.squeeze(head_out.output, axis=-1)
            new_head_states[k] = head_out.next_state
        network_state = {**network_state, "critic_heads": new_head_states}

        return network_state, PPONetworkOutput(
            actions=action,
            raw_actions=new_raw_action,
            loglikelihoods=loglikelihood,
            regularization_loss=regularization_loss,
            value_estimates=value_estimates,
            metrics={},
        )

    def initialize_state(self, batch_size: int) -> dict:
        state = {
            "actor": self.actor.initialize_state(batch_size),
            "action_sampler": self.action_sampler.initialize_state(batch_size),
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
    actor_hidden_sizes=[1024] * 4,
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

# Determine reward keys from a sample reset
_sample_state = jax.jit(base_env.reset)(jax.random.key(0))
reward_keys = list(_sample_state.reward.keys())
del _sample_state

rngs = nnx.Rngs(SEED)
nets = MLPMultiCriticNetwork(
    obs_size=int(train_env.observation_size),
    action_size=int(train_env.action_size),
    reward_keys=reward_keys,
    rngs=rngs,
    **net_config,
)

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"MLPMultiCritic-{timestamp}"
wandb.init(
    project="nnx-ppo-modular-rodent-imitation",
    config={
        "env": "ModularImitation",
        "SEED": SEED,
        "config": dataclasses.asdict(config),
        "net_params": net_config.to_dict(),
        "env_params": env_config.to_dict(),
        "reward_keys": reward_keys,
    },
    name=exp_name,
    tags=("MLP", "MultiCritic", "warp", "Modular"),
    notes="MLP actor + per-module critics. Tests multi-objective PPO independently of NerveNet.",
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