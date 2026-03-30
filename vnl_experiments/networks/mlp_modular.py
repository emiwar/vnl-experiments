"""MLP-based modular network and env wrapper for multi-module rodent imitation."""

from typing import Optional, Union
from collections.abc import Mapping

import jax
import jax.flatten_util
import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.types import RLEnv, EnvState, Transition
from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.factories import make_mlp
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.containers import Flattener


class FlatObsMultiRewardWrapper(RLEnv):
    """Flattens obs; actions are a dict passed directly to the env; keeps reward as dict."""

    def __init__(self, base_env: RLEnv) -> None:
        self.base_env = base_env

    def reset(self, rng) -> EnvState:
        return self._flatten_obs(self.base_env.reset(rng))

    def step(self, state: EnvState, action) -> EnvState:
        return self._flatten_obs(self.base_env.step(state, action))

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
        obs_sizes: Mapping[str, int],
        action_sizes: Mapping[str, int],
        actor_hidden_sizes: list[int],
        critic_hidden_sizes: list[int],
        rngs: nnx.Rngs,
        activation: Union[str, callable] = nnx.swish,
        normalize_obs: bool = True,
        entropy_weight: float = 1e-2,
        min_std: float = 1e-1,
        std_scale: float = 1.0,
        initializer_scale: float = 1.0,
        reveal_targets: str = "all",
    ):
        if isinstance(activation, str):
            activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[activation]

        self.reveal_targets = reveal_targets
        flat_obs_size = int(sum(obs_sizes.values()))
        kernel_init = nnx.initializers.variance_scaling(initializer_scale, "fan_in", "uniform")

        # Actor: shared encoder + per-module heads
        self.actor_encoder = make_mlp(
            [flat_obs_size] + actor_hidden_sizes, rngs, activation,
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

        # Critic: shared encoder + per-module heads
        self.critic_encoder = make_mlp(
            [flat_obs_size] + critic_hidden_sizes, rngs, activation,
            activation_last_layer=True, kernel_init=kernel_init,
        )
        critic_feature_size = critic_hidden_sizes[-1]
        self.critic_heads = nnx.Dict({
            k: Dense(critic_feature_size, 1, rngs, kernel_init=kernel_init)
            for k in obs_sizes.keys()
        })

        self.preprocessor: Optional[Normalizer] = Normalizer(flat_obs_size) if normalize_obs else None

    def __call__(
        self,
        network_state: dict,
        obs: Mapping[str, jax.Array],
        raw_action: Optional[Mapping[str, jax.Array]] = None,
    ) -> tuple[dict, PPONetworkOutput]:
        regularization_loss = jp.array(0.0)

        # Filter and flatten obs
        obs_filtered = self._filter_obs(obs)
        flattener = Flattener()
        obs_flat = flattener((), obs_filtered).output

        # Obs normalization
        if self.preprocessor is not None:
            prep_out = self.preprocessor(network_state["preprocessor"], obs_flat)
            obs_proc = prep_out.output
            network_state = {**network_state, "preprocessor": prep_out.next_state}
        else:
            obs_proc = obs_flat

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
            obs_filtered = self._filter_obs(last_rollout.obs)
            flat_obs = jp.concatenate(
                [obs_filtered[k] for k in obs_filtered], axis=-1
            )
            self.preprocessor.update_statistics(
                last_rollout.replace(obs=flat_obs), total_steps
            )

    def _filter_obs(self, obs):
        if self.reveal_targets != "all":
            obs = obs.copy()
            for k, o in obs.items():
                if k != "root":
                    obs[k] = o["proprioception"]
            if self.reveal_targets != "root_only":
                batch_dim = obs["root"]["current_target"].shape
                obs["root"] = jp.zeros(batch_dim, 0)
        return obs
