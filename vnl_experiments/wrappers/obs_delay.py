"""Observation delay wrapper for PPONetwork.

Wraps any PPONetwork so the inner network receives k-step-old observations.
The delay buffer lives in the network carry state, so it is handled correctly
by the rollout infrastructure (carried across steps, reset on episode end) and
by the distillation loss replay (full time sequence re-scanned per minibatch).

Obs delay is functionally equivalent to action delay during steady-state
episodes (both result in π(o_{t-k}) being applied to physics at time t).
They differ only at episode resets, a negligible transient for long episodes.

Usage::

    sample_obs = jax.jit(train_env.reset)(jax.random.key(0)).obs
    student = DelayedObsNetwork(base_student, delay_k=5, sample_obs=sample_obs)

The sample_obs is a single (unbatched) observation used solely to infer the
PyTree structure and leaf shapes for buffer initialisation.
"""

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput, ModuleState


class DelayedObsNetwork(PPONetwork):
    """Wraps a PPONetwork to deliver k-step-delayed observations.

    The obs buffer is stored as part of the network carry state:
      state["inner"]      — carry state of the wrapped network
      state["obs_buffer"] — PyTree mirroring obs, leaves shape [B, k, *leaf]
      state["buf_idx"]    — circular write pointer, shape [B], dtype int32

    At each step the wrapper reads the oldest slot (= obs from k steps ago),
    writes the current obs into that slot, advances the pointer, then calls
    the wrapped network with the delayed obs.

    Args:
        network: Any PPONetwork to wrap.
        delay_k: Number of steps to delay. Must be >= 1.
        sample_obs: A single unbatched observation PyTree used to determine
            the buffer structure and leaf shapes at construction time.
    """

    def __init__(self, network: PPONetwork, delay_k: int, sample_obs) -> None:
        if delay_k < 1:
            raise ValueError(f"delay_k must be >= 1, got {delay_k}")
        self.network = network
        self.delay_k = delay_k
        # Store pytree structure and leaf metadata as pure Python objects so
        # that NNX does not try to treat them as tracked Variables.
        leaves, self._obs_treedef = jax.tree_util.tree_flatten(sample_obs)
        self._obs_leaf_shapes = tuple(leaf.shape for leaf in leaves)
        self._obs_leaf_dtypes = tuple(leaf.dtype for leaf in leaves)

    # ------------------------------------------------------------------
    # PPONetwork interface
    # ------------------------------------------------------------------

    def initialize_state(self, batch_size: int) -> dict:
        buffer_leaves = [
            jp.zeros((batch_size, self.delay_k) + shape, dtype)
            for shape, dtype in zip(self._obs_leaf_shapes, self._obs_leaf_dtypes)
        ]
        obs_buffer = jax.tree_util.tree_unflatten(self._obs_treedef, buffer_leaves)
        return {
            "inner": self.network.initialize_state(batch_size),
            "obs_buffer": obs_buffer,
            "buf_idx": jp.zeros(batch_size, jp.int32),
        }

    def reset_state(self, state: dict) -> dict:
        return {
            "inner": self.network.reset_state(state["inner"]),
            "obs_buffer": jax.tree.map(jp.zeros_like, state["obs_buffer"]),
            "buf_idx": jp.zeros_like(state["buf_idx"]),
        }

    def __call__(
        self,
        state: dict,
        obs,
        raw_action=None,
    ) -> tuple[dict, PPONetworkOutput]:
        idx = state["buf_idx"]           # [B]
        arange = jp.arange(idx.shape[0]) # [B]

        # Read the oldest slot — obs from delay_k steps ago.
        delayed_obs = jax.tree.map(lambda b: b[arange, idx], state["obs_buffer"])

        # Overwrite the oldest slot with the current obs, then advance the pointer.
        new_buffer = jax.tree.map(
            lambda b, o: b.at[arange, idx].set(o),
            state["obs_buffer"], obs,
        )
        new_idx = (idx + 1) % self.delay_k

        inner_state, output = self.network(state["inner"], delayed_obs, raw_action)

        new_state = {
            "inner": inner_state,
            "obs_buffer": new_buffer,
            "buf_idx": new_idx,
        }
        return new_state, output

    def update_statistics(self, rollout, total_steps) -> None:
        # rollout.obs has shape [T, B, *leaf] per leaf.
        # The inner network's normaliser must see the same delayed obs it
        # receives during __call__, so we shift by delay_k along axis 0
        # (pad the start with zeros, drop the last delay_k steps).
        delayed_obs = jax.tree.map(
            lambda o: jp.concatenate(
                [jp.zeros_like(o[: self.delay_k]), o[: -self.delay_k]], axis=0
            ),
            rollout.obs,
        )
        self.network.update_statistics(rollout.replace(obs=delayed_obs), total_steps)
