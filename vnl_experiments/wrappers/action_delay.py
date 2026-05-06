"""Action delay wrapper for JAX/MuJoCo RL environments.

Delays all actions by k steps, applying null_action() for the first k steps
of each episode. A circular buffer stored in state.info provides O(1) per-step
cost with no array shifting. Because the buffer lives in state.info, the
rollout infrastructure's tree_where(done, reset_states, current_states)
automatically reinitialises it on episode reset.
"""

import jax
import jax.numpy as jnp

from nnx_ppo.algorithms.types import RLEnv, EnvState


class ActionDelayWrapper:
    """Wraps an RLEnv and delays all actions by k steps.

    For the first k steps of each episode the base env receives null_action()
    instead of the agent's action. Supports any action pytree (vectors, dicts,
    nested dicts).

    The buffer is stored in state.info under two private keys:
      "_action_delay_buffer": pytree-of-arrays, shape (k, *leaf_shape) per leaf
      "_action_delay_idx":    jnp.int32 circular write pointer

    Circular-buffer invariant: buf[idx] is always the *oldest* entry, i.e. the
    action submitted k steps ago. We read it (to apply), overwrite it with the
    incoming action, then advance idx.

    Args:
        env: Base environment that implements null_action().
        k: Number of steps to delay. k=0 is a zero-overhead passthrough.
    """

    def __init__(self, env: RLEnv, k: int) -> None:
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.env = env
        self.k = k

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, rng) -> EnvState:
        state = self.env.reset(rng)
        if self.k == 0:
            return state
        null = self.env.null_action()
        buffer = jax.tree.map(lambda a: jnp.stack([a] * self.k), null)
        state.info["_action_delay_buffer"] = buffer
        state.info["_action_delay_idx"] = jnp.int32(0)
        return state

    def step(self, state: EnvState, action) -> EnvState:
        if self.k == 0:
            return self.env.step(state, action)

        buffer = state.info["_action_delay_buffer"]
        idx = state.info["_action_delay_idx"]

        # Read oldest slot — the action from k steps ago.
        delayed_action = jax.tree.map(lambda b: b[idx], buffer)

        # Overwrite oldest slot with the incoming action, advance pointer.
        new_buffer = jax.tree.map(
            lambda b, a: b.at[idx].set(a), buffer, action
        )
        new_idx = (idx + 1) % self.k

        # Step with delayed action. Write buffer to new state's info afterwards
        # so this works whether the base env preserves or replaces info.
        new_state = self.env.step(state, delayed_action)
        new_state.info["_action_delay_buffer"] = new_buffer
        new_state.info["_action_delay_idx"] = new_idx
        return new_state

    # ------------------------------------------------------------------
    # Proxied interface
    # ------------------------------------------------------------------

    @property
    def observation_size(self):
        return self.env.observation_size
    
    @property
    def non_flattened_observation_size(self):
        return self.env.non_flattened_observation_size

    @property
    def action_size(self):
        return self.env.action_size

    def null_action(self):
        return self.env.null_action()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
