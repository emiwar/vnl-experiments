"""Tests for ActionDelayWrapper.

All test environments are minimal in-memory implementations with no MuJoCo
dependency, so these tests run quickly and without GPU.
"""

import dataclasses
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from nnx_ppo.jax_dataclass import JaxDataclass
from vnl_experiments.wrappers.action_delay import ActionDelayWrapper


# ---------------------------------------------------------------------------
# Minimal test environments
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class SimpleEnvState(JaxDataclass):
    obs: jax.Array           # step counter
    reward: jax.Array
    done: jax.Array
    info: dict[str, Any]
    metrics: dict[str, Any]


class SimpleVecEnv:
    """Vector-action env (action_dim=3). Records the applied action in info."""

    ACTION_DIM = 3

    def reset(self, rng) -> SimpleEnvState:
        return SimpleEnvState(
            obs=jnp.array(0.0),
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            info={"applied_action": jnp.zeros(self.ACTION_DIM)},
            metrics={},
        )

    def step(self, state: SimpleEnvState, action: jax.Array) -> SimpleEnvState:
        return SimpleEnvState(
            obs=state.obs + 1.0,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            info={"applied_action": action},
            metrics={},
        )

    def null_action(self) -> jax.Array:
        return jnp.zeros(self.ACTION_DIM)

    @property
    def observation_size(self):
        return 1

    @property
    def action_size(self):
        return self.ACTION_DIM


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DictEnvState(JaxDataclass):
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict[str, Any]
    metrics: dict[str, Any]


class DictTrackerEnv:
    """Dict-action env (force: [2], torque: [1]). Records applied action in info."""

    def reset(self, rng) -> DictEnvState:
        return DictEnvState(
            obs=jnp.array(0.0),
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            info={"applied_action": self.null_action()},
            metrics={},
        )

    def step(self, state: DictEnvState, action: dict) -> DictEnvState:
        return DictEnvState(
            obs=state.obs + 1.0,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            info={"applied_action": action},
            metrics={},
        )

    def null_action(self) -> dict:
        return {"force": jnp.zeros(2), "torque": jnp.zeros(1)}

    @property
    def observation_size(self):
        return 1

    @property
    def action_size(self):
        return {"force": 2, "torque": 1}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ActionDelayWrapperTest(absltest.TestCase):

    def _vec_env(self, k):
        return ActionDelayWrapper(SimpleVecEnv(), k)

    def _dict_env(self, k):
        return ActionDelayWrapper(DictTrackerEnv(), k)

    # --- k=2 vector: first two steps use null ---

    def test_k2_first_two_steps_use_null(self):
        env = self._vec_env(2)
        state = env.reset(jax.random.PRNGKey(0))
        null = jnp.zeros(3)

        state = env.step(state, jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(state.info["applied_action"], null)

        state = env.step(state, jnp.array([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(state.info["applied_action"], null)

    # --- k=2 vector: correct delayed actions ---

    def test_k2_delayed_actions_correct(self):
        env = self._vec_env(2)
        state = env.reset(jax.random.PRNGKey(0))

        a1 = jnp.array([1.0, 2.0, 3.0])
        a2 = jnp.array([4.0, 5.0, 6.0])
        a3 = jnp.array([7.0, 8.0, 9.0])
        a4 = jnp.array([10.0, 11.0, 12.0])

        state = env.step(state, a1)   # applies null
        state = env.step(state, a2)   # applies null
        state = env.step(state, a3)   # applies a1
        np.testing.assert_allclose(state.info["applied_action"], a1)

        state = env.step(state, a4)   # applies a2
        np.testing.assert_allclose(state.info["applied_action"], a2)

    # --- k=0 passthrough ---

    def test_k0_action_applied_immediately(self):
        env = self._vec_env(0)
        state = env.reset(jax.random.PRNGKey(0))
        a = jnp.array([1.0, 2.0, 3.0])
        state = env.step(state, a)
        np.testing.assert_allclose(state.info["applied_action"], a)

    def test_k0_no_buffer_keys_in_info(self):
        env = self._vec_env(0)
        state = env.reset(jax.random.PRNGKey(0))
        self.assertNotIn("_action_delay_buffer", state.info)
        self.assertNotIn("_action_delay_idx", state.info)

    # --- k=1 single-step delay ---

    def test_k1_single_step_delay(self):
        env = self._vec_env(1)
        state = env.reset(jax.random.PRNGKey(0))
        null = jnp.zeros(3)

        a1 = jnp.array([1.0, 0.0, 0.0])
        state = env.step(state, a1)           # applies null
        np.testing.assert_allclose(state.info["applied_action"], null)

        a2 = jnp.array([2.0, 0.0, 0.0])
        state = env.step(state, a2)           # applies a1
        np.testing.assert_allclose(state.info["applied_action"], a1)

        a3 = jnp.array([3.0, 0.0, 0.0])
        state = env.step(state, a3)           # applies a2
        np.testing.assert_allclose(state.info["applied_action"], a2)

    # --- Reset reinitialises buffer ---

    def test_reset_reinitialises_buffer(self):
        env = self._vec_env(2)
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Advance two steps to dirty the buffer
        state = env.step(state, jnp.array([1.0, 2.0, 3.0]))
        state = env.step(state, jnp.array([4.0, 5.0, 6.0]))

        # Reset — buffer must reinitialise
        state = env.reset(rng)
        null = jnp.zeros(3)

        state = env.step(state, jnp.array([9.0, 9.0, 9.0]))
        np.testing.assert_allclose(state.info["applied_action"], null)

        state = env.step(state, jnp.array([9.0, 9.0, 9.0]))
        np.testing.assert_allclose(state.info["applied_action"], null)

    # --- JIT compatibility ---

    def test_jit_compatible(self):
        env = self._vec_env(2)
        step_jit = jax.jit(env.step)
        state = env.reset(jax.random.PRNGKey(0))

        a = jnp.array([1.0, 2.0, 3.0])
        state = step_jit(state, a)   # null
        state = step_jit(state, a)   # null
        state = step_jit(state, a)   # applies a
        np.testing.assert_allclose(state.info["applied_action"], a)

    # --- vmap compatibility ---

    def test_vmap_compatible(self):
        N = 4
        env = self._vec_env(2)
        keys = jax.random.split(jax.random.PRNGKey(0), N)
        states = jax.vmap(env.reset)(keys)

        actions = jnp.ones((N, 3))

        states = jax.vmap(env.step)(states, actions)  # null applied
        np.testing.assert_allclose(
            states.info["applied_action"], jnp.zeros((N, 3))
        )

        states = jax.vmap(env.step)(states, actions)  # null applied
        states = jax.vmap(env.step)(states, actions)  # a applied
        np.testing.assert_allclose(
            states.info["applied_action"], jnp.ones((N, 3))
        )

    # --- Dict actions ---

    def test_dict_actions_k2_first_steps_use_null(self):
        env = self._dict_env(2)
        state = env.reset(jax.random.PRNGKey(0))

        a1 = {"force": jnp.array([1.0, 2.0]), "torque": jnp.array([0.5])}
        state = env.step(state, a1)
        np.testing.assert_allclose(
            state.info["applied_action"]["force"], jnp.zeros(2)
        )
        np.testing.assert_allclose(
            state.info["applied_action"]["torque"], jnp.zeros(1)
        )

    def test_dict_actions_k2_delayed_correctly(self):
        env = self._dict_env(2)
        state = env.reset(jax.random.PRNGKey(0))

        a1 = {"force": jnp.array([1.0, 2.0]), "torque": jnp.array([0.5])}
        a2 = {"force": jnp.array([3.0, 4.0]), "torque": jnp.array([1.0])}
        a3 = {"force": jnp.array([5.0, 6.0]), "torque": jnp.array([1.5])}

        state = env.step(state, a1)   # null applied
        state = env.step(state, a2)   # null applied
        state = env.step(state, a3)   # a1 applied
        np.testing.assert_allclose(
            state.info["applied_action"]["force"], a1["force"]
        )
        np.testing.assert_allclose(
            state.info["applied_action"]["torque"], a1["torque"]
        )

    # --- null_action proxy ---

    def test_null_action_proxy_vector(self):
        base = SimpleVecEnv()
        wrapper = ActionDelayWrapper(base, k=2)
        np.testing.assert_allclose(wrapper.null_action(), base.null_action())

    def test_null_action_proxy_dict(self):
        base = DictTrackerEnv()
        wrapper = ActionDelayWrapper(base, k=2)
        np.testing.assert_allclose(
            wrapper.null_action()["force"], base.null_action()["force"]
        )
        np.testing.assert_allclose(
            wrapper.null_action()["torque"], base.null_action()["torque"]
        )

    # --- Buffer shape ---

    def test_buffer_shape_vector(self):
        env = self._vec_env(3)
        state = env.reset(jax.random.PRNGKey(0))
        buf = state.info["_action_delay_buffer"]
        self.assertEqual(buf.shape, (3, 3))   # k=3, action_dim=3
        np.testing.assert_allclose(buf, jnp.zeros((3, 3)))

    def test_buffer_shape_dict(self):
        env = self._dict_env(2)
        state = env.reset(jax.random.PRNGKey(0))
        buf = state.info["_action_delay_buffer"]
        self.assertEqual(buf["force"].shape, (2, 2))   # k=2, force_dim=2
        self.assertEqual(buf["torque"].shape, (2, 1))  # k=2, torque_dim=1

    # --- Invalid k ---

    def test_negative_k_raises(self):
        with self.assertRaises(ValueError):
            ActionDelayWrapper(SimpleVecEnv(), k=-1)


if __name__ == "__main__":
    absltest.main()
