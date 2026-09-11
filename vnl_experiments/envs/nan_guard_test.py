"""Tests for NaNGuardWrapper, against the real dm_control_suite task."""

import jax
import jax.numpy as jp
import mujoco_playground
import numpy as np
from absl.testing import absltest

from nnx_ppo.wrappers import episode_wrapper
from vnl_experiments.envs import registry as env_registry
from vnl_experiments.envs.nan_guard import NaNGuardWrapper


def _humanoid():
    return mujoco_playground.registry.load("HumanoidWalk")


def _diverged_state(env, state):
    """A state as MJX leaves it when the *final* substep's integrator diverges:
    qpos/qvel non-finite, but xpos/xmat/sensordata still finite because they are
    filled by `forward` before the integration and so lag one step behind."""
    d = state.data
    return state.replace(
        data=d.replace(qpos=d.qpos.at[9].set(jp.nan), qvel=d.qvel.at[8].set(jp.nan))
    )


class NaNGuardWrapperTest(absltest.TestCase):

    def setUp(self):
        self.env = _humanoid()
        self.guarded = NaNGuardWrapper(_humanoid())
        self.state = jax.jit(self.env.reset)(jax.random.key(0))
        self.action = jp.zeros(self.env.action_size)

    def test_clean_step_is_unchanged(self):
        a = jax.jit(self.env.step)(self.state, self.action)
        b = jax.jit(self.guarded.step)(self.state, self.action)
        np.testing.assert_allclose(np.asarray(a.reward), np.asarray(b.reward))
        np.testing.assert_allclose(np.asarray(a.done), np.asarray(b.done))

    def test_unguarded_divergence_yields_a_nonfinite_reward(self):
        """The failure this wrapper exists for. If this ever stops holding, the
        wrapper is no longer needed -- but check why before deleting it."""
        s = jax.jit(self.env.step)(_diverged_state(self.env, self.state), self.action)
        self.assertFalse(np.isfinite(float(s.reward)))
        self.assertEqual(float(s.done), 1.0)

    def test_guard_makes_the_reward_finite_and_zero(self):
        s = jax.jit(self.guarded.step)(
            _diverged_state(self.guarded, self.state), self.action
        )
        self.assertTrue(np.isfinite(float(s.reward)))
        self.assertEqual(float(s.reward), 0.0)

    def test_guard_terminates(self):
        s = jax.jit(self.guarded.step)(
            _diverged_state(self.guarded, self.state), self.action
        )
        self.assertEqual(float(s.done), 1.0)

    def test_observation_is_left_non_finite_on_purpose(self):
        """It is harmless once done=1 (rollout resets the state, GAE drops the
        bootstrap) and it is what keeps nnx-ppo's nonfinite_next_obs counter
        usable as a divergence-rate signal."""
        s = jax.jit(self.guarded.step)(
            _diverged_state(self.guarded, self.state), self.action
        )
        self.assertGreater(int(np.isnan(np.asarray(s.obs)).sum()), 0)

    def test_reward_only_divergence_is_caught(self):
        """A non-finite reward with finite qpos/qvel -- which the task's own
        `done = isnan(qpos) | isnan(qvel)` would miss entirely."""

        class NaNRewardEnv:
            def __init__(self, env):
                self.env = env
            def reset(self, rng):
                return self.env.reset(rng)
            def step(self, state, action):
                s = self.env.step(state, action)
                return s.replace(reward=jp.full_like(s.reward, jp.nan))
            def __getattr__(self, name):
                return getattr(self.env, name)

        guarded = NaNGuardWrapper(NaNRewardEnv(_humanoid()))
        s = jax.jit(guarded.step)(self.state, self.action)
        self.assertEqual(float(s.reward), 0.0)
        self.assertEqual(float(s.done), 1.0)
        self.assertEqual(int(np.isnan(np.asarray(s.obs)).sum()), 0)

    def test_forwards_attributes_to_the_wrapped_task(self):
        self.assertEqual(self.guarded.action_size, self.env.action_size)
        self.assertEqual(self.guarded.observation_size, self.env.observation_size)
        self.assertIsNotNone(self.guarded.mj_model)


class BuilderWiringTest(absltest.TestCase):
    """The guard must sit inside EpisodeWrapper, so a divergence reads as a
    termination (done=1, truncated=False) and not as a truncation."""

    def _built(self, for_eval):
        spec = env_registry.get("HumanoidWalk")
        return spec.build(spec.default_config(), for_eval=for_eval)

    def test_training_env_has_the_guard(self):
        env = self._built(for_eval=False)
        inner = env
        seen = []
        while hasattr(inner, "env"):
            seen.append(type(inner).__name__)
            inner = inner.env
        self.assertIn("NaNGuardWrapper", seen)
        self.assertLess(
            seen.index("NaNGuardWrapper"), len(seen),
            "guard must be inside EpisodeWrapper",
        )
        self.assertEqual(seen[-1], "NaNGuardWrapper")

    def test_eval_env_does_not(self):
        env = self._built(for_eval=True)
        self.assertFalse(hasattr(env, "env"))

    def test_divergence_reads_as_termination_not_truncation(self):
        env = self._built(for_eval=False)
        state = jax.jit(env.reset)(jax.random.key(0))
        bad = _diverged_state(env, state)
        s = jax.jit(env.step)(bad, jp.zeros(env.action_size))
        self.assertEqual(float(s.done), 1.0)
        self.assertFalse(bool(s.info["truncated"]))
        self.assertTrue(np.isfinite(float(s.reward)))
        self.assertEqual(float(s.reward), 0.0)


if __name__ == "__main__":
    absltest.main()
