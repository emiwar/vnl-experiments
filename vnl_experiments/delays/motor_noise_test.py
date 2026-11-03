"""Unit tests for :mod:`vnl_experiments.delays.motor_noise`.

The network-level behaviour (where the module sits, what the efference queue records,
loss replay over several steps) is tested in ``network_builders_test.MotorNoiseTest``.
"""

import math

from absl.testing import absltest

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import PPONetworkOutput

from vnl_experiments.delays.motor_noise import (
    MotorNoise,
    perturb_actions,
    validate_motor_noise_site,
)

BATCH, ACTION = 4096, 3


def sampler_dict(action):
    return {"action": action, "log_likelihood": jp.arange(action.shape[0], dtype=float)}


class PerturbActionsTest(absltest.TestCase):

    def test_std_and_clipping(self):
        zeros = jp.zeros((BATCH, ACTION))
        noisy = perturb_actions(zeros, 0.1, jax.random.key(0))
        self.assertEqual(noisy.shape, zeros.shape)
        self.assertAlmostEqual(float(jp.std(noisy)), 0.1, delta=0.005)

        saturated = perturb_actions(jp.ones((BATCH, ACTION)), 0.5, jax.random.key(1))
        self.assertLessEqual(float(jp.max(saturated)), 1.0)
        self.assertGreaterEqual(float(jp.min(saturated)), -1.0)

    def test_leaves_get_independent_noise(self):
        tree = {"a": jp.zeros((BATCH, ACTION)), "b": jp.zeros((BATCH, ACTION))}
        noisy = perturb_actions(tree, 0.1, jax.random.key(0))
        self.assertFalse(jp.allclose(noisy["a"], noisy["b"]))

    def test_matches_the_eval_rollout_key_schedule(self):
        """``evaluation._rollout`` used to inline this; its outputs must not move."""
        actions = {"a": jp.zeros((8, ACTION)), "b": jp.zeros((8, 2))}
        key = jax.random.key(3)
        leaves, treedef = jax.tree.flatten(actions)
        inlined = jax.tree.unflatten(treedef, [
            jp.clip(a + 0.2 * jax.random.normal(k, a.shape), -1.0, 1.0)
            for a, k in zip(leaves, jax.random.split(key, len(leaves)))
        ])
        helper = perturb_actions(actions, 0.2, key)
        for x, y in zip(jax.tree.leaves(inlined), jax.tree.leaves(helper)):
            self.assertTrue(jp.array_equal(x, y))


class MotorNoiseModuleTest(absltest.TestCase):

    def setUp(self):
        self.noise = MotorNoise(0.2, nnx.Rngs(0))
        self.action = jp.zeros((8, ACTION))

    def test_sampler_dict_passes_log_likelihood_through(self):
        x = sampler_dict(self.action)
        out = self.noise((), x)
        self.assertTrue(jp.array_equal(out.output["log_likelihood"], x["log_likelihood"]))
        self.assertFalse(jp.allclose(out.output["action"], self.action))
        self.assertTrue(jp.array_equal(out.rollout_extras, out.output["action"]))

    def test_replay_reuses_the_stored_noise(self):
        x = sampler_dict(self.action)
        rollout = self.noise((), x)
        replay = self.noise((), x, rollout.rollout_extras)
        self.assertTrue(jp.array_equal(replay.output["action"], rollout.output["action"]))
        fresh = self.noise((), x)
        self.assertFalse(jp.allclose(fresh.output["action"], rollout.output["action"]))

    def test_ppo_network_output(self):
        x = PPONetworkOutput(actions=self.action, loglikelihoods=jp.ones(8),
                             value_estimates=jp.full(8, 2.0))
        out = self.noise((), x)
        self.assertIsInstance(out.output, PPONetworkOutput)
        self.assertFalse(jp.allclose(out.output.actions, self.action))
        self.assertTrue(jp.array_equal(out.output.loglikelihoods, x.loglikelihoods))
        self.assertTrue(jp.array_equal(out.output.value_estimates, x.value_estimates))

    def test_pytree_of_sampler_dicts(self):
        x = {"left": sampler_dict(self.action), "right": sampler_dict(self.action)}
        out = self.noise((), x)
        self.assertEqual(set(out.output), {"left", "right"})
        self.assertFalse(jp.allclose(out.output["left"]["action"],
                                     out.output["right"]["action"]))

    def test_unaffected_by_eval(self):
        """No ``deterministic`` attribute, so ``eval()`` leaves the noise on."""
        self.assertFalse(hasattr(self.noise, "deterministic"))
        self.noise.eval()
        out = self.noise((), sampler_dict(self.action))
        self.assertFalse(jp.allclose(out.output["action"], self.action))

    def test_validation(self):
        with self.assertRaises(ValueError):
            MotorNoise(-0.1, nnx.Rngs(0))
        with self.assertRaises(ValueError):
            MotorNoise(0.1, nnx.Rngs(0), tau=-1.0)
        with self.assertRaises(ValueError):
            MotorNoise(0.1, nnx.Rngs(0), tau=5.0)  # no action_size
        with self.assertRaises(ValueError):
            validate_motor_noise_site("before_copy")


class CorrelatedMotorNoiseTest(absltest.TestCase):
    """``tau > 0``: AR(1) noise carried per env."""

    def run_steps(self, noise, steps, batch=BATCH, reset_at=None):
        state = noise.initialize_state(batch)
        action = jp.zeros((batch, ACTION))
        values, states = [], [state]
        for t in range(steps):
            if t == reset_at:
                state = noise.reset_state(state)
            out = noise(state, sampler_dict(action))
            values.append(out.rollout_extras)
            state = out.next_state
            states.append(state)
        return jp.stack(values), states

    def test_iid_is_stateless(self):
        noise = MotorNoise(0.2, nnx.Rngs(0))
        self.assertEqual(noise.initialize_state(8), ())
        self.assertEqual(noise.rho, 0.0)

    def test_stationary_std_and_autocorrelation(self):
        tau = 5.0
        noise = MotorNoise(0.2, nnx.Rngs(0), tau=tau, action_size=ACTION)
        n, _ = self.run_steps(noise, 20)
        # Stationary from the first step, because a fresh carry is drawn at full size.
        for t in (0, 19):
            self.assertAlmostEqual(float(jp.std(n[t])), 0.2, delta=0.01)
        lag1 = float(jp.mean(n[1:] * n[:-1]) / jp.mean(n[:-1] ** 2))
        self.assertAlmostEqual(lag1, math.exp(-1 / tau), delta=0.02)

    def test_inf_tau_is_constant_until_reset(self):
        noise = MotorNoise(0.2, nnx.Rngs(0), tau=math.inf, action_size=ACTION)
        self.assertEqual(noise.rho, 1.0)
        n, _ = self.run_steps(noise, 6, batch=8, reset_at=3)
        for t in (1, 2):
            self.assertTrue(jp.array_equal(n[t], n[0]))
        self.assertFalse(jp.allclose(n[3], n[0]))
        self.assertTrue(jp.array_equal(n[5], n[3]))

    def test_executed_action_is_clipped_action_plus_noise(self):
        noise = MotorNoise(0.5, nnx.Rngs(0), tau=5.0, action_size=ACTION)
        action = jp.full((8, ACTION), 0.8)
        out = noise(noise.initialize_state(8), sampler_dict(action))
        self.assertTrue(jp.allclose(out.output["action"],
                                    jp.clip(action + out.rollout_extras, -1, 1)))

    def test_replay_reproduces_actions_and_carry(self):
        noise = MotorNoise(0.2, nnx.Rngs(0), tau=5.0, action_size=ACTION)
        action = jp.zeros((8, ACTION))
        state0 = noise.initialize_state(8)
        state, rollout = state0, []
        for _ in range(4):
            out = noise(state, sampler_dict(action))
            rollout.append(out)
            state = out.next_state
        state = state0
        for r in rollout:
            out = noise(state, sampler_dict(action), r.rollout_extras)
            self.assertTrue(jp.array_equal(out.output["action"], r.output["action"]))
            self.assertTrue(jp.array_equal(out.next_state["n"], r.next_state["n"]))
            state = out.next_state


if __name__ == "__main__":
    absltest.main()
