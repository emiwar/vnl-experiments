"""Tests for the eval-time action-noise path in ``evaluation``.

These use a toy env and a constant-zero-action network rather than the rodent, so
the executed action is exactly the perturbation and its statistics can be asserted
on directly.
"""

import dataclasses

from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import struct

from nnx_ppo.networks.types import (
    PPONetworkOutput,
    StatefulModule,
    StatefulModuleOutput,
)

from vnl_experiments.delays.evaluation import eval_dataset

N_CLIPS = 4
N_STEPS = 256
ACTION_SIZE = 3


@struct.dataclass
class _State:
    obs: jp.ndarray
    reward: jp.ndarray
    done: jp.ndarray
    metrics: dict


@dataclasses.dataclass(frozen=True)
class _ActionProbeEnv:
    """Toy env whose metrics expose the action it was actually stepped with.

    Never terminates, so every clip runs the full ``n_steps``. All probes are
    ``rewards/`` metrics because those are the ones ``eval_dataset`` surfaces as
    per-episode sums; they do not feed ``state.reward`` and so do not affect
    ``episode_reward``.
    """

    action_size: int = ACTION_SIZE

    def _metrics(self, action, key_probe):
        return {
            "rewards/a_sq": jp.mean(action**2),
            "rewards/a_abs": jp.mean(jp.abs(action)),
            "rewards/key_probe": key_probe,
            "terminations/any": jp.zeros(()),
        }

    def reset(self, key, clip_idx=0, start_frame=0):
        zeros = jp.zeros(self.action_size)
        return _State(
            obs=zeros,
            reward=jp.zeros(()),
            done=jp.zeros((), dtype=bool),
            # A deterministic function of the reset key, carried unchanged
            # through the episode, so a test can assert the per-clip reset keys
            # are what a noise-free eval has always used.
            metrics=self._metrics(zeros, jax.random.uniform(key)),
        )

    def step(self, state, action):
        return state.replace(
            reward=jp.zeros(()),
            done=jp.zeros((), dtype=bool),
            metrics=self._metrics(action, state.metrics["rewards/key_probe"]),
        )


class _ZeroActionNet(StatefulModule):
    """Emits a zero action every step, so the executed action *is* the noise."""

    def __init__(self, action_size: int = ACTION_SIZE):
        self.action_size = action_size
        self.deterministic = False

    def __call__(self, state, obs, rollout_extras=None):
        batch = obs.shape[0]
        return StatefulModuleOutput(
            next_state=state,
            output=PPONetworkOutput(
                actions=jp.zeros((batch, self.action_size)),
                loglikelihoods=jp.zeros(batch),
                value_estimates=jp.zeros(batch),
            ),
            regularization_loss=jp.zeros(batch),
            metrics={},
        )

    def initialize_state(self, batch_size: int):
        return jp.zeros((batch_size, 1))


def _run(action_noise, *, seed: int = 0) -> dict:
    return eval_dataset(_ActionProbeEnv(), _ZeroActionNet(), N_CLIPS, N_STEPS,
                        0.01, jax.random.key(seed), None, action_noise)


def _per_step(result: dict, term: str) -> float:
    """Per-step mean of a ``rewards/`` probe (they accumulate as episode sums)."""
    return result["reward_terms"][term]["mean"] / N_STEPS


class ActionNoiseTest(absltest.TestCase):

    def test_no_noise_leaves_the_action_untouched(self):
        for action_noise in (None, 0.0):
            result = _run(action_noise)
            self.assertEqual(_per_step(result, "a_sq"), 0.0)
            self.assertEqual(_per_step(result, "a_abs"), 0.0)

    def test_reset_keys_are_unchanged_by_the_noise_stream(self):
        """The noise key is folded in, not split off the eval key, so the
        per-clip reset keys are identical with and without noise."""
        expected = float(jp.mean(jax.vmap(jax.random.uniform)(
            jax.random.split(jax.random.key(0), N_CLIPS))))
        for action_noise in (None, 0.0, 0.25):
            self.assertAlmostEqual(_per_step(_run(action_noise), "key_probe"),
                                   expected, places=5)

    def test_noise_has_the_requested_standard_deviation(self):
        sigma = 0.25
        # E[a^2] = sigma^2 for an unclipped N(0, sigma); the clip at +-1 is four
        # sigma away and contributes nothing at this magnitude.
        self.assertAlmostEqual(_per_step(_run(sigma), "a_sq"), sigma**2,
                               delta=0.1 * sigma**2)

    def test_noise_is_clipped_into_the_action_range(self):
        # A sigma far beyond the range: nearly every sample saturates, so the
        # mean |a| must approach but never exceed 1.
        mean_abs = _per_step(_run(5.0), "a_abs")
        self.assertLessEqual(mean_abs, 1.0)
        self.assertGreater(mean_abs, 0.8)

    def test_noise_is_reproducible_for_a_given_seed(self):
        self.assertEqual(_per_step(_run(0.25, seed=3), "a_sq"),
                         _per_step(_run(0.25, seed=3), "a_sq"))
        self.assertNotEqual(_per_step(_run(0.25, seed=3), "a_sq"),
                            _per_step(_run(0.25, seed=4), "a_sq"))


if __name__ == "__main__":
    absltest.main()
