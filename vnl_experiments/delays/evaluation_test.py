"""Tests for the eval-time action-noise and per-clip paths in ``evaluation``.

These use a toy env and a constant-zero-action network rather than the rodent, so
the executed action is exactly the perturbation and its statistics can be asserted
on directly.
"""

import dataclasses

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np
from flax import struct

from nnx_ppo.networks.types import (
    PPONetworkOutput,
    StatefulModule,
    StatefulModuleOutput,
)

from vnl_experiments.delays.evaluation import (
    _TERMINATION_REASONS,
    eval_dataset,
    flat_summary,
    trace_dataset,
)

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


@struct.dataclass
class _VaryingState:
    obs: jp.ndarray
    reward: jp.ndarray
    done: jp.ndarray
    metrics: dict
    clip_idx: jp.ndarray
    t: jp.ndarray


@dataclasses.dataclass(frozen=True)
class _VaryingEnv:
    """Toy env that terminates at a different step in every clip.

    The per-clip identities are only worth asserting on data that actually varies
    per clip: if the per-clip block were (say) the aggregate broadcast back out, a
    test on a constant-across-clips env would pass. So this env gives clip ``c`` a
    lifespan of ``(c + 1) * stride`` steps, a unit reward every step (making
    ``episode_reward`` equal to the lifespan), a per-step error of ``c + 1`` (so the
    divide-by-lifespan path in ``errors`` has a distinct value per clip), and a
    termination reason that alternates between two of the four.
    """

    stride: int = 17
    action_size: int = ACTION_SIZE

    def _metrics(self, clip_idx, terminated, t):
        # `terminations/*` are 0/1 flags that fire only on the terminating step;
        # everything else accumulates every step.
        reasons = {f"terminations/{r}": jp.zeros(()) for r in _TERMINATION_REASONS}
        fires = jp.astype(terminated, float)
        # Even clips end one way, odd clips another -- so a per-reason vector is not
        # reconstructable from the rate alone.
        reasons["terminations/root_too_far"] = jp.where(clip_idx % 2 == 0, fires, 0.0)
        reasons["terminations/root_too_rotated"] = jp.where(clip_idx % 2 == 1, fires, 0.0)
        return {
            **reasons,
            "terminations/any": fires,
            "rewards/unit": jp.ones(()),
            "joint_l2_error": jp.astype(clip_idx, float) + 1.0,
            # The rodent env reports the reference frame it is tracking; the trace passes
            # whatever the env reports straight through, so the toy env reports one too.
            "current_frame": jp.astype(t, float),
        }

    def reset(self, key, clip_idx=0, start_frame=0):
        clip_idx = jp.asarray(clip_idx)
        return _VaryingState(
            obs=jp.zeros(self.action_size),
            reward=jp.zeros(()),
            done=jp.zeros((), dtype=bool),
            metrics=self._metrics(clip_idx, jp.zeros((), dtype=bool),
                                  jp.zeros((), dtype=jp.int32)),
            clip_idx=clip_idx,
            t=jp.zeros((), dtype=jp.int32),
        )

    def step(self, state, action):
        t = state.t + 1
        terminated = t >= (state.clip_idx + 1) * self.stride
        return state.replace(
            reward=jp.ones(()),
            done=terminated,
            metrics=self._metrics(state.clip_idx, terminated, t),
            t=t,
        )


def _run(action_noise, *, seed: int = 0) -> dict:
    return eval_dataset(_ActionProbeEnv(), _ZeroActionNet(), N_CLIPS, N_STEPS,
                        0.01, jax.random.key(seed), None, action_noise)


def _run_varying(*, per_clip: bool = True, clip_names=None,
                 limit_clips: int | None = None, seed: int = 0) -> dict:
    return eval_dataset(_VaryingEnv(), _ZeroActionNet(), N_CLIPS, N_STEPS,
                        0.01, jax.random.key(seed), limit_clips, None,
                        per_clip=per_clip, clip_names=clip_names)


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


class PerClipTest(absltest.TestCase):
    """The per-clip block must be a *view* of the aggregates, not a second computation.

    Every assertion here is an identity between the block and a number the record
    already published. Together they are the evidence for ``EvalProducer.VERSION``
    staying at 3 across this change: the aggregates are untouched, and the block adds
    no number that is not already implied by them.
    """

    def test_the_block_is_absent_unless_asked_for(self):
        self.assertNotIn("per_clip", _run_varying(per_clip=False))
        # The default, which is what every already-stored artifact was made with.
        self.assertNotIn("per_clip", eval_dataset(
            _VaryingEnv(), _ZeroActionNet(), N_CLIPS, N_STEPS, 0.01,
            jax.random.key(0), None, None))

    def test_asking_for_it_changes_nothing_else(self):
        """The published aggregates are identical with and without the block.

        This is the load-bearing one. If it fails, the ``spec_id`` for the per-clip spec
        is no longer poolable with the default one and ``VERSION`` has to move.
        """
        without = _run_varying(per_clip=False)
        with_block = dict(_run_varying(per_clip=True))
        with_block.pop("per_clip")
        self.assertEqual(without, with_block)

    def test_the_data_actually_varies_per_clip(self):
        """Guards the identities below from being vacuous on constant data."""
        block = _run_varying()["per_clip"]
        self.assertLen(set(block["lifespan_steps"]), N_CLIPS)
        self.assertLen(set(block["episode_reward"]), N_CLIPS)

    def test_reward_and_lifespan_means_match(self):
        result = _run_varying()
        block = result["per_clip"]
        self.assertAlmostEqual(float(np.mean(block["episode_reward"])),
                               result["episode_reward"]["mean"], places=5)
        self.assertAlmostEqual(float(np.std(block["episode_reward"])),
                               result["episode_reward"]["std"], places=5)
        self.assertAlmostEqual(float(np.mean(block["lifespan_steps"])),
                               result["lifespan_steps"]["mean"], places=5)

    def test_reward_term_episode_totals_match(self):
        result = _run_varying()
        for term, stats in result["reward_terms"].items():
            self.assertAlmostEqual(
                float(np.mean(result["per_clip"]["env"][f"rewards/{term}"])),
                stats["mean"], places=5, msg=term)

    def test_termination_flags_match_the_rates(self):
        result = _run_varying()
        env = result["per_clip"]["env"]
        for reason in _TERMINATION_REASONS:
            self.assertAlmostEqual(float(np.mean(env[f"terminations/{reason}"])),
                                   result["termination_rate"][reason],
                                   places=5, msg=reason)
        any_rate = float(np.mean(env["terminations/any"]))
        self.assertAlmostEqual(any_rate, result["termination_rate"]["any"], places=5)
        self.assertAlmostEqual(1.0 - any_rate,
                               result["termination_rate"]["survived"], places=5)
        # And the two reasons really do split across clips, so the per-reason vectors
        # carry information the scalar rate does not.
        self.assertNotEqual(list(env["terminations/root_too_far"]),
                            list(env["terminations/root_too_rotated"]))

    def test_errors_are_per_step_sums_needing_the_lifespan_divide(self):
        """The documented unit of the `env` block: a masked sum, not a mean."""
        result = _run_varying()
        block = result["per_clip"]
        lifespan = np.asarray(block["lifespan_steps"], dtype=float)
        raw = np.asarray(block["env"]["joint_l2_error"], dtype=float)
        per_step = raw / np.maximum(lifespan, 1.0)
        self.assertAlmostEqual(float(per_step.mean()),
                               result["errors"]["joint_l2_error"]["mean"], places=5)
        # The sum is genuinely not already a mean -- otherwise the divide would be a
        # no-op and the docstring would be wrong.
        self.assertGreater(float(raw.max()), float(per_step.max()) * 10)

    def test_lengths_and_limit_clips(self):
        block = _run_varying()["per_clip"]
        self.assertEqual(block["clip_ids"], list(range(N_CLIPS)))
        self.assertLen(block["episode_reward"], N_CLIPS)
        self.assertLen(block["env"]["rewards/unit"], N_CLIPS)

        limited = _run_varying(limit_clips=2)
        self.assertEqual(limited["n_clips"], 2)
        self.assertEqual(limited["per_clip"]["clip_ids"], [0, 1])
        self.assertLen(limited["per_clip"]["episode_reward"], 2)

    def test_clip_names_round_trip(self):
        names = ["Walk", "Rear", "FaceGroom", "LGroom"]
        self.assertEqual(_run_varying(clip_names=names)["per_clip"]["clip_names"],
                         names)
        # Truncated with the clips, so name i always belongs to clip i.
        self.assertEqual(
            _run_varying(clip_names=names, limit_clips=2)["per_clip"]["clip_names"],
            names[:2])
        # numpy string arrays are what `ReferenceClips.clip_names` actually is.
        self.assertEqual(
            _run_varying(clip_names=np.array(names))["per_clip"]["clip_names"], names)

    def test_absent_clip_names_are_recorded_as_none(self):
        """`new_eval` has no labels, and the artifact should say so rather than omit it."""
        block = _run_varying(clip_names=None)["per_clip"]
        self.assertIn("clip_names", block)
        self.assertIsNone(block["clip_names"])

    def test_the_block_cannot_reach_a_wandb_summary(self):
        """`flat_summary` enumerates keys, so a 169-element list cannot leak into it."""
        record = {"datasets": {"old_eval": _run_varying()}}
        flat = flat_summary(record)
        self.assertTrue(flat, "expected some summary keys")
        self.assertFalse([k for k in flat if "per_clip" in k])
        for value in flat.values():
            self.assertIsInstance(value, float)


def _trace_varying(*, limit_clips=None, max_steps=None, seed: int = 0) -> dict:
    return trace_dataset(_VaryingEnv(), _ZeroActionNet(), N_CLIPS, N_STEPS,
                         0.01, jax.random.key(seed), limit_clips, max_steps)


class TraceTest(absltest.TestCase):
    """The trace must be the same rollout as the eval, step by step.

    ``_trace_rollout`` is a second copy of ``_rollout``'s step function -- the
    accumulate-vs-emit difference is structural in ``nnx.scan`` and cannot be factored
    out -- so the guarantee that matters is behavioural: summing a trace over time has to
    reproduce the eval's accumulators. These tests are what keeps the two from drifting.
    """

    def test_shape_is_clip_major(self):
        trace = _trace_varying()
        for name, arr in trace.items():
            self.assertEqual(np.shape(arr), (N_CLIPS, N_STEPS), msg=name)
            self.assertEqual(np.asarray(arr).dtype, np.float32, msg=name)

    def test_summing_over_time_reproduces_the_eval(self):
        trace = _trace_varying()
        result = _run_varying()
        block = result["per_clip"]

        np.testing.assert_allclose(trace["reward"].sum(axis=1),
                                   block["episode_reward"], rtol=1e-5)
        # `running`, not `alive`, is the lifespan predicate -- see `_trace_rollout`.
        np.testing.assert_allclose(trace["running"].sum(axis=1),
                                   block["lifespan_steps"], rtol=1e-5)
        for name, per_clip in block["env"].items():
            np.testing.assert_allclose(trace[name].sum(axis=1), per_clip,
                                       rtol=1e-5, err_msg=name)

    def test_post_termination_steps_are_zeroed_and_flagged(self):
        trace = _trace_varying()
        alive = trace["alive"]
        # Clip c runs for (c + 1) * stride steps, so every clip has dead steps here and
        # the mask is actually exercised.
        self.assertTrue((alive.sum(axis=1) < N_STEPS).all())
        dead = alive == 0.0
        self.assertTrue(dead.any())
        # Every traced quantity is zero wherever the episode was already over.
        for name, arr in trace.items():
            if name == "alive":
                continue
            self.assertEqual(float(np.abs(arr[dead]).max()), 0.0, msg=name)
        # `alive` is monotone non-increasing: `done` latches, so no clip comes back.
        self.assertTrue((np.diff(alive, axis=1) <= 0).all())

    def test_the_two_flags_differ_by_exactly_the_terminating_step(self):
        """The subtlety `_trace_rollout` documents, pinned so it cannot drift silently.

        `_rollout` masks its accumulators on the pre-step `done` but increments
        `lifespan` from the post-step one, so the terminating step contributes to every
        reward term while not counting towards the lifespan.
        """
        trace = _trace_varying()
        terminated = _run_varying()["per_clip"]["env"]["terminations/any"]
        delta = trace["alive"].sum(axis=1) - trace["running"].sum(axis=1)
        np.testing.assert_allclose(delta, terminated, rtol=1e-5)
        self.assertTrue((np.asarray(terminated) == 1.0).all(),
                        "every clip in this env terminates, so every delta should be 1")

    def test_every_env_metric_is_traced(self):
        """Including `current_frame`, so a behaviour label is read rather than derived
        from `step * ctrl_dt * mocap_hz`."""
        trace = _trace_varying()
        for name in _run_varying()["per_clip"]["env"]:
            self.assertIn(name, trace)
        self.assertIn("current_frame", trace)

    def test_limit_clips_and_max_steps(self):
        trace = _trace_varying(limit_clips=2, max_steps=40)
        for name, arr in trace.items():
            self.assertEqual(np.shape(arr), (2, 40), msg=name)
        # Truncation makes the record incomplete, which is the documented cost: clip 1
        # would have run 34 steps and clip 2 would not have finished by step 40.
        self.assertEqual(float(trace["alive"][1].sum()), 34.0)


if __name__ == "__main__":
    absltest.main()
