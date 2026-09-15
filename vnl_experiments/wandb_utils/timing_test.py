"""Tests for the wall-clock reconstruction in :mod:`vnl_experiments.wandb_utils.timing`.

The reconstruction is a residual of a residual -- checkpoint time is what is left of an
iteration after three measured rates are subtracted -- so it is tested against a
*simulated* training loop whose true costs are known, including the awkward parts: the
row lag on ``runtime_s``, an eval cadence that is not a whole number of iterations, and
a stall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnl_experiments.wandb_utils import timing

CFG = timing.LoopConfig(
    n_envs=8192, rollout_length=30,
    eval_n_envs=256, eval_episode_length=1000, eval_every_steps=600_000,
    video_episode_length=1000, video_every_steps=6_000_000,
    checkpoint_every_steps=10_000_000,
)


def simulate(n_iterations: int = 400, *, lag: int = 1, train_s: float = 15.0,
             eval_s: float = 1.5, video_s: float = 8.0, checkpoint_s: float = 4.0,
             log_s: float = 0.005, stall: tuple[int, float] | None = None):
    """A fake run, returned as the frame a ``timing`` artifact would hold, plus truth.

    ``lag`` is how many rows late wandb stamps ``_runtime`` relative to the row's own
    metrics -- the thing :func:`timing.iteration_costs` has to discover.
    """
    step = CFG.steps_per_iteration
    steps = np.arange(1, n_iterations + 1) * step
    ckpt = timing.checkpoint_mask(steps, CFG.checkpoint_every_steps)

    last_eval = last_video = 0
    rows, wall = [], []
    for i, s in enumerate(steps):
        did_eval = (s // CFG.eval_every_steps) > (last_eval // CFG.eval_every_steps)
        did_video = (s // CFG.video_every_steps) > (last_video // CFG.video_every_steps)
        if did_eval:
            last_eval = s
        if did_video:
            last_video = s
        extra = checkpoint_s if ckpt[i] else 0.0
        if stall is not None and i == stall[0]:
            extra += stall[1]
        rows.append({
            "_step": float(s),
            "train_sps": CFG.steps_per_iteration / train_s,
            "eval_sps": (CFG.eval_env_steps / eval_s) if did_eval else np.nan,
            "video_sps": (CFG.video_episode_length / video_s) if did_video else np.nan,
        })
        wall.append(train_s + (eval_s if did_eval else 0.0)
                    + (video_s if did_video else 0.0) + extra + log_s)

    startup = 300.0
    ends = startup + np.cumsum(wall)          # true wall clock at the end of each iter
    frame = pd.DataFrame(rows)
    # wandb stamps row i with the clock as of `lag` rows later.
    stamps = np.full(len(ends), np.nan)
    stamps[: len(ends) - lag] = ends[lag:] if lag else ends
    frame["runtime_s"] = stamps
    frame = frame.dropna(subset=["runtime_s"]).reset_index(drop=True)
    frame = frame[["_step", "runtime_s", "train_sps", "eval_sps", "video_sps"]]
    return frame, {"wall": np.array(wall), "startup": startup, "ckpt": ckpt,
                   "runtime_s": float(ends[-1])}


def test_checkpoint_mask_matches_should_run():
    """The mask is `_should_run`, not divisibility -- no step lands on the 10 M grid."""
    steps = np.arange(1, 90) * CFG.steps_per_iteration
    mask = timing.checkpoint_mask(steps, 10_000_000)
    assert not (steps % 10_000_000 == 0).any(), "premise: no step is a multiple"
    fired = steps[mask]
    assert len(fired) == 2                      # crosses 10 M and 20 M in 90 iterations
    assert list(fired // 10_000_000) == [1, 2]
    # first step past each boundary, and only the first
    assert fired[0] - CFG.steps_per_iteration < 10_000_000 <= fired[0]
    # The step-0 checkpoint has already been written when these rows start, so the very
    # first iteration -- the one carrying XLA compilation -- must NOT be marked.
    assert not mask[0]
    assert timing.checkpoint_mask(steps, 10_000_000, last_step=15_000_000).sum() == 1


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_alignment_is_recovered(lag):
    frame, _ = simulate(lag=lag)
    _, align = timing.iteration_costs(frame, CFG)
    assert align.shift == lag
    # `log_s` is genuine per-iteration overhead the gauges do not measure, so it is the
    # floor on the residual, not an alignment error.
    assert align.ok and align.score_s == pytest.approx(0.005, abs=1e-6)


def test_buckets_partition_runtime_and_match_truth():
    frame, truth = simulate()
    costs, align = timing.iteration_costs(frame, CFG)
    b = timing.budget(costs, truth["runtime_s"])

    assert sum(b.values()) == pytest.approx(truth["runtime_s"])
    assert set(b) == set(timing.BUCKETS)

    n_ckpt = int(costs["is_checkpoint"][costs["wall_s"].notna()].sum())
    assert b["checkpoint_s"] == pytest.approx(4.0 * n_ckpt + 0.005 * n_ckpt, abs=1e-6)
    # startup, the first iterations that have no earlier stamp to difference, and the
    # tail all land in one bucket -- and it is positive, not a fudge absorbing error.
    assert b["startup_tail_s"] > truth["startup"]
    assert b["train_s"] == pytest.approx(15.0 * int(costs["wall_s"].notna().sum()))


def test_stall_shows_as_excess_not_as_a_higher_typical_cost():
    """An outage must not inflate the per-event cost the projection is built from."""
    quiet, q_truth = simulate()
    stalled, s_truth = simulate(stall=(199, 3600.0))   # iteration 200 hangs for an hour

    q = timing.per_event(timing.iteration_costs(quiet, CFG)[0])
    s = timing.per_event(timing.iteration_costs(stalled, CFG)[0])

    assert s["checkpoint_s_median"] == pytest.approx(q["checkpoint_s_median"])
    assert s["checkpoint_excess_s"] == pytest.approx(q["checkpoint_excess_s"])
    # iteration 200 is not on the checkpoint grid, so the hour lands in `overhead_s`
    costs, _ = timing.iteration_costs(stalled, CFG)
    b = timing.budget(costs, s_truth["runtime_s"])
    assert b["overhead_s"] > 3600.0
    assert timing.budget(timing.iteration_costs(quiet, CFG)[0],
                         q_truth["runtime_s"])["overhead_s"] < 60.0


def test_stalls_flags_only_the_hang():
    quiet, _ = simulate()
    assert timing.stalls(timing.iteration_costs(quiet, CFG)[0]).empty

    stalled, _ = simulate(stall=(199, 3600.0))
    costs, _ = timing.iteration_costs(stalled, CFG)
    found = timing.stalls(costs)
    assert len(found) == 1
    assert found["kind"].iloc[0] == "other"
    assert found["excess_s"].iloc[0] == pytest.approx(3600.0, abs=1.0)
    # An 8 s video is 0.5x the iteration and well under the floor -- not a stall.
    assert (costs["video_s"] > 0).sum() > 1


def test_stall_floor_protects_the_fast_tasks():
    """CartpoleSwingup iterates in 0.4 s, so 5x its median is 2 s -- not a hang."""
    quick, _ = simulate(train_s=0.4, eval_s=0.15, video_s=8.0, checkpoint_s=0.4)
    costs, _ = timing.iteration_costs(quick, CFG)
    # the 8 s video is 20x the median iteration, but nowhere near the 60 s floor
    assert timing.stalls(costs).empty


def test_worst_stall_locates_the_hang():
    stalled, _ = simulate(stall=(199, 3600.0))
    costs, _ = timing.iteration_costs(stalled, CFG)
    worst = timing.worst_stall(costs)
    assert worst["worst_wall_s"] > 3600.0
    assert worst["worst_kind"] == "other"
    assert worst["worst_at_step"] == pytest.approx(200 * CFG.steps_per_iteration)


def test_cadence_saving_is_priced_at_typical_cost():
    stalled, _ = simulate(stall=(199, 3600.0))
    costs, _ = timing.iteration_costs(stalled, CFG)
    events = timing.per_event(costs)

    halved = timing.cadence_saving(costs, CFG, eval_every=2 * CFG.eval_every_steps)
    assert halved == pytest.approx(0.5 * events["n_eval"] * events["eval_s_median"])

    # a cadence that is not slower saves nothing, and never goes negative
    assert timing.cadence_saving(costs, CFG, eval_every=CFG.eval_every_steps) == 0.0
    assert timing.cadence_saving(costs, CFG, eval_every=1) == 0.0


def test_missing_video_gauge_is_zero_not_nan():
    """Twelve 2026-09-09 runs logged no video at all; they must still decompose."""
    frame, truth = simulate()
    frame = frame.drop(columns=["video_sps"])
    costs, align = timing.iteration_costs(frame, CFG)
    assert align.shift == 1
    assert (costs["video_s"] == 0).all()
    b = timing.budget(costs, truth["runtime_s"])
    assert b["video_s"] == 0.0
    assert sum(b.values()) == pytest.approx(truth["runtime_s"])


def test_healthy_budget_returns_the_stall_to_where_it_was_charged():
    """A hang during a checkpoint must not read as "checkpointing is expensive"."""
    quiet, q_truth = simulate()
    # stall iteration 200 -> not on the checkpoint grid; and iteration 40, which is.
    on_grid = int(np.flatnonzero(timing.checkpoint_mask(
        np.arange(1, 401) * CFG.steps_per_iteration, CFG.checkpoint_every_steps))[0])
    stalled, s_truth = simulate(stall=(on_grid, 3600.0))

    costs, _ = timing.iteration_costs(stalled, CFG)
    raw = timing.budget(costs, s_truth["runtime_s"])
    healthy = timing.healthy_budget(raw, timing.stalls(costs))

    assert raw["checkpoint_s"] > 3600.0
    quiet_costs, _ = timing.iteration_costs(quiet, CFG)
    quiet_ckpt = timing.budget(quiet_costs, q_truth["runtime_s"])["checkpoint_s"]
    assert healthy["checkpoint_s"] == pytest.approx(quiet_ckpt, abs=20.0)
    # nothing else moved
    assert healthy["train_s"] == pytest.approx(raw["train_s"])
    assert healthy["eval_s"] == pytest.approx(raw["eval_s"])
