"""Reconstruct where a training run's wall clock went, from its ``timing`` artifact.

``nnx_ppo.algorithms.ppo.train_ppo`` times three things and logs them as *rates*:

===================== ================================================================
``throughput/train_sps``  ``ppo.n_envs * ppo.rollout_length / t`` for one PPO step,
                          measured around ``ppo_step_jit`` and the host-sync on
                          ``steps_taken`` -- so it excludes eval, video and checkpoint.
``throughput/eval_sps``   ``eval.n_envs * eval.max_episode_length / t`` for one eval
                          rollout, with an explicit ``block_until_ready``.
``throughput/video_sps``  ``video.episode_length / t`` for one video, covering the
                          render rollout, the MuJoCo render itself *and* ``video_fn``
                          (the upload).
===================== ================================================================

Multiplying each back out by its own constant recovers the seconds that piece cost, and
the constants come from the run's config. Checkpointing is **not** timed -- it is
recovered here as the part of an iteration's wall clock that the three gauges do not
explain, on the iterations where ``_should_run`` says a checkpoint was written.

The one subtlety is aligning the gauges to wall clock. A history row's ``_runtime`` is
stamped when wandb *commits* the row, which happens on a later ``log`` call, so the
``_runtime`` series lags the metric series by a fixed whole number of rows. The lag is
not documented and could change with the wandb client, so it is measured per run
(:func:`iteration_costs` picks the shift minimising the trimmed mean absolute
unexplained time) and reported alongside the result. On the 2026-09 dm_control runs the
answer is ``+1`` and the residual is a few milliseconds against ~15 s iterations, which
is what makes the checkpoint bucket -- a residual of a residual -- worth believing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

#: Row lags tried when aligning ``runtime_s`` to the gauges. ``+1`` is what the wandb
#: client has done throughout this project and is tried first, so a run whose residual
#: is degenerate (a run too short to discriminate) falls back to the usual answer rather
#: than to whichever shift won by a rounding error.
ALIGN_CANDIDATES: tuple[int, ...] = (1, 0, -1, 2, -2)

#: Above this, the alignment is not trustworthy and the per-iteration residual is noise
#: rather than checkpoint time. Iterations are 10-30 s here; the measured score on the
#: 2026-09 dm_control runs is single-digit milliseconds.
ALIGN_RESID_WARN_S = 0.5

#: Fraction of the largest absolute residuals ignored when scoring an alignment. The
#: genuinely unexplained iterations -- the checkpoints, and any stall -- are exactly the
#: ones with a large residual *at the correct* alignment, so scoring on the untrimmed
#: mean would reward hiding them. The median is not enough on its own: most iterations
#: are plain training, identical under every shift, so on a run where fewer than half of
#: the rows carry an eval the median is flat across shifts and picks a lag by rounding
#: error (this is not hypothetical -- it is what an eval cadence of 2.44 iterations
#: does).
ALIGN_TRIM = 0.10

#: The buckets, in the order they are plotted and reported. They partition ``runtime_s``
#: exactly: see :func:`budget`.
BUCKETS: tuple[str, ...] = ("train_s", "eval_s", "video_s", "checkpoint_s",
                            "overhead_s", "startup_tail_s")


@dataclass(frozen=True)
class LoopConfig:
    """The training-loop constants needed to turn the logged rates back into seconds.

    Read these off the run's own config (``config.ppo.*``, ``config.eval.*``,
    ``config.video.*``, ``config.checkpoint_every_steps``), never off a default -- they
    are exactly the knobs a "should I do this less often?" question is about.
    """

    n_envs: int
    rollout_length: int
    eval_n_envs: int
    eval_episode_length: int
    eval_every_steps: int
    video_episode_length: int
    video_every_steps: int
    checkpoint_every_steps: int

    @property
    def steps_per_iteration(self) -> int:
        return int(self.n_envs) * int(self.rollout_length)

    @property
    def eval_env_steps(self) -> int:
        return int(self.eval_n_envs) * int(self.eval_episode_length)

    @classmethod
    def from_index_row(cls, row: Mapping[str, object]) -> "LoopConfig":
        """Build from a row of the run index (dotted ``config.*`` columns)."""
        def need(key: str) -> int:
            value = row.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                raise KeyError(f"run config has no {key}; this analysis needs the "
                               f"Hydra-era schema (see dm_control_suite/README.md)")
            return int(value)

        return cls(
            n_envs=need("config.ppo.n_envs"),
            rollout_length=need("config.ppo.rollout_length"),
            eval_n_envs=need("config.eval.n_envs"),
            eval_episode_length=need("config.eval.max_episode_length"),
            eval_every_steps=need("config.eval.every_steps"),
            video_episode_length=need("config.video.episode_length"),
            video_every_steps=need("config.video.every_steps"),
            checkpoint_every_steps=need("config.checkpoint_every_steps"),
        )


def checkpoint_mask(steps: Sequence[float], every_steps: int,
                    *, last_step: int = 0) -> np.ndarray:
    """Which of these logged steps wrote a checkpoint.

    Replicates ``nnx_ppo.algorithms.ppo._should_run`` exactly -- ``(s // every) > (last
    // every)``, carrying ``last`` forward -- rather than testing divisibility, which
    would find no checkpoints at all: a step count is ``n_envs * rollout_length`` per
    iteration and never lands on the 10 M grid.

    ``last_step`` is the loop's ``last_checkpoint_step`` when the first of these rows was
    reached. The default 0 is the fresh-start case: ``initial_eval`` is ``True`` whenever
    ``train.py`` starts a run from scratch, and it writes a checkpoint at step 0. Getting
    this wrong is not cosmetic -- with ``-every_steps`` instead the very first iteration
    is marked as a checkpoint, and that iteration carries the XLA compilation, so an
    extra minute of compile time would be reported as the cost of writing a checkpoint.
    A run resumed from a checkpoint (``initial_eval=False``) should pass the step it
    resumed at.

    The step-0 checkpoint itself is not represented, because the step-0 row carries no
    ``train_sps`` and so is not in the artifact; its cost is part of the startup bucket.
    """
    if every_steps <= 0:
        return np.zeros(len(steps), dtype=bool)
    out = np.zeros(len(steps), dtype=bool)
    last = int(last_step)
    for i, step in enumerate(steps):
        step = int(step)
        if (step // int(every_steps)) > (last // int(every_steps)):
            out[i] = True
            last = step
    return out


@dataclass
class Alignment:
    """How well the wall clock lines up with the gauges, and at which row lag.

    ``score_s`` is what the lag was chosen on: the mean absolute unexplained time per
    iteration over the quietest ``1 - ALIGN_TRIM`` of them. ``median_resid_s`` and
    ``p90_resid_s`` are untrimmed, for reporting.
    """

    shift: int
    score_s: float
    median_resid_s: float
    p90_resid_s: float
    ok: bool = field(init=False)

    def __post_init__(self) -> None:
        self.ok = bool(self.score_s <= ALIGN_RESID_WARN_S)


def iteration_costs(series: pd.DataFrame, cfg: LoopConfig) -> tuple[pd.DataFrame, Alignment]:
    """Per-iteration seconds, from a ``timing`` artifact frame and the run's config.

    ``series`` is the artifact as written: ``_step``, ``runtime_s`` and the three
    ``*_sps`` gauges, one row per logged iteration, ascending.

    Returns the frame with the derived columns below, plus the alignment that was used:

    ``train_s`` / ``eval_s`` / ``video_s``
        seconds that piece cost in this iteration (0 where it did not run).
    ``accounted_s``
        their sum.
    ``wall_s``
        the iteration's true duration, from the lag-corrected ``runtime_s`` difference.
        NaN on the first rows, where there is no earlier stamp to difference.
    ``other_s``
        ``wall_s - accounted_s``: checkpoint writing, the ``wandb.log`` call, and any
        time the process spent blocked on something outside the loop.
    ``is_checkpoint``
        whether this iteration wrote a checkpoint.
    """
    frame = series.copy()
    for column in ("train_sps", "eval_sps", "video_sps"):
        if column not in frame:
            frame[column] = np.nan

    # A rate of exactly 0 would be a divide-by-zero; it never occurs, but a NaN is the
    # honest answer if it ever does.
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["train_s"] = cfg.steps_per_iteration / frame["train_sps"]
        frame["eval_s"] = (cfg.eval_env_steps / frame["eval_sps"]).fillna(0.0)
        frame["video_s"] = (cfg.video_episode_length / frame["video_sps"]).fillna(0.0)
    frame["accounted_s"] = frame[["train_s", "eval_s", "video_s"]].sum(axis=1)

    diff = frame["runtime_s"].diff()
    best: Alignment | None = None
    for shift in ALIGN_CANDIDATES:
        resid = (diff.shift(shift) - frame["accounted_s"]).abs().dropna()
        if len(resid) < 3:
            continue
        keep = resid.nsmallest(max(3, int(round(len(resid) * (1 - ALIGN_TRIM)))))
        candidate = Alignment(shift=shift, score_s=float(keep.mean()),
                              median_resid_s=float(resid.median()),
                              p90_resid_s=float(resid.quantile(0.9)))
        # Strictly less, and ALIGN_CANDIDATES leads with +1, so a tie keeps the lag the
        # wandb client has actually had rather than whichever shift sorted first.
        if best is None or candidate.score_s < best.score_s:
            best = candidate
    if best is None:  # a run with one or two logged iterations
        best = Alignment(shift=ALIGN_CANDIDATES[0], score_s=float("nan"),
                         median_resid_s=float("nan"), p90_resid_s=float("nan"))

    frame["wall_s"] = diff.shift(best.shift)
    frame["other_s"] = frame["wall_s"] - frame["accounted_s"]
    frame["is_checkpoint"] = checkpoint_mask(frame["_step"].to_numpy(),
                                             cfg.checkpoint_every_steps)
    return frame, best


def budget(frame: pd.DataFrame, runtime_s: float) -> dict[str, float]:
    """Partition ``runtime_s`` into :data:`BUCKETS`. The six sum to it exactly.

    Only iterations with a measured ``wall_s`` contribute to the first five; everything
    else -- process start, imports, the env build, XLA compilation, the step-0 eval,
    video and checkpoint, the first iteration or two, and whatever happens after the
    last logged row -- is what is left, and lands in ``startup_tail_s``. That bucket is
    therefore a genuine "everything not inside a measured iteration", not a fudge, but
    it is a *mixture* and should not be read as any one thing.
    """
    valid = frame["wall_s"].notna()
    out = {
        "train_s": float(frame.loc[valid, "train_s"].sum()),
        "eval_s": float(frame.loc[valid, "eval_s"].sum()),
        "video_s": float(frame.loc[valid, "video_s"].sum()),
        "checkpoint_s": float(frame.loc[valid & frame["is_checkpoint"], "other_s"].sum()),
        "overhead_s": float(frame.loc[valid & ~frame["is_checkpoint"], "other_s"].sum()),
    }
    out["startup_tail_s"] = float(runtime_s) - sum(out.values())
    return out


def per_event(frame: pd.DataFrame) -> dict[str, float]:
    """Count and typical (median) cost of each periodic event, and the stalled excess.

    ``*_excess_s`` is ``total - n * median``: what the run spent above a steady-state
    version of itself. It is the handle on an outage. A run that never stalled has an
    excess of a few seconds; a run whose checkpoint hung for an hour has one hour of
    ``checkpoint_excess_s`` and an unchanged ``checkpoint_s_typical``.
    """
    valid = frame["wall_s"].notna()
    out: dict[str, float] = {}

    def summarise(name: str, costs: pd.Series) -> None:
        costs = costs.dropna()
        n = int(len(costs))
        median = float(costs.median()) if n else float("nan")
        total = float(costs.sum())
        out[f"n_{name}"] = n
        out[f"{name}_s_median"] = median
        out[f"{name}_s_total"] = total
        out[f"{name}_s_typical"] = median * n if n else 0.0
        out[f"{name}_excess_s"] = total - (median * n if n else 0.0)

    summarise("eval", frame.loc[valid & (frame["eval_s"] > 0), "eval_s"])
    summarise("video", frame.loc[valid & (frame["video_s"] > 0), "video_s"])
    summarise("checkpoint", frame.loc[valid & frame["is_checkpoint"], "other_s"])
    summarise("iteration", frame.loc[valid, "train_s"])
    return out


def worst_stall(frame: pd.DataFrame) -> dict[str, float]:
    """The single slowest iteration, and how much of it was unexplained.

    Reported so a suspected outage can be placed on the clock: add ``at_runtime_s`` to
    the run's ``created_at`` and compare with when the filesystem was down.
    """
    valid = frame["wall_s"].notna()
    if not valid.any():
        return {"worst_wall_s": float("nan"), "worst_at_runtime_s": float("nan"),
                "worst_at_step": float("nan"), "worst_kind": ""}
    sub = frame.loc[valid]
    row = sub.loc[sub["wall_s"].idxmax()]
    parts = {"eval": row["eval_s"], "video": row["video_s"],
             "checkpoint": row["other_s"] if row["is_checkpoint"] else 0.0,
             "other": 0.0 if row["is_checkpoint"] else row["other_s"],
             "train": row["train_s"]}
    return {
        "worst_wall_s": float(row["wall_s"]),
        "worst_at_runtime_s": float(row["runtime_s"]),
        "worst_at_step": float(row["_step"]),
        "worst_kind": max(parts, key=lambda k: parts[k]),
    }


#: An iteration counts as stalled when it took both this many seconds and this multiple
#: of the run's own median iteration. A healthy iteration in these tasks is 0.4-30 s and
#: a healthy checkpoint under 1 s, so this catches minute-scale hangs and nothing else;
#: the ratio keeps it fair to the slow tasks, the floor keeps it fair to the fast ones.
STALL_MIN_S = 60.0
STALL_MIN_RATIO = 5.0


def stalls(frame: pd.DataFrame, *, min_s: float = STALL_MIN_S,
           min_ratio: float = STALL_MIN_RATIO) -> pd.DataFrame:
    """The individual iterations that hung, with how much longer they took than usual.

    A per-run total cannot tell "this task is slow" from "the filesystem went away for an
    hour"; individually timestamped hangs can, because an outage puts them in the same
    wall-clock window across unrelated runs on unrelated nodes.

    Adds ``excess_s`` (over the run's median iteration) and ``kind``: which component of
    the iteration was the largest, i.e. what the process was doing when it stopped.
    Whichever it is says more about the outage than about the component -- during the
    2026-09-14 one, ``checkpoint``, ``video`` and ``eval`` all appear, because all three
    touch the filesystem or the network.
    """
    valid = frame["wall_s"].notna()
    if not valid.any():
        return frame.iloc[:0].assign(excess_s=[], excess_in_kind_s=[], kind=[])
    median_wall = float(frame.loc[valid, "wall_s"].median())
    hit = valid & (frame["wall_s"] > max(min_s, min_ratio * median_wall))
    out = frame[hit].copy()
    out["excess_s"] = out["wall_s"] - median_wall

    # What each component usually costs on the iterations where it runs at all, so the
    # excess can be charged to the component rather than to the whole iteration.
    typical = {
        "eval": float(frame.loc[valid & (frame["eval_s"] > 0), "eval_s"].median() or 0.0),
        "video": float(frame.loc[valid & (frame["video_s"] > 0), "video_s"].median() or 0.0),
        "checkpoint": float(frame.loc[valid & frame["is_checkpoint"], "other_s"].median()
                            or 0.0),
        "other": float(frame.loc[valid & ~frame["is_checkpoint"], "other_s"].median()
                       or 0.0),
    }
    kinds, in_kind = [], []
    for r in out.itertuples():
        parts = {"eval": r.eval_s, "video": r.video_s,
                 "checkpoint": r.other_s if r.is_checkpoint else 0.0,
                 "other": 0.0 if r.is_checkpoint else r.other_s}
        kind = max(parts, key=lambda k: parts[k])
        kinds.append(kind)
        # `excess_s` measures the hang against a whole normal iteration and is the right
        # thing to read; `excess_in_kind_s` measures it against what that component
        # normally costs, and is what `healthy_budget` takes back out -- charging the
        # stalled iteration's ordinary training time to the component as well would make
        # the bucket it is subtracted from go slightly negative across a cohort.
        in_kind.append(max(0.0, parts[kind] - typical[kind]))
    out["kind"] = kinds
    out["excess_in_kind_s"] = in_kind
    return out


#: Which bucket a hang of each ``kind`` was charged to, so it can be taken back out.
STALL_BUCKET = {"eval": "eval_s", "video": "video_s",
                "checkpoint": "checkpoint_s", "other": "overhead_s"}


def healthy_budget(buckets: Mapping[str, float], stalled: pd.DataFrame) -> dict[str, float]:
    """``buckets`` with the time lost to hangs removed from the bucket it landed in.

    Subtracting the stall only from the total -- and leaving it inside, say,
    ``checkpoint_s`` -- reads as "checkpointing costs 12 % of this run", which is the
    opposite of true: it cost 0.1 %, and then the filesystem went away while a checkpoint
    happened to be the thing waiting on it. The healthy buckets are what the run would
    have cost had it not stalled, and are what the cadence advice is read off.
    """
    out = dict(buckets)
    if len(stalled):
        for kind, group in stalled.groupby("kind"):
            out[STALL_BUCKET[str(kind)]] -= float(group["excess_in_kind_s"].sum())
    return out


def cadence_saving(frame: pd.DataFrame, cfg: LoopConfig, *,
                   eval_every: int | None = None,
                   video_every: int | None = None,
                   checkpoint_every: int | None = None) -> float:
    """Seconds this run would have saved at slower cadences, at its *typical* costs.

    Deliberately built from the median cost per event, not the total: the point of the
    projection is what a steady-state run would save, and a run that lost an hour to a
    filesystem outage must not be credited with saving that hour again.

    Fewer checkpoints also means a longer replay on preemption, and fewer eval points
    means a coarser curve; this function prices only the wall clock.
    """
    events = per_event(frame)
    total = 0.0
    for name, current, proposed in (
        ("eval", cfg.eval_every_steps, eval_every),
        ("video", cfg.video_every_steps, video_every),
        ("checkpoint", cfg.checkpoint_every_steps, checkpoint_every),
    ):
        if proposed is None or proposed <= current:
            continue
        n = events[f"n_{name}"]
        median = events[f"{name}_s_median"]
        if not n or not np.isfinite(median):
            continue
        total += median * n * (1.0 - current / proposed)
    return float(total)
