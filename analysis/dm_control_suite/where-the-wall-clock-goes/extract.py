"""Where does a dm_control run's wall clock actually go -- and are the eval, video and
checkpoint cadences worth what they cost?

A 480 M-step CheetahRun takes ten hours and a 1 G-step WalkerWalk six, which felt long
for tasks this small, and a cluster filesystem outage on 2026-09-14 is a candidate
explanation for part of it. This folder separates the two questions: how much of a
*healthy* run is spent on something other than training, and how much of these particular
runs was lost to the outage.

Nothing in the training loop logs a time budget, but enough is logged to reconstruct one.
``train_ppo`` emits three throughput gauges, each a known constant divided by the seconds
that piece took, and each timed around exactly one thing:

* ``throughput/train_sps`` -- the PPO step alone, excluding eval, video and checkpoint;
* ``throughput/eval_sps`` -- one eval rollout, with an explicit ``block_until_ready``;
* ``throughput/video_sps`` -- one video: the render rollout, the render, and the upload.

Multiplying each back out by its own config constant recovers seconds. What is left of an
iteration's wall clock (from the row's wandb ``_runtime`` stamp) after subtracting those
three is everything else -- and on the iterations where ``_should_run`` says a checkpoint
was written, that remainder *is* the checkpoint, since checkpointing is synchronous
(``StandardCheckpointer.close()`` blocks) and nothing else in the loop takes measurable
time. The reconstruction, the row-lag correction it needs, and its tests live in
``vnl_experiments/wandb_utils/timing.py``; this script only applies it.

The budget partitions ``runtime_s`` exactly, into six buckets: ``train``, ``eval``,
``video``, ``checkpoint``, ``overhead`` (unexplained time on non-checkpoint iterations --
near zero in a healthy run, and where a stall that is not a checkpoint lands), and
``startup_tail`` (process start, imports, the env build, XLA compilation, the step-0
eval/video/checkpoint, and anything after the last logged row -- a mixture, not one
thing).

Conditions are ``task@gpu``. GPU is a throughput confound worth 1.6-1.8x on this cluster
(``../README.md`` §6) and cannot be pooled; task changes every one of the six buckets.
Network class varies *within* a condition and is a caveat, not a condition: it moves
``train_s`` but leaves the periodic costs nearly alone, and splitting on it as well would
leave most cells at n = 1.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/where-the-wall-clock-goes/extract.py
    ../.venv/bin/python analysis/dm_control_suite/where-the-wall-clock-goes/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/where-the-wall-clock-goes/extract.py --check

Writes ``data.csv`` (one row per run: the six buckets, the per-event costs, the alignment
quality) and ``stalls.csv`` (every individual iteration that hung, with its UTC clock
time, which is what places the outage).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, index, pipeline, timing

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: The `timing` producer takes the project as a spec field defaulting to the rodent one,
#: exactly as `history` does (../README.md), so a control-suite artifact must be made
#: with an explicit override:
#:
#:     python -m vnl_experiments.artifacts ensure --kind timing \
#:         --runs analysis/dm_control_suite/where-the-wall-clock-goes/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
#:
#: Pinned here and asserted in main(), so a producer VERSION bump cannot quietly
#: re-resolve this folder onto artifacts made by different code.
TIMING_SPEC_ID = "timing20000-3089bf7d"

REQUIRES = ["index", f"timing:{TIMING_SPEC_ID}"]

#: "The past couple of weeks", and also the whole Hydra era in this project: the
#: pre-Hydra runs (<= 2026-07-08) log no `config.eval.*`, `config.video.*` or
#: `config.checkpoint_every_steps` at all, so a budget cannot be built for them.
COHORT_START = "2026-09-01"

#: Excluded: the twelve runs at this commit evaluated on the *training* env, which is
#: wrapped (`RewardScalingWrapper -> EpisodeWrapper`), and rendered no video because a
#: wrapped env has no `render`. That is a different eval configuration costing a
#: different amount, so their eval seconds are not the eval seconds of every run since.
#: Gated on the commit, not the date: both sides of the fix were created on 2026-09-09.
PRE_EVAL_FIX_COMMIT = "d168093"

#: Tasks and GPUs actually present in the cohort. main() checks that every cohort run
#: landed in a condition, so a task or a GPU model appearing later fails loudly instead
#: of being dropped.
TASKS = ("CartpoleSwingup", "WalkerWalk", "HumanoidWalk", "CheetahRun", "HopperHop")
GPUS = {"a100": "NVIDIA A100-SXM4-80GB", "h200": "NVIDIA H200"}

#: MuJoCo Warp and MuJoCo JAX are different physics backends with different throughput,
#: so the one warp run ("Hopper Hop in warp.") gets its own cell rather than being
#: averaged into the jax HopperHop median or silently dropped.
IMPLS = ("jax", "warp")


def _cohort(df: pd.DataFrame) -> pd.Series:
    """The runs this analysis is about, before they are split into conditions."""
    return (
        (pd.to_datetime(df["created_at"]) >= pd.Timestamp(COHORT_START, tz="UTC"))
        & (~df["git_commit"].fillna("").str.startswith(PRE_EVAL_FIX_COMMIT))
        # Two runs died in setup with zero steps and no gauges; there is no budget to
        # build for a run that never reached an iteration.
        & (df["summary._step"].fillna(0) > 0)
    )


def _cell(task: str, gpu: str, impl: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        return (_cohort(df) & (df["env"] == task) & (df["gpu"] == gpu)
                & (df["env_params.impl"] == impl))
    return selector


CONDITIONS = {
    f"{task}{'' if impl == 'jax' else '-warp'}@{key}": _cell(task, gpu, impl)
    for task in TASKS for key, gpu in GPUS.items() for impl in IMPLS
}

#: A time budget is only comparable across runs if the loop being budgeted is the same
#: loop. Every cadence and every size that enters the reconstruction is here, plus the
#: stack: a CUDA or MuJoCo bump moves throughput, and `repos.*.dirty` voids the commits.
INVARIANTS = [
    "config.ppo.n_envs",
    "config.ppo.rollout_length",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "config.eval.every_steps",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "config.video.every_steps",
    "config.video.episode_length",
    "config.video.render_kwargs.height",
    "config.video.render_kwargs.width",
    "config.video.render_kwargs.camera",
    "config.checkpoint_every_steps",
    "env_params.impl",
    "env_params.episode_length",
    "net_params.network_class",
    "net_params.critic_hidden_sizes",
    "cuda_version",
    "os",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "repos.nnx_ppo.dirty",
    "repos.vnl_playground.dirty",
    "git_commit",
]

#: Cadences the report prices, against the current 0.6 M / 6 M / 10 M.
#:
#: `eval_4p8M` is not an arbitrary round number: `conf/train/dmc.yaml` says in its own
#: comment that `eval.every_steps: 600_000` is brax's 60 M budget divided by the old
#: script's `num_evals = 100`, and that it was *not* rescaled when `total_steps` was
#: multiplied by 8. 4.8 M is the cadence that gives the 100 evals the number was chosen
#: to give. `checkpoint_50M` is priced in order to be able to say what it buys, which is
#: the question asked; it is not a recommendation.
SCENARIOS = {
    "eval_4p8M": {"eval_every": 4_800_000},
    "eval_10M": {"eval_every": 10_000_000},
    "video_30M": {"video_every": 30_000_000},
    "checkpoint_50M": {"checkpoint_every": 50_000_000},
    "intended": {"eval_every": 4_800_000, "video_every": 30_000_000},
}


def timing_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("timing", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def _wall_clock_s(run: pd.Series, series: pd.DataFrame) -> float:
    """The wall clock the budget partitions.

    Two snapshots of the same quantity disagree, and which is later depends on the run:
    the index's ``runtime_s`` was read when the index was last synced, and the artifact's
    last ``runtime_s`` stamp when the artifact was produced. For a finished run they
    agree to a second or two; for a run that is **still training** the artifact is minutes
    ahead, and taking the index value would make the startup/tail bucket -- the one
    defined as the remainder -- come out negative, which is how this was noticed.
    """
    stamps = series["runtime_s"].dropna()
    return float(max(float(run["runtime_s"]), float(stamps.max()) if len(stamps) else 0.0))


def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    """One row of data.csv: the run's identity, its cadences, and its time budget."""
    row: dict = {
        "condition": run["condition"],
        "task": run["env"],
        "gpu": run["gpu"],
        "impl": run.get("env_params.impl"),
        "network_class": pipeline.first_present(run, "net_params.network_class",
                                                "network_class"),
        "delay_k": pipeline.first_present(run, "net_params.delay_k", "config.delay_k"),
        "seed": pipeline.first_present(run, "config.seed", "seed"),
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run.get("created_at"),
        "state": run.get("state"),
        "git_commit": str(run.get("git_commit", ""))[:8],
        "total_steps": run.get("config.ppo.total_steps"),
        "actual_step": run.get("summary._step"),
        "runtime_s": run.get("runtime_s"),
        "eval_every_steps": run.get("config.eval.every_steps"),
        "video_every_steps": run.get("config.video.every_steps"),
        "checkpoint_every_steps": run.get("config.checkpoint_every_steps"),
        "has_timing": series is not None,
    }
    if series is None or series.empty:
        return row

    cfg = timing.LoopConfig.from_index_row(run)
    costs, align = timing.iteration_costs(series, cfg)
    runtime_s = _wall_clock_s(run, series)

    row["wall_clock_s"] = runtime_s
    row["steps_per_iteration"] = cfg.steps_per_iteration
    buckets = timing.budget(costs, runtime_s)
    row.update(buckets)
    row.update(timing.per_event(costs))
    row.update(timing.worst_stall(costs))
    row["align_shift"] = align.shift
    row["align_score_s"] = align.score_s
    row["align_p90_s"] = align.p90_resid_s
    row["align_ok"] = align.ok
    row["n_iterations_logged"] = int(len(costs))
    # The step the artifact actually reaches. Not the same as `actual_step` for a run
    # that is still training: that comes from the index, which was synced earlier.
    row["logged_max_step"] = float(series["_step"].max())

    for bucket in timing.BUCKETS:
        row[bucket.replace("_s", "_frac")] = row[bucket] / runtime_s if runtime_s else np.nan
    # Fractions partition the wall clock, so a negative one means the reconstruction has
    # overspent the clock it was given -- a bug, not a result. Fail rather than plot it.
    if row["startup_tail_frac"] < -0.01:
        raise AssertionError(
            f"{run['wandb_id']}: buckets exceed the wall clock by "
            f"{-row['startup_tail_s']:.0f} s ({-100 * row['startup_tail_frac']:.1f} %)")
    # What a run of this length *would* cost with no stall: the buckets with every
    # periodic cost priced at its own median. This is the number the cadence advice is
    # built on, and it is what makes the 2026-09-14 runs comparable with the rest.
    # Time lost to hangs, by the same rule that builds stalls.csv, so the per-run number
    # and the timestamped table are the same measurement. NOT the sum of the `*_excess_s`
    # columns: those also carry the ordinary jitter of 800 evals, which is not a stall.
    stalled = timing.stalls(costs)
    row["stall_s"] = float(stalled["excess_in_kind_s"].sum())
    row["n_stalls"] = int(len(stalled))
    row["healthy_wall_s"] = runtime_s - row["stall_s"]
    # The same six buckets with the hangs taken back out of whichever one they landed in.
    # These, not the raw buckets, are what "where does a healthy run's time go" means.
    healthy = timing.healthy_budget(buckets, stalled)
    for bucket, value in healthy.items():
        row[f"healthy_{bucket}"] = value
        row[f"healthy_{bucket.replace('_s', '_frac')}"] = (
            value / row["healthy_wall_s"] if row["healthy_wall_s"] else np.nan)
    for name, kwargs in SCENARIOS.items():
        row[f"save_{name}_s"] = timing.cadence_saving(costs, cfg, **kwargs)
    return row


def build_stalls(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    """Every individual iteration that hung, with the UTC clock time it hung at.

    A per-run total cannot distinguish "this task is slow" from "the filesystem went
    away for an hour"; a timestamped list of the hangs can, because an outage puts them
    in the same wall-clock window across unrelated runs on unrelated nodes.
    """
    if series is None or series.empty:
        return []
    cfg = timing.LoopConfig.from_index_row(run)
    costs, _ = timing.iteration_costs(series, cfg)
    started = pd.Timestamp(run["created_at"])

    return [{
        "condition": run["condition"],
        "wandb_id": run["wandb_id"],
        "task": run["env"],
        "gpu": run["gpu"],
        "step": float(r["_step"]),
        # `runtime_s` is the *commit* stamp of the row the gap was measured from, so
        # this is the end of the stalled iteration to within one iteration.
        "at_utc": (started + pd.Timedelta(seconds=float(r["runtime_s"]))
                   ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_s": float(r["wall_s"]),
        "excess_s": float(r["excess_s"]),
        "excess_in_kind_s": float(r["excess_in_kind_s"]),
        "kind": r["kind"],
        "is_checkpoint": bool(r["is_checkpoint"]),
    } for _, r in timing.stalls(costs).iterrows()]


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)

    # The conditions enumerate the tasks and GPUs seen so far. If the cohort has grown a
    # cell they do not cover, say so rather than reporting on the part that happened to
    # match -- a new GPU model is exactly the confound this analysis exists to respect.
    if args.refresh:
        full = index.load(args.project or PROJECT)
        missed = full[_cohort(full) & ~full["wandb_id"].isin(runs["wandb_id"])]
        if not missed.empty:
            cells = sorted({f"{r['env']}/{r['gpu']}/{r['env_params.impl']}"
                            for _, r in missed.iterrows()})
            raise SystemExit(
                f"{len(missed)} run(s) are in the cohort but match no condition: "
                f"{cells}.\nAdd the task / GPU / impl to TASKS, GPUS or IMPLS.")

    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    producer = get_producer("timing")
    spec_id = producer.spec_id(producer.spec(project=PROJECT))
    if spec_id != TIMING_SPEC_ID:
        raise SystemExit(
            f"timing spec_id has drifted: got {spec_id}, expected {TIMING_SPEC_ID}.\n"
            f"The budgets in data.csv were made by a different producer version; either "
            f"re-produce against the new id and update TIMING_SPEC_ID, or keep the pin "
            f"and do not mix the two.")

    rows, stalls = [], []
    for _, run in runs.iterrows():
        series = timing_of(store, run["wandb_id"], TIMING_SPEC_ID)
        rows.append(build_row(run, series))
        stalls.extend(build_stalls(run, series))

    df = pd.DataFrame(rows).sort_values(["condition", "wandb_id"], ignore_index=True)
    stalls_df = pd.DataFrame(stalls)
    if not stalls_df.empty:
        stalls_df = stalls_df.sort_values(["at_utc", "wandb_id"], ignore_index=True)

    missing = int((~df["has_timing"]).sum())
    if missing:
        print(f"\n*** {missing}/{len(df)} runs have no timing artifact and contribute no "
              f"budget. They are still rows in data.csv. Produce them with:\n"
              f"    python -m vnl_experiments.artifacts ensure --kind timing \\\n"
              f"        --runs {HERE.relative_to(HERE.parents[2])}/runs.csv \\\n"
              f"        --set project='\"{PROJECT}\"'\n")

    bad = df[df["align_ok"] == False] if "align_ok" in df else df.iloc[:0]  # noqa: E712
    if len(bad):
        print(f"*** {len(bad)} run(s) could not be aligned to better than "
              f"{timing.ALIGN_RESID_WARN_S} s per iteration; their checkpoint and "
              f"overhead buckets are not trustworthy: {list(bad['wandb_id'])}\n")

    # Independent check on the reconstruction. The number of evals, videos and
    # checkpoints found is not fitted to anything -- evals and videos are counted from
    # the gauges the run logged, and checkpoints from a replay of `_should_run` -- so
    # comparing all three against `floor(steps / every_steps)` tests the step-to-event
    # bookkeeping (and `checkpoint_mask` in particular) against the config alone. An
    # event landing on one of the first rows, which have no differenced wall clock, is
    # legitimately absent, hence the tolerance of one.
    counted = df[df["has_timing"]]
    checks = []
    for name, every in (("eval", "eval_every_steps"), ("video", "video_every_steps"),
                        ("checkpoint", "checkpoint_every_steps")):
        expected = np.floor(counted["logged_max_step"] / counted[every])
        off = (counted[f"n_{name}"] - expected).abs()
        checks.append(f"{name}: {int((off <= 1).sum())}/{len(counted)} within 1 "
                      f"(worst {off.max():.0f})")
    print("\nevent-count check vs config: " + "; ".join(checks))
    if any("within 1" in c and not c.startswith(c.split(":")[0] + f": {len(counted)}/")
           for c in checks):
        print("*** some runs disagree with the configured cadence; see logged_max_step")

    align = df[df["has_timing"]]
    print(f"alignment: lag {sorted(align['align_shift'].dropna().unique().tolist())}, "
          f"median unexplained {align['align_score_s'].median() * 1000:.1f} ms/iteration, "
          f"worst {align['align_score_s'].max() * 1000:.1f} ms\n")

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
    print(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(stalls_df, HERE / "stalls.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
