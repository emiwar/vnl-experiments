"""First look at ReacherHard: delay tolerance, seed spread, and what a forward model buys.

ReacherHard is new to this track -- it is not in the nine tasks the track README lists,
and nothing has been written about it yet. It is the two-joint planar arm the
equilibrium-point (lambda) model was formulated about, which is why it was added. This
folder is the first pass over the 37 runs that exist, and answers four questions off one
cohort:

1. **How much delay does it tolerate?**  ``reward_final`` against ``delay``, baseline arm.
2. **How large is the seed-to-seed spread?**  Three independent seed settings of the
   baseline at every delay, so the spread is measured rather than assumed. This is the
   number every other conclusion in the folder has to clear.
3. **Does an explicit forward model help?**  ``FlatForwardModel`` against ``DelayedMLP``.
4. **How many steps to reach 900 eval reward, with and without the forward model?**
   Threshold crossings read off the eval series in ``curves.csv``.

The four are one folder rather than four because they are one cohort, one pair of CSVs
and one comparability verdict; splitting them would mean four identical ``runs.csv``.

The two arms
------------
* ``DelayedMLP`` -- the baseline. The actor sees the observation delayed by ``delay_k``
  control steps plus an efference copy of the ``delay_k`` actions still in flight; the
  critic sees the fresh observation.
* ``FlatForwardModel`` -- identical actor and critic, plus a 4x256 predictor mapping
  (delayed obs, action buffer) -> current obs, whose output is handed to the actor. It is
  trained by its own loss (``fm_loss_weight = 1``, ``detach_prediction = True``) rather
  than by the policy gradient.

``efference_length == delay_k`` in every run, so the two arms have the same information
available and differ only in whether the state estimate is built explicitly.

Seeds: there are two of them, and they are not the same number
--------------------------------------------------------------
``train.py`` draws network initialisation from the top-level ``seed`` (wandb ``seed``,
Hydra default 42) and hands ``train.seed`` (wandb ``config.seed``, Hydra default 1234) to
PPO, where it keys env resets, rollouts and the eval episodes. The two were overridden
together for most of this cohort but *not* for the first baseline batch, which ran at the
default ``train.seed`` while setting ``seed`` implicitly to 42. So there are four distinct
seed settings here, not three:

    (init 42, ppo 1234)  baseline batch 1, 9 runs   <- `seed=` not passed at all
    (init 43, ppo 43)    baseline batch 2, 9 runs
    (init 45, ppo 45)    baseline batch 3, 9 runs
    (init 42, ppo 42)    forward model,   10 runs

The forward-model arm therefore does **not** share a seed with any baseline run, which is
the right situation for question 3 but has to be said out loud: reading ``seed`` alone
would call the FM arm and baseline batch 1 "seed 42" and make the contrast look paired
when it is not. ``data.csv`` carries ``seed_init``, ``seed_ppo`` and the combined label
``seed``, and every seed-aware figure groups on the label.

Preemption is not balanced across the arms
------------------------------------------
Every run was launched with ``requeue.enabled=true`` on ``gpu_requeue``. All nine
baseline batch-1 runs and three of batch 2 were killed and requeued; batch 3 and all ten
forward-model runs ran straight through. A light checkpoint does not save the env states,
so on resume each env redraws its episode phase and the actor's delay/efference queues
start empty (track README) -- a transient of at most one episode against a 480 M budget,
but one that lands on one arm and not the other. ``data.csv`` records ``n_restarts`` and
``resumed_from_step`` so the figures can be redrawn on the never-preempted subset, and
``plot.py`` draws that check.

Traps this folder gates on
--------------------------
* **The eval-env fix (``971ab99``, 2026-09-09).** Every run here is at
  ``7bedb5cc`` and post-dates it, so ``eval/*`` is the bare, unscaled, full-length
  series. Asserted rather than assumed: ``EVAL_FIX_MIN_DATE`` below, and
  ``reward_source`` is a column.
* **Control step.** ReacherHard runs at ``ctrl_dt = 0.02``, i.e. 20 ms per delay step --
  not the 25 ms of the Walker/Humanoid tasks and not ``style.CTRL_DT_MS``'s rodent 10.
  It is read off ``env_params.ctrl_dt`` and carried into ``data.csv`` rather than
  hard-coded in ``plot.py``.
* **The final eval point is not a measurement** (README §6). ``reward_final`` is the mean
  of the eval points in the last 50 M steps (~11 points at the 4.8 M cadence), not the
  run summary's last value, which is also kept as ``reward_last_point`` for contrast.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/reacher-hard-first-look/extract.py
    ../.venv/bin/python analysis/dm_control_suite/reacher-hard-first-look/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/reacher-hard-first-look/extract.py --check

Writes ``data.csv`` (one row per run) and ``curves.csv`` (the eval series). ``plot.py``
reads only those.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: The `history` producer takes the project as a **spec field** defaulting to the rodent
#: project, so a control-suite curve must be produced with an explicit override:
#:
#:     python -m vnl_experiments.artifacts ensure --kind history \
#:         --runs analysis/dm_control_suite/reacher-hard-first-look/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
#:
#: Pinned here and asserted in main(): a producer VERSION bump must be a deliberate
#: repoint, not a silent one.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

TASK = "ReacherHard"

#: Runs created before the `971ab99` eval-env fix (2026-09-09) report a scaled and
#: truncated `eval/*`. The whole cohort post-dates it; this asserts that rather than
#: trusting it, because a --refresh could pull in an older run.
EVAL_FIX_MIN_DATE = "2026-09-10"

#: End-of-training competence is the mean of the eval points in the last this-many steps
#: (README §6: one inline eval moves by a few percent). ~11 points at the 4.8 M cadence.
FINAL_WINDOW_STEPS = 50_000_000

#: The threshold question 4 asks about, plus two neighbours so the report can say whether
#: the answer is an artefact of where the line was drawn.
THRESHOLDS = (800, 900, 950)

#: A crossing is called on a trailing mean of this many eval points, not on a single
#: point, for the same reason `reward_final` is a window: the series is noisy at the few-
#: percent level and a single lucky eval would date the crossing ~10 M steps early.
CROSSING_WINDOW = 3


def _arm(network_class: str):
    """One architecture on ReacherHard.

    `state == "finished"` is safe to use here -- unlike the rodent torque sweep in
    README §6, every run in this cohort both finished and reached `total_steps`, which
    the `summary._step` invariant re-checks. The `|` form is kept anyway so a --refresh
    that picks up a run which died in its final eval does not silently drop it.
    """
    def selector(df: pd.DataFrame) -> pd.Series:
        reached_total = df["summary._step"].ge(df["config.ppo.total_steps"])
        return (
            df["env"].eq(TASK)
            & df["net_params.network_class"].eq(network_class)
            & (df["state"].eq("finished") | reached_total)
            & df["created_at"].ge(EVAL_FIX_MIN_DATE)
        )
    return selector


CONDITIONS = {
    "delayed_mlp": _arm("DelayedMLP"),
    "flat_forward_model": _arm("FlatForwardModel"),
}

#: Must be constant across the whole cohort for the arms to be comparable. `ctrl_dt`,
#: `reward_scale` and `episode_length` are here because they are what "reward" means;
#: `config.eval.*` because they are what "eval reward" means; the three repo commits
#: because `git_commit` covers only vnl-experiments.
INVARIANTS = [
    "env",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "repos.nnx_ppo.dirty",
    "repos.vnl_playground.dirty",
    "repos.vnl_experiments.dirty",
    "config.ppo.n_envs",
    "config.ppo.total_steps",
    "config.ppo.learning_rate",
    "config.ppo.rollout_length",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "config.ppo.discounting_factor",
    "summary._step",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "env_params.servo_kp",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.normalize_obs",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "config.eval.every_steps",
    "os",
    "cuda_version",
    "python",
]

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")
LIFESPAN_KEYS = ("eval/lifespan/mean", "lifespan_mean")


# --------------------------------------------------------------------------------------
# history -> series
# --------------------------------------------------------------------------------------


def history_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def series_of(hist: pd.DataFrame | None, keys) -> pd.DataFrame | None:
    """One logged series as a tidy step/value frame, or None if the run never logged it."""
    if hist is None:
        return None
    key = next((k for k in keys if k in hist.columns), None)
    if key is None:
        return None
    out = hist.dropna(subset=[key])[["_step", key]]
    out = out.rename(columns={"_step": "step", key: "value"}).sort_values("step")
    # A preempted run's curve is the concatenation of its attempts, so the same step can
    # appear twice with different values. Average the duplicates rather than letting the
    # threshold search see a sawtooth.
    return out.groupby("step", as_index=False)["value"].mean()


def crossing_step(series: pd.DataFrame, threshold: float,
                  window: int = CROSSING_WINDOW) -> int | None:
    """First step at which the trailing `window`-point mean of the series reaches
    `threshold`, or None if it never does.

    The trailing mean is the point of this function: at the 4.8 M eval cadence a single
    point above the line dates the crossing one eval early roughly a third of the time
    near threshold, which is 4.8 M steps of spurious difference between two arms.
    """
    if series is None or len(series) < window:
        return None
    smoothed = series["value"].rolling(window, min_periods=window).mean()
    hits = series.loc[smoothed.ge(threshold).fillna(False), "step"]
    return None if hits.empty else int(hits.iloc[0])


# --------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------


def seed_label(run: pd.Series) -> str:
    """The pair of seeds that actually differ between batches, as one label.

    `seed` alone is 42 for both baseline batch 1 and the whole forward-model arm, but
    those two batches ran at ppo seed 1234 and 42 respectively. Grouping on `seed` would
    pool them into one "seed 42" cell and hide a replicate.
    """
    return f"i{int(run['seed'])}/p{int(run['config.seed'])}"


def build_row(run: pd.Series, reward: pd.DataFrame | None,
              lifespan: pd.DataFrame | None) -> dict:
    resumed = run.get("requeue.resumed_from_step")
    row = {
        "condition": run["condition"],
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "vnlx_dirty": bool(run.get("repos.vnl_experiments.dirty")),
        "gpu": run["gpu"],
        "delay": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        "seed": seed_label(run),
        "seed_init": int(run["seed"]),
        "seed_ppo": int(run["config.seed"]),
        "ctrl_dt": float(run["env_params.ctrl_dt"]),
        "actual_step": int(run["summary._step"]),
        # Preemption bookkeeping: not balanced across the arms, so it is a column rather
        # than a footnote. `n_restarts` counts requeues; `resumed_from_step` is null for
        # a run that was killed before its first checkpoint and simply started over.
        "n_restarts": int(run.get("requeue.restart_count") or 0),
        "resumed_from_step": None if pd.isna(resumed) else int(resumed),
        # Post-`971ab99` throughout (asserted by the selector), so one value -- recorded
        # so a cohort that later straddles the fix cannot pool the two silently.
        "reward_source": "dmc_eval",
        "reward_last_point": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
    }

    if reward is None or reward.empty:
        row.update(reward_final=None, reward_final_n=0, reward_max=None,
                   reward_max_step=None, lifespan_final=None,
                   curve_points=0, curve_max_step=None)
        for threshold in THRESHOLDS:
            row[f"steps_to_{threshold}"] = None
        row["steps_to_900_first_point"] = None
        return row

    cutoff = reward["step"].max() - FINAL_WINDOW_STEPS
    window = reward[reward["step"] >= cutoff]
    row.update(
        reward_final=float(window["value"].mean()),
        reward_final_n=int(len(window)),
        reward_max=float(reward["value"].max()),
        reward_max_step=int(reward.loc[reward["value"].idxmax(), "step"]),
        curve_points=int(len(reward)),
        curve_max_step=int(reward["step"].max()),
    )
    if lifespan is not None and not lifespan.empty:
        tail = lifespan[lifespan["step"] >= lifespan["step"].max() - FINAL_WINDOW_STEPS]
        row["lifespan_final"] = float(tail["value"].mean())
    else:
        row["lifespan_final"] = None

    for threshold in THRESHOLDS:
        row[f"steps_to_{threshold}"] = crossing_step(reward, threshold)
    # The naive criterion, kept so the report can quote how much the smoothing moved the
    # answer rather than asserting that it was the right choice.
    row["steps_to_900_first_point"] = crossing_step(reward, 900, window=1)
    return row


def build_curves(run: pd.Series, reward: pd.DataFrame | None) -> list[dict]:
    if reward is None:
        return []
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "delay": int(run["net_params.delay_k"]), "seed": seed_label(run),
             "step": int(r.step), "reward_mean": float(r.value)}
            for r in reward.itertuples()]


# --------------------------------------------------------------------------------------


def seed_spread_report(df: pd.DataFrame) -> str:
    """Question 2, as a number, computed here so `plot.py` does not have to.

    The quantity is the spread across *seed settings within one delay* in the baseline
    arm -- pooled over delays as a root-mean-square of the per-delay standard deviations,
    which is the residual an arm-vs-arm difference has to clear.
    """
    base = df[(df.condition == "delayed_mlp") & df.reward_final.notna()]
    lines = ["Seed-to-seed spread, delayed_mlp arm (reward_final, dmc_eval)", ""]
    lines.append(f"{'delay':>6} {'n':>3} {'mean':>8} {'sd':>7} {'range':>7}  seeds")
    sds, means = [], []
    for delay, group in base.groupby("delay"):
        values = group["reward_final"]
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        spread = float(values.max() - values.min())
        sds.append(sd)
        means.append(float(values.mean()))
        seeds = ", ".join(f"{s}={v:.0f}" for s, v in
                          zip(group["seed"], group["reward_final"]))
        lines.append(f"{delay:>6} {len(values):>3} {values.mean():>8.1f} {sd:>7.1f} "
                     f"{spread:>7.1f}  {seeds}")

    pooled = float(np.sqrt(np.nanmean(np.square(sds))))
    grand = float(np.nanmean(means))
    lines += [
        "",
        f"pooled within-delay sd (rms over delays) : {pooled:.1f} reward",
        f"grand mean over delays                   : {grand:.1f} reward",
        f"pooled sd as a fraction of the mean      : {100 * pooled / grand:.1f} %",
        f"sd of the mean of {len(base.seed.unique())} seeds               "
        f"          : {pooled / np.sqrt(len(base.seed.unique())):.1f} reward",
        "",
        "Any arm-vs-arm difference smaller than the pooled sd is inside seed noise.",
        "The forward-model arm has ONE seed setting, so its points carry the full",
        "single-run spread, not the smaller spread of a 3-seed mean.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    producer = get_producer("history")
    spec_id = producer.spec_id(producer.spec(project=PROJECT))
    if spec_id != HISTORY_SPEC_ID:
        raise SystemExit(
            f"history spec_id has drifted: got {spec_id}, expected {HISTORY_SPEC_ID}.\n"
            f"The curves this analysis reads were made by a different spec or producer "
            f"VERSION. Update HISTORY_SPEC_ID deliberately, re-produce, and say so in "
            f"report.md -- do not silently repoint at a different generation of data.")

    rows, curves = [], []
    for _, run in runs.iterrows():
        hist = history_of(store, run["wandb_id"], spec_id)
        reward = series_of(hist, REWARD_KEYS)
        rows.append(build_row(run, reward, series_of(hist, LIFESPAN_KEYS)))
        curves.extend(build_curves(run, reward))

    df = pd.DataFrame(rows).sort_values(
        ["condition", "delay", "seed", "wandb_id"], ignore_index=True)
    curves_df = pd.DataFrame(
        curves, columns=["wandb_id", "condition", "delay", "seed", "step", "reward_mean"])
    if not curves_df.empty:
        curves_df = curves_df.sort_values(
            ["condition", "delay", "seed", "step"], ignore_index=True)

    # A stale `history` made while a run was still training stays in the store for ever
    # and `ensure` reports it as present (README §6). Compare what the artifact saw with
    # what the index says the run reached.
    stale = df[df.curve_max_step.notna() &
               (df.curve_max_step < df.actual_step - FINAL_WINDOW_STEPS)]
    if not stale.empty:
        print(f"\n*** {len(stale)} history artifact(s) stop well short of the run's "
              f"final step -- they were probably made mid-training. Re-produce with "
              f"`artifacts ensure --kind history --override`:\n"
              f"{stale[['wandb_id', 'curve_max_step', 'actual_step']].to_string(index=False)}\n")

    missing = int((df["curve_points"] == 0).sum())
    if missing:
        print(f"\n*** {missing}/{len(df)} runs have no history artifact; their "
              f"reward_final and steps_to_* are blank. Run:\n"
              f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
              f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
              f"        --set project='\"{PROJECT}\"'\n")

    report = comparability_report(runs, invariant_cols=INVARIANTS,
                                  group_col="condition")
    spread = seed_spread_report(df)
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
        (HERE / "seed_spread.txt").write_text(spread)
    print(report)
    print()
    print(spread)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
