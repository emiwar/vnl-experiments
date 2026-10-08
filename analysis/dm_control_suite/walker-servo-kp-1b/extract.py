"""On WalkerWalk at 1e9 steps, how do final reward and time-to-solve depend on ``servo_kp``
-- and does the stiffness at which the servo starts to help move with the delay?

The predecessor, ``../walker-joint-stiffness/``, swept ``servo_kp`` at 4.8e8 steps and
delays 0/5/10, every servo cell n=1. Its two open problems were both budget problems: at
delay 10 the torque baseline had not converged, so "stiffness raises the asymptote" and
"stiffness speeds learning" could not be separated; and the servo looked *slower* than
torque at delay 0, which a 4.8e8 window cannot distinguish from "worse".

The 2026-10-07 sweep (note *"Sweeping kp again"*, init/PPO seed 45/45) re-runs the
stiffness axis at **1e9** with a wider range -- ``servo_kp`` 0.25 to 512, twelve values at
delay 7 and four at delay 0. A second launch on 2026-10-08 (same seed) added all twelve
values at **delay 5**, the delay at which ``servo_kp = 64`` had turned out to learn
significantly *slower* than torque. Delay 7 (175 ms) is where torque control is still solvable but
already slowed, so it asks the question in the regime the predecessor never sampled.
Under the predecessor's settling-time account, the hip servo (the slowest joint) settles
faster than 175 ms only above ``servo_kp`` ~20 (``t_settle`` is exactly proportional to
``servo_kp ** -0.5``: 195.7 ms at 16, 138.3 ms at 32), so that is where any delay-specific
benefit should begin.

What is pooled, and how
-----------------------
Every WalkerWalk ``DelayedMLP`` run at delays 0/5/7/10 -- the delays where ``servo_kp``
varies at all -- at budgets 4.8e8 and 1e9, both actuators, every stiffness. That brings in:

* the new 1e9 kp sweep (16 runs that reached budget);
* the 1e9 ``servo_kp = 64`` and torque seeds from ``../walker-servo-x-forward-model/``,
  which give the two ends of the stiffness axis 3-5 seeds each at 1e9;
* the 4.8e8 kp sweep and torque runs from ``../walker-joint-stiffness/``.

The budget is a **column, not a confound**, because nothing in nnx-ppo or ``train.py``
depends on ``total_steps`` except the loop bound -- no learning-rate or entropy schedule
(checked 2026-10-08). A 4.8e8 run is therefore the first 4.8e8 steps of a 1e9 run, up to
GPU nondeterminism. That licenses two readouts that span budgets:

* ``steps_to_<thr>``, censored at each run's own budget. A 4.8e8 run that never got there
  says "> 4.8e8", which is weaker than a 1e9 run's "> 1e9" but not different in kind.
* ``reward_480``, the eval-window mean ending at 4.8e8 -- read off the 1e9 curves too, so
  the old and new sweeps can be compared at one matched budget.

``reward_final`` is at the run's *own* budget and is only ever compared within a budget.

Efference copy
--------------
The new sweep sets ``efference_length == delay_k``, as do the predecessor and the 1e9
servo-x-forward-model arms. A later torque/kp64 batch (2026-09-25, 2026-10-06/07) ran
``efference_length = 0`` at delays 5-10. Those are kept, flagged ``efference = "none"``,
and drawn hollow; every headline contrast is on ``efference = "copy"`` runs.
``../efference-copy-across-tasks/`` found no WalkerWalk efference effect out to 125 ms, and
``efference_check`` in `main` re-measures it at delay 7, where it matters here.

Readouts
--------
As in the predecessor, so the numbers are directly comparable with its tables:
``reward_final`` is the mean of the eval points in the last 5e7 steps (README §6), and
``steps_to_<thr>`` is the first step at which the trailing 2.4e7-step mean on a common
4.8e6-step grid reaches ``thr`` (`on_common_grid` -- the cohort mixes 6e5 and 4.8e6 eval
cadences, and an un-regridded trailing mean would average 8x more points for the older
runs).

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-servo-kp-1b/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-servo-kp-1b/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-servo-kp-1b/extract.py --check

Writes ``data.csv`` (one row per run), ``curves.csv`` (eval series on the common grid) and
``comparability.txt``. ``plot.py`` reads only the CSVs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: Same spec as every sibling WalkerWalk folder, so a curve here is the same bytes they
#: read. Produce missing ones with
#:
#:     python -m vnl_experiments.artifacts ensure --kind history \
#:         --runs analysis/dm_control_suite/walker-servo-kp-1b/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

TASK = "WalkerWalk"
NETWORK = "DelayedMLP"
MIN_STD = 0.001
BUDGETS = (480_000_000, 1_000_000_000)

#: The delays at which servo_kp takes more than one value. Torque and kp64 exist at many
#: more delays at 1e9; that delay axis is ../walker-servo-x-forward-model/'s question.
DELAYS = (0, 5, 7, 10)

#: Runs at or before this commit evaluated on the *training* env (track README).
PRE_EVAL_FIX_COMMIT = "d168093"

FINAL_WINDOW_STEPS = 50_000_000
LATE_WINDOW_STEPS = 50_000_000
DRAWDOWN_WINDOW_STEPS = 200_000_000
MATCHED_BUDGET = 480_000_000

GRID_STEPS = 4_800_000
SMOOTH_WINDOW_STEPS = 24_000_000

#: 900 is the project's WalkerWalk convention (``walker-joint-stiffness``,
#: ``walker-forward-model-1b``); 800 and 950 test that the ordering is not an artefact of it.
THRESHOLDS = (800, 900, 950)

SERVO_JOINTS = ("right_hip", "right_knee", "right_ankle",
                "left_hip", "left_knee", "left_ankle")

#: Two runs agreeing on all of these are one experiment launched twice, not two seeds.
#: Same rule as ``../walker-servo-x-forward-model/extract.py``.
DEDUPE_KEY = [
    "net_params.network_class",
    "net_params.delay_k",
    "net_params.efference_length",
    "net_params.min_std",
    "config.ppo.total_steps",
    "env_params.servo_kp",
    "env_params.servo_damping_ratio",
    "env_params.servo_center",
    "seed",
    "config.seed",
]


def _kp(df: pd.DataFrame) -> pd.Series:
    """``servo_kp`` with *absent* read as 0: a pre-``540e356`` run is the unpatched task."""
    return df["env_params.servo_kp"].fillna(0.0)


def _reached_budget(df: pd.DataFrame) -> pd.Series:
    # `state == "finished"` alone drops runs that trained fully and died in the final eval
    # (README §6). Gate on the property meant.
    return df["state"].eq("finished") | df["summary._step"].ge(df["config.ppo.total_steps"])


def _base(df: pd.DataFrame) -> pd.Series:
    """Every gate except de-duplication and the budget-reached gate."""
    return (
        df["env"].eq(TASK)
        & df["net_params.network_class"].eq(NETWORK)
        & df["net_params.min_std"].eq(MIN_STD)
        & df["config.ppo.total_steps"].isin(BUDGETS)
        & df["net_params.delay_k"].isin(DELAYS)
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
    )


def _first_of_duplicates(df: pd.DataFrame, candidates: pd.Series) -> pd.Series:
    """One run per DEDUPE_KEY group, the earliest launched (stable under refresh).

    The servo columns are filled first: a torque run has them absent, and groupby would
    otherwise drop those rows and exempt the whole torque arm from de-duplication.
    """
    pool = df[candidates].copy()
    for col in ("env_params.servo_kp", "env_params.servo_damping_ratio"):
        pool[col] = pool[col].fillna(0.0)
    pool["env_params.servo_center"] = pool["env_params.servo_center"].fillna("absent")
    keep = (pool.sort_values(["created_at", "wandb_id"])
                .groupby(DEDUPE_KEY, dropna=False)["wandb_id"].first())
    return df["wandb_id"].isin(set(keep))


def _cohort(df: pd.DataFrame) -> pd.Series:
    base = _base(df) & _reached_budget(df)
    return base & _first_of_duplicates(df, base)


CONDITIONS = {
    "torque": lambda df: _cohort(df) & _kp(df).eq(0.0),
    "servo": lambda df: _cohort(df) & _kp(df).gt(0.0),
}

INVARIANTS = [
    "config.ppo.n_envs",
    "config.ppo.rollout_length",
    "config.ppo.learning_rate",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "config.ppo.clip_range",
    "config.ppo.discounting_factor",
    "config.ppo.gae_lambda",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "env_params.sim_dt",
    "env_params.servo_damping_ratio",
    "env_params.servo_center",
    "env_params.servo_unlimited_half_range",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.normalize_obs",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "repos.vnl_playground.commit",
    # Design axes / known-varying, listed so their spread is on the record:
    "config.ppo.total_steps",
    "summary._step",
    "config.eval.every_steps",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_experiments.dirty",
]

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")


# --------------------------------------------------------------------------------------
# Curve readouts. Copied from ../walker-joint-stiffness/extract.py (not imported: question
# folders are not packages, and that folder is frozen) -- keep them identical so the
# numbers here and there mean the same thing.
# --------------------------------------------------------------------------------------

def history_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def reward_series(hist: pd.DataFrame | None) -> pd.DataFrame | None:
    if hist is None:
        return None
    key = next((k for k in REWARD_KEYS if k in hist.columns), None)
    if key is None:
        return None
    out = hist.dropna(subset=[key])[["_step", key]]
    return out.rename(columns={"_step": "step", key: "value"}).sort_values("step")


def on_common_grid(series: pd.DataFrame, cadence: int = GRID_STEPS) -> pd.DataFrame:
    """Nearest eval sample to each multiple of ``cadence``: one 256-episode eval per grid
    point whatever the run's own cadence (6e5 or 4.8e6), so the trailing mean averages the
    same number of samples for every run."""
    steps = series["step"].to_numpy()
    grid = np.arange(cadence, steps.max() + 1, cadence)
    idx = np.abs(steps[None, :] - grid[:, None]).argmin(axis=1)
    keep = series.iloc[np.unique(idx)]
    return keep.sort_values("step").reset_index(drop=True)


def trailing_mean(steps: np.ndarray, values: np.ndarray,
                  window: int = SMOOTH_WINDOW_STEPS) -> np.ndarray:
    """Mean over ``(s - window, s]``; NaN until the window is fully covered."""
    lo = np.searchsorted(steps, steps - window, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= window, out, np.nan)


def time_to_criterion(series: pd.DataFrame) -> dict:
    """``steps_to_<thr>`` (blank = censored at the run's budget) and ``smooth_max``."""
    grid = on_common_grid(series)
    steps = grid["step"].to_numpy()
    smooth = trailing_mean(steps, grid["value"].to_numpy())
    out = {"smooth_max": float(np.nanmax(smooth)) if np.isfinite(smooth).any() else None}
    for thr in THRESHOLDS:
        reached = np.where(smooth >= thr)[0]
        out[f"solved_{thr}"] = bool(len(reached))
        out[f"steps_to_{thr}"] = int(steps[reached[0]]) if len(reached) else None
    return out


def window_mean(series: pd.DataFrame, end: int, width: int = FINAL_WINDOW_STEPS):
    w = series[(series["step"] > end - width) & (series["step"] <= end + GRID_STEPS / 2)]
    return (float(w["value"].mean()), int(len(w))) if len(w) else (None, 0)


# --------------------------------------------------------------------------------------

def _seed_pair(run: pd.Series) -> str:
    return f"i{int(run['seed'])}/p{int(run['config.seed'])}"


def _launch(run: pd.Series) -> str:
    """A readable launch label: creation date + commit. Several launches are pooled."""
    return f"{str(run['created_at'])[:10]}/{str(run['git_commit'])[:7]}"


def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    delay = int(run["net_params.delay_k"])
    eff = int(run["net_params.efference_length"])
    kp = 0.0 if pd.isna(run.get("env_params.servo_kp")) else float(run["env_params.servo_kp"])
    settling = [run.get(f"servo.{j}.settling_time") for j in SERVO_JOINTS]
    settling = [s for s in settling if s is not None and np.isfinite(s)]
    row = {
        "condition": run["condition"],
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "launch": _launch(run),
        # The kp sweep at 1e9, init/PPO seed 45/45: delays 0/7 launched 2026-10-07,
        # delay 5 added 2026-10-08. (The 10-07 4.8e8 torque runs are not part of it.)
        "is_new_sweep": str(run["created_at"])[:10] in ("2026-10-07", "2026-10-08")
                        and kp > 0 and int(run["config.ppo.total_steps"]) == 1_000_000_000,
        "git_commit": run["git_commit"],
        "vnl_experiments_dirty": run.get("repos.vnl_experiments.dirty"),
        "gpu": run["gpu"],
        "budget": int(run["config.ppo.total_steps"]),
        "actual_step": run.get("summary._step"),
        "eval_every_steps": run.get("config.eval.every_steps"),
        "servo_kp": kp,
        "servo_center": run.get("env_params.servo_center"),
        "delay": delay,
        "efference_length": eff,
        # delay 0 has eff = delay = 0 trivially; it is "copy" in the sense of eff == delay.
        "efference": "copy" if eff == delay else ("none" if eff == 0 else "other"),
        "seed": int(run["seed"]),
        "ppo_seed": int(run["config.seed"]),
        "seed_pair": _seed_pair(run),
        "ctrl_dt": run.get("env_params.ctrl_dt"),
        # Slowest joint's 2 % settling time of the servo (the hip), from the run record.
        # Blank for torque runs: there is no servo.
        "settling_max_s": max(settling) if settling else None,
        "reward_source": "dmc_eval",
        "reward_last_point": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
    }
    blank = {"reward_final": None, "reward_final_n": 0, "reward_480": None,
             "reward_480_n": 0, "late_gain": None, "max_drawdown_late": None,
             "curve_points": 0, "curve_max_step": None, "smooth_max": None}
    blank.update({f"steps_to_{t}": None for t in THRESHOLDS})
    blank.update({f"solved_{t}": None for t in THRESHOLDS})
    if series is None or series.empty:
        row.update(blank)
        return row

    steps = series["step"].to_numpy()
    values = series["value"].to_numpy()
    end = int(steps.max())
    final, final_n = window_mean(series, end)
    r480, r480_n = window_mean(series, MATCHED_BUDGET)
    late = series[series["step"] >= end - LATE_WINDOW_STEPS]["value"]
    earlier = series[(series["step"] >= end - 3 * LATE_WINDOW_STEPS)
                     & (series["step"] < end - 2 * LATE_WINDOW_STEPS)]["value"]
    tail = values[steps >= end - DRAWDOWN_WINDOW_STEPS]
    row.update(
        reward_final=final, reward_final_n=final_n,
        reward_480=r480, reward_480_n=r480_n,
        late_gain=float(late.mean() - earlier.mean()) if len(earlier) else None,
        max_drawdown_late=float(np.max(np.maximum.accumulate(tail) - tail)),
        curve_points=int(len(steps)), curve_max_step=end,
    )
    row.update(time_to_criterion(series))
    return row


def build_curves(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    """The eval series on the common grid plus the trailing mean the criterion reads."""
    if series is None or series.empty:
        return []
    grid = on_common_grid(series)
    steps = grid["step"].to_numpy()
    values = grid["value"].to_numpy()
    smooth = trailing_mean(steps, values)
    kp = 0.0 if pd.isna(run.get("env_params.servo_kp")) else float(run["env_params.servo_kp"])
    eff = int(run["net_params.efference_length"])
    delay = int(run["net_params.delay_k"])
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "servo_kp": kp, "delay": delay, "budget": int(run["config.ppo.total_steps"]),
             "efference": "copy" if eff == delay else "none",
             "seed_pair": _seed_pair(run),
             "step": int(s), "reward_mean": float(v),
             "reward_smooth": None if not np.isfinite(m) else float(m)}
            for s, v, m in zip(steps, values, smooth)]


# --------------------------------------------------------------------------------------
# Reports written into comparability.txt
# --------------------------------------------------------------------------------------

def excluded_runs(full: pd.DataFrame, selected: pd.DataFrame) -> str:
    """Runs that pass every config gate but are not in the cohort, and why -- so a gap in
    the design is never silent (README §3)."""
    base = full[_base(full)]
    out = base[~base["wandb_id"].isin(selected["wandb_id"])].copy()
    if out.empty:
        return "\n  none\n"
    reached = _reached_budget(out)
    out["why"] = np.where(~reached, "did not reach budget", "duplicate of an earlier run")
    out["kp"] = _kp(out)
    cols = ["wandb_id", "created_at", "kp", "net_params.delay_k",
            "net_params.efference_length", "seed", "config.seed", "state",
            "summary._step", "config.ppo.total_steps", "why"]
    return "\n" + out[cols].sort_values(["why", "wandb_id"]).to_string(index=False) + "\n"


def design_grid(df: pd.DataFrame) -> str:
    lines = []
    for (budget, eff), g in df.groupby(["budget", "efference"]):
        tab = pd.crosstab(g["servo_kp"], g["delay"])
        lines.append(f"\nbudget {budget:.1e}, efference = {eff}: runs per servo_kp x delay\n"
                     f"{tab.to_string()}\n")
    return "".join(lines)


def seed_spread(df: pd.DataFrame) -> str:
    g = df.groupby(["budget", "efference", "servo_kp", "delay"])["reward_final"]
    tab = g.agg(["count", "mean", "min", "max"])
    tab["spread"] = tab["max"] - tab["min"]
    return "\nreward_final by cell (own budget)\n" + tab.round(1).to_string() + "\n"


def efference_check(df: pd.DataFrame) -> str:
    """Torque and kp64 at delays 5-10 exist with and without an efference copy. If the two
    agree within seed spread, the hollow (eff = 0) points are a fair side-by-side."""
    sub = df[(df.delay > 0) & df.servo_kp.isin([0.0, 64.0]) & (df.budget == 1_000_000_000)]
    tab = sub.groupby(["servo_kp", "delay", "efference"])["reward_final"].agg(
        ["count", "mean", "min", "max"]).round(1)
    t900 = sub.groupby(["servo_kp", "delay", "efference"])["steps_to_900"].agg(
        lambda s: sorted(round(v / 1e6, 1) if pd.notna(v) else float("inf") for v in s))
    return ("\n1e9, reward_final by efference\n" + tab.to_string()
            + "\n\n1e9, steps_to_900 (M; inf = censored) by efference\n"
            + t900.to_string() + "\n")


def budget_overlap(df: pd.DataFrame) -> str:
    """Where a (kp, delay, efference) cell exists at both budgets, compare reward_480 --
    which reads both at the same step -- to see what pooling launches costs."""
    sub = df[df.efference == "copy"]
    tab = sub.pivot_table(index=["servo_kp", "delay"], columns="budget",
                          values="reward_480", aggfunc=["count", "mean"])
    both = tab.dropna()
    return ("\nreward_480 where a cell exists at both budgets (eff = copy)\n"
            + (both.round(1).to_string() if len(both) else "  none") + "\n")


def main() -> None:
    args = pipeline.parse_args(__doc__)
    project = args.project or PROJECT

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    producer = get_producer("history")
    spec_id = producer.spec_id(producer.spec(project=PROJECT))
    if spec_id != HISTORY_SPEC_ID:
        raise SystemExit(f"history spec_id has drifted: got {spec_id}, expected "
                         f"{HISTORY_SPEC_ID}. Repoint deliberately, not silently.")

    rows, curves = [], []
    for _, run in runs.iterrows():
        series = reward_series(history_of(store, run["wandb_id"], spec_id))
        rows.append(build_row(run, series))
        curves.extend(build_curves(run, series))

    df = pd.DataFrame(rows).sort_values(
        ["budget", "delay", "servo_kp", "efference", "seed", "wandb_id"], ignore_index=True)
    curves_df = pd.DataFrame(curves).sort_values(
        ["budget", "delay", "servo_kp", "seed_pair", "step"], ignore_index=True)

    missing = int((df["curve_points"] == 0).sum())
    if missing:
        raise SystemExit(
            f"\n*** {missing}/{len(df)} runs have no history artifact. Produce them:\n"
            f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
            f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
            f"        --set project='\"{PROJECT}\"'\n")
    # A history artifact made mid-training is a snapshot (README §6). The last eval of a
    # complete run can sit up to one eval interval short of `_step` (998.5e6 vs 1000.2e6
    # at the 4.8e6 cadence), so that is the tolerance, not a percentage.
    stale = df[df["curve_max_step"] < df["actual_step"] - df["eval_every_steps"]]
    if len(stale):
        print(stale[["wandb_id", "curve_max_step", "actual_step"]].to_string(index=False))
        raise SystemExit("*** stale history artifacts; re-produce with --override")

    full = index.load(project=project)
    sections = [
        ("DESIGN GRID", design_grid(df)),
        ("WITHIN-CELL SPREAD", seed_spread(df)),
        ("EFFERENCE COPY CHECK", efference_check(df)),
        ("BUDGET OVERLAP (matched 4.8e8 readout)", budget_overlap(df)),
        ("EXCLUDED (pass the config gates, not in the cohort)", excluded_runs(full, runs)),
    ]
    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    for title, body in sections:
        print(f"\n# {title}{body}")
        report += "\n\n" + "#" * 78 + f"\n# {title}\n" + "#" * 78 + body
    if not args.check:
        (HERE / "comparability.txt").write_text(report + "\n")

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
