"""Overview: no efference copy vs efference copy vs forward model, on six dm_control tasks.

A summary-figure dataset, not a new question. Every arm of the three-way architecture
comparison that the sibling folders analyse one task at a time
(``efference-copy-across-tasks``, ``ball-in-cup-first-look``, ``reacher-hard-first-look``,
``walker-forward-model-1b``) is put into **one** ``data.csv`` with **one** readout
definition, so a single overview figure can be drawn from it. Consistency across tasks
is preferred over the best per-task readout: BallInCup is bimodal and its cell means
will look odd, and that is accepted here (see ``ball-in-cup-first-look`` for the
readouts that task actually needs).

Tasks and conditions
--------------------
Tasks: CartpoleSwingup, ReacherHard, BallInCup, CheetahRun, WalkerWalk (torque only,
``servo_kp`` absent or 0), HumanoidWalk. These are the only current-era variant of each
environment.

``condition`` is one of

* ``no_efference`` -- ``DelayedMLP``, ``delay_k > 0``, ``efference_length = 0``;
* ``efference`` -- ``DelayedMLP``, ``delay_k > 0``, ``efference_length == delay_k``;
* ``forward_model`` -- ``FlatForwardModel``, ``efference_length == delay_k``, **including
  delay 0** (where its predictor has nothing to predict);
* ``undelayed`` -- ``DelayedMLP`` at ``delay_k = 0``, where the two MLP arms are the same
  network. A shared reference for both MLP arms.

Readouts (identical rules on every task)
----------------------------------------
Every eval series is first put on a common 4.8 M-step grid (nearest sample to each grid
step), because the eval cadence is 0.6 M on runs before 2026-09-15 and 4.8 M after, and
that is partly confounded with the arm (``efference-copy-across-tasks``).

* ``reward_480M`` -- mean of the grid points in (430 M, 480 M]: the last-50 M window of
  README §6, read at 480 M on every run, whatever its budget. There is no LR or entropy
  schedule in nnx-ppo (``total_steps`` only bounds the training loop), so a 1 G run read
  at 480 M is the same experiment as a 480 M run.
* ``reward_1B`` -- the same window ending at 1 G. Filled only where the run trained
  >= 1 G steps, on any task: most of WalkerWalk, and the HumanoidWalk runs at 1 G / 2 G.
  Blank for every 480 M run, so compare at 1 G only within the rows that have it.
* ``steps_to_solve`` -- first grid step at which the trailing 24 M-step mean (5 grid
  points) reaches ``solve_threshold``; blank if never (**censored, not missing** -- the
  run's ``budget`` says how long it had). Read over the run's whole curve.
* ``solve_threshold`` -- 90 % of the task's *anchor*, the mean ``reward_480M`` of that
  task's ``undelayed`` runs. Same rule as ``efference-copy-across-tasks`` except that the
  anchor is read at 480 M on every task (that folder used 1 G for WalkerWalk; the two
  thresholds differ by < 1 %).

Selection
---------
* Current era only (``net_params.*`` present), ``min_std = 0.001`` (excludes ten
  CartpoleSwingup ``min_std`` sweep runs), and not at ``d168093``, the last commit before
  the ``971ab99`` eval fix (excludes twelve CartpoleSwingup runs).
* Trained to budget: ``state == finished`` or ``summary._step >= total_steps``. Crashed /
  failed runs are out.
* ``servo_kp`` absent or 0. The six ``servo_center = "range"`` WalkerWalk runs are the
  ``walker-joint-stiffness`` kp-sweep baselines (note *"Sweeping kp."*): physically
  torque, but part of the servo cohort, which is being treated separately -- excluded.
* :data:`EXCLUDE` -- runs that pass the gates but need special treatment.
* ``has_readout``: the eval series must reach the run's own budget (within one grid
  step). Three HumanoidWalk runs trained to budget but stopped logging ``eval/*`` partway
  (``efference-copy-across-tasks`` caveat 1); this drops them.
* **De-duplicated** after all of the above, so an excluded run can't displace a usable
  twin. Two runs agreeing on task, network, delay, efference length and *both* seeds
  (init ``seed`` and PPO ``config.seed``) are one experiment run twice. Kept: the longest
  budget (so a WalkerWalk run keeps its 1 G readout), then the earliest launched.
  ``duplicates.txt`` lists what was dropped.

``runs.csv`` is the candidate pool (before the readout gate and de-duplication);
``data.csv`` is the final one-row-per-seed table.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/three-arm-overview/extract.py
    ../.venv/bin/python analysis/dm_control_suite/three-arm-overview/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/three-arm-overview/extract.py --check

History artifacts:
    python -m vnl_experiments.artifacts ensure --kind history \\
        --runs analysis/dm_control_suite/three-arm-overview/runs.csv \\
        --set project='"emiwar-team/nnx-ppo-delays"'
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

#: (key, registry task, ms per control step). ctrl_dt is asserted against the run record.
TASKS = (
    ("cartpole", "CartpoleSwingup", 10),
    ("reacher", "ReacherHard", 20),
    ("ball_in_cup", "BallInCup", 20),
    ("cheetah", "CheetahRun", 10),
    ("walker", "WalkerWalk", 25),
    ("humanoid", "HumanoidWalk", 25),
)
TASK_KEY = {task: key for key, task, _ in TASKS}
CTRL_DT_MS = {task: ms for _, task, ms in TASKS}

ARMS = ("undelayed", "no_efference", "efference", "forward_model")

MIN_STD = 0.001
PRE_EVAL_FIX_COMMIT = "d168093"

#: Passes every gate but needs special treatment.
EXCLUDE = {
    # Recorded `finished`, but a NaN reward at ~849.6 M steps killed the network (note on
    # the run; pre-9e216ae NaN guard only zeroed obs). Its 480 M readout would be fine,
    # but it is a crashed run.
    "xj19crft": "NaN reward at ~849.6M killed the run (see run note)",
}

GRID_STEPS = 4_800_000
FINAL_WINDOW_STEPS = 50_000_000
SMOOTH_WINDOW_STEPS = 24_000_000
READOUT_480M = 480_000_000
READOUT_1B = 1_000_000_000
SOLVE_FRACTION = 0.9

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _base(df: pd.DataFrame) -> pd.Series:
    reached = df["summary._step"].ge(df["config.ppo.total_steps"])
    return (
        df["net_params.delay_k"].notna()
        & df["net_params.min_std"].eq(MIN_STD)
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
        & df["env_params.servo_kp"].fillna(0.0).eq(0.0)
        & df["env_params.servo_center"].ne("range")
        & (df["state"].eq("finished") | reached)
    )


def _cell(task: str, arm: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        mask = _base(df) & df["env"].eq(task)
        delay = df["net_params.delay_k"]
        eff = df["net_params.efference_length"]
        cls = df["net_params.network_class"]
        if arm == "undelayed":
            return mask & cls.eq("DelayedMLP") & delay.eq(0)
        if arm == "forward_model":
            return mask & cls.eq("FlatForwardModel") & eff.eq(delay)
        mask &= cls.eq("DelayedMLP") & delay.gt(0)
        if arm == "efference":
            return mask & eff.eq(delay)
        return mask & eff.eq(0)
    return selector


CONDITIONS = {f"{TASK_KEY[task]}__{arm}": _cell(task, arm)
              for _, task, _ in TASKS for arm in ARMS}

DEDUPE_KEY = ["task", "network_class", "delay", "efference_length",
              "seed_init", "seed_ppo"]

#: Must hold within a task. comparability.txt is grouped by `{task}__{arm}`.
INVARIANTS = [
    "config.ppo.n_envs",
    "config.ppo.rollout_length",
    "config.ppo.learning_rate",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "config.ppo.clip_range",
    "config.ppo.discounting_factor",
    "config.ppo.gae_lambda",
    "config.ppo.gradient_clipping",
    "config.ppo.weight_decay",
    "config.ppo.normalize_advantages",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "env_params.sim_dt",
    "env_params.impl",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.predictor_hidden_sizes",
    "net_params.fm_loss_weight",
    "net_params.detach_prediction",
    "net_params.normalize_obs",
    "net_params.activation",
    "net_params.initializer_scale",
    "net_params.std_scale",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "repos.vnl_playground.commit",
    # Expected to vary; listed so the spread is on record.
    "config.ppo.total_steps",
    "config.eval.every_steps",
    "git_commit",
    "repos.nnx_ppo.commit",
]


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def reward_series(store: Store, wandb_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, HISTORY_SPEC_ID)
    if entry is None:
        raise SystemExit(f"no history artifact for {wandb_id}; run the `artifacts ensure` "
                         "command in the docstring (README §3: never drop silently)")
    try:
        hist = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None
    key = next((k for k in REWARD_KEYS if k in hist.columns), None)
    if key is None:
        return None
    out = hist.dropna(subset=[key])[["_step", key]]
    out = out.rename(columns={"_step": "step", key: "value"})
    # A requeued run can repeat a step across attempts; average the duplicates.
    return out.groupby("step", as_index=False)["value"].mean().sort_values("step")


def on_grid(series: pd.DataFrame) -> pd.DataFrame:
    """One point per GRID_STEPS: the sample nearest each grid step."""
    steps = series["step"].to_numpy()
    grid = np.arange(GRID_STEPS, steps.max() + 1, GRID_STEPS)
    idx = np.abs(steps[None, :] - grid[:, None]).argmin(axis=1)
    return series.iloc[np.unique(idx)].reset_index(drop=True)


def trailing_mean(steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean over (s - window, s]; NaN until the window is fully covered."""
    lo = np.searchsorted(steps, steps - SMOOTH_WINDOW_STEPS, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= SMOOTH_WINDOW_STEPS, out, np.nan)


def window_mean(grid: pd.DataFrame, upper: int) -> float | None:
    sub = grid[(grid["step"] > upper - FINAL_WINDOW_STEPS) & (grid["step"] <= upper)]
    return float(sub["value"].mean()) if len(sub) else None


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    task_key, arm = run["condition"].split("__")
    task = run["env"]
    budget = int(run["config.ppo.total_steps"])
    delay = int(run["net_params.delay_k"])
    row = {
        "task": task_key,
        "condition": arm,
        "network_class": run["net_params.network_class"],
        "delay": delay,
        "delay_ms": delay * CTRL_DT_MS[task],
        "efference_length": int(run["net_params.efference_length"]),
        "seed": f"i{int(run['seed'])}/p{int(run['config.seed'])}",
        "seed_init": int(run["seed"]),
        "seed_ppo": int(run["config.seed"]),
        "budget": budget,
        "reward_480M": None,
        "reward_1B": None,
        "steps_to_solve": None,
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        # Not written to data.csv; used for gating.
        "_curve_max_step": None,
        "_steps": None,
    }
    if series is None or series.empty:
        return row
    grid = on_grid(series)
    steps = grid["step"].to_numpy()
    row["_curve_max_step"] = int(series["step"].max())
    row["reward_480M"] = window_mean(grid, READOUT_480M)
    if budget >= READOUT_1B:
        row["reward_1B"] = window_mean(grid, READOUT_1B)
    row["_steps"] = steps
    row["_smooth"] = trailing_mean(steps, grid["value"].to_numpy())
    return row


def add_steps_to_solve(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    anchors = (df[df["condition"].eq("undelayed")]
               .groupby("task")["reward_480M"].agg(["mean", "count"]))
    missing = set(TASK_KEY.values()) - set(anchors.index)
    assert not missing, f"no undelayed anchor for {missing}"
    df["solve_threshold"] = df["task"].map(anchors["mean"] * SOLVE_FRACTION)
    out = []
    for _, r in df.iterrows():
        if r["_steps"] is None:
            out.append(None)
            continue
        hit = np.flatnonzero(r["_smooth"] >= r["solve_threshold"])
        out.append(int(r["_steps"][hit[0]]) if hit.size else None)
    df["steps_to_solve"] = pd.array(out, dtype="Int64")
    df["solved"] = df["steps_to_solve"].notna()
    print("\nanchors (mean reward_480M of undelayed runs) and thresholds:")
    for task, a in anchors.iterrows():
        print(f"  {task:12s} anchor {a['mean']:7.1f} (n={int(a['count'])})  "
              f"threshold {a['mean'] * SOLVE_FRACTION:7.1f}")
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ordered = df.sort_values(["budget", "created_at", "wandb_id"],
                             ascending=[False, True, True])
    keep = ~ordered.duplicated(DEDUPE_KEY, keep="first")
    lines = ["Duplicate groups (same task, network, delay, efference length and both "
             "seeds).",
             "Kept: longest budget, then earliest launched.", ""]
    cols = ["wandb_id", "budget", "created_at", "reward_480M", "reward_1B"]
    n = 0
    for key, g in ordered.groupby(DEDUPE_KEY, sort=True):
        if len(g) < 2:
            continue
        n += 1
        lines.append("{} {} delay {} eff {} seed i{}/p{}".format(*key))
        for idx, r in g[cols].iterrows():
            tag = "KEEP" if keep.loc[idx] else "drop"
            lines.append(f"  {tag}  " + "  ".join(f"{c}={r[c]}" for c in cols))
        lines.append("")
    lines.insert(2, f"{n} groups, {int((~keep).sum())} runs dropped.")
    return ordered[keep], "\n".join(lines) + "\n"


OUTPUT_COLUMNS = [
    "task", "condition", "network_class", "delay", "delay_ms", "efference_length",
    "seed", "seed_init", "seed_ppo", "budget",
    "reward_480M", "reward_1B", "solve_threshold", "steps_to_solve", "solved",
    "wandb_id", "wandb_name",
]


def main() -> None:
    args = pipeline.parse_args(__doc__)
    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    for task, ms in CTRL_DT_MS.items():
        dts = runs.loc[runs["env"].eq(task), "env_params.ctrl_dt"].unique()
        assert np.allclose(dts * 1000, ms), (task, dts)

    store = Store()
    rows = [build_row(run, reward_series(store, run["wandb_id"]))
            for _, run in runs.iterrows()]
    df = pd.DataFrame(rows)

    heartbeat = runs.set_index("wandb_id")["heartbeat_at"]
    excluded, stale = [], []
    for _, r in df.iterrows():
        if r["wandb_id"] in EXCLUDE:
            excluded.append((r, EXCLUDE[r["wandb_id"]]))
        elif r["_curve_max_step"] is None or r["_curve_max_step"] < r["budget"] - GRID_STEPS:
            # README §6: a history made while the run was training is a snapshot. Only an
            # artifact made after the run's last heartbeat proves the series really ends.
            made = store.lookup("history", r["wandb_id"], HISTORY_SPEC_ID).created_at
            if str(made) < str(heartbeat[r["wandb_id"]]):
                stale.append(r["wandb_id"])
            excluded.append((r, f"eval series ends at {r['_curve_max_step']} "
                                f"of {r['budget']}"))
    if stale:
        raise SystemExit(f"stale history snapshots for {stale}: re-produce with "
                         "`artifacts ensure ... --override`")
    gone = {r["wandb_id"] for r, _ in excluded}
    df = df[~df["wandb_id"].isin(gone)]

    df, dup_text = deduplicate(df)
    # Anchors from the final, de-duplicated undelayed rows only.
    df = add_steps_to_solve(df)
    df = df[OUTPUT_COLUMNS].sort_values(
        ["task", "condition", "delay", "seed"], ignore_index=True)

    ex_text = "Runs in runs.csv excluded from data.csv (before de-duplication):\n\n" + \
        "\n".join(f"  {r['wandb_id']}  {r['task']:12s} {r['condition']:14s} "
                  f"delay {r['delay']:2d}  {r['seed']:10s}  {why}"
                  for r, why in excluded) + "\n"
    print("\n" + ex_text)
    print(dup_text.split("\n\n")[0])
    print("\nrows per task x condition:")
    print(df.pivot_table(index="task", columns="condition", values="wandb_id",
                         aggfunc="count", fill_value=0).to_string())

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
        (HERE / "excluded.txt").write_text(ex_text)
        (HERE / "duplicates.txt").write_text(dup_text)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
