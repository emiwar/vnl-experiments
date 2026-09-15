"""On WalkerWalk at a 1e9-step budget, how much delay can each architecture absorb, and
how long does each take to solve the task?

The sibling folder [`../explicit-forward-model/`](../explicit-forward-model/) asked the
same architecture question across three tasks at 4.8e8 steps and came back with a
WalkerWalk answer it could not defend: the baseline was *still climbing at the end of
training*, so the gap it measured was "the forward model gets there sooner", not "the
baseline cannot get there". Its own first follow-up was to run longer. This folder is
that follow-up, on WalkerWalk only, at 1e9 steps -- and it adds the readout the speed
claim actually needs, which is a time-to-criterion rather than a level.

The two arms, matched in everything but the state estimate
----------------------------------------------------------
* ``DelayedMLP`` -- the actor sees the ``delay_k``-step-stale observation plus an
  efference copy of the ``delay_k`` actions still in flight; the critic sees the fresh
  observation. Any state estimate is implicit in the actor MLP.
* ``FlatForwardModel`` -- identical, plus a supervised predictor
  (delayed obs, action buffer) -> current obs whose output is handed to the actor.
  ``fm_loss_weight = 1`` and ``detach_prediction = True``, so the predictor is trained by
  its own regression loss and *not* by the policy gradient.

``efference_length == delay_k`` in every run, so the two arms are given exactly the same
information and differ only in whether the state estimate is built explicitly.

Two readouts, and why the second one exists
-------------------------------------------
``reward_final`` -- the mean of the eval points in the last 5e7 steps (README §6: a single
inline eval point moves by a few percent, so the last point is not a measurement).

``steps_to_<thr>`` -- the first step at which the **trailing 1e7-step mean** of eval reward
reaches ``thr``. This is the "how long to solve it" axis. Three things about it:

* **The 900 criterion is this project's convention, not a published standard.** dm_control
  returns are bounded in [0, 1000] by construction (1000 steps x a reward in [0, 1]), and
  the benchmark literature reports curves rather than declaring a task solved, so there is
  no number to inherit. 900 is chosen because it sits clearly above where these runs
  plateau when they fail (5.2e2-7.6e2) and clearly below where they plateau when they
  succeed (9.5e2-9.8e2), and the delay-0 runs of *both* arms reach ~9.8e2 -- so it is not
  a bar that the architecture question itself decides. ``steps_to_800`` and
  ``steps_to_950`` are extracted alongside it so ``plot.py`` can show that the ordering of
  the arms does not depend on the choice.
* **Trailing, not centred, and not a single point.** A criterion read off raw eval points
  fires on noise; a centred window reads the future. The trailing window means
  "by this step, recent average performance is at criterion", which is what a
  time-to-solve is supposed to say. The window is defined in *steps*, not samples, so it
  means the same thing at both budgets.
* **A run that never reaches the criterion is censored, not missing.** ``solved_<thr>`` is
  False and ``steps_to_<thr>`` is blank. ``smooth_max`` records how close it got, so the
  censored runs can be drawn rather than silently dropped -- the MLP arm never reaches 900
  at delay >= 15, and a plot that just ends its line there would read as absent data.

Cohort: prefer 1e9, and what the 4.8e8 runs are doing here
----------------------------------------------------------
The 1e9 cohort (18 runs, seeds 43 and 46) is the primary dataset and carries both
headline figures. It has exactly one hole: **no forward-model run at delay 5**, in either
seed.

The 4.8e8 cohort (14 runs, seed 42 -- the dataset of the sibling folder) is kept as a
`budget` column, for two jobs and no others:

1. It is a complete, single-launch grid over delays 0/2/5/7/10/15/20 for both arms, so it
   **fills the delay-5 hole** and adds the intermediate delays the 1e9 grid skips.
2. It is an independent third seed, so it says whether the 1e9 result replicates.

**The two budgets are never pooled into one line**, and this is not fussiness. The MLP is
the slower-learning arm, so a shared 4.8e8 readout is biased in favour of the conclusion
this folder is testing -- exactly README §6's "common mid-training budget" trap. Measured
here: the MLP at delay 10 finishes 4.8e8 at 6.9e2 and 1e9 at 9.4e2, because its learning
curve steps up at ~6.2e8 -- *past* the shorter budget -- while the forward model at delay
10 is at ~9.7e2 under either. So the arm gap at delay 10 reads +2.7e2 at 4.8e8 against
+2.6e1 at 1e9, an order of magnitude, almost all of it budget rather than architecture.
(Those two gaps are also different seeds, so the factor is indicative, not measured; the
MLP's own 6.9e2 -> 9.4e2 is the part that is unambiguous.) ``budget`` is therefore a
column, ``plot.py`` facets on it, and the reader is never shown a mean taken across it.

Selection gates that no run name or tag would reveal
----------------------------------------------------
* **The eval-env fix (``971ab99``, 2026-09-09).** Before it the eval env inherited the
  training wrappers, so ``eval/*`` was reward-scaled x10 *and* truncated by
  ``EpisodeWrapper``'s random start phase (track README). Gated on the commit, not the
  date -- both sides of the fix were created on 2026-09-09. This also excludes the entire
  pre-Hydra 9e1-run era, whose WalkerWalk runs have no ``net_params.*`` at all.
* **min_std.** The project contains a ``min_std`` sweep (0.01-0.2) whose runs are named
  exactly like the defaults. Gated to ``min_std == 0.001``.
* **Budget.** Only 4.8e8 and 1e9 are admitted. The 6e7 / 2.4e8 WalkerWalk runs are legacy
  and pre-fix on both counts.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/extract.py --check

Writes ``data.csv`` (one row per run) and ``curves.csv`` (the eval series). ``plot.py``
reads only those. See also ``nan_guard_inert.py``, which checks the one code difference
inside the 1e9 cohort that could in principle have changed the physics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: The `history` producer takes the project as a **spec field** defaulting to the *rodent*
#: project, so a dm_control history artifact only exists under an explicit override:
#:
#:     python -m vnl_experiments.artifacts ensure --kind history \
#:         --runs analysis/dm_control_suite/walker-forward-model-1b/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
#:
#: Because the project is hashed into the spec, control-suite and rodent curves can never
#: be pooled by accident. Same id as the sibling folder, so the 4.8e8 curves are literally
#: the same bytes that analysis read. Asserted in `main`.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

TASK = "WalkerWalk"

#: Default exploration width; the project sweeps this elsewhere under identical run names.
MIN_STD = 0.001

#: Runs at or before this commit evaluated on the *training* env, wrappers and all.
PRE_EVAL_FIX_COMMIT = "d168093"

#: The two admitted budgets. 1e9 is primary; 4.8e8 fills the delay-5 hole and adds a third
#: seed. Never averaged together -- see the module docstring.
PRIMARY_BUDGET = 1_000_000_000
SUPPORT_BUDGET = 480_000_000
BUDGETS = (PRIMARY_BUDGET, SUPPORT_BUDGET)

#: End-of-training reward is the mean of the eval points in the last this-many steps.
FINAL_WINDOW_STEPS = 50_000_000

#: Width of the trailing window that defines "solved", in environment steps. At the 6e5
#: eval cadence this is ~17 points. Defined in steps so it means the same at both budgets.
SMOOTH_WINDOW_STEPS = 10_000_000

#: Reward criteria for time-to-solve. 900 is the headline; the other two are extracted so
#: `plot.py` can show the arm ordering does not depend on the choice. See the docstring --
#: none of these is a published standard.
THRESHOLDS = (800, 900, 950)
HEADLINE_THRESHOLD = 900


def _arm(network_class: str):
    """One architecture, with all three selection gates applied."""
    def selector(df: pd.DataFrame) -> pd.Series:
        return (
            df["net_params.network_class"].eq(network_class)
            & df["env"].eq(TASK)
            & df["net_params.min_std"].eq(MIN_STD)
            & df["config.ppo.total_steps"].isin(BUDGETS)
            & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
            & df["state"].eq("finished")
        )
    return selector


#: The two arms. `budget` is a **column**, not part of the condition, so that an arm keeps
#: one colour across both panels (README §7) -- but see the docstring: the two budgets are
#: faceted, never averaged.
CONDITIONS = {
    "delayed_mlp": _arm("DelayedMLP"),
    "flat_forward_model": _arm("FlatForwardModel"),
}

#: Must hold *within one budget*. `config.ppo.total_steps` and `summary._step` are
#: deliberately included even though they define the facet: their appearance under
#: *** VARIES *** in the pooled section is the check that the facet is the only thing
#: separating the two cohorts, and their constancy within each section is the check that
#: no run was preempted short.
INVARIANTS = [
    "config.ppo.total_steps",
    "summary._step",
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
    "env_params.impl",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.normalize_obs",
    "net_params.activation",
    "net_params.initializer_scale",
    "config.eval.every_steps",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "repos.vnl_experiments.dirty",
]

#: Asserted per arm rather than reported: these are what *make* an arm, so a stray value
#: here would mean the condition label is wrong, which is a bug and not a caveat.
ARM_SIGNATURE = {
    "flat_forward_model": {"net_params.fm_loss_weight": 1.0,
                           "net_params.detach_prediction": True,
                           "net_params.predictor_hidden_sizes": "[256, 256, 256, 256]"},
    "delayed_mlp": {"net_params.fm_loss_weight": None,
                    "net_params.detach_prediction": None,
                    "net_params.predictor_hidden_sizes": None},
}

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")


def history_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def reward_series(hist: pd.DataFrame | None) -> pd.DataFrame | None:
    """The eval-reward series as a tidy step/value frame, or None."""
    if hist is None:
        return None
    key = next((k for k in REWARD_KEYS if k in hist.columns), None)
    if key is None:
        return None
    out = hist.dropna(subset=[key])[["_step", key]]
    return out.rename(columns={"_step": "step", key: "value"}).sort_values("step")


def trailing_mean(steps: np.ndarray, values: np.ndarray,
                  window: int = SMOOTH_WINDOW_STEPS) -> np.ndarray:
    """Mean of every sample in ``(s - window, s]``, for each s in ``steps``.

    A window in *steps* rather than samples, so the smoothing means the same thing
    whatever the eval cadence or the budget. Positions whose window is not yet fully
    covered by the run (i.e. ``s < window``) are NaN, so a criterion cannot be met on two
    early samples -- without that, a run whose first eval point happens to land high
    "solves" the task at step 0.
    """
    lo = np.searchsorted(steps, steps - window, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= window, out, np.nan)


def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    row = {
        "condition": run["condition"],
        "budget": int(run["config.ppo.total_steps"]),
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "nnx_ppo_commit": run.get("repos.nnx_ppo.commit"),
        "vnl_experiments_dirty": run.get("repos.vnl_experiments.dirty"),
        "gpu": run["gpu"],
        "delay": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        "seed": int(run["seed"]),
        "min_std": run["net_params.min_std"],
        "actual_step": run.get("summary._step"),
        "ctrl_dt": run.get("env_params.ctrl_dt"),
        # Post-`971ab99` only, so one value -- recorded as a column so a future cohort
        # that straddles the fix cannot pool the two silently (track README).
        "reward_source": "dmc_eval",
        # The run summary's single final eval point. Kept for reference only;
        # `reward_final` is what the figures use.
        "reward_last_point": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
    }
    blank = {"reward_final": None, "reward_final_n": 0, "reward_max": None,
             "reward_max_step": None, "smooth_max": None, "curve_points": 0,
             "curve_max_step": None}
    blank.update({f"steps_to_{t}": None for t in THRESHOLDS})
    blank.update({f"solved_{t}": None for t in THRESHOLDS})
    if series is None or series.empty:
        row.update(blank)
        return row

    steps = series["step"].to_numpy()
    values = series["value"].to_numpy()
    smooth = trailing_mean(steps, values)

    window = series[series["step"] >= steps.max() - FINAL_WINDOW_STEPS]
    row.update(
        reward_final=float(window["value"].mean()),
        reward_final_n=int(len(window)),
        reward_max=float(values.max()),
        reward_max_step=int(steps[values.argmax()]),
        # How high the *smoothed* curve ever got. This is the y-value a censored run
        # should be plotted at in the sensitivity panel, and it is what says whether a
        # run that missed the criterion missed it narrowly or by 4e2.
        smooth_max=float(np.nanmax(smooth)) if np.isfinite(smooth).any() else None,
        curve_points=int(len(steps)),
        curve_max_step=int(steps.max()),
    )
    for thr in THRESHOLDS:
        reached = np.where(smooth >= thr)[0]
        row[f"solved_{thr}"] = bool(len(reached))
        row[f"steps_to_{thr}"] = int(steps[reached[0]]) if len(reached) else None
    return row


def build_curves(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    """The eval series, plus the trailing mean the criterion is read off.

    ``reward_smooth`` is written here rather than recomputed in ``plot.py`` so that the
    curve a figure draws and the number ``steps_to_900`` reports are the same object.
    """
    if series is None or series.empty:
        return []
    steps = series["step"].to_numpy()
    smooth = trailing_mean(steps, series["value"].to_numpy())
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "budget": int(run["config.ppo.total_steps"]),
             "delay": int(run["net_params.delay_k"]), "seed": int(run["seed"]),
             "step": int(s), "reward_mean": float(v),
             "reward_smooth": None if not np.isfinite(m) else float(m)}
            for s, v, m in zip(steps, series["value"].to_numpy(), smooth)]


def assert_arm_signature(runs: pd.DataFrame) -> None:
    """The arm-defining net_params really do split along the condition label."""
    for condition, expected in ARM_SIGNATURE.items():
        sub = runs[runs["condition"] == condition]
        for col, want in expected.items():
            got = sub[col].dropna().unique() if col in sub else np.array([])
            if want is None:
                if len(got):
                    raise SystemExit(
                        f"{condition}: expected {col} to be absent, found {got!r}. The "
                        f"condition label does not describe the network that ran.")
            elif list(got) != [want]:
                raise SystemExit(
                    f"{condition}: expected {col} == {want!r}, found {got!r}. The "
                    f"condition label does not describe the network that ran.")


def comparability(runs: pd.DataFrame) -> str:
    """One section per budget, plus the pooled section.

    Per budget is where the invariants have to hold, because that is the only grouping a
    figure ever averages within. The pooled section is kept deliberately: it should flag
    `total_steps` and `summary._step` and nothing else, which is the check that budget is
    the *only* axis separating the two cohorts.
    """
    parts = []
    for budget in BUDGETS:
        sub = runs[runs["config.ppo.total_steps"] == budget]
        label = "primary" if budget == PRIMARY_BUDGET else "support: delay-5 hole + 3rd seed"
        parts.append(f"\n{'#' * 78}\n# budget = {budget:,} steps  ({label})  "
                     f"n={len(sub)}\n{'#' * 78}")
        parts.append(comparability_report(sub, invariant_cols=INVARIANTS,
                                          group_col="condition"))
    parts.append(f"\n{'#' * 78}\n# BOTH BUDGETS POOLED  n={len(runs)}\n"
                 f"# Expected to flag exactly config.ppo.total_steps and summary._step.\n"
                 f"# Anything else here is a second difference between the cohorts and\n"
                 f"# must be written up as a caveat in report.md.\n{'#' * 78}")
    parts.append(comparability_report(runs, invariant_cols=INVARIANTS,
                                      group_col="condition"))
    return "\n".join(parts)


def grid_report(df: pd.DataFrame) -> str:
    """Runs per (budget, arm, delay), so a hole in the design is visible as a hole."""
    parts = []
    for budget in BUDGETS:
        sub = df[df["budget"] == budget]
        table = pd.crosstab(sub["condition"], sub["delay"])
        parts.append(f"\nbudget = {budget:,}  (runs per arm x delay)\n{table}")
        empties = [(c, d) for c in table.index for d in table.columns
                   if table.loc[c, d] == 0]
        parts.append(f"  holes: {empties if empties else 'none'}")
    return "\n".join(parts)


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)
    assert_arm_signature(runs)

    store = Store()
    producer = get_producer("history")
    history_spec_id = producer.spec_id(producer.spec(project=PROJECT))
    if history_spec_id != HISTORY_SPEC_ID:
        raise SystemExit(
            f"history spec_id has drifted: got {history_spec_id}, expected "
            f"{HISTORY_SPEC_ID}.\nThe curves this analysis reads were made by a different "
            f"spec or producer VERSION. Update HISTORY_SPEC_ID deliberately, re-produce, "
            f"and say so in report.md -- do not silently repoint at a different "
            f"generation of data.")

    rows, curves = [], []
    for _, run in runs.iterrows():
        series = reward_series(history_of(store, run["wandb_id"], history_spec_id))
        rows.append(build_row(run, series))
        curves.extend(build_curves(run, series))

    df = pd.DataFrame(rows).sort_values(
        ["budget", "condition", "delay", "seed", "wandb_id"], ignore_index=True)
    curves_df = pd.DataFrame(curves).sort_values(
        ["budget", "condition", "delay", "seed", "step"], ignore_index=True)

    # A `history` artifact made while a run was still training is a snapshot, and
    # `ensure` will not replace it (README §6). Compare what the curve reaches against
    # what the index says the run reached.
    stale = df[df["curve_max_step"].notna()
               & (df["curve_max_step"] < df["actual_step"] * 0.999)]
    if len(stale):
        print("\n*** history artifacts are short of the run's own _step -- produced "
              "mid-training. Re-produce with `artifacts ensure --override`:")
        print(stale[["wandb_id", "curve_max_step", "actual_step"]].to_string(index=False))
        raise SystemExit(1)

    missing = int((df["curve_points"] == 0).sum())
    if missing:
        raise SystemExit(
            f"\n*** {missing}/{len(df)} runs have no history artifact. Runs are never "
            f"silently dropped for a missing artifact (README §3); produce them:\n"
            f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
            f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
            f"        --set project='\"{PROJECT}\"'\n")

    grid = grid_report(df)
    print(grid)
    solved = df.groupby(["budget", "condition"])[f"solved_{HEADLINE_THRESHOLD}"].agg(
        ["sum", "count"])
    print(f"\nruns reaching the {HEADLINE_THRESHOLD} criterion:\n{solved}\n")

    report = comparability(runs) + "\n\n" + "#" * 78 + "\n# DESIGN GRID\n" + \
        "#" * 78 + "\n" + grid + "\n"
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
