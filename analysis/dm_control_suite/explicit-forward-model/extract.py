"""Does an explicit forward model help under observation delay, on three dm_control tasks?

Two flat-observation architectures, swept over delay on CartpoleSwingup, WalkerWalk and
HumanoidWalk:

* ``DelayedMLP`` -- the baseline. The actor sees the delayed observation plus an efference
  copy of the ``delay_k`` actions still in flight; the critic sees the fresh observation.
* ``FlatForwardModel`` -- identical, except an explicit predictor maps (delayed obs,
  action buffer) -> current obs, and the actor is handed that prediction. The predictor is
  trained by its own loss (``fm_loss_weight = 1``) rather than by the policy gradient.

So the arms differ in exactly one factor: whether the state estimate is built explicitly
by a supervised predictor, or left implicit in the actor MLP. ``efference_length == delay``
in every run, so both arms have the same information available.

The three tasks are chosen to span the difficulty range: CartpoleSwingup is small and
trains in ~35 min, WalkerWalk is the standard locomotion benchmark, HumanoidWalk is the
hardest thing in the suite that still trains at this budget.

Selection traps this folder gates on -- neither is visible in a run name or tag
----------------------------------------------------------------------------------
* **min_std.** The 2026-09-09 cartpole batch includes a ``min_std`` experiment
  (0.01 / 0.02 / 0.05 / 0.1 / 0.2) whose runs are named exactly like the defaults. Pooling
  them would compare exploration-width settings while claiming to compare architectures.
  Gated to the default ``min_std == 0.001``.
* **The eval-env fix (``971ab99``, 2026-09-09).** Before it, the eval env inherited the
  training wrappers, so ``eval/*`` was reward-scaled x10 *and* the episodes were truncated
  by ``EpisodeWrapper``'s random phase. Twelve cartpole runs at ``d168093`` are on the
  wrong side and are excluded; one of them reports 6147 where its post-fix twins report
  ~830, which is what that bug looks like. Commit, not ``created_at``, is the
  discriminator -- both sides were created on 2026-09-09.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/explicit-forward-model/extract.py
    ../.venv/bin/python analysis/dm_control_suite/explicit-forward-model/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/explicit-forward-model/extract.py --check

Writes ``data.csv`` (one row per run) and ``curves.csv`` (the eval series, for the two
supplementary figures). ``plot.py`` reads only those.
"""

from pathlib import Path

import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: The `history` producer takes the project as a **spec field**, defaulting to the rodent
#: one -- so a dm_control history artifact must be produced with an explicit override:
#:
#:     python -m vnl_experiments.artifacts ensure --kind history \
#:         --runs analysis/dm_control_suite/explicit-forward-model/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
#:
#: Because the project is in the spec, the override changes the spec_id, so control-suite
#: and rodent curves can never be pooled by accident. Pinned here and asserted below.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

#: The three tasks, in increasing order of difficulty -- also the panel order in plot.py.
TASKS = ["CartpoleSwingup", "WalkerWalk", "HumanoidWalk"]

#: Default exploration width. See the docstring: the cartpole batch sweeps this.
MIN_STD = 0.001

#: Runs before `971ab99` evaluated on the training env, wrappers and all.
PRE_EVAL_FIX_COMMIT = "d168093"

#: End-of-training reward is the mean of the eval points in the last this-many steps, not
#: the final point -- a single inline eval moves by a few percent (README §6). At the
#: 600 k eval cadence this averages ~83 points.
FINAL_WINDOW_STEPS = 50_000_000


def _arm(network_class: str):
    """Select one architecture, with both selection gates applied."""
    def selector(df: pd.DataFrame) -> pd.Series:
        return (
            df["net_params.network_class"].eq(network_class)
            & df["env"].isin(TASKS)
            & df["net_params.min_std"].eq(MIN_STD)
            & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
            & df["state"].eq("finished")
        )
    return selector


#: The two arms. `task` is a column rather than part of the condition, because the
#: contrast is within a task and the figure facets on it.
CONDITIONS = {
    "delayed_mlp": _arm("DelayedMLP"),
    "flat_forward_model": _arm("FlatForwardModel"),
}

#: Must hold *within a task*. `env_params.ctrl_dt` is deliberately absent: it is 0.01 on
#: CartpoleSwingup and 0.025 on the other two, so it varies across tasks by design and is
#: asserted per task below instead.
INVARIANTS = [
    "git_commit",
    "config.ppo.n_envs",
    "config.ppo.total_steps",
    "config.ppo.learning_rate",
    "summary._step",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.critic_hidden_sizes",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
]

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


def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    row = {
        "condition": run["condition"],
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "gpu": run["gpu"],
        "delay": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        "seed": int(run["seed"]),
        "min_std": run["net_params.min_std"],
        "actual_step": run.get("summary._step"),
        "ctrl_dt": run.get("env_params.ctrl_dt"),
        # Post-`971ab99` only, so a single value -- recorded so a future cohort that
        # spans the fix cannot pool the two silently.
        "reward_source": "dmc_eval",
        # The run summary's final eval point. Kept for reference; `reward_final` below
        # is what the figures use.
        "reward_last_point": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
    }
    if series is None or series.empty:
        row.update(reward_final=None, reward_max=None, reward_final_n=0,
                   curve_points=0, curve_max_step=None)
        return row

    cutoff = series["step"].max() - FINAL_WINDOW_STEPS
    window = series[series["step"] >= cutoff]
    row.update(
        # End-of-training competence.
        reward_final=float(window["value"].mean()),
        reward_final_n=int(len(window)),
        # Best the run ever reached: on CartpoleSwingup a low final score is often
        # catastrophic forgetting rather than a failure to find the solution, and the
        # difference between these two columns is what shows it.
        reward_max=float(series["value"].max()),
        reward_max_step=int(series.loc[series["value"].idxmax(), "step"]),
        curve_points=int(len(series)),
        curve_max_step=int(series["step"].max()),
    )
    return row


def build_curves(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    if series is None:
        return []
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "task": run["env"], "delay": int(run["net_params.delay_k"]),
             "seed": int(run["seed"]), "step": int(r.step),
             "reward_mean": float(r.value)}
            for r in series.itertuples()]


def comparability(runs: pd.DataFrame) -> str:
    """One section per task: the contrast is within a task, so that is where the
    invariants have to hold. Comparing `ctrl_dt` across tasks would only report the
    designed difference."""
    parts = []
    for task in TASKS:
        sub = runs[runs["env"] == task]
        parts.append(f"\n{'#' * 70}\n# {task}  (n={len(sub)})\n{'#' * 70}")
        parts.append(comparability_report(sub, invariant_cols=INVARIANTS,
                                          group_col="condition"))
    return "\n".join(parts)


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    producer = get_producer("history")
    history_spec_id = producer.spec_id(producer.spec(project=PROJECT))
    if history_spec_id != HISTORY_SPEC_ID:
        raise SystemExit(
            f"history spec_id has drifted: got {history_spec_id}, expected "
            f"{HISTORY_SPEC_ID}.\nThe curves this analysis reads were made by a "
            f"different spec or producer VERSION. Update HISTORY_SPEC_ID deliberately, "
            f"re-produce, and say so in report.md -- do not silently repoint at a "
            f"different generation of data.")

    rows, curves = [], []
    for _, run in runs.iterrows():
        series = reward_series(history_of(store, run["wandb_id"], history_spec_id))
        rows.append(build_row(run, series))
        curves.extend(build_curves(run, series))

    df = pd.DataFrame(rows).sort_values(
        ["task", "condition", "delay", "seed", "wandb_id"], ignore_index=True)
    curve_cols = ["wandb_id", "condition", "task", "delay", "seed", "step", "reward_mean"]
    curves_df = pd.DataFrame(curves, columns=curve_cols)
    if not curves_df.empty:
        curves_df = curves_df.sort_values(
            ["task", "condition", "delay", "seed", "step"], ignore_index=True)

    missing = int((df["curve_points"] == 0).sum())
    if missing:
        print(f"\n*** {missing}/{len(df)} runs have no history artifact; their "
              f"reward_final/reward_max are blank. Run:\n"
              f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
              f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
              f"        --set project='\"{PROJECT}\"'\n")

    report = comparability(runs)
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
    print(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
