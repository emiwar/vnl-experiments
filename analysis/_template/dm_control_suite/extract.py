"""<One-line question this folder answers.>

<A paragraph or two: what is being compared, why, and anything a reader needs to know to
interpret the conditions below. This docstring is the primary record of the analysis's
intent -- report.md quotes conclusions, this explains the design.>

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/<question-slug>/extract.py
    ../.venv/bin/python analysis/dm_control_suite/<question-slug>/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/<question-slug>/extract.py --check

Frozen is the default: it rebuilds ``data.csv`` from exactly the runs listed in the
committed ``runs.csv``. ``--refresh`` re-runs the ``CONDITIONS`` selectors against the
run index, prints which runs entered or left, and rewrites both files. ``--sync``
refreshes the index from WandB first.

This is the only script in the folder that reads the index or the artifact store;
``plot.py`` reads the CSVs it writes and nothing else.

Read ``analysis/dm_control_suite/README.md`` first. The traps that bite here are not the
rodent's: the reward scale, the 2026-09-10 eval-wrapper change, and the two config-schema
eras.
"""

from pathlib import Path

import pandas as pd

from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent

# Pinned, not defaulted. `--project` has no default precisely so a dm_control folder
# cannot silently read the rodent index. Confirm the entity once with
# `python -m vnl_experiments.wandb_utils.index sync --project <entity>/nnx-ppo-delays`.
PROJECT = "emiwar-team/nnx-ppo-delays"

REQUIRES = ["index", "history"]

# The nine registered tasks (vnl_experiments/envs/registry.py). Raw reward is NOT
# comparable across them -- plot one panel per task rather than normalising (README §7).
TASKS = ["WalkerWalk", "WalkerRun", "WalkerStand", "CheetahRun",
         "HumanoidWalk", "HumanoidStand", "CartpoleBalance", "CartpoleSwingup",
         "BallInCup"]

# The eval-wrapper fix: before this date `eval/*` was scaled x10 AND truncated by the
# training EpisodeWrapper's random phase. Runs either side are not poolable.
EVAL_FIX_DATE = "2026-09-10"

# One entry per experimental cell; keys are index columns (`index.load(PROJECT).columns`).
# Conditions must be mutually exclusive -- select_conditions raises if a run matches two.
#
# Beware the two eras (README): a legacy run (<= 2026-07-08) has NO `env_params.*` and no
# `net_params.*` columns at all, so selecting on them silently returns current runs only.
# The task is `config.env` in legacy and `env_spec.task` now; `delay_k` is top-level in
# legacy and `net_params.delay_k` now. Use a callable selector and coalesce if a cohort
# has to span both -- and if it does, `created_at` belongs in INVARIANTS.
CONDITIONS = {
    "mlp": {
        "tags": ["WalkerWalk", "DelayedMLP"],
        "state": "finished",
    },
    "recurrent": {
        "tags": ["WalkerWalk", "FlatRecurrent"],
        "state": "finished",
    },
}

# Columns that must be constant for the comparison to be fair. comparability.txt flags
# any that vary, overall and within each condition. `reward_scale` and `episode_length`
# are here because they are the two env knobs that change what "reward" *means*.
INVARIANTS = [
    "config.ppo.n_envs",
    "config.ppo.total_steps",
    "config.ppo.learning_rate",
    "summary._step",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "git_commit",
]


def build_row(run: pd.Series) -> dict:
    """One row of data.csv from one index row. Add the metrics this question needs."""
    return {
        "condition": run["condition"],
        "task": pipeline.first_present(run, "env_spec.task", "config.env"),
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run.get("created_at"),
        "git_commit": run["git_commit"],
        "gpu": run["gpu"],
        "delay_k": pipeline.first_present(run, "net_params.delay_k", "config.delay_k"),
        "efference_length": pipeline.first_present(
            run, "net_params.efference_length", "config.efference_length"),
        "seed": pipeline.first_present(run, "config.seed", "config.ppo.seed"),
        "actual_step": run.get("summary._step"),
        # Which reward this is, recorded as a column rather than left to the reader.
        # `eval/*` is unscaled and full-length only from EVAL_FIX_DATE; before it the
        # series is x10 and the episodes are cut short.
        "reward_source": ("dmc_eval" if str(run.get("created_at", "")) >= EVAL_FIX_DATE
                          else "dmc_eval_prefix"),
        "reward_mean": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
        "lifespan_mean": pipeline.first_present(
            run, "summary.eval/lifespan/mean", "summary.lifespan_mean"),
    }


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    df = pd.DataFrame([build_row(run) for _, run in runs.iterrows()])
    df = df.sort_values(["condition", "task", "wandb_id"], ignore_index=True)

    if df["reward_source"].nunique() > 1:
        print("\n*** WARNING: this cohort straddles the {} eval-wrapper fix. The two "
              "reward_source values are in different units AND different episode "
              "lengths; do not pool them. ***\n".format(EVAL_FIX_DATE))

    report = comparability_report(runs, invariant_cols=INVARIANTS,
                                  group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
    print(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
