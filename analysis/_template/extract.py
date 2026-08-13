"""<One-line question this folder answers.>

<A paragraph or two: what is being compared, why, and anything a reader needs to know to
interpret the conditions below. This docstring is the primary record of the analysis's
intent -- report.md quotes conclusions, this explains the design.>

Run it
------
    ../.venv/bin/python analysis/<question-slug>/extract.py            # frozen rebuild
    ../.venv/bin/python analysis/<question-slug>/extract.py --sync --refresh
    ../.venv/bin/python analysis/<question-slug>/extract.py --check    # CI-style drift check

Frozen is the default: it rebuilds ``data.csv`` from exactly the runs listed in the
committed ``runs.csv``. ``--refresh`` re-runs the ``CONDITIONS`` selectors against the
run index, prints which runs entered or left, and rewrites both files. ``--sync``
refreshes the index from WandB first.

This is the only script in the folder that reads the index or the artifact store;
``plot.py`` reads the CSVs it writes and nothing else.
"""

from pathlib import Path

import pandas as pd

from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

# Data this analysis needs per run. "index" is always available; every other entry names
# an artifact kind (optionally `kind:spec_id`) and is checked into coverage.txt, so a run
# that is missing one is reported rather than silently dropped.
REQUIRES = ["index", "history"]

XML_ROOT = ("/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/vnl-playground/"
            "vnl_playground/tasks/rodent/xmls")
OLD_XML = f"{XML_ROOT}/rodent.xml"
NEW_XML = f"{XML_ROOT}/rodent_no_tail_collisions.xml"

# One entry per experimental cell. Keys are index columns (see `index.load().columns`);
# `tags` requires *all* the listed tags. Conditions must be mutually exclusive --
# select_conditions raises if a run matches two of them.
#
# Filter on `env_params.*` for anything the environment consumes. Never infer an env knob
# from the training script at the run's commit: the cluster working copy drifts from what
# is committed, and `env_params` is the only record of what actually ran.
CONDITIONS = {
    "old_xml": {
        "tags": ["TrainEvalSplit"],
        "state": "finished",
        "env_params.walker_xml_path": OLD_XML,
        "env_params.torque_actuators": True,
    },
    "new_xml": {
        "tags": ["TrainEvalSplit"],
        "state": "finished",
        "env_params.walker_xml_path": NEW_XML,
        "env_params.torque_actuators": True,
    },
}

# Columns that must be constant for the comparison to be fair. comparability.txt flags
# any that vary, overall and within each condition.
INVARIANTS = [
    "env",
    "config.ppo.n_envs",
    "config.ppo.total_steps",
    "summary._step",
    "net_params.latent_size",
    "net_params.kl_weight",
    "env_params.body_target_frame",
    "env_params.ctrl_dt",
    "git_commit",
]


def build_row(run: pd.Series) -> dict:
    """One row of data.csv from one index row. Add the metrics this question needs."""
    return {
        "condition": run["condition"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "git_commit": run["git_commit"],
        "gpu": run["gpu"],
        "delay_k": run.get("delay_k"),
        "actual_step": run.get("summary._step"),
        # Logging was renamed mid-project (`x` -> `eval/x`); coalesce so cohorts that
        # span the rename stay comparable.
        "reward_mean": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
        "lifespan_mean": pipeline.first_present(
            run, "summary.eval/lifespan/mean", "summary.lifespan_mean"),
    }


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    df = pd.DataFrame([build_row(run) for _, run in runs.iterrows()])
    df = df.sort_values(["condition", "wandb_id"], ignore_index=True)

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
