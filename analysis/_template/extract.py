"""Extract data for <QUESTION> from WandB into data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/<question-slug>/extract.py

This is the ONLY script that talks to WandB. It fetches the relevant runs, assigns a
``condition`` to each, runs the programmatic comparability check, and writes:
  - data.csv          one row per included run (see analysis/README.md §3)
  - comparability.txt the programmatic comparability report (§4)
"""

from pathlib import Path

from vnl_experiments.wandb_utils import (
    comparability_report,
    fetch_runs,
    git_commit_summary,
    records_to_df,
    run_record,
)

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/<project>"            # TODO
REQUIRE_TAGS = ["TrainEvalSplit"]            # TODO: tags every included run must have

# Columns to pull from each run (TODO: adjust to the question).
CONFIG_KEYS = ["env", "delay_k", "efference_length"]
NET_PARAM_KEYS = ["latent_size", "kl_weight", "body_target_frame"]
PPO_KEYS = ["n_envs", "total_steps"]
METRICS = ["episode_reward/mean", "episode_reward/std"]


def condition_of(run) -> str | None:
    """Map a run to a condition label, or None to exclude it. TODO: implement."""
    raise NotImplementedError


def main() -> None:
    runs = fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)
    print(f"Fetched {len(runs)} finished runs with tags {REQUIRE_TAGS}")

    records = []
    for r in runs:
        cond = condition_of(r)
        if cond is None:
            continue
        rec = run_record(
            r,
            config_keys=CONFIG_KEYS,
            net_param_keys=NET_PARAM_KEYS,
            ppo_keys=PPO_KEYS,
            metrics=METRICS,
            extra={"condition": cond},
        )
        records.append(rec)

    df = records_to_df(records)
    df = df.sort_values(["condition", "delay_k"]).reset_index(drop=True)  # TODO sort keys

    report = comparability_report(df, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
