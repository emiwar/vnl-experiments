"""Extract the action-buffer-length comparison into data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/action-buffer-length/extract.py

Does the efference-copy network use the *whole* action buffer (all actions taken since the
delayed observation) or just the few most recent ones? We compare, across delays:
  - efference        full buffer: efference_length == delay_k (all intervening actions)
  - efference_trunc  truncated buffer: efference_length == 5, fixed ("Fixed efference length.")
  - no_efference     no buffer at all: efference_length == 0

All EncDec (no forward model), standard architecture. The truncated runs are git 5464376;
the full / no-buffer references are git 1cd5838 (additive-only diff — see report.md).
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

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRE_TAGS = ["TrainEvalSplit"]  # excludes the older, non-comparable pre-split runs

CONFIG_KEYS = ["env", "delay_k", "efference_length"]
NET_PARAM_KEYS = [
    "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
]
PPO_KEYS = ["n_envs", "total_steps"]
METRICS = ["episode_reward/mean", "episode_reward/std"]

STD_ARCH = {
    "enc_hidden_sizes": [512, 512, 512, 512],
    "dec_hidden_sizes": [512, 512, 512, 512],
    "critic_hidden_sizes": [1024, 1024],
}

# efference_length and delay_k are the variables; git varies by condition (documented).
INVARIANTS = [
    "git_commit", "env", "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
    "n_envs", "total_steps", "actual_step",
]


def condition_of(run) -> str | None:
    if "ForwardModel" in run.tags:
        return None
    c = run.config
    net = c.get("net_params", {}) or {}
    if any(net.get(k) != v for k, v in STD_ARCH.items()):
        return None
    delay = c.get("delay_k")
    eff = c.get("efference_length")
    if run.notes == "Fixed efference length.":
        return "efference_trunc"
    if eff == delay:
        return "efference"
    if eff == 0 and (delay or 0) > 0:
        return "no_efference"
    return None


def main() -> None:
    runs = fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)
    print(f"Fetched {len(runs)} finished runs with tags {REQUIRE_TAGS}")

    records = []
    for r in runs:
        cond = condition_of(r)
        if cond is None:
            continue
        records.append(
            run_record(
                r,
                config_keys=CONFIG_KEYS,
                net_param_keys=NET_PARAM_KEYS,
                ppo_keys=PPO_KEYS,
                metrics=METRICS,
                extra={"condition": cond},
            )
        )

    df = records_to_df(records)
    df = df.sort_values(["condition", "delay_k"]).reset_index(drop=True)

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
