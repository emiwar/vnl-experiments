"""Extract forward-model prediction accuracy vs imitation performance into data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-accuracy-vs-imitation/extract.py

For every forward-model run we record the feedforward prediction error
(`net/3/action/1/fm_pred_mse`, the self-supervised MSE between the predictor's output and
the true current proprioception) alongside the imitation performance
(`episode_reward/mean`), plus the two sources of variation, `delay_k` and `fm_loss_weight`.

All runs are git 5464376, standard architecture, full 600M steps — a single comparable set.
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
REQUIRE_TAGS = ["TrainEvalSplit", "ForwardModel"]

CONFIG_KEYS = ["env", "delay_k", "efference_length", "fm_loss_weight"]
NET_PARAM_KEYS = [
    "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes", "fm_hidden_sizes",
]
PPO_KEYS = ["n_envs", "total_steps"]
# fm prediction MSE (median + IQR) and imitation reward.
FM_MSE = "net/3/action/1/fm_pred_mse"
METRICS = [
    "episode_reward/mean", "episode_reward/std",
    f"{FM_MSE}/p25", f"{FM_MSE}/p50", f"{FM_MSE}/p75",
]

STD_ENC = [512, 512, 512, 512]
STD_DEC = [512, 512, 512, 512]
STD_CRIT = [1024, 1024]
STD_FM = [512, 512, 512, 512]

# delay_k and fm_loss_weight are the variables; everything else should be constant.
INVARIANTS = [
    "git_commit", "env", "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes", "fm_hidden_sizes",
    "n_envs", "total_steps", "actual_step",
]


def is_standard(run) -> bool:
    net = run.config.get("net_params", {}) or {}
    return (
        net.get("enc_hidden_sizes") == STD_ENC
        and net.get("dec_hidden_sizes") == STD_DEC
        and net.get("critic_hidden_sizes") == STD_CRIT
        and net.get("fm_hidden_sizes") == STD_FM
    )


def main() -> None:
    runs = fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)
    print(f"Fetched {len(runs)} finished runs with tags {REQUIRE_TAGS}")

    records = []
    for r in runs:
        if not is_standard(r):
            continue
        records.append(
            run_record(
                r,
                config_keys=CONFIG_KEYS,
                net_param_keys=NET_PARAM_KEYS,
                ppo_keys=PPO_KEYS,
                metrics=METRICS,
                extra={"condition": "forward_model",
                       "fm_trained": (r.config.get("fm_loss_weight") or 0) > 0},
            )
        )

    df = records_to_df(records)
    df = df.rename(columns={f"{FM_MSE}/p50": "fm_pred_mse",
                            f"{FM_MSE}/p25": "fm_pred_mse_p25",
                            f"{FM_MSE}/p75": "fm_pred_mse_p75"})
    df = df.sort_values(["delay_k", "fm_loss_weight"]).reset_index(drop=True)

    report = comparability_report(df, invariant_cols=INVARIANTS)
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    # Headline correlations (Spearman, computed with pandas — no scipy dependency).
    print("\nSpearman corr (fm_pred_mse vs reward):")
    print("  all FM runs        :", round(df["fm_pred_mse"].corr(df["episode_reward/mean"], method="spearman"), 3))
    can = df[df["fm_loss_weight"] == 1]
    print("  canonical (w=1)    :", round(can["fm_pred_mse"].corr(can["episode_reward/mean"], method="spearman"), 3))
    d10 = df[df["delay_k"] == 10]
    print("  delay=10 (w sweep) :", round(d10["fm_pred_mse"].corr(d10["episode_reward/mean"], method="spearman"), 3))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
