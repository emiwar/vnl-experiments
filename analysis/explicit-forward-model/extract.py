"""Extract the explicit-forward-model comparison into data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/explicit-forward-model/extract.py

Three conditions, all AbsoluteImitation / TrainEvalSplit, standard architecture:
  - forward_model: ForwardModel tag (learned predictor + self-supervised L2 loss)
  - efference:     efference_length == delay_k (plain efference copy, no predictor)
  - no_efference:  efference_length == 0 and delay_k > 0 (no action history at all)

The forward-model runs are git 5464376; the efference/no_efference baselines are git
1cd5838. The only shared-code change between those commits is the backward-compatible
`inject_key` parameter on EfferenceCopy — all env/network/PPO/reward code is identical —
so the conditions are comparable (documented as a caveat in report.md).
"""

from pathlib import Path

from vnl_experiments.wandb_utils import (
    comparability_report,
    fetch_runs,
    git_commit_summary,
    pipeline,
    records_to_df,
    run_record,
)

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRE_TAGS = ["TrainEvalSplit"]

CONFIG_KEYS = ["env", "delay_k", "efference_length"]
NET_PARAM_KEYS = [
    "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes", "fm_hidden_sizes",
]
PPO_KEYS = ["n_envs", "total_steps"]
METRICS = ["episode_reward/mean", "episode_reward/std"]

# Standard baseline architecture (excludes the "Larger/Deeper decoder" sweeps).
STD_ARCH = {
    "enc_hidden_sizes": [512, 512, 512, 512],
    "dec_hidden_sizes": [512, 512, 512, 512],
    "critic_hidden_sizes": [1024, 1024],
}


def condition_of(run) -> str | None:
    c = run.config
    net = c.get("net_params", {}) or {}
    if not pipeline.full_decoder_inputs(net):
        return None  # a decoder-input ablation is a different question
    if any(net.get(k) != v for k, v in STD_ARCH.items()):
        return None  # non-standard architecture — different question
    delay = c.get("delay_k")
    eff = c.get("efference_length")
    if "ForwardModel" in run.tags:
        return "forward_model"
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

    report = comparability_report(df, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
