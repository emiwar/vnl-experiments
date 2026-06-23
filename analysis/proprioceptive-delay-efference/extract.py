"""Extract the proprioceptive delay sweep (efference copy vs none) into data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/proprioceptive-delay-efference/extract.py

Two conditions, both AbsoluteImitation / TrainEvalSplit (forward-model runs excluded):
  - efference:     efference_length == delay_k
  - no_efference:  efference_length == 0 and delay_k > 0
    (delay_k == 0 with eff == 0 is identical to the efference baseline, so excluded)
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
REQUIRE_TAGS = ["TrainEvalSplit"]

CONFIG_KEYS = ["env", "delay_k", "efference_length"]
NET_PARAM_KEYS = [
    "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
]
PPO_KEYS = ["n_envs", "total_steps"]
METRICS = ["episode_reward/mean", "episode_reward/std"]

# Standard baseline architecture. Runs with a different encoder/decoder/critic (e.g.
# the "Larger decoder"/"Deeper decoder" sweeps at a later commit) are a *different*
# question and must be excluded so this comparison holds the network fixed.
STD_ARCH = {
    "enc_hidden_sizes": [512, 512, 512, 512],
    "dec_hidden_sizes": [512, 512, 512, 512],
    "critic_hidden_sizes": [1024, 1024],
}


def condition_of(run) -> str | None:
    if "ForwardModel" in run.tags:
        return None
    c = run.config
    net = c.get("net_params", {}) or {}
    if any(net.get(k) != v for k, v in STD_ARCH.items()):
        return None  # non-standard architecture (decoder-size sweep) — different question
    delay = c.get("delay_k")
    eff = c.get("efference_length")
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
