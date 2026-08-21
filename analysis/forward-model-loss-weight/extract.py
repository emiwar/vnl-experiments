"""Extract the forward-model loss-weight analysis into data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-loss-weight/extract.py

Questions: how important is the self-supervised forward-model *loss* (vs just having the
predictor architecture), and how does its weight affect performance?

Conditions (all AbsoluteImitation / TrainEvalSplit, eff_length == delay_k, standard
architecture enc/dec/fm [512]x4, critic [1024,1024]):
  - forward_model  every ForwardModel run, with its top-level `fm_loss_weight` recorded.
                   This spans the canonical sweep (weight == 1, delays 0-100) AND the
                   loss-weight sweep (weights 0 .. 10, mostly delay 10, plus weight==0 at
                   delays 5/20/50). weight == 0 == "architecture present, no FM training".
  - efference      plain efference copy (reference).
  - no_efference   no efference copy at all (floor reference).

Forward-model runs are git 5464376; the efference/no_efference references are git 1cd5838
(additive-only diff — see report.md).
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

# fm_loss_weight is a TOP-LEVEL config key (not under net_params).
CONFIG_KEYS = ["env", "delay_k", "efference_length", "fm_loss_weight"]
NET_PARAM_KEYS = [
    "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes", "fm_hidden_sizes",
]
PPO_KEYS = ["n_envs", "total_steps"]
METRICS = ["episode_reward/mean", "episode_reward/std"]

STD_ENC = [512, 512, 512, 512]
STD_DEC = [512, 512, 512, 512]
STD_CRIT = [1024, 1024]
STD_FM = [512, 512, 512, 512]

# fm_loss_weight / delay / efference are the variables, so excluded from invariants.
# fm_hidden_sizes is structurally present only for FM, so also excluded.
INVARIANTS = [
    "git_commit", "env", "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
    "n_envs", "total_steps", "actual_step",
]


def condition_of(run) -> str | None:
    c = run.config
    net = c.get("net_params", {}) or {}
    if not pipeline.full_decoder_inputs(net):
        return None  # a decoder-input ablation is a different question
    if net.get("enc_hidden_sizes") != STD_ENC:
        return None
    if net.get("dec_hidden_sizes") != STD_DEC:
        return None
    if net.get("critic_hidden_sizes") != STD_CRIT:
        return None
    delay = c.get("delay_k")
    eff = c.get("efference_length")
    if "ForwardModel" in run.tags:
        if net.get("fm_hidden_sizes") != STD_FM:
            return None
        if eff != delay:
            return None
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
    df = df.sort_values(["condition", "delay_k", "fm_loss_weight"]).reset_index(drop=True)

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))
    fm = df[df["condition"] == "forward_model"]
    print("\nforward_model fm_loss_weight values:",
          sorted(fm["fm_loss_weight"].dropna().unique().tolist()))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
