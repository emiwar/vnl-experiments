"""Extract the "is the forward-model benefit just more weights?" comparison.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-vs-bigger-decoder/extract.py

Four conditions, all AbsoluteImitation / TrainEvalSplit, eff_length == delay_k, standard
encoder ([512]x4) and critic ([1024,1024]); the DECODER architecture is the variable:
  - efference         standard decoder [512]x4              (baseline)
  - efference_larger  wider decoder    [1024]x4  ("Larger decoder")
  - efference_deeper  deeper decoder   [512]x8   ("Deeper decoder")
  - forward_model     standard decoder + learned predictor  (canonical FM only)

We also record `extra_hidden_params` — the number of weights in the decoder hidden stack
plus (for FM) the predictor hidden stack — so the figure can ask whether reward tracks
parameter count. Forward-model runs are git 5464376; the standard-decoder efference
baseline is git 1cd5838 (additive-only diff — see report.md).
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

# Held fixed across all conditions; the decoder is the only architectural variable.
FIXED_ENC = [512, 512, 512, 512]
FIXED_CRITIC = [1024, 1024]

STD_DEC = [512, 512, 512, 512]
LARGER_DEC = [1024, 1024, 1024, 1024]
DEEPER_DEC = [512, 512, 512, 512, 512, 512, 512, 512]

# Invariants that SHOULD be constant (decoder/fm sizes are the variable, so excluded).
INVARIANTS = [
    "git_commit", "env", "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "critic_hidden_sizes", "n_envs", "total_steps", "actual_step",
]


def hidden_stack_params(sizes) -> int:
    """Weights+biases between consecutive hidden layers (delay-independent).

    Excludes the input layer (whose width depends on the delay-scaled efference buffer)
    and the small output layer, so it cleanly characterises how many extra weights each
    architecture variant adds.
    """
    if not sizes:
        return 0
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def condition_of(run) -> str | None:
    c = run.config
    net = c.get("net_params", {}) or {}
    if not pipeline.full_decoder_inputs(net):
        return None  # a decoder-input ablation is a different question
    # Hold the rest of the architecture + hyperparameters fixed.
    if net.get("enc_hidden_sizes") != FIXED_ENC:
        return None
    if net.get("critic_hidden_sizes") != FIXED_CRITIC:
        return None
    if c.get("efference_length") != c.get("delay_k"):
        return None
    dec = net.get("dec_hidden_sizes")
    if "ForwardModel" in run.tags:
        # Canonical forward model only (exclude the loss-weight sweep).
        if dec == STD_DEC and run.notes == "Explicit forward model.":
            return "forward_model"
        return None
    if dec == STD_DEC:
        return "efference"
    if dec == LARGER_DEC:
        return "efference_larger"
    if dec == DEEPER_DEC:
        return "efference_deeper"
    return None


def main() -> None:
    runs = fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)
    print(f"Fetched {len(runs)} finished runs with tags {REQUIRE_TAGS}")

    records = []
    for r in runs:
        cond = condition_of(r)
        if cond is None:
            continue
        net = r.config.get("net_params", {}) or {}
        extra = hidden_stack_params(net.get("dec_hidden_sizes"))
        if "ForwardModel" in r.tags:
            extra += hidden_stack_params(net.get("fm_hidden_sizes"))
        records.append(
            run_record(
                r,
                config_keys=CONFIG_KEYS,
                net_param_keys=NET_PARAM_KEYS,
                ppo_keys=PPO_KEYS,
                metrics=METRICS,
                extra={"condition": cond, "extra_hidden_params": extra},
            )
        )

    df = records_to_df(records)
    df = df.sort_values(["condition", "delay_k"]).reset_index(drop=True)

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))
    print("\nextra_hidden_params per condition:")
    print(df.groupby("condition")["extra_hidden_params"].first().to_string())

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
