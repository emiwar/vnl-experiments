"""Reference-root vs current-root imitation frame: does it help, and does the forward model still win?

Run from the repo root::

    ../.venv/bin/python analysis/reference-root-frame/extract.py

This is the ONLY script that talks to WandB. It fetches the relevant runs, assigns a
``condition``, runs the programmatic comparability check, and writes ``data.csv`` +
``comparability.txt``.

Two questions, one 2x2 (frame x network) design over a delay sweep {0,2,5,10,20,50},
efference-matched (efference_length == delay_k), seed 42, standard architecture:

  Q1. Is the ``reference_root`` body-target frame better or worse than ``current_root``?
  Q2. Is an explicit forward model still advantageous under ``reference_root``?

**The authoritative frame is ``env_params.body_target_frame``**, NOT ``net_params`` (the
net_params copy is inert -- see analysis/README.md "body_target_frame bug"). Network type is
read from the run **tags** (``ForwardModel`` vs ``EncDec``); ``network_class`` is not a
top-level config key on these runs.

Conditions and their (pinned) commits -- the frame is confounded with the code commit, so
each condition is pinned to a single commit and the training-relevant diffs were checked by
hand (see report.md "Comparability"):
  - reference_efference      reference_root, EncDec        git 909e774d (the new cohort)
  - reference_forward_model  reference_root, ForwardModel  git 909e774d (the new cohort)
  - current_efference        current_root,   EncDec        git 1cd5838f (canonical eff sweep)
  - current_forward_model    current_root,   ForwardModel  git 54643764 (canonical FM sweep)

Reward metric: ``body_target_frame`` changes only the *observation* the policy sees, never
the reward, so eval reward is directly comparable across frames. The "new logging api"
renamed ``episode_reward/mean`` -> ``eval/episode_reward/mean`` (same eval-on-train-clips
protocol); we coalesce the two keys into ``reward_mean``.
"""

from pathlib import Path

from vnl_experiments.wandb_utils import (
    comparability_report,
    fetch_runs,
    git_commit_summary,
    records_to_df,
)

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRE_TAGS = ["TrainEvalSplit"]

DELAYS = {0, 2, 5, 10, 20, 50}
STD_ARCH = {
    "enc_hidden_sizes": [512] * 4,
    "dec_hidden_sizes": [512] * 4,
    "critic_hidden_sizes": [1024, 1024],
}
REF_GIT = "909e774d"   # body_target_frame-fix cohort (reference_root)
CUR_EFF_GIT = "1cd5838f"   # canonical current_root efference sweep
CUR_FM_GIT = "54643764"    # canonical current_root forward-model sweep

# Metric keys, old-logging first then new-logging alias (coalesced left-to-right).
REWARD_MEAN_KEYS = ["episode_reward/mean", "eval/episode_reward/mean"]
REWARD_STD_KEYS = ["episode_reward/std", "eval/episode_reward/std"]
LIFESPAN_KEYS = ["lifespan_mean", "eval/lifespan/mean"]
FM_MSE_KEYS = ["eval/net/3/action/1/fm_pred_mse/mean"]

# Invariants that must be single-valued WITHIN a condition (git_commit, body_target_frame,
# network, delay_k are the experimental / by-design-varying axes and are excluded).
INVARIANTS = [
    "latent_size", "kl_weight", "enc_hidden_sizes", "dec_hidden_sizes",
    "critic_hidden_sizes", "seed", "clip_length", "actual_step",
]


def git8(run) -> str:
    return (((run.metadata or {}).get("git", {}) or {}).get("commit", "") or "")[:8]


def std_arch(net: dict) -> bool:
    return all(list(net.get(k, [])) == v for k, v in STD_ARCH.items())


def coalesce(summary, keys):
    for k in keys:
        v = summary.get(k)
        if v is not None:
            return v
    return None


def condition_of(run):
    """Map a run to (condition, frame, network) or None to exclude it."""
    c = run.config
    net = c.get("net_params", {}) or {}
    env = c.get("env_params", {}) or {}
    tags = set(run.tags)
    frame = env.get("body_target_frame")          # authoritative frame
    delay = c.get("delay_k")
    eff = c.get("efference_length")

    # Common gates: efference-matched, seed 42, standard architecture, delay in sweep.
    if eff != delay or c.get("seed") != 42 or not std_arch(net) or delay not in DELAYS:
        return None

    is_fm = "ForwardModel" in tags
    is_enc = "EncDec" in tags and not is_fm
    # Canonical explicit forward model only: trained L2 loss, detached prediction.
    fm_ok = is_fm and c.get("fm_loss_weight") == 1 and c.get("detach_prediction") in (None, True)
    g = git8(run)

    if frame == "reference_root" and g == REF_GIT:
        if fm_ok:
            return ("reference_forward_model", "reference_root", "forward_model")
        if is_enc:
            return ("reference_efference", "reference_root", "efference")
    if frame == "current_root":
        if is_enc and g == CUR_EFF_GIT:
            return ("current_efference", "current_root", "efference")
        if fm_ok and g == CUR_FM_GIT:
            return ("current_forward_model", "current_root", "forward_model")
    return None


def record_of(run, cond, frame, network) -> dict:
    c = run.config
    net = c.get("net_params", {}) or {}
    env = c.get("env_params", {}) or {}
    s = run.summary
    return {
        # provenance
        "wandb_id": run.id,
        "wandb_name": run.name,
        "wandb_project": PROJECT,
        "git_commit": git8(run),
        "tags": ",".join(sorted(run.tags)),
        # experimental axes
        "condition": cond,
        "frame": frame,
        "network": network,
        "delay_k": c.get("delay_k"),
        "efference_length": c.get("efference_length"),
        # authoritative frame + fm knobs (sanity)
        "body_target_frame": env.get("body_target_frame"),
        "fm_loss_weight": c.get("fm_loss_weight"),
        "detach_prediction": c.get("detach_prediction"),
        # invariants
        "seed": c.get("seed"),
        "latent_size": net.get("latent_size"),
        "kl_weight": net.get("kl_weight"),
        "enc_hidden_sizes": tuple(net.get("enc_hidden_sizes") or []),
        "dec_hidden_sizes": tuple(net.get("dec_hidden_sizes") or []),
        "critic_hidden_sizes": tuple(net.get("critic_hidden_sizes") or []),
        "clip_length": env.get("clip_length"),
        "actual_step": s.get("_step"),
        # metrics (coalesced across the old/new logging key names)
        "reward_mean": coalesce(s, REWARD_MEAN_KEYS),
        "reward_std": coalesce(s, REWARD_STD_KEYS),
        "lifespan_mean": coalesce(s, LIFESPAN_KEYS),
        "fm_pred_mse": coalesce(s, FM_MSE_KEYS),
    }


def main() -> None:
    runs = fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)
    print(f"Fetched {len(runs)} finished runs with tags {REQUIRE_TAGS}")

    records = []
    for r in runs:
        hit = condition_of(r)
        if hit is None:
            continue
        cond, frame, network = hit
        records.append(record_of(r, cond, frame, network))

    df = records_to_df(records)
    df = df.sort_values(["frame", "network", "delay_k", "wandb_id"]).reset_index(drop=True)

    print(f"\nCohort ({len(df)} rows): condition / delay / frame / btf / git")
    for _, row in df.iterrows():
        print(f"  {row['wandb_id']}  {row['condition']:24s} delay={row['delay_k']:<3} "
              f"btf={row['body_target_frame']:14s} git={row['git_commit']}")

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
