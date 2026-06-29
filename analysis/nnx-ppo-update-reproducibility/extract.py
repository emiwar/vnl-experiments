"""Reproducibility check: does the delay-sweep curve survive the nnx-ppo update + a new seed?

Run from the repo root::

    ../.venv/bin/python analysis/nnx-ppo-update-reproducibility/extract.py

The standard with-efference delay sweep was re-run on the cluster with the updated nnx-ppo
(repo commit ``714cc735``, "new version of nnx-ppo [git 4ed1f36]"). We compare three conditions,
all standard-architecture efference runs (``efference_length == delay_k``, AbsoluteImitation,
``body_target_frame=reference_root``, 600 M steps):

  - ``baseline``           — the original committed sweep (git ``1cd5838``, the old code, seed 42).
  - ``new_seed``           — the full re-run sweep with the updated nnx-ppo AND a new seed (43).
  - ``new_code_old_seed``  — a single delay-0 test run with the updated nnx-ppo but the OLD seed
                             (42); isolates the *code* change from the *seed* change.

The updated code logs the eval reward under ``eval/episode_reward/mean``; the old code used
``episode_reward/mean``. Both are the same quantity (eval episode reward), so we coalesce them
into ``episode_reward_mean`` / ``episode_reward_std``.

git_commit and seed VARY by design here (that *is* the thing being tested); everything else
(network, env, PPO, trained steps) must stay constant for the comparison to be meaningful.
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

BASELINE_GIT = "1cd5838f"
NEW_GIT = "714cc735"
OLD_SEED = 42
NEW_SEED = 43

CONFIG_KEYS = ["env", "delay_k", "efference_length", "seed"]
NET_PARAM_KEYS = [
    "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
]
PPO_KEYS = ["n_envs", "total_steps"]
# Pull both the old and new reward keys; coalesce below.
METRICS = [
    "episode_reward/mean", "episode_reward/std",
    "eval/episode_reward/mean", "eval/episode_reward/std",
    "eval/lifespan/mean",
]

# Standard baseline architecture (same filter as the original sweep): exclude the
# decoder-size sweeps etc. so the network is held fixed.
STD_ARCH = {
    "enc_hidden_sizes": [512, 512, 512, 512],
    "dec_hidden_sizes": [512, 512, 512, 512],
    "critic_hidden_sizes": [1024, 1024],
}


def git_short(run) -> str:
    return ((run.metadata or {}).get("git", {}) or {}).get("commit", "")[:8]


def condition_of(run) -> str | None:
    if "ForwardModel" in run.tags:
        return None
    c = run.config
    net = c.get("net_params", {}) or {}
    if any(net.get(k) != v for k, v in STD_ARCH.items()):
        return None  # non-standard architecture -> different question
    delay = c.get("delay_k")
    eff = c.get("efference_length")
    if eff != delay:           # efference condition only (eff == delay)
        return None
    g = git_short(run)
    seed = c.get("seed")
    if g == BASELINE_GIT:
        return "baseline"
    if g == NEW_GIT and seed == NEW_SEED:
        return "new_seed"
    if g == NEW_GIT and seed == OLD_SEED:
        return "new_code_old_seed"
    return None


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
            extra={"condition": cond, "git_short": git_short(r)},
        )
        # Coalesce old/new reward keys into one column.
        rec["episode_reward_mean"] = (
            rec.get("eval/episode_reward/mean")
            if rec.get("eval/episode_reward/mean") is not None
            else rec.get("episode_reward/mean")
        )
        rec["episode_reward_std"] = (
            rec.get("eval/episode_reward/std")
            if rec.get("eval/episode_reward/std") is not None
            else rec.get("episode_reward/std")
        )
        rec["lifespan_mean"] = rec.get("eval/lifespan/mean")
        records.append(rec)

    df = records_to_df(records)
    df = df.sort_values(["condition", "delay_k"]).reset_index(drop=True)

    # Drop any baseline duplicate delays (the original sweep has 2 seeds at delay 0):
    # keep them — they show the within-condition spread at delay 0. Just report counts.
    print("\nrows per condition:")
    print(df.groupby("condition")["delay_k"].agg(["count", "min", "max"]).to_string())
    print("\ndelay coverage:")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:18s}: {sorted(sub['delay_k'].unique())}")

    # Comparability: network/env/PPO/steps must be constant; git & seed vary by design.
    inv = ["env", "latent_size", "kl_weight", "body_target_frame",
           "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
           "n_envs", "total_steps", "actual_step", "git_short", "seed"]
    report = comparability_report(df, invariant_cols=inv, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
