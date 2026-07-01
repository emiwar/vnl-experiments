"""Forward-model *loss* vs *architecture*: is the benefit the self-supervised loss, or can the
policy gradient alone make the forward-model architecture useful?

Run from the repo root::

    ../.venv/bin/python analysis/forward-loss-vs-architecture/extract.py

The explicit forward model adds a predictor architecture trained by a self-supervised L2 loss
(``fm_loss_weight``, logged as ``fm_pred_mse``). A separate ablation set turns that loss OFF
(``fm_loss_weight = 0``) but removes the stop-gradient so the **policy gradient** can train the
predictor (``--no-detach-prediction``, i.e. ``detach_prediction = False``). Does this
policy-gradient forward model recover the performance of the explicit one — and does it do so by
*implicitly* learning to predict proprioception (i.e. reducing the same L2)?

Conditions (all standard-arch, ``efference_length == delay_k``, AbsoluteImitation,
``body_target_frame=reference_root``, latent 32, 600 M steps):
  - ``forward_model``           — the canonical explicit FM sweep, ``fm_loss_weight = 1``,
                                  detached predictor (git ``54643764``, full delay coverage).
  - ``pg_forward_model``        — the new baseline: ``fm_loss_weight = 0`` AND
                                  ``detach_prediction = False`` so the policy gradient trains the
                                  predictor (git ``d4bd4dc0``; delays 0,1,2,5,10,20,50,80,100).
  - ``forward_model_nnxupdate`` — bridge check: ``fm_loss_weight = 1``, detached, at the SAME
                                  commit as pg (git ``d4bd4dc0``, delays 0,5,10). Confirms the
                                  commit difference between the canonical sweep and the new
                                  baseline is benign.
  - ``fm0_untrained``           — reference: ``fm_loss_weight = 0`` with the predictor DETACHED
                                  (old behaviour, git ``54643764``) — an untrained predictor. Its
                                  L2 sets the "no forward learning at all" level.

Reward key differs by code version (old ``episode_reward/mean`` vs new ``eval/episode_reward/mean``)
and so does the FM-error logging; both are coalesced below. The L2 error we compare across
conditions is the training-time median ``net/3/action/1/fm_pred_mse/p50`` (present for every run),
with the p25/p75 as a spread band.
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
REQUIRE_TAGS = ["ForwardModel", "TrainEvalSplit"]

CANON_GIT = "54643764"   # canonical FM sweep + old untrained (old nnx-ppo)
NEW_GIT = "d4bd4dc0"     # new baseline (nodetach) + bridge (new nnx-ppo, detach flag added)

CONFIG_KEYS = ["env", "delay_k", "efference_length", "seed",
               "fm_loss_weight", "detach_prediction"]
NET_PARAM_KEYS = [
    "latent_size", "kl_weight", "body_target_frame",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
]
FM_MSE = "net/3/action/1/fm_pred_mse"
METRICS = [
    "episode_reward/mean", "episode_reward/std",
    "eval/episode_reward/mean", "eval/episode_reward/std",
    f"{FM_MSE}/p25", f"{FM_MSE}/p50", f"{FM_MSE}/p75",
    f"eval/{FM_MSE}/mean", f"eval/{FM_MSE}/std",
]

STD_ARCH = {
    "enc_hidden_sizes": [512, 512, 512, 512],
    "dec_hidden_sizes": [512, 512, 512, 512],
    "critic_hidden_sizes": [1024, 1024],
}


def git8(run) -> str:
    return ((run.metadata or {}).get("git", {}) or {}).get("commit", "")[:8]


def _is(x, v):  # tolerant numeric equality (1 == 1.0, 0 == 0.0)
    try:
        return float(x) == float(v)
    except (TypeError, ValueError):
        return x == v


def condition_of(run) -> str | None:
    c = run.config
    net = c.get("net_params", {}) or {}
    if any(net.get(k) != v for k, v in STD_ARCH.items()):
        return None
    if c.get("efference_length") != c.get("delay_k"):
        return None
    g = git8(run)
    fmw = c.get("fm_loss_weight")
    det = c.get("detach_prediction")
    if g == CANON_GIT and _is(fmw, 1):
        return "forward_model"
    if g == CANON_GIT and _is(fmw, 0) and det in (None, True):
        return "fm0_untrained"
    if g == NEW_GIT and det is True and _is(fmw, 1):
        return "forward_model_nnxupdate"
    if g == NEW_GIT and det is False and _is(fmw, 0):
        return "pg_forward_model"
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
            metrics=METRICS,
            extra={"condition": cond, "git_short": git8(r)},
        )
        rec["episode_reward_mean"] = (
            rec.get("eval/episode_reward/mean")
            if rec.get("eval/episode_reward/mean") is not None
            else rec.get("episode_reward/mean")
        )
        # L2 forward-prediction error: training-time median (all runs) + eval mean (new runs).
        rec["fm_mse_p50"] = rec.get(f"{FM_MSE}/p50")
        rec["fm_mse_p25"] = rec.get(f"{FM_MSE}/p25")
        rec["fm_mse_p75"] = rec.get(f"{FM_MSE}/p75")
        rec["fm_mse_eval_mean"] = rec.get(f"eval/{FM_MSE}/mean")
        records.append(rec)

    df = records_to_df(records)
    df = df.sort_values(["condition", "delay_k"]).reset_index(drop=True)

    print("\nrows/condition and delay coverage:")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:24s} n={len(sub):2d}  git={sorted(sub['git_short'].unique())}  "
              f"delays={sorted(sub['delay_k'].unique())}")

    inv = ["env", "latent_size", "kl_weight", "body_target_frame",
           "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes"]
    report = comparability_report(df, invariant_cols=inv, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
