"""Explicit vs implicit forward model at 2 G steps: how much is left after convergence?

The question
------------
[`forward-loss-vs-architecture/`](../forward-loss-vs-architecture/) concluded, from 600 M-
step runs, that the **explicit** forward model (a predictor trained by a self-supervised
L2 loss) and the **implicit** one (same architecture, ``fm_loss_weight = 0`` and
``detach_prediction = False``, so the predictor is shaped only by the policy gradient) are
indistinguishable up to delay ~10-15, after which the explicit one pulls away. Given that
[`xml-ceiling-vs-convergence/`](../xml-ceiling-vs-convergence/) showed 600 M is well short
of convergence at these delays, that crossover may be an artefact of budget rather than a
statement about the converged networks.

This folder asks four things of the eight 2 G-step runs:

1. How large is the explicit-vs-implicit reward difference **at 2 G**?
2. Does the crossover delay move when the budget grows? (It is read *within* these runs:
   a 2 G run's curve at 600 M is the 600 M run, see below.)
3. Does the **forward-model prediction error** (``fm_pred_mse``) saturate earlier or later
   than reward?
4. Is there any sign of **overfitting** — training reward still rising while held-out
   performance flattens or falls?

Design: eight runs, nothing else
--------------------------------
``expfm_2g`` (4) and ``pgfm_2g`` (4), delays 0/10/20/50, all new XML + ``reference_root``
+ torque, seed 42, launched 2026-08-13 at ``25732c42``. The budget axis is read *along*
each run rather than across separate cohorts, which is legitimate here: ``total_steps``
only bounds PPO's training loop and the learning rate is constant
(``nnx_ppo/algorithms/ppo.py``), so a 2 G run's state at 600 M *is* a 600 M run.
``xml-ceiling-vs-convergence`` verified this empirically against the separately launched
600 M twins: the eight matched pairs agree to **±2.9 %**, which is the noise floor used
throughout this analysis.

Reading budget within-run has a second advantage over comparing separate cohorts: at every
budget the two arms differ *only* in the forward-model knobs. No commit, GPU, or launch
date varies with the x-axis.

Metrics
-------
Beyond reward and lifespan, the ``history`` artifact here carries
``eval/net/3/action/1/fm_pred_mse/mean`` -- the L2 between the predictor's output and the
true current proprioception. It is logged by **both** arms: in the implicit arm the target
is never used in the loss, so it is the "would-be" prediction error, which is exactly the
quantity that says whether the policy gradient learns to predict on its own. Also carried:
the policy sigma, the encoder KL, and the tracking error, as diagnostics.

The overfitting question, and what these runs cannot answer
-----------------------------------------------------------
``eval_env == train_env`` in these runs, so the logged ``eval/*`` curve is a
*deterministic evaluation on the training clips*, not held-out. There is therefore **no
held-out curve over training**. The only held-out numbers are the single end-of-training
``final_eval/*`` points, which do exist for all eight runs (and, for the implicit arm
only, for the separate 600 M runs, giving one before/after comparison of the
generalisation gap). A checkpoint-sweep offline eval is what would answer the question
properly; see ``report.md``.

Run it
------
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-2g/extract.py           # frozen
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-2g/extract.py --refresh
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-2g/extract.py --check
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: A non-default ``history`` spec: the same sampled endpoint, but asking for the
#: forward-model and policy diagnostics as well. Reproduce with::
#:
#:     python -m vnl_experiments.artifacts ensure --kind history --runs <ids> \
#:         --set keys='["eval/episode_reward/mean", ...]'
HISTORY_SPEC_ID = "hist2000-fc46b078"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

NEW_XML = "rodent_no_tail_collisions.xml"
BUDGET = 2_000_000_000
EXPECTED_STEP = 2_000_076_800
DELAYS = (0, 10, 20, 50)

STD_ARCH = {
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
}

#: Curve column names in the artifact -> the short names used in ``curves.csv``.
CURVE_COLUMNS = {
    "eval/episode_reward/mean": "reward_mean",
    "eval/lifespan/mean": "lifespan_mean",
    "eval/net/3/action/1/fm_pred_mse/mean": "fm_mse_eval",
    "net/3/action/1/fm_pred_mse/p50": "fm_mse_train_p50",
    "net/3/action/1/fm_pred_mse/p25": "fm_mse_train_p25",
    "net/3/action/1/fm_pred_mse/p75": "fm_mse_train_p75",
    "eval/net/3/action/1/decoder/5/sigma/mean": "action_sigma",
    "eval/net/3/action/0/task_obs/6/kl_divergence/mean": "encoder_kl",
    "eval/env/joint_l2_error/mean": "joint_l2_error",
    "eval/env/terminations/any/mean": "termination_rate",
}

#: Budgets the curves are read at. Doublings, so "gain per doubling" is a difference
#: between adjacent columns and the scaling behaviour is legible as a table.
MILESTONES = (125_000_000, 250_000_000, 500_000_000, 600_000_000,
              1_000_000_000, 2_000_000_000)

#: Averaging window for every "value at step X". Eval runs every 10 M steps and a single
#: point moves ~1-2 %, so five points are averaged. See analysis/README.md §6.
WINDOW = 50_000_000

EVAL_DATASETS = ("train", "old_eval", "new_eval")


# --------------------------------------------------------------------------------------
# condition selectors
# --------------------------------------------------------------------------------------


def _cell(network: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        is_fm = df["tags"].fillna("").str.split(",").apply(lambda t: "ForwardModel" in t)
        if network == "expfm":
            # `ne(False)` rather than `fillna(True)`: unset means the constructor
            # default, which is True.
            arm = (df["fm_loss_weight"] == 1) & df["detach_prediction"].ne(False)
        else:
            arm = (df["fm_loss_weight"] == 0) & (df["detach_prediction"] == False)  # noqa: E712
        mask = (
            (df["env"] == "AbsoluteImitation")
            & (df["seed"] == 42)
            & (df["config.ppo.total_steps"] == BUDGET)
            & (df["summary._step"] == EXPECTED_STEP)
            & (df["state"] == "finished")
            & is_fm & arm
            & df["env_params.walker_xml_path"].astype(str).str.contains(NEW_XML)
            & (df["env_params.body_target_frame"] == "reference_root")
            & (df["env_params.torque_actuators"] == True)  # noqa: E712
            & (df["delay_k"] == df["efference_length"])
            & df["delay_k"].isin(DELAYS)
        )
        for column, value in STD_ARCH.items():
            mask &= df[column] == value
        return mask

    return selector


CONDITIONS = {
    "expfm_2g": _cell("expfm"),   # fm_loss_weight = 1, prediction detached
    "pgfm_2g": _cell("pgfm"),     # fm_loss_weight = 0, predictor trained by the PG
}

#: The two arms differ *only* in these two fields; everything else must match. Including
#: `git_commit` and `summary._step` as invariants is meaningful here rather than pro
#: forma: this cohort was launched as one batch.
INVARIANTS = [
    "env", "seed", "net_params.latent_size", "net_params.kl_weight",
    "net_params.min_std", "net_params.latent_min_std", "net_params.std_scale",
    "net_params.enc_hidden_sizes", "net_params.dec_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "env_params.clip_length", "env_params.ctrl_dt", "env_params.sim_dt",
    "env_params.solver", "env_params.iterations", "env_params.ls_iterations",
    "env_params.njmax", "env_params.naconmax", "env_params.rescale_factor",
    "env_params.mujoco_impl", "env_params.walker_xml_path",
    "env_params.torque_actuators", "env_params.body_target_frame",
    "config.ppo.n_envs", "config.ppo.learning_rate", "config.ppo.rollout_length",
    "config.ppo.n_epochs", "config.ppo.n_minibatches", "config.ppo.clip_range",
    "config.ppo.discounting_factor", "config.ppo.gae_lambda",
    "config.ppo.total_steps", "summary._step", "git_commit",
    "config.ppo.logging_level", "config.eval.logging_level",
]

NETWORK_OF = {"expfm_2g": "forward_model", "pgfm_2g": "pg_forward_model"}


# --------------------------------------------------------------------------------------
# curve reductions
# --------------------------------------------------------------------------------------


def load_curve(store: Store, wandb_id: str) -> pd.DataFrame:
    entry = store.lookup("history", wandb_id, HISTORY_SPEC_ID)
    if entry is None:
        raise FileNotFoundError(
            f"no history:{HISTORY_SPEC_ID} for {wandb_id}. Produce it with the "
            f"`artifacts ensure --set keys=...` command in this module's docstring.")
    frame = pd.read_csv(store.root / entry.path).sort_values("_step")
    out = pd.DataFrame({"step": frame["_step"].astype(int)})
    for source, name in CURVE_COLUMNS.items():
        out[name] = frame[source] if source in frame.columns else np.nan
    return out.reset_index(drop=True)


def window_mean(curve: pd.DataFrame, column: str, step: int,
                window: int = WINDOW) -> float | None:
    """Mean of the eval points in ``(step - window, step]``; ``None`` if fewer than 3."""
    sub = curve[(curve["step"] > step - window) & (curve["step"] <= step)]
    return float(sub[column].mean()) if len(sub) >= 3 else None


def smoothed(curve: pd.DataFrame, column: str) -> pd.Series:
    return curve[column].rolling(int(WINDOW / 10_000_000), min_periods=1).mean()


def saturation_step(curve: pd.DataFrame, column: str, fraction: float,
                    start: int = 100_000_000) -> float | None:
    """First step at which ``column`` has completed ``fraction`` of its total change.

    Total change is measured from ``start`` (past the initial transient, which otherwise
    dominates and makes every metric look saturated) to the end of the run. Returns
    ``None`` for a metric whose net change is negligible, where the fraction is
    meaningless. **Monotonicity is not assumed and not checked** -- for a curve that
    overshoots and comes back (the implicit arm at delay 50 does both, in reward and in
    prediction error) this reports when the *net* change was first completed, which is
    not a saturation time; ``peak_step`` in ``data.csv`` is what to read there.
    """
    sub = curve[curve["step"] >= start]
    values = smoothed(sub, column).to_numpy()
    steps = sub["step"].to_numpy()
    if len(values) < 10:
        return None
    total = values[-1] - values[0]
    if abs(total) < 1e-12 or abs(total) < 0.02 * abs(values[0]):
        return None
    progress = (values - values[0]) / total
    hit = np.nonzero(progress >= fraction)[0]
    return None if len(hit) == 0 else float(steps[hit[0]])


def build_row(run: pd.Series, curve: pd.DataFrame) -> dict:
    condition = run["condition"]
    row = {
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "wandb_project": PROJECT,
        "state": run["state"],
        "git_commit": (run["git_commit"] or "")[:8],
        "created_at": run["created_at"],
        "condition": condition,
        "network": NETWORK_OF[condition],
        "arm": "explicit" if condition == "expfm_2g" else "implicit",
        "delay_k": int(run["delay_k"]),
        "fm_loss_weight": run["fm_loss_weight"],
        "detach_prediction": run["detach_prediction"],
        "budget": int(run["config.ppo.total_steps"]),
        "actual_step": run["summary._step"],
        "gpu": run["gpu"],
        "runtime_s": run.get("summary._runtime"),
    }
    for milestone in MILESTONES:
        label = f"{milestone // 1_000_000}M"
        row[f"reward_at_{label}"] = window_mean(curve, "reward_mean", milestone)
        row[f"fm_mse_at_{label}"] = window_mean(curve, "fm_mse_eval", milestone)

    smooth_reward = smoothed(curve, "reward_mean")
    peak = int(smooth_reward.idxmax())
    row["reward_peak"] = float(smooth_reward.max())
    row["peak_step"] = int(curve.loc[peak, "step"])
    for fraction in (0.90, 0.95):
        tag = int(fraction * 100)
        row[f"reward_steps_to_{tag}pct"] = saturation_step(curve, "reward_mean", fraction)
        row[f"fm_mse_steps_to_{tag}pct"] = saturation_step(curve, "fm_mse_eval", fraction)

    for column in ("action_sigma", "encoder_kl", "joint_l2_error", "fm_mse_eval",
                   "fm_mse_train_p50", "termination_rate"):
        row[f"{column}_at_600M"] = window_mean(curve, column, 600_000_000)
        row[f"{column}_final"] = window_mean(curve, column, 2_000_000_000)

    # Inline end-of-training eval: the only held-out numbers these runs carry. A
    # different measurement from the batch `eval` artifacts (in-memory weights vs the
    # newest checkpoint on disk) -- never mix them in one figure.
    for dataset in EVAL_DATASETS:
        row[f"inline_{dataset}_reward"] = run.get(
            f"summary.final_eval/{dataset}/episode_reward/mean")
        row[f"inline_{dataset}_survived"] = run.get(
            f"summary.final_eval/{dataset}/termination_rate/survived")
    train = row["inline_train_reward"]
    row["generalization_ratio"] = (
        None if not train else row["inline_old_eval_reward"] / train)
    return row


#: The implicit arm's separately launched 600 M runs. They are **not** part of this
#: analysis's cohort -- the budget axis is read within the 2 G runs -- but they are the
#: only way to see the generalisation gap at two budgets, since the inline eval did not
#: exist when the explicit arm's 600 M runs were launched and the eval numbers cannot be
#: reconstructed from a training curve. Read from the index only, one scalar each.
PGFM_600M_RUNS = {0: ["efc0z8fe"], 10: ["cgs8q5gj", "kwk401pl"],
                  20: ["yt776s6d"], 50: ["lxgp2zfn"]}


def build_generalization_rows(index_frame: pd.DataFrame) -> list[dict]:
    rows = []
    for delay, ids in PGFM_600M_RUNS.items():
        for wandb_id in ids:
            run = index_frame.loc[wandb_id]
            row = {"wandb_id": wandb_id, "condition": "pgfm_600m_reference",
                   "arm": "implicit", "delay_k": delay, "budget": 600_000_000}
            for dataset in EVAL_DATASETS:
                row[f"inline_{dataset}_reward"] = run.get(
                    f"summary.final_eval/{dataset}/episode_reward/mean")
                row[f"inline_{dataset}_survived"] = run.get(
                    f"summary.final_eval/{dataset}/termination_rate/survived")
            row["generalization_ratio"] = (row["inline_old_eval_reward"]
                                           / row["inline_train_reward"])
            rows.append(row)
    return rows


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    rows, curves = [], []
    for _, run in runs.iterrows():
        curve = load_curve(store, run["wandb_id"])
        rows.append(build_row(run, curve))
        tagged = curve.copy()
        tagged.insert(0, "delay_k", int(run["delay_k"]))
        tagged.insert(0, "condition", run["condition"])
        tagged.insert(0, "wandb_id", run["wandb_id"])
        curves.append(tagged)

    df = pd.DataFrame(rows).sort_values(["condition", "delay_k"], ignore_index=True)
    curves_df = pd.concat(curves, ignore_index=True).sort_values(
        ["condition", "delay_k", "step"], ignore_index=True)

    from vnl_experiments.wandb_utils import index as index_module
    index_frame = index_module.load(args.project).set_index("wandb_id")
    gen_df = pd.DataFrame(build_generalization_rows(index_frame) +
                          df[["wandb_id", "condition", "arm", "delay_k", "budget",
                              *[f"inline_{d}_{m}" for d in EVAL_DATASETS
                                for m in ("reward", "survived")],
                              "generalization_ratio"]].to_dict("records"))
    gen_df = gen_df.sort_values(["arm", "budget", "delay_k"], ignore_index=True)

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report + "\n")

    print("\nCohort:")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:10s} n={len(sub)}  arm={sub['arm'].iat[0]:8s} "
              f"fm_loss_weight={sub['fm_loss_weight'].iat[0]} "
              f"detach={sub['detach_prediction'].iat[0]} "
              f"delays={sorted(sub['delay_k'])} git={sorted(sub['git_commit'].unique())}")

    print("\nReward by budget:")
    print(df.pivot_table(index="delay_k", columns="arm",
                         values=[f"reward_at_{m // 1_000_000}M"
                                 for m in (600_000_000, 2_000_000_000)])
          .round(0).to_string())
    print("\nForward-prediction MSE (eval) by budget:")
    print(df.pivot_table(index="delay_k", columns="arm",
                         values=[f"fm_mse_at_{m // 1_000_000}M"
                                 for m in (600_000_000, 2_000_000_000)])
          .round(4).to_string())

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    ok &= pipeline.write_csv(gen_df, HERE / "data_generalization.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
