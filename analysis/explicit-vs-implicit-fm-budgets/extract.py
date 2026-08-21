"""Explicit vs implicit forward model across three training budgets: 600 M, 2 G, 4 G.

The question
------------
Same contrast as [`explicit-vs-implicit-fm-2g/`](../explicit-vs-implicit-fm-2g/) -- the
**explicit** forward model (predictor trained by a self-supervised L2 loss) against the
**implicit** one (same architecture, ``fm_loss_weight = 0`` and
``detach_prediction = False``, so the predictor is shaped only by the policy gradient) --
but now with a delay sweep at every budget we have, rather than four delays at one budget.

That analysis found the crossover delay drifting later as the budget grew (~10 at 500 M,
~13 at 1 G, ~17 at 2 G) with nothing sampled between delay 20 and 50 to pin it down. The
2026-08-18/19 batch of **4 G-step** runs samples delays 20/30/40/50 in both arms and lands
in exactly that gap.

Design: three budget tiers, read within-run
-------------------------------------------
``total_steps`` only bounds PPO's training loop and the learning rate is constant
(``nnx_ppo/algorithms/ppo.py`` -- no schedule, no annealing), so a run's state at step *s*
is an *s*-step run. Every run therefore contributes to **every tier it reached**, and a
4 G run appears in the 600 M, 2 G and 4 G panels alike.
``xml-ceiling-vs-convergence`` verified this against separately launched twins: matched
pairs agree to ±2.9 %, the noise floor used throughout.

Reading budget within-run also means that at a given tier the two arms differ *only* in
the forward-model knobs -- no commit, GPU or launch-date rides along with the x-axis.

Two things are deliberately looser here than in a strict comparison
-------------------------------------------------------------------
1. **The 4 G runs are seed 43; everything else is seed 42.** They were launched as a
   second seed, not as a budget extension, so a 600 M-to-4 G reading crosses a seed
   boundary. This is handled rather than ignored: the 4 G runs also *pass through* 600 M
   and 2 G, so the seed effect is measured directly at matched budget and delay
   (``seed_check`` in ``plot.py``) and quoted alongside every 4 G number.
2. **Crashed and still-running runs are included**, contributing to the tiers they
   reached and no further. ``max_step`` is read from the run's own curve, not from
   ``summary._step``, because a running run's artifact is fresher than the index. This is
   what makes a 2.9 G tier worth having: it is the largest budget at which all four
   seed-43 delay pairs are complete, while 4 G has complete pairs only at delays 40 and
   50. Cells with one arm are drawn as examples and labelled as such, never differenced.

One run, ``jhghg9vt`` (4 G, implicit, delay 20, seed 43), crashed before logging anything;
its history artifact is empty and it drops out of every figure. It is kept in ``runs.csv``
so the selection stays a faithful record of what the query matched.

Data sources
------------
Configs and summaries from the committed run index; curves from ``history`` artifacts
under a non-default spec that also carries ``fm_pred_mse``, the policy sigma, the encoder
KL and the tracking error (see ``HISTORY_SPEC_ID`` below). Nothing here touches the WandB
API.

Run it
------
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-budgets/extract.py       # frozen
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-budgets/extract.py --refresh
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-budgets/extract.py --check
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: Non-default ``history`` spec: same sampled endpoint, extra keys. Reproduce with
#: ``artifacts ensure --kind history --runs <file> --set keys='[...]'`` using the names in
#: ``CURVE_COLUMNS`` below.
HISTORY_SPEC_ID = "hist2000-fc46b078"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

NEW_XML = "rodent_no_tail_collisions.xml"
STD_MIN_STD = 0.1
STD_ARCH = {
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
}

#: The budgets the curves are read at. 2.9 G is not a round number by accident: it is the
#: largest tier at which every seed-43 delay has both arms (two of those runs are still
#: training and were at ~2.94-2.98 G when this was extracted).
BUDGET_TIERS = (600_000_000, 1_000_000_000, 2_000_000_000, 2_900_000_000,
                4_000_000_000)

#: Averaging window for every "value at step X". Eval runs every 10 M steps and a single
#: point moves ~1-2 %, so five points are averaged. See analysis/README.md §6.
WINDOW = 50_000_000

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

EVAL_DATASETS = ("train", "old_eval", "new_eval")


# --------------------------------------------------------------------------------------
# condition selectors
# --------------------------------------------------------------------------------------


def _shared(df: pd.DataFrame) -> pd.Series:
    """Everything every included run has in common: the going-forward configuration."""
    mask = (
        (df["env"] == "AbsoluteImitation")
        & df["tags"].fillna("").str.split(",").apply(lambda t: "ForwardModel" in t)
        & (df["delay_k"] == df["efference_length"])
        & pipeline.full_decoder_inputs_mask(df)
        & df["env_params.walker_xml_path"].astype(str).str.contains(NEW_XML)
        & (df["env_params.body_target_frame"] == "reference_root")
        & (df["env_params.torque_actuators"] == True)  # noqa: E712
        & (df["net_params.min_std"] == STD_MIN_STD)
    )
    for column, value in STD_ARCH.items():
        mask &= df[column] == value
    return mask


def _arm(df: pd.DataFrame, arm: str) -> pd.Series:
    """No single config field separates the two arms, so both knobs are needed.

    ``ne(False)`` rather than ``fillna(True)`` for the detach flag: unset means the
    ``ForwardModel`` constructor default, which is ``True``.
    """
    if arm == "explicit":
        return (df["fm_loss_weight"] == 1) & df["detach_prediction"].ne(False)
    return (df["fm_loss_weight"] == 0) & (df["detach_prediction"] == False)  # noqa: E712


def _cell(arm: str, *, budget: int, seed: int):
    def selector(df: pd.DataFrame) -> pd.Series:
        # No `state` filter and no `summary._step` filter: crashed and running runs are
        # wanted, and how far each got is decided per budget tier from its own curve.
        return (_shared(df) & _arm(df, arm)
                & (df["config.ppo.total_steps"] == budget)
                & (df["seed"] == seed))

    return selector


CONDITIONS = {
    # --- the budget tiers, min_std 0.1 ------------------------------------------------
    # 600 M: the widest delay coverage we have (0-100 in both arms).
    "expfm_600m": _cell("explicit", budget=600_000_000, seed=42),
    "pgfm_600m": _cell("implicit", budget=600_000_000, seed=42),
    # 2 G: delays 0/10/20/50, the cohort of explicit-vs-implicit-fm-2g.
    "expfm_2g": _cell("explicit", budget=2_000_000_000, seed=42),
    "pgfm_2g": _cell("implicit", budget=2_000_000_000, seed=42),
    # 4 G: delays 20/30/40/50, seed 43. Two per arm are still running or crashed.
    "expfm_4g": _cell("explicit", budget=4_000_000_000, seed=43),
    "pgfm_4g": _cell("implicit", budget=4_000_000_000, seed=43),
}

#: ``arm`` and ``tier`` are encoded in the condition name; ``delay_k`` is the sweep axis.
#: Everything here must hold within a condition. ``seed`` and
#: ``config.ppo.total_steps`` vary *between* conditions by design and are listed so the
#: report can point at where.
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
    "config.ppo.total_steps", "fm_loss_weight", "detach_prediction", "git_commit",
]


def arm_of(condition: str) -> str:
    return "explicit" if condition.startswith("expfm") else "implicit"


def tier_of(condition: str) -> str:
    return condition.split("_", 1)[1]


# --------------------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------------------


def load_curve(store: Store, wandb_id: str) -> pd.DataFrame | None:
    """Tidy curve for one run, or ``None`` if the run logged nothing.

    A run that died before its first eval leaves a zero-byte artifact (``jhghg9vt``), so
    the empty case is expected rather than exceptional.
    """
    entry = store.lookup("history", wandb_id, HISTORY_SPEC_ID)
    if entry is None:
        raise FileNotFoundError(
            f"no history:{HISTORY_SPEC_ID} for {wandb_id}; see this module's docstring")
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None
    if "eval/episode_reward/mean" not in frame.columns:
        return None
    frame = frame.dropna(subset=["eval/episode_reward/mean"]).sort_values("_step")
    if frame.empty:
        return None
    out = pd.DataFrame({"step": frame["_step"].astype(int)})
    for source, name in CURVE_COLUMNS.items():
        out[name] = frame[source].to_numpy() if source in frame.columns else np.nan
    return out.reset_index(drop=True)


def window_mean(curve: pd.DataFrame, column: str, step: int) -> float | None:
    """Mean of the eval points in ``(step - WINDOW, step]``.

    ``None`` when fewer than three points land in the window, which is also how a run
    that never reached ``step`` is excluded from that budget tier -- no separate
    max-step test is needed at the call sites.
    """
    sub = curve[(curve["step"] > step - WINDOW) & (curve["step"] <= step)]
    return float(sub[column].mean()) if len(sub) >= 3 else None


def build_row(run: pd.Series, curve: pd.DataFrame | None) -> dict:
    condition = run["condition"]
    row = {
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "wandb_project": PROJECT,
        "state": run["state"],
        "git_commit": (run["git_commit"] or "")[:8],
        "created_at": run["created_at"],
        "condition": condition,
        "arm": arm_of(condition),
        "tier": tier_of(condition),
        "budget": int(run["config.ppo.total_steps"]),
        "delay_k": int(run["delay_k"]),
        "seed": int(run["seed"]),
        "min_std": run["net_params.min_std"],
        "fm_loss_weight": run["fm_loss_weight"],
        "detach_prediction": run["detach_prediction"],
        "gpu": run["gpu"],
        "runtime_s": run.get("summary._runtime"),
        # From the curve, not summary._step: a running run's artifact is fresher than the
        # index, and this is the number every tier test is made against.
        "max_step": None if curve is None else int(curve["step"].max()),
        "n_eval_points": 0 if curve is None else len(curve),
    }
    for tier in BUDGET_TIERS:
        label = (f"{tier // 1_000_000}M" if tier < 1_000_000_000
                 else f"{tier / 1e9:g}G".replace(".", "p"))
        row[f"reward_at_{label}"] = (
            None if curve is None else window_mean(curve, "reward_mean", tier))
        row[f"fm_mse_at_{label}"] = (
            None if curve is None else window_mean(curve, "fm_mse_eval", tier))
    if curve is not None:
        last = int(curve["step"].max())
        for column in ("reward_mean", "fm_mse_eval", "action_sigma", "encoder_kl",
                       "joint_l2_error", "lifespan_mean"):
            row[f"{column}_final"] = window_mean(curve, column, last)
    for dataset in EVAL_DATASETS:
        row[f"inline_{dataset}_reward"] = run.get(
            f"summary.final_eval/{dataset}/episode_reward/mean")
    return row


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
        if curve is None:
            continue
        tagged = curve.copy()
        tagged.insert(0, "delay_k", int(run["delay_k"]))
        tagged.insert(0, "seed", int(run["seed"]))
        tagged.insert(0, "condition", run["condition"])
        tagged.insert(0, "wandb_id", run["wandb_id"])
        curves.append(tagged)

    df = pd.DataFrame(rows).sort_values(["condition", "delay_k", "wandb_id"],
                                       ignore_index=True)
    curves_df = pd.concat(curves, ignore_index=True).sort_values(
        ["condition", "delay_k", "wandb_id", "step"], ignore_index=True)

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report + "\n")

    print("\nCohort:")
    for cond, sub in df.groupby("condition"):
        reached = (sub["max_step"] / 1e6).round(0).astype("Int64").tolist()
        print(f"  {cond:18s} n={len(sub):2d} arm={sub['arm'].iat[0]:8s} "
              f"seed={sub['seed'].iat[0]} min_std={sub['min_std'].iat[0]} "
              f"budget={sub['budget'].iat[0] // 1_000_000:>4d}M")
        print(f"  {'':18s} delays={sorted(sub['delay_k'])}")
        print(f"  {'':18s} reached(M)={reached}  states={sorted(set(sub['state']))}")

    empty = df[df["max_step"].isna()]
    if not empty.empty:
        print(f"\nNo curve at all (dropped from every figure): "
              f"{empty['wandb_id'].tolist()}")

    print("\nRuns per (tier, arm) reaching each budget:")
    counts = {}
    for tier in BUDGET_TIERS:
        label = (f"{tier // 1_000_000}M" if tier < 1_000_000_000
                 else f"{tier / 1e9:g}G".replace(".", "p"))
        ok = df[df[f"reward_at_{label}"].notna()]
        counts[label] = ok.groupby(["tier", "arm"]).size()
    print(pd.DataFrame(counts).fillna(0).astype(int).to_string())

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
