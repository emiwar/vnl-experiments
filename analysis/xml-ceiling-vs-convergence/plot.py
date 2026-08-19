"""Figures for "is the new-XML deficit a ceiling or a slower climb?".

Run from the repo root::

    ../.venv/bin/python analysis/xml-ceiling-vs-convergence/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/xml-ceiling-vs-convergence/plot.py

Reads ONLY the committed CSVs (never WandB, never the artifact store). See
analysis/README.md §3.

Encoding, used in every figure: **grey dashed = the old-XML baseline at 600 M**,
**thin network-coloured = the new XML at 600 M**, **thick network-coloured = the new XML
at 2 G**. Colour is the canonical per-network colour, so these line up with the other
questions' figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import (
    apply_style,
    color_for,
    marker_for,
    provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"
EVAL = HERE / "data_eval.csv"

_MANIFEST: dict[str, str] = {}


def stamp(fig, name: str, *inputs) -> None:
    _MANIFEST[f"{name}.png"] = provenance(fig, HERE, *(inputs or (DATA,)))


DELAYS = (0, 10, 20, 50)
STANDARD_BUDGET = 600_000_000

NETWORKS = {"expfm": "forward_model", "pgfm": "pg_forward_model"}
NETWORK_LABEL = {"expfm": "Explicit forward model",
                 "pgfm": "Policy-gradient forward model"}

#: ``network -> (baseline, new @600M, new @2G)``.
TIERS = {
    "expfm": ("expfm_old_600m", "expfm_new_600m", "expfm_new_2g"),
    "pgfm": ("pgfm_old_600m", "pgfm_new_600m", "pgfm_new_2g"),
}
TIER_LABEL = ("old XML, current_root — 600 M",
              "new XML, reference_root — 600 M",
              "new XML, reference_root — 2 G")


def tier_style(network: str, tier: int) -> dict:
    """0 = baseline, 1 = new @600 M, 2 = new @2 G."""
    if tier == 0:
        return dict(color="0.35", ls="--", lw=1.3)
    return dict(color=color_for(NETWORKS[network]), ls="-",
                lw=1.1 if tier == 1 else 2.1, alpha=0.65 if tier == 1 else 1.0)


# --------------------------------------------------------------------------------------
# curve helpers -- everything is read off the smoothed curve, never a single eval point
# --------------------------------------------------------------------------------------

WINDOW_POINTS = 5  # eval every 10 M steps -> a 50 M-step trailing window


def smooth(group: pd.DataFrame) -> pd.DataFrame:
    out = group.sort_values("step").copy()
    out["reward_mean"] = out["reward_mean"].rolling(WINDOW_POINTS, min_periods=1).mean()
    out["lifespan_mean"] = (out["lifespan_mean"]
                            .rolling(WINDOW_POINTS, min_periods=1).mean())
    return out


def curve(curves: pd.DataFrame, condition: str, delay: int) -> pd.DataFrame:
    """The smoothed curve for one cell. Where a cell holds two runs at the same delay
    (pgfm_new_600m at delay 10), they are averaged step-wise."""
    sub = curves[(curves["condition"] == condition) & (curves["delay_k"] == delay)]
    if sub.empty:
        return sub
    if sub["wandb_id"].nunique() > 1:
        sub = (sub.groupby("step", as_index=False)[["reward_mean", "lifespan_mean"]]
                  .mean())
    return smooth(sub)


def value_at(curves: pd.DataFrame, condition: str, delay: int, step: int,
             metric: str = "reward_mean") -> float:
    """The smoothed value at (or just before) ``step``; NaN if the run never got there."""
    c = curve(curves, condition, delay)
    if c.empty:
        return np.nan
    c = c[c["step"] <= step]
    return np.nan if c.empty else float(c[metric].iloc[-1])


def final_value(curves: pd.DataFrame, condition: str, delay: int,
                metric: str = "reward_mean") -> float:
    c = curve(curves, condition, delay)
    return np.nan if c.empty else float(c[metric].iloc[-1])


def steps_to(curves: pd.DataFrame, condition: str, delay: int, target: float) -> float:
    """First step whose smoothed reward reaches ``target``; NaN if it never does."""
    c = curve(curves, condition, delay)
    if c.empty or not np.isfinite(target):
        return np.nan
    hit = c[c["reward_mean"] >= target]
    return np.nan if hit.empty else float(hit["step"].iloc[0])


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def contrast_table(curves: pd.DataFrame) -> pd.DataFrame:
    """The whole analysis in one frame: baseline, new @600 M, new @2 G, per cell."""
    rows = []
    for network, (base, new600, new2g) in TIERS.items():
        for delay in DELAYS:
            b = value_at(curves, base, delay, STANDARD_BUDGET)
            n6 = value_at(curves, new600, delay, STANDARD_BUDGET)
            n2_at600 = value_at(curves, new2g, delay, STANDARD_BUDGET)
            n2 = final_value(curves, new2g, delay)
            rows.append({
                "network": network,
                "delay_k": delay,
                "baseline_600M": b,
                "new_600M": n6,
                "new_2G_at_600M": n2_at600,
                "new_2G_final": n2,
                # the deficit at the standard budget, and after 3.3x the budget
                "deficit_600M_pct": 100 * (n6 / b - 1),
                "deficit_2G_pct": 100 * (n2 / b - 1),
                # same configuration, two runs, same step: pure noise
                "replicate_pct": 100 * (n2_at600 / n6 - 1),
                # is the long run itself finished?
                "new_2G_gain_last_500M_pct": 100 * (
                    n2 / value_at(curves, new2g, delay, 1_500_000_000) - 1),
                "baseline_gain_last_100M_pct": 100 * (
                    b / value_at(curves, base, delay, 500_000_000) - 1),
                # convergence speed, at a target both arms can be asked about
                "steps_to_baseline_600M": steps_to(curves, new2g, delay, b),
                "base_steps_to_90pct": steps_to(curves, base, delay, 0.9 * b),
                "new_steps_to_90pct": steps_to(curves, new2g, delay, 0.9 * b),
            })
    out = pd.DataFrame(rows)
    out["slowdown_factor"] = out["new_steps_to_90pct"] / out["base_steps_to_90pct"]
    return out


# --------------------------------------------------------------------------------------
# Figure 1 (headline): the three tiers, one panel per network x delay
# --------------------------------------------------------------------------------------


def fig_budget(curves: pd.DataFrame, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(DELAYS), figsize=(3.2 * len(DELAYS), 6.4),
                             sharex=True)
    for row, network in enumerate(TIERS):
        for col, delay in enumerate(DELAYS):
            ax = axes[row, col]
            base = value_at(curves, TIERS[network][0], delay, STANDARD_BUDGET)
            ax.axhline(base, color="0.35", lw=0.7, ls=":", zorder=0)
            ax.axvline(600, color="0.8", lw=0.7, zorder=0)
            for tier, condition in enumerate(TIERS[network]):
                c = curve(curves, condition, delay)
                if c.empty:
                    continue
                ax.plot(c["step"] / 1e6, c["reward_mean"],
                        label=TIER_LABEL[tier] if (row, col) == (0, 0) else None,
                        **tier_style(network, tier))
            row_ = table[(table["network"] == network) & (table["delay_k"] == delay)]
            pct600, pct2g = row_["deficit_600M_pct"].iat[0], row_["deficit_2G_pct"].iat[0]
            ax.set_title(f"delay {delay}\n{pct600:+.1f} % @600 M  →  {pct2g:+.1f} % @2 G",
                         fontsize=8.5)
            ax.set_ylim(bottom=0)
            if row == 1:
                ax.set_xlabel("Env steps (millions)")
            sns.despine(ax=ax)
        axes[row, 0].set_ylabel(f"{NETWORK_LABEL[network]}\nmean episode reward",
                                fontsize=8.5)
    axes[0, 0].legend(loc="lower right", fontsize=6.5, frameon=False)
    fig.suptitle("Does 3.3x the training budget close the new-XML gap?  "
                 "Dotted line = the baseline's 600 M reward.", fontsize=10)
    fig.tight_layout()
    stamp(fig, "budget", CURVES)
    fig.savefig(FIGURES / "budget.png")
    print("Saved", FIGURES / "budget.png")


# --------------------------------------------------------------------------------------
# Figure 2: the answer -- deficit at 600 M vs after 2 G, against the noise floor
# --------------------------------------------------------------------------------------


def fig_closure(curves: pd.DataFrame, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    noise = table["replicate_pct"].abs().max()

    for ax, network in zip(axes, TIERS):
        sub = table[table["network"] == network].sort_values("delay_k")
        color = color_for(NETWORKS[network])
        ax.axhspan(-noise, noise, color="0.88", zorder=0, lw=0)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.plot(sub["delay_k"], sub["deficit_600M_pct"], color=color,
                marker=marker_for(NETWORKS[network]), ls="--", mfc="none",
                label="at the standard 600 M budget")
        ax.plot(sub["delay_k"], sub["deficit_2G_pct"], color=color,
                marker=marker_for(NETWORKS[network]), ls="-",
                label="after 2 G steps")
        for _, r in sub.iterrows():
            ax.annotate("", xy=(r["delay_k"], r["deficit_2G_pct"]),
                        xytext=(r["delay_k"], r["deficit_600M_pct"]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.9,
                                        alpha=0.55, shrinkA=3, shrinkB=3))
        ax.set_title(NETWORK_LABEL[network])
        ax.set_xlabel("Observation delay (steps)")
        ax.legend(loc="lower left", fontsize=8)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Reward vs the old-XML baseline at 600 M (%)")
    axes[0].annotate(f"run-to-run noise (±{noise:.1f} %)", (0, noise), fontsize=7,
                     color="0.4", va="bottom")
    fig.suptitle("Above zero = the new XML has caught the baseline. "
                 "Arrows show what the extra 1.4 G steps bought.", fontsize=10)
    fig.tight_layout()
    stamp(fig, "closure", CURVES)
    fig.savefig(FIGURES / "closure.png")
    print("Saved", FIGURES / "closure.png")


# --------------------------------------------------------------------------------------
# Figure 3: is the long run itself finished, and how much slower is the climb?
# --------------------------------------------------------------------------------------


def fig_convergence(curves: pd.DataFrame, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) remaining slope along the 2 G runs: gain over the trailing 300 M steps
    ax = axes[0]
    span = 300_000_000
    for network in TIERS:
        for delay in DELAYS:
            c = curve(curves, TIERS[network][2], delay)
            if c.empty:
                continue
            steps, values = c["step"].to_numpy(), c["reward_mean"].to_numpy()
            earlier = np.interp(steps - span, steps, values)
            gain = 100 * (values / earlier - 1)
            keep = steps >= span
            ax.plot(steps[keep] / 1e6, gain[keep], color=color_for(NETWORKS[network]),
                    lw=1.2, alpha=0.3 + 0.7 * DELAYS.index(delay) / (len(DELAYS) - 1),
                    label=f"{network} d{delay}")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.axvline(600, color="0.8", lw=0.7)
    ax.set_xlabel("Env steps (millions)")
    ax.set_ylabel("Reward gained over the\nprevious 300 M steps (%)")
    ax.set_title("Are the 2 G runs converged?")
    # The first few hundred million steps are off the scale (>100 % gains) and are not
    # what this panel is about; clip so the tail is readable.
    ax.set_ylim(-4, 26)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    sns.despine(ax=ax)

    # (b) steps the new arm needs to reach the baseline's 600 M reward
    ax = axes[1]
    width = 0.36
    x = np.arange(len(DELAYS))
    for i, network in enumerate(TIERS):
        sub = table[table["network"] == network].set_index("delay_k").loc[list(DELAYS)]
        steps = sub["steps_to_baseline_600M"].to_numpy() / 1e6
        bars = ax.bar(x + (i - 0.5) * width, np.nan_to_num(steps, nan=0.0), width,
                      color=color_for(NETWORKS[network]), label=NETWORK_LABEL[network])
        for bar, value in zip(bars, steps):
            if np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, 30,
                        "never, in 2 G steps", rotation=90, ha="center", va="bottom",
                        fontsize=7, color=color_for(NETWORKS[network]))
    ax.axhline(600, color="0.35", lw=1.0, ls="--")
    ax.text(len(DELAYS) - 0.55, 620, "baseline budget", fontsize=7, color="0.35",
            ha="right")
    ax.set_xticks(x, [str(d) for d in DELAYS])
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Steps for the new XML to reach the\nbaseline's 600 M reward (millions)")
    ax.set_title("The cost of the new collision model,\nin training steps")
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax)

    # (c) slowdown factor at a target both arms reach
    ax = axes[2]
    for network in TIERS:
        sub = table[table["network"] == network].sort_values("delay_k")
        ax.plot(sub["delay_k"], sub["slowdown_factor"], color=color_for(NETWORKS[network]),
                marker=marker_for(NETWORKS[network]), label=NETWORK_LABEL[network])
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Steps to 90 % of the baseline's 600 M reward,\nnew / old")
    ax.set_title("How much slower is the climb?")
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "convergence", CURVES)
    fig.savefig(FIGURES / "convergence.png")
    print("Saved", FIGURES / "convergence.png")


# --------------------------------------------------------------------------------------
# Figure 4: the noise floor -- same configuration, two runs, compared at 600 M
# --------------------------------------------------------------------------------------


def fig_replication(curves: pd.DataFrame, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    for network in TIERS:
        sub = table[table["network"] == network]
        ax.scatter(sub["new_600M"], sub["new_2G_at_600M"],
                   color=color_for(NETWORKS[network]), marker=marker_for(NETWORKS[network]),
                   label=NETWORK_LABEL[network], zorder=3)
    values = table[["new_600M", "new_2G_at_600M"]].to_numpy()
    # Zoomed to the data, not to the origin: at this scale a 3 % miss is still only a
    # few pixels off the diagonal, which is the point.
    lims = [0.9 * values.min(), 1.05 * values.max()]
    ax.plot(lims, lims, color="0.6", lw=0.8, ls=":")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Reward at 600 M, the 600 M run")
    ax.set_ylabel("Reward at 600 M, the 2 G run")
    ax.set_title("Same configuration, two runs")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    sns.despine(ax=ax)

    ax = axes[1]
    order = table.sort_values(["network", "delay_k"])
    labels = [f"{n} d{d}" for n, d in zip(order["network"], order["delay_k"])]
    colors = [color_for(NETWORKS[n]) for n in order["network"]]
    ax.barh(range(len(order)), order["replicate_pct"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(order)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Difference between the two runs at 600 M (%)")
    ax.set_title(f"Noise floor: worst |Δ| = "
                 f"{order['replicate_pct'].abs().max():.1f} %")
    sns.despine(ax=ax)

    fig.suptitle("A 2 G run and a 600 M run of the same configuration are the same "
                 "process up to 600 M, so this is run-to-run noise", fontsize=9.5)
    fig.tight_layout()
    stamp(fig, "replication", CURVES)
    fig.savefig(FIGURES / "replication.png")
    print("Saved", FIGURES / "replication.png")


# --------------------------------------------------------------------------------------
# Figure 5: does the extra budget also buy held-out performance?
# --------------------------------------------------------------------------------------

#: The held-out dataset. ``old_eval`` is the 169-clip held-out split of the same reference
#: data the runs trained on, so it is the one dataset comparable across every tier here.
#: (``new_eval`` is 32 fresh 30 s clips at one seed and moves ~10 %; ``train`` is the
#: training split, i.e. the same thing the reward curve measures.)
HELD_OUT = "old_eval"


def held_out_table(evals: pd.DataFrame, dataset: str = HELD_OUT,
                   metric: str = "episode_reward") -> pd.DataFrame:
    """The three-tier contrast again, measured on held-out clips instead of the curve.

    One batch pass per run against the newest checkpoint on disk, so unlike the training
    curve (which each run computes on its own training clips) every tier is the *same*
    measurement, and new-vs-old is finally sayable on data no run trained on.
    """
    sub = evals[evals["dataset"] == dataset]
    rows = []
    for network, (base, new600, new2g) in TIERS.items():
        # delay 10 has two pgfm_new_600m runs; average them, as the curves do.
        means = (sub[sub["condition"].isin((base, new600, new2g))]
                 .groupby(["condition", "delay_k"])[metric].mean())
        for delay in DELAYS:
            b = means.get((base, delay), np.nan)
            n6 = means.get((new600, delay), np.nan)
            n2 = means.get((new2g, delay), np.nan)
            rows.append({"network": network, "delay_k": delay,
                         f"{metric}_baseline_600M": b,
                         f"{metric}_new_600M": n6,
                         f"{metric}_new_2G": n2,
                         f"{metric}_deficit_600M_pct": 100 * (n6 / b - 1),
                         f"{metric}_deficit_2G_pct": 100 * (n2 / b - 1)})
    return pd.DataFrame(rows)


def fig_held_out(evals: pd.DataFrame, table: pd.DataFrame) -> None:
    """Raw held-out reward per tier (top), and the same as a deficit (bottom)."""
    held = held_out_table(evals)
    noise = table["replicate_pct"].abs().max()
    sub_all = evals[evals["dataset"] == HELD_OUT]

    # sharey per row: the two networks are only comparable on a common scale, and the
    # bottom row's whole point is where each curve sits relative to the noise band.
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), sharex=True, sharey="row")
    for col, network in enumerate(TIERS):
        # -- raw reward, three tiers ----------------------------------------------------
        ax = axes[0, col]
        for tier, condition in enumerate(TIERS[network]):
            means = (sub_all[sub_all["condition"] == condition]
                     .groupby("delay_k")["episode_reward"].mean().reindex(DELAYS))
            ax.plot(DELAYS, means.to_numpy(), label=TIER_LABEL[tier],
                    marker=marker_for(NETWORKS[network]), ms=4,
                    mfc="none" if tier == 1 else None, **tier_style(network, tier))
        ax.set_title(NETWORK_LABEL[network])
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)

        # -- the same, relative to the baseline, against the noise floor ----------------
        ax = axes[1, col]
        h = held[held["network"] == network].sort_values("delay_k")
        color = color_for(NETWORKS[network])
        ax.axhspan(-noise, noise, color="0.88", zorder=0, lw=0)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.plot(h["delay_k"], h["episode_reward_deficit_600M_pct"], color=color,
                marker=marker_for(NETWORKS[network]), ls="--", mfc="none",
                label="new @600 M vs baseline")
        ax.plot(h["delay_k"], h["episode_reward_deficit_2G_pct"], color=color,
                marker=marker_for(NETWORKS[network]), ls="-",
                label="new @2 G vs baseline")
        ax.set_xlabel("Observation delay (steps)")
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
        sns.despine(ax=ax)

    axes[0, 0].set_ylabel(f"Held-out reward\n({HELD_OUT}, 169 clips)")
    axes[1, 0].set_ylabel("Held-out reward vs the\nold-XML baseline @600 M (%)")
    axes[0, 0].legend(loc="lower left", fontsize=6.5, frameon=False)
    axes[1, 1].annotate(f"run-to-run noise (±{noise:.1f} %)", (DELAYS[-1], noise),
                        fontsize=7, color="0.4", va="bottom", ha="right")
    fig.suptitle("The primary result on clips no run trained on. One batch eval pass per "
                 "run,\nthe same measurement for every tier -- unlike the training curve.",
                 fontsize=10)
    fig.tight_layout()
    stamp(fig, "held_out", EVAL, CURVES)
    fig.savefig(FIGURES / "held_out.png")
    print("Saved", FIGURES / "held_out.png")


# --------------------------------------------------------------------------------------


def last_point_table(df: pd.DataFrame) -> pd.DataFrame:
    """The same 600 M deficit computed from the single final eval point.

    ``collision-model-xml`` used the run-summary reward, which is one eval point at one
    step; on GPU-nondeterministic physics that point moves ~1-2 %, and more at short
    lifespans. Reproduced here so the difference between the two figures is visibly a
    windowing choice and not a different cohort.
    """
    rows = []
    for network, (base, _, _) in TIERS.items():
        new600 = TIERS[network][1]
        for delay in DELAYS:
            a = df[(df["condition"] == base) & (df["delay_k"] == delay)]["summary_reward"]
            b = df[(df["condition"] == new600)
                   & (df["delay_k"] == delay)]["summary_reward"]
            if a.empty or b.empty:
                continue
            rows.append({"network": network, "delay_k": delay,
                         "baseline_last_point": a.max(), "new_last_point": b.max(),
                         "deficit_last_point_pct": 100 * (b.max() / a.max() - 1)})
    return pd.DataFrame(rows)


def print_inline_crosscheck(df: pd.DataFrame, evals: pd.DataFrame) -> None:
    """Batch eval vs the run's own inline end-of-training eval, where both exist.

    Two *different* measurements -- in-memory weights at ``total_steps`` vs the newest
    checkpoint on disk -- so they are never mixed in a figure. Their agreement is the
    check that the re-produced (post walker-XML-fix) batch evals are sane; before the fix
    this ratio ran 0.58-0.98 on the new-XML runs, scaling with delay.
    """
    batch = (evals[evals["dataset"] == HELD_OUT]
             .set_index("wandb_id")["episode_reward"])
    both = df.dropna(subset=["inline_old_eval_reward"]).copy()
    both = both[both["wandb_id"].isin(batch.index)]
    if both.empty:
        return
    both["batch"] = both["wandb_id"].map(batch)
    both["ratio"] = both["batch"] / both["inline_old_eval_reward"]
    print("\n=== Batch eval vs inline end-of-training eval (old_eval; different weights, "
          "so expect ~1 with no delay trend) ===")
    print(both[["condition", "delay_k", "inline_old_eval_reward", "batch", "ratio"]]
          .sort_values(["condition", "delay_k"]).round(3).to_string(index=False))
    print(f"  ratio: {both['ratio'].min():.3f} - {both['ratio'].max():.3f} "
          f"(n={len(both)}); worst |1 - ratio| = "
          f"{(both['ratio'] - 1).abs().max() * 100:.1f} %")


def print_tables(table: pd.DataFrame, df: pd.DataFrame, evals: pd.DataFrame) -> None:
    show = ["network", "delay_k", "baseline_600M", "new_600M", "new_2G_final",
            "deficit_600M_pct", "deficit_2G_pct", "replicate_pct",
            "new_2G_gain_last_500M_pct", "baseline_gain_last_100M_pct"]
    print("\n=== Reward, deficit and remaining slope ===")
    print(table[show].round(2).to_string(index=False))

    print("\n=== Convergence speed ===")
    speed = table[["network", "delay_k", "steps_to_baseline_600M",
                   "base_steps_to_90pct", "new_steps_to_90pct", "slowdown_factor"]].copy()
    for column in ("steps_to_baseline_600M", "base_steps_to_90pct",
                   "new_steps_to_90pct"):
        speed[column] = speed[column] / 1e6
    print(speed.round(2).to_string(index=False))

    print("\n=== The 600 M deficit: 50 M-step window vs the single final eval point ===")
    merged = table[["network", "delay_k", "deficit_600M_pct"]].merge(
        last_point_table(df), on=["network", "delay_k"])
    print(merged.round(2).to_string(index=False))

    print("\n=== Held-out clips (old_eval, 169 clips), reward and survival ===")
    held = held_out_table(evals).merge(
        held_out_table(evals, metric="survived"), on=["network", "delay_k"])
    print(held.round(3).to_string(index=False))

    print("\n=== The curve and the held-out eval, side by side (deficit %) ===")
    both = table[["network", "delay_k", "deficit_600M_pct", "deficit_2G_pct"]].merge(
        held_out_table(evals)[["network", "delay_k",
                               "episode_reward_deficit_600M_pct",
                               "episode_reward_deficit_2G_pct"]],
        on=["network", "delay_k"])
    print(both.round(1).to_string(index=False))

    print("\n=== Inline end-of-training eval (only runs from 2026-08-10 onward have it; "
          "both baselines predate it) ===")
    inline = df.dropna(subset=["inline_old_eval_reward"])
    if not inline.empty:
        print(inline[["condition", "delay_k", "inline_train_reward",
                      "inline_old_eval_reward", "inline_old_eval_survived",
                      "inline_new_eval_reward", "inline_new_eval_survived"]]
              .round(3).to_string(index=False))


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    evals = pd.read_csv(EVAL)

    table = contrast_table(curves)

    fig_budget(curves, table)
    fig_closure(curves, table)
    fig_convergence(curves, table)
    fig_replication(curves, table)
    fig_held_out(evals, table)
    print_tables(table, df, evals)
    print_inline_crosscheck(df, evals)

    # The curve-based contrast and its held-out counterpart, in one committed table.
    table = table.merge(held_out_table(evals), on=["network", "delay_k"]).merge(
        held_out_table(evals, metric="survived"), on=["network", "delay_k"])
    table.to_csv(HERE / "contrast_table.csv", index=False)
    write_figure_manifest(HERE, _MANIFEST)


if __name__ == "__main__":
    main()
