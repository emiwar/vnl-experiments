"""Figures for "explicit vs implicit forward model at 2 G steps".

Run from the repo root::

    ../.venv/bin/python analysis/explicit-vs-implicit-fm-2g/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/explicit-vs-implicit-fm-2g/plot.py

Reads ONLY the committed CSVs (never WandB, never the artifact store). See
analysis/README.md §3.

Encoding: **colour = network** (the canonical `CONDITION_STYLE` colours — green for the
explicit forward model, purple for the policy-gradient one), **line weight = delay**
(darker/heavier is a longer delay). The ±2.9 % run-to-run noise floor measured in
`xml-ceiling-vs-convergence/` is drawn as a grey band wherever a difference is plotted.
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
GENERALIZATION = HERE / "data_generalization.csv"

_MANIFEST: dict[str, str] = {}


def stamp(fig, name: str, *inputs) -> None:
    _MANIFEST[f"{name}.png"] = provenance(fig, HERE, *(inputs or (DATA,)))


DELAYS = (0, 10, 20, 50)
ARMS = {"expfm_2g": "explicit", "pgfm_2g": "implicit"}
NETWORK_OF = {"expfm_2g": "forward_model", "pgfm_2g": "pg_forward_model"}
ARM_LABEL = {"expfm_2g": "Explicit FM (L2 loss, detached)",
             "pgfm_2g": "Implicit FM (policy gradient only)"}

#: Run-to-run noise, measured in `xml-ceiling-vs-convergence/` from eight pairs of runs
#: that share a configuration. A difference inside this is not a result.
NOISE_FLOOR_PCT = 2.9

#: Budgets the curves are read at. Doublings from 125 M, plus the standard 600 M budget.
MILESTONES = (125_000_000, 250_000_000, 500_000_000, 1_000_000_000, 2_000_000_000)
DOUBLINGS = MILESTONES
WINDOW_POINTS = 5  # eval every 10 M steps -> a 50 M-step trailing window


def delay_alpha(delay: int) -> float:
    return 0.35 + 0.65 * DELAYS.index(delay) / (len(DELAYS) - 1)


def budget_axis(ax, steps=MILESTONES) -> None:
    """Log x-axis labelled at the sampled budgets only.

    ``minorticks_off`` matters: matplotlib's log locator otherwise writes 3x10^2,
    4x10^2 ... between the labels we set, which collides with them.
    """
    ax.set_xscale("log")
    ax.set_xticks([s / 1e6 for s in steps],
                  [f"{s // 1_000_000} M" if s < 1_000_000_000
                   else f"{s / 1e9:g} G" for s in steps])
    ax.minorticks_off()


#: Below this the predictor has essentially nothing to predict (at delay 0 the target is
#: the current proprioception, which is also the input), the MSE sits at ~1e-3 and its
#: trace is pure noise. Normalised-progress and saturation numbers are meaningless there.
PREDICTION_FLOOR = 0.005


# --------------------------------------------------------------------------------------
# curve helpers
# --------------------------------------------------------------------------------------


def curve(curves: pd.DataFrame, condition: str, delay: int) -> pd.DataFrame:
    sub = curves[(curves["condition"] == condition)
                 & (curves["delay_k"] == delay)].sort_values("step").copy()
    for column in ("reward_mean", "fm_mse_eval", "fm_mse_train_p50", "action_sigma",
                   "encoder_kl", "joint_l2_error", "lifespan_mean"):
        sub[column] = sub[column].rolling(WINDOW_POINTS, min_periods=1).mean()
    return sub


def value_at(curves: pd.DataFrame, condition: str, delay: int, step: int,
             column: str = "reward_mean") -> float:
    c = curve(curves, condition, delay)
    c = c[c["step"] <= step]
    return np.nan if c.empty else float(c[column].iloc[-1])


def advantage_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Explicit-minus-implicit reward, in per cent, at each budget and delay."""
    rows = []
    for step in MILESTONES:
        for delay in DELAYS:
            e = value_at(curves, "expfm_2g", delay, step)
            i = value_at(curves, "pgfm_2g", delay, step)
            rows.append({"step": step, "delay_k": delay, "explicit": e, "implicit": i,
                         "advantage_pct": 100 * (e / i - 1)})
    return pd.DataFrame(rows)


def crossover_delay(advantage: pd.DataFrame, step: int,
                    threshold: float = NOISE_FLOOR_PCT) -> float:
    """The delay at which the explicit arm's advantage first exceeds ``threshold``.

    Linear interpolation between the four sampled delays, walking up from delay 0. With
    only four delays this locates the crossing inside a 10- or 30-step bracket, no
    better; it is plotted with that bracket shown. ``NaN`` if the advantage is already
    above the threshold at delay 0, or never reaches it.
    """
    sub = advantage[advantage["step"] == step].sort_values("delay_k")
    delays = sub["delay_k"].to_numpy(dtype=float)
    values = sub["advantage_pct"].to_numpy()
    if values[0] >= threshold:
        return np.nan
    for i in range(len(values) - 1):
        if values[i] < threshold <= values[i + 1]:
            span = values[i + 1] - values[i]
            frac = 0.0 if span == 0 else (threshold - values[i]) / span
            return float(delays[i] + frac * (delays[i + 1] - delays[i]))
    return np.nan


def doubling_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Reward gain from each doubling of the training budget."""
    rows = []
    for condition in ARMS:
        for delay in DELAYS:
            values = [value_at(curves, condition, delay, s) for s in DOUBLINGS]
            for i in range(len(values) - 1):
                rows.append({
                    "condition": condition, "arm": ARMS[condition], "delay_k": delay,
                    "from_step": DOUBLINGS[i], "to_step": DOUBLINGS[i + 1],
                    "gain_pct": 100 * (values[i + 1] / values[i] - 1)})
    return pd.DataFrame(rows)


def geometric_extrapolation(values: list[float], steps: list[int],
                            target: float) -> tuple[float, float]:
    """Where a geometrically decaying quantity would fall below ``target``.

    ``values`` are measured at successive *doublings* of the budget. Their ratio is
    roughly constant here (0.4-0.7 per doubling), so fitting a line to ``log(value)``
    against doubling number and solving for ``target`` gives the number of further
    doublings. Returns ``(decay_per_doubling, budget)``.

    This is an extrapolation of three or four points, and the report treats it as an
    order of magnitude, not a number. It is meaningless for a non-positive series (the
    implicit arm at delay 50 loses reward per doubling), which returns NaN.
    """
    positive = [(i, v) for i, v in enumerate(values) if v > 0]
    if len(positive) < 3 or values[-1] <= 0:
        return float("nan"), float("nan")
    x = np.array([i for i, _ in positive], dtype=float)
    y = np.log(np.array([v for _, v in positive]))
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return float(np.exp(slope)), float("inf")
    n = (np.log(target) - intercept) / slope
    return float(np.exp(slope)), float(steps[0] * 2 ** n)


def extrapolation_table(doublings: pd.DataFrame,
                        advantage: pd.DataFrame) -> pd.DataFrame:
    """When would (a) each run stop gaining, (b) each delay's contrast stop moving?"""
    rows = []
    for condition in ARMS:
        for delay in DELAYS:
            sub = (doublings[(doublings["condition"] == condition)
                             & (doublings["delay_k"] == delay)].sort_values("to_step"))
            decay, budget = geometric_extrapolation(
                sub["gain_pct"].tolist(), sub["to_step"].tolist(), NOISE_FLOOR_PCT)
            rows.append({"quantity": "reward gain per doubling", "arm": ARMS[condition],
                         "delay_k": delay, "value_at_2G": sub["gain_pct"].iloc[-1],
                         "decay_per_doubling": decay, "budget_below_noise": budget})
    for delay in DELAYS:
        sub = advantage[(advantage["delay_k"] == delay)
                        & (advantage["step"] >= 500_000_000)].sort_values("step")
        decay, budget = geometric_extrapolation(
            sub["advantage_pct"].tolist(), sub["step"].tolist(), NOISE_FLOOR_PCT)
        rows.append({"quantity": "explicit advantage", "arm": "contrast",
                     "delay_k": delay, "value_at_2G": sub["advantage_pct"].iloc[-1],
                     "decay_per_doubling": decay, "budget_below_noise": budget})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figure 1: the raw curves -- reward over training, one panel per delay
# --------------------------------------------------------------------------------------


def fig_reward_curves(curves: pd.DataFrame) -> None:
    """The plain picture everything else is a reduction of.

    Shared y-axis across the four panels: the delay effect is a large part of what there
    is to see, and per-panel autoscaling would hide it.
    """
    fig, axes = plt.subplots(1, len(DELAYS), figsize=(3.3 * len(DELAYS), 3.9),
                             sharex=True, sharey=True)
    for ax, delay in zip(axes, DELAYS):
        for condition in ARMS:
            c = curve(curves, condition, delay)
            ax.plot(c["step"] / 1e6, c["reward_mean"], lw=1.6,
                    color=color_for(NETWORK_OF[condition]),
                    label=ARM_LABEL[condition] if delay == DELAYS[0] else None)
        ax.axvline(600, color="0.8", lw=0.8, zorder=0)
        explicit = value_at(curves, "expfm_2g", delay, 2_000_000_000)
        implicit = value_at(curves, "pgfm_2g", delay, 2_000_000_000)
        ax.set_title(f"delay {delay}\nexplicit {explicit:.0f} vs implicit {implicit:.0f} "
                     f"({100 * (explicit / implicit - 1):+.1f} %)", fontsize=9)
        ax.set_xlabel("Env steps (millions)")
        sns.despine(ax=ax)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Mean episode reward (eval on train clips)")
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    # In the last panel, where the low half of the axes is empty.
    axes[-1].text(630, 120, "standard\nbudget", fontsize=6.5, color="0.5")
    fig.suptitle("Reward over training at each delay, 2 G-step runs", fontsize=10)
    fig.tight_layout()
    stamp(fig, "reward_curves", CURVES)
    fig.savefig(FIGURES / "reward_curves.png")
    print("Saved", FIGURES / "reward_curves.png")


# --------------------------------------------------------------------------------------
# Figure 2: the advantage, and where the crossover sits, as budget grows
# --------------------------------------------------------------------------------------


def fig_advantage(advantage: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    ax = axes[0]
    ax.axhspan(-NOISE_FLOOR_PCT, NOISE_FLOOR_PCT, color="0.88", zorder=0, lw=0)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    for delay in DELAYS:
        sub = advantage[advantage["delay_k"] == delay].sort_values("step")
        color = plt.get_cmap("viridis")(DELAYS.index(delay) / (len(DELAYS) - 1))
        ax.plot(sub["step"] / 1e6, sub["advantage_pct"], marker="o", ms=4, color=color)
        # Labelled at the line end rather than in a legend: four lines spanning
        # -8 % to +91 % leave no empty corner for a legend box.
        # delay 0 and delay 10 both end within a per cent of zero; nudge them apart.
        ax.annotate(f"delay {delay}", (2080, sub["advantage_pct"].iloc[-1]),
                    textcoords="offset points", xytext=(0, {0: -7, 10: 7}.get(delay, 0)),
                    fontsize=7.5, va="center", color=color)
    budget_axis(ax)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([-10, 0, 10, 25, 50, 100], ["−10", "0", "10", "25", "50", "100"])
    ax.set_xlabel("Training budget")
    ax.set_ylabel("Explicit FM advantage over implicit (%)")
    ax.set_title("The advantage shrinks with budget — except at delay 50")
    ax.set_xlim(110, 3600)
    ax.annotate(f"±{NOISE_FLOOR_PCT} % noise floor", (135, NOISE_FLOOR_PCT), fontsize=7,
                color="0.4", va="bottom")
    sns.despine(ax=ax)

    ax = axes[1]
    # Below 500 M both arms are still in the initial transient and "the crossover
    # delay" is not a meaningful quantity -- at 125 M the explicit arm is behind at
    # delay 0 and 24 % ahead at delay 10.
    steps = [s for s in MILESTONES if s >= 500_000_000]
    crossings = [crossover_delay(advantage, s) for s in steps]
    ax.plot([s / 1e6 for s in steps], crossings, marker="o", color="C3")
    for s, c in zip(steps, crossings):
        if not np.isnan(c):
            lower = max(d for d in DELAYS if d <= c)
            upper = min(d for d in DELAYS if d >= c)
            ax.plot([s / 1e6, s / 1e6], [lower, upper], color="C3", lw=6, alpha=0.18,
                    solid_capstyle="butt")
    budget_axis(ax, steps)
    ax.set_ylim(0, 55)
    ax.set_xlabel("Training budget")
    ax.set_ylabel("Delay at which the explicit FM\npulls ahead by more than the noise floor")
    ax.set_title("The crossover delay moves later as the budget grows\n"
                 "(shaded = the bracket between sampled delays)", fontsize=9.5)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "advantage", CURVES)
    fig.savefig(FIGURES / "advantage.png")
    print("Saved", FIGURES / "advantage.png")


# --------------------------------------------------------------------------------------
# Figure 2: reward scaling, and what another doubling would buy
# --------------------------------------------------------------------------------------


def fig_scaling(curves: pd.DataFrame, doublings: pd.DataFrame,
                extrapolation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    ax = axes[0]
    for condition in ARMS:
        for delay in DELAYS:
            c = curve(curves, condition, delay)
            ax.plot(c["step"] / 1e6, c["reward_mean"], lw=1.4,
                    color=color_for(NETWORK_OF[condition]), alpha=delay_alpha(delay),
                    label=ARM_LABEL[condition] if delay == DELAYS[-1] else None)
            ax.annotate(f"d{delay}", (2050, c["reward_mean"].iloc[-1]),
                        textcoords="offset points",
                        xytext=(0, 8 if condition == "expfm_2g" else -8),
                        fontsize=6.5, va="center",
                        color=color_for(NETWORK_OF[condition]))
    ax.axvline(600, color="0.75", lw=0.8)
    ax.text(610, 300, "standard budget", fontsize=7, color="0.5", rotation=90)
    budget_axis(ax)
    ax.set_xlim(100, 2400)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Mean episode reward (eval on train clips)")
    ax.set_title("Reward vs budget, log axis")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    sns.despine(ax=ax)

    ax = axes[1]
    for condition in ARMS:
        for delay in DELAYS:
            sub = doublings[(doublings["condition"] == condition)
                            & (doublings["delay_k"] == delay)].sort_values("to_step")
            ax.plot(sub["to_step"] / 1e6, sub["gain_pct"], marker="o", ms=3.5, lw=1.3,
                    color=color_for(NETWORK_OF[condition]), alpha=delay_alpha(delay),
                    label=f"{ARMS[condition]} d{delay}")
    # Dashed: the geometric decay of the per-doubling gain, continued to where it
    # enters the noise band. That intersection is the "how long should I train?" answer,
    # and it is an extrapolation of three or four points -- read the order, not the digit.
    rows = extrapolation[extrapolation["quantity"] == "reward gain per doubling"]
    for _, row in rows.iterrows():
        if not np.isfinite(row["budget_below_noise"]):
            continue
        condition = "expfm_2g" if row["arm"] == "explicit" else "pgfm_2g"
        last = doublings[(doublings["condition"] == condition)
                         & (doublings["delay_k"] == row["delay_k"])]["gain_pct"].iloc[-1]
        end = row["budget_below_noise"]
        ax.plot([2000, end / 1e6], [last, NOISE_FLOOR_PCT], ls=":", lw=1.0,
                color=color_for(NETWORK_OF[condition]), alpha=delay_alpha(row["delay_k"]))
        if end > 4e9:
            ax.annotate(f"{row['arm'][:3]} d{row['delay_k']}: ~{end / 1e9:.0f} G",
                        (end / 1e6, NOISE_FLOOR_PCT), fontsize=6.5, va="bottom",
                        ha="right", color=color_for(NETWORK_OF[condition]))
    ax.axhspan(-NOISE_FLOOR_PCT, NOISE_FLOOR_PCT, color="0.88", zorder=0, lw=0)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    budget_axis(ax, DOUBLINGS[1:] + (4_000_000_000, 8_000_000_000, 16_000_000_000,
                                     32_000_000_000))
    ax.set_xlabel("Budget after the doubling  (dotted = extrapolated)")
    ax.set_ylabel("Reward gained by doubling the budget (%)")
    ax.set_title("Marginal return per doubling\n(inside the grey band = converged)",
                 fontsize=9.5)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "scaling", CURVES)
    fig.savefig(FIGURES / "scaling.png")
    print("Saved", FIGURES / "scaling.png")


# --------------------------------------------------------------------------------------
# Figure 3: the forward-model prediction itself
# --------------------------------------------------------------------------------------


def fig_fm_prediction(curves: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    ax = axes[0]
    for condition in ARMS:
        for delay in DELAYS:
            c = curve(curves, condition, delay)
            ax.plot(c["step"] / 1e6, c["fm_mse_eval"], lw=1.4,
                    color=color_for(NETWORK_OF[condition]), alpha=delay_alpha(delay),
                    label=ARM_LABEL[condition] if delay == DELAYS[-1] else None)
            ax.annotate(f"d{delay}", (2050, c["fm_mse_eval"].iloc[-1]),
                        textcoords="offset points",
                        xytext=(0, 4 if condition == "expfm_2g" else -6),
                        fontsize=6.5, va="center",
                        color=color_for(NETWORK_OF[condition]))
    budget_axis(ax)
    ax.set_yscale("log")
    ax.set_xlim(100, 2400)
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Forward-prediction MSE (eval)")
    ax.set_title("The implicit arm never predicts —\nand at delay 50 it gets worse",
                 fontsize=9.5)
    ax.legend(fontsize=7, loc="center left", frameon=False)
    sns.despine(ax=ax)

    # Progress fraction: reward vs prediction error on one normalised axis, explicit arm.
    ax = axes[1]
    for delay in DELAYS:
        c = curve(curves, "expfm_2g", delay)
        c = c[c["step"] >= 100_000_000]
        for column, style in (("reward_mean", "-"), ("fm_mse_eval", "--")):
            values = c[column].to_numpy()
            total = values[-1] - values[0]
            if column == "fm_mse_eval" and values[0] < PREDICTION_FLOOR:
                continue  # delay 0 has no prediction problem; its MSE trace is noise
            ax.plot(c["step"] / 1e6, (values - values[0]) / total, style, lw=1.3,
                    color=plt.get_cmap("viridis")(DELAYS.index(delay) / (len(DELAYS) - 1)),
                    label=f"delay {delay}" if column == "reward_mean" else None)
    ax.axhline(0.95, color="0.5", lw=0.8, ls=":")
    ax.text(120, 0.96, "95 % of the total change", fontsize=7, color="0.45")
    budget_axis(ax)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Fraction of the run's total change completed")
    ax.set_title("Explicit FM: reward (solid) vs\nprediction error (dashed)", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    sns.despine(ax=ax)

    ax = axes[2]
    sub = df[df["condition"] == "expfm_2g"].sort_values("delay_k").copy()
    # Blank the delay-0 prediction bar: its MSE is at the ~1e-3 noise floor, so "95 % of
    # its total change" is a number about noise.
    sub.loc[sub["fm_mse_at_600M"] < PREDICTION_FLOOR, "fm_mse_steps_to_95pct"] = np.nan
    width = 0.36
    x = np.arange(len(sub))
    ax.bar(x - width / 2, sub["reward_steps_to_95pct"] / 1e6, width,
           color=color_for("forward_model"), label="reward")
    ax.bar(x + width / 2, sub["fm_mse_steps_to_95pct"] / 1e6, width,
           color=color_for("forward_model"), alpha=0.45, hatch="//",
           label="prediction error")
    if sub["fm_mse_steps_to_95pct"].isna().any():
        blank = int(np.flatnonzero(sub["fm_mse_steps_to_95pct"].isna())[0])
        ax.text(blank + width / 2, 40, "nothing to predict", rotation=90, fontsize=7,
                ha="center", va="bottom", color="0.4")
    ax.axhline(600, color="0.35", lw=1.0, ls="--")
    ax.text(len(sub) - 0.5, 620, "standard budget", fontsize=7, color="0.35", ha="right")
    ax.set_xticks(x, [f"d{d}" for d in sub["delay_k"]])
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Steps to 95 % of the total change (millions)")
    ax.set_title("Prediction saturates later than reward\n(delay 0: nothing to predict)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "fm_prediction", DATA, CURVES)
    fig.savefig(FIGURES / "fm_prediction.png")
    print("Saved", FIGURES / "fm_prediction.png")


# --------------------------------------------------------------------------------------
# Figure 4: overfitting, and what the delay-50 implicit run is actually doing
# --------------------------------------------------------------------------------------


def fig_generalization(gen: pd.DataFrame, curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    ax = axes[0]
    for condition, budget, style in (("expfm_2g", 2_000_000_000, dict(ls="-")),
                                     ("pgfm_2g", 2_000_000_000, dict(ls="-")),
                                     ("pgfm_600m_reference", 600_000_000,
                                      dict(ls="--", mfc="none"))):
        sub = (gen[gen["condition"] == condition]
               .groupby("delay_k", as_index=False)["generalization_ratio"].mean())
        color = color_for(NETWORK_OF.get(condition, "pg_forward_model"))
        label = ("Explicit, 2 G" if condition == "expfm_2g" else
                 "Implicit, 2 G" if condition == "pgfm_2g" else "Implicit, 600 M")
        ax.plot(sub["delay_k"], sub["generalization_ratio"], marker="o", ms=4,
                color=color, label=label, **style)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Held-out reward / training-clip reward")
    ax.set_title("Generalisation gap at the end of training\n"
                 "(inline eval; below 1 = held-out is worse)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    sns.despine(ax=ax)

    ax = axes[1]
    pairs = gen[gen["arm"] == "implicit"].groupby(["budget", "delay_k"],
                                                  as_index=False).mean(numeric_only=True)
    width = 0.26
    x = np.arange(len(DELAYS))
    for i, (column, label) in enumerate([("inline_train_reward", "train (80 % split)"),
                                         ("inline_old_eval_reward", "old_eval (held-out)"),
                                         ("inline_new_eval_reward", "new_eval (30 s)")]):
        pct = []
        for delay in DELAYS:
            a = pairs[(pairs["budget"] == 600_000_000)
                      & (pairs["delay_k"] == delay)][column]
            b = pairs[(pairs["budget"] == 2_000_000_000)
                      & (pairs["delay_k"] == delay)][column]
            pct.append(np.nan if a.empty or b.empty
                       else 100 * (b.iat[0] / a.iat[0] - 1))
        ax.bar(x + (i - 1) * width, pct, width, label=label, alpha=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"d{d}" for d in DELAYS])
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Reward change, 2 G vs 600 M (%)")
    ax.set_title("Implicit arm: the extra budget moves train and\nheld-out together, "
                 "not apart", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax)

    # What the delay-50 implicit run does over training: not overfitting, degradation.
    ax = axes[2]
    diagnostics = [("fm_mse_eval", "prediction MSE"), ("action_sigma", "action σ"),
                   ("joint_l2_error", "joint tracking error"), ("encoder_kl", "encoder KL")]
    for i, (column, label) in enumerate(diagnostics):
        c = curve(curves, "pgfm_2g", 50)
        base = c[c["step"] <= 300_000_000][column].mean()
        ax.plot(c["step"] / 1e6, c[column] / base, lw=1.4, color=f"C{i}", label=label)
    reward = curve(curves, "pgfm_2g", 50)
    ax.plot(reward["step"] / 1e6, reward["reward_mean"]
            / reward[reward["step"] <= 300_000_000]["reward_mean"].mean(),
            lw=2.0, color="k", label="reward")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Env steps (millions)")
    ax.set_ylabel("Relative to the mean over the first 300 M steps")
    ax.set_title("Implicit FM, delay 50: the predictor degrades\nwhile the policy loses "
                 "reward", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "generalization", GENERALIZATION, CURVES)
    fig.savefig(FIGURES / "generalization.png")
    print("Saved", FIGURES / "generalization.png")


# --------------------------------------------------------------------------------------


def print_tables(advantage: pd.DataFrame, doublings: pd.DataFrame,
                 df: pd.DataFrame, gen: pd.DataFrame) -> None:
    print("\n=== Explicit FM advantage over implicit (%), by budget ===")
    print(advantage.pivot_table(index="delay_k", columns="step",
                                values="advantage_pct").round(1).to_string())

    print("\n=== Crossover delay (advantage exceeds the ±2.9 % noise floor) ===")
    for step in MILESTONES:
        value = crossover_delay(advantage, step)
        print(f"  {step // 1_000_000:>5d} M: "
              + ("already ahead at delay 0" if np.isnan(value) else f"delay ≈ {value:.0f}"))

    print("\n=== Reward gained per doubling of the budget (%) ===")
    print(doublings.pivot_table(index=["arm", "delay_k"], columns="to_step",
                                values="gain_pct").round(1).to_string())

    print("\n=== Steps to 95 % of the run's total change (millions) ===")
    show = df[["arm", "delay_k", "reward_steps_to_95pct", "fm_mse_steps_to_95pct",
               "fm_mse_at_600M", "fm_mse_at_2000M", "peak_step"]].copy()
    for column in ("reward_steps_to_95pct", "fm_mse_steps_to_95pct", "peak_step"):
        show[column] = show[column] / 1e6
    print(show.round(3).to_string(index=False))

    print("\n=== Extrapolation: budget at which the quantity drops below the "
          "±2.9 % noise floor ===")
    show = extrapolation_table(doublings, advantage).copy()
    show["budget_below_noise"] = show["budget_below_noise"] / 1e9
    print(show.round(2).to_string(index=False))

    print("\n=== Generalisation (inline end-of-training eval) ===")
    print(gen[["arm", "budget", "delay_k", "inline_train_reward",
               "inline_old_eval_reward", "generalization_ratio"]]
          .round(3).to_string(index=False))


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    gen = pd.read_csv(GENERALIZATION)

    advantage = advantage_table(curves)
    doublings = doubling_table(curves)

    extrapolation = extrapolation_table(doublings, advantage)

    fig_reward_curves(curves)
    fig_advantage(advantage)
    fig_scaling(curves, doublings, extrapolation)
    fig_fm_prediction(curves, df)
    fig_generalization(gen, curves)
    print_tables(advantage, doublings, df, gen)

    advantage.to_csv(HERE / "advantage_table.csv", index=False)
    doublings.to_csv(HERE / "doubling_table.csv", index=False)
    extrapolation.to_csv(HERE / "extrapolation_table.csv", index=False)
    write_figure_manifest(HERE, _MANIFEST)


if __name__ == "__main__":
    main()
