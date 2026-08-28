"""Talk figures for the August 2026 lab meeting: five claims about proprioceptive delay.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

These are **presentation** figures, so the provenance footer is off by default (the string
is still written to ``figures/manifest.json``, which is what lets a slide be traced back to
committed data). Set ``VNL_NO_FOOTER=0`` to stamp them.

Two conventions used throughout, both requested for the talk:

* **Log delay axis with a special-cased 0.** Delay 0 is drawn at a fixed position left of
  delay 1, separated by a dotted rule, because ``log(0)`` has nowhere to go and the point
  matters -- it is the no-delay control. The gap is a visual break, not a scale.
* **Dashed, semi-transparent = the reference condition** in the same colour as its solid
  partner: the shorter budget, or the noise-free baseline. Solid is always the condition
  the slide is about.

    ../.venv/bin/python analysis/aug-2026-labmeeting-summary/plot.py
    ../.venv/bin/python analysis/aug-2026-labmeeting-summary/plot.py --only c3 c5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

# Presentation default. Set VNL_NO_FOOTER=0 in the environment to get the stamp back.
os.environ.setdefault("VNL_NO_FOOTER", "1")

from vnl_experiments.wandb_utils.style import (  # noqa: E402
    apply_style, color_for, provenance, write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

DELAY = HERE / "data_delay.csv"
BUDGET = HERE / "data_budget.csv"
CURVES = HERE / "curves.csv"
NOISE = HERE / "data_noise.csv"
PROBE = HERE / "data_probe.csv"
GROUPS = HERE / "data_probe_groups.csv"

PRIMARY_DATASET = "old_eval"

#: Condition -> the shared style key, so a colour means the same thing in every folder.
STYLE_KEY = {
    "efference_old": "efference", "no_efference_old": "no_efference",
    "ablate_intention": "ablate_intention",
    "ablate_proprioception": "ablate_proprioception",
    "ablate_efference": "ablate_efference",
    "encdec": "encdec", "expfm": "forward_model", "pgfm": "pg_forward_model",
    "expfm_2g": "forward_model", "pgfm_2g": "pg_forward_model",
    "expfm_4g": "forward_model", "pgfm_4g": "pg_forward_model",
}
LABEL = {
    "ablate_intention": "No intention",
    "ablate_proprioception": "No proprioception",
    "ablate_efference": "No efference copy",
    "efference_old": "With efference copy",
    "no_efference_old": "No efference copy",
    "encdec": "Enc-dec (efference copy)",
    "expfm": "Explicit forward model",
    "pgfm": "Policy-gradient forward model",
}
MARKER = {"efference_old": "o", "no_efference_old": "s", "encdec": "o",
          "expfm": "^", "pgfm": "D"}
ARM_LABEL = {"explicit": "Explicit forward model",
             "implicit": "Policy-gradient forward model"}
ARM_COLOR = {"explicit": color_for("forward_model"),
             "implicit": color_for("pg_forward_model")}
ARM_MARKER = {"explicit": "^", "implicit": "D"}


# --------------------------------------------------------------------------------------
# The log delay axis
# --------------------------------------------------------------------------------------

#: Where delay 0 is drawn on the log axis, and where the break rule goes.
ZERO_X = 0.45
BREAK_X = 0.70
DELAY_TICKS = (0, 1, 2, 5, 10, 20, 50, 100)


def xpos(delays) -> np.ndarray:
    """Delay -> x position: identity, except 0, which is parked left of the break."""
    values = np.asarray(delays, dtype=float)
    return np.where(values == 0, ZERO_X, values)


def log_delay_axis(ax, *, ticks=DELAY_TICKS, xmax=115.0, ms_axis=True, xlabel=True):
    ax.set_xscale("log")
    ax.set_xlim(0.36, xmax)
    ax.xaxis.set_major_locator(FixedLocator(xpos(ticks)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    # The break: everything left of it is the single special-cased delay-0 point.
    ax.axvline(BREAK_X, color="0.75", lw=0.9, ls=(0, (2, 2)), zorder=0)
    if xlabel:
        ax.set_xlabel("Observation delay (control steps)")
    if ms_axis:
        top = ax.twiny()
        top.set_xscale("log")
        top.set_xlim(ax.get_xlim())
        top.xaxis.set_major_locator(FixedLocator(xpos(ticks)))
        top.xaxis.set_major_formatter(FixedFormatter([str(int(t * 10)) for t in ticks]))
        top.xaxis.set_minor_locator(NullLocator())
        top.set_xlabel("Observation delay (ms)", labelpad=6)
        for side in ("right", "left", "bottom"):
            top.spines[side].set_visible(False)
        return top
    return None


def by_delay(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean over repeats so each delay contributes one point, sorted for plotting."""
    out = (frame.dropna(subset=[column])
                .groupby("delay_k", as_index=False)
                .agg(value=(column, "mean"), n=(column, "size")))
    return out.sort_values("delay_k")


def line(ax, frame, column, *, condition=None, color=None, marker=None, label=None,
         dashed=False, alpha=1.0, **kw):
    points = by_delay(frame, column)
    if points.empty:
        return points
    style = dict(color=color or color_for(STYLE_KEY[condition]),
                 marker=marker or (MARKER.get(condition, "o")),
                 label=label if label is not None else LABEL.get(condition, condition),
                 alpha=alpha)
    if dashed:
        style.update(ls=(0, (5, 2)), lw=1.4, markersize=3, markerfacecolor="none")
    ax.plot(xpos(points["delay_k"]), points["value"], **style, **kw)
    return points


def held_out(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["dataset"] == PRIMARY_DATASET) & frame["have_eval"]]


def finish(fig, ax=None, *, legend_loc="best", legend_ncol=1):
    if ax is not None:
        ax.legend(loc=legend_loc, ncol=legend_ncol, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Claim 1 -- the efference copy is necessary
# --------------------------------------------------------------------------------------

def c1_reward(data) -> plt.Figure:
    """The contrast itself. Old walker XML + current_root -- the only cohort that has it."""
    delay = held_out(data["delay"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for condition in ("efference_old", "no_efference_old"):
        line(ax, delay[delay["condition"] == condition], "episode_reward",
             condition=condition)
    log_delay_axis(ax)
    ax.set_ylabel("Held-out episode reward")
    ax.set_ylim(0, None)
    ax.set_title("An efference copy is what makes delay survivable", pad=28)
    return finish(fig, ax, legend_loc="lower left")


def c1_tracking(data) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    encdec = held_out(data["delay"])
    encdec = encdec[encdec["condition"] == "encdec"]
    line(ax, encdec, "joint_l2_error", condition="encdec",
         label="Enc-dec with efference copy")
    log_delay_axis(ax)
    ax.set_ylabel("Joint tracking error (L2, per step)")
    ax.set_ylim(0, None)
    ax.set_title("Tracking error doubles across the delay range", pad=28)
    return finish(fig, ax, legend_loc="upper left")


def c1_lifetime(data) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    encdec = held_out(data["delay"])
    encdec = encdec[encdec["condition"] == "encdec"]
    line(ax, encdec, "lifespan_s", condition="encdec",
         label="Enc-dec with efference copy")
    clip_s = float(encdec["n_steps"].iloc[0]) / 100.0
    ax.axhline(clip_s, color="0.6", lw=1.0, ls=":")
    ax.text(1.0, clip_s, "clip length (measurement is censored here)", color="0.5",
            va="bottom", ha="left", fontsize=8)
    log_delay_axis(ax)
    ax.set_ylabel("Time before failure (s)")
    ax.set_ylim(0, clip_s * 1.08)
    ax.set_title("...and the animal falls sooner", pad=28)
    return finish(fig, ax, legend_loc="lower left")


# --------------------------------------------------------------------------------------
# Claim 2 -- a forward model improves learning
# --------------------------------------------------------------------------------------

#: The ablations exist at one delay only. Their control is the delay-10 member of the
#: `encdec` sweep -- the same run claims 1, 2 and 4 already use.
ABLATION_DELAY = 10
ABLATION_ORDER = ("ablate_intention", "ablate_proprioception", "ablate_efference")


def c1_decoder_inputs(data) -> plt.Figure:
    """All three decoder inputs, one at a time, against the intact baseline.

    Training curves rather than endpoints because the endpoints alone would not show that
    none of the three arms is merely *slower* -- all three flatten early and stay flat.
    """
    curves = data["curves"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    control = curves[(curves["condition"] == "encdec")
                     & (curves["delay_k"] == ABLATION_DELAY)]
    for wandb_id, curve in control.groupby("wandb_id"):
        curve = curve.sort_values("step")
        ax.plot(curve["step"] / 1e6, curve["reward_mean"],
                color=color_for("encdec"), lw=2.6,
                label="All three inputs (baseline)", zorder=3)
    for condition in ABLATION_ORDER:
        cell = curves[curves["condition"] == condition]
        for wandb_id, curve in cell.groupby("wandb_id"):
            curve = curve.sort_values("step")
            ax.plot(curve["step"] / 1e6, curve["reward_mean"],
                    color=color_for(STYLE_KEY[condition]), lw=1.7,
                    label=LABEL[condition])

    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Episode reward")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.2)
    ax.set_title(f"Drop any one decoder input and the policy loses half its reward\n"
                 f"delay {ABLATION_DELAY} steps ({ABLATION_DELAY * 10} ms), "
                 f"one run per condition", fontsize=10)
    return finish(fig, ax, legend_loc="upper left")


def c2_three_arms(data) -> plt.Figure:
    delay = held_out(data["delay"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for condition in ("encdec", "pgfm", "expfm"):
        line(ax, delay[delay["condition"] == condition], "episode_reward",
             condition=condition)
    log_delay_axis(ax)
    ax.set_ylabel("Held-out episode reward")
    ax.set_ylim(0, None)
    ax.set_title("The predictor buys nothing; the loss that trains it buys everything",
                 pad=28)
    return finish(fig, ax, legend_loc="lower left")


# --------------------------------------------------------------------------------------
# Claim 3 -- part of the gain is convergence speed
# --------------------------------------------------------------------------------------

C3_DELAYS = (0, 10, 20, 30, 40, 50)


def c3_curves(data) -> plt.Figure:
    """Reward against training step, one panel per delay, both forward-model arms.

    Every run at that delay is drawn; the longest is solid and the shorter ones are thin,
    which shows the budget cross-check (same configuration at a smaller budget lies on top
    of the longer run's curve) rather than asserting it.
    """
    curves = data["curves"]
    budget = data["budget"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), sharex=True, sharey=True)
    for ax, delay in zip(axes.ravel(), C3_DELAYS):
        cell = curves[curves["delay_k"] == delay]
        seeds, reached = set(), {}
        for arm in ("explicit", "implicit"):
            runs = cell[cell["arm"] == arm]
            if runs.empty:
                continue
            longest = (runs.groupby("wandb_id")["step"].max().idxmax())
            for wandb_id, curve in runs.groupby("wandb_id"):
                is_main = wandb_id == longest
                ax.plot(curve["step"] / 1e9, curve["reward_mean"],
                        color=ARM_COLOR[arm],
                        lw=2.0 if is_main else 0.9,
                        alpha=1.0 if is_main else 0.45,
                        label=ARM_LABEL[arm] if is_main else None)
                if is_main:
                    seeds.add(int(curve["seed"].iloc[0]))
                    reached[arm[:3]] = curve["step"].max()
        seed_text = "/".join(str(s) for s in sorted(seeds)) or "-"
        ends = ", ".join(f"{name} {value / 1e9:.1f} B"
                         for name, value in sorted(reached.items()))
        ax.set_title(f"delay {delay} steps ({delay * 10} ms)\n"
                     f"{ends} — seed {seed_text}", fontsize=10)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Training steps (billions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode reward")
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.suptitle("Up to delay ~30 the explicit model's lead is a head start; "
                 "from 40 the implicit one stops improving", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def c3_reward_at_4g(data) -> plt.Figure:
    """Delay vs reward at the largest budget, with the 600 M curves behind it."""
    budget = data["budget"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm in ("explicit", "implicit"):
        rows = budget[budget["arm"] == arm]
        # 600 M reference: the full delay sweep, seed 42.
        short = rows[rows["condition"].isin(("expfm", "pgfm"))]
        points = short.dropna(subset=["reward_at_600M"]).groupby(
            "delay_k", as_index=False)["reward_at_600M"].mean().sort_values("delay_k")
        ax.plot(xpos(points["delay_k"]), points["reward_at_600M"],
                color=ARM_COLOR[arm], ls=(0, (5, 2)), lw=1.4, alpha=0.5,
                marker=ARM_MARKER[arm], markersize=3, markerfacecolor="none",
                label=f"{ARM_LABEL[arm]}, 600 M")
        # 4 G, falling back to the largest tier a delay actually reached.
        full, partial = [], []
        for delay, group in rows.groupby("delay_k"):
            at_4g = group["reward_at_4G"].dropna()
            if len(at_4g):
                full.append((delay, at_4g.max()))
                continue
            at_3g = group["reward_at_2p9G"].dropna()
            if len(at_3g) and group["max_step"].max() > 2.5e9:
                partial.append((delay, at_3g.max(), group["max_step"].max()))
        combined = sorted(full + [(d, v) for d, v, _ in partial])
        if combined:
            ax.plot(xpos([d for d, _ in combined]), [v for _, v in combined],
                    color=ARM_COLOR[arm], marker=ARM_MARKER[arm], lw=2.0,
                    label=f"{ARM_LABEL[arm]}, 4 B")
        for delay, value, reached in partial:
            # An open marker means "this arm never reached 4 B at this delay"; the point
            # is its largest completed tier, so the line is not silently interpolated.
            ax.plot(xpos([delay]), [value], marker=ARM_MARKER[arm], ls="none",
                    markerfacecolor="white", markeredgecolor=ARM_COLOR[arm],
                    markersize=9, markeredgewidth=1.6, zorder=5)
            ax.annotate(f"{reached / 1e9:.1f} B only", (xpos([delay])[0], value),
                        textcoords="offset points",
                        xytext=(-10, 12 if arm == "explicit" else -20),
                        ha="right", fontsize=7.5, color=ARM_COLOR[arm])
    log_delay_axis(ax)
    ax.set_ylabel("Episode reward (training clips)")
    ax.set_ylim(0, None)
    ax.set_title("More compute closes most of the gap at delay 20-30 and widens it "
                 "at 40-50\n(dashed = the same runs' 600 M curves; open marker = that "
                 "arm stopped short of 4 B)", fontsize=10, pad=26)
    return finish(fig, ax, legend_loc="lower left")


# --------------------------------------------------------------------------------------
# Claim 4 -- forward models are sensitive to motor noise
# --------------------------------------------------------------------------------------

SIGMA_SHOWN = 0.02
C4_DELAYS = (0, 5, 10, 20)


def _noise(data) -> pd.DataFrame:
    frame = data["noise"]
    return frame[(frame["dataset"] == PRIMARY_DATASET) & frame["have_eval"]]


def c4_reward_vs_delay(data) -> plt.Figure:
    noise = _noise(data)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for condition in ("encdec", "expfm"):
        arm = noise[noise["condition"] == condition]
        line(ax, arm[arm["action_noise"] == 0.0], "episode_reward",
             condition=condition, dashed=True, alpha=0.5,
             label=f"{LABEL[condition]}, no noise")
        line(ax, arm[arm["action_noise"] == SIGMA_SHOWN], "episode_reward",
             condition=condition, label=f"{LABEL[condition]}, σ = {SIGMA_SHOWN}")
    log_delay_axis(ax)
    ax.set_ylabel("Held-out episode reward")
    ax.set_ylim(0, None)
    ax.set_title(f"σ = {SIGMA_SHOWN} motor noise costs the forward model far more\n"
                 f"(dashed = the same networks with no noise)", fontsize=10, pad=26)
    return finish(fig, ax, legend_loc="lower left")


def c4_reward_vs_sigma(data) -> plt.Figure:
    noise = _noise(data)
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.8), sharey=True)
    for ax, delay in zip(axes, C4_DELAYS):
        cell = noise[noise["delay_k"] == delay]
        for condition in ("encdec", "expfm"):
            arm = (cell[cell["condition"] == condition]
                   .groupby("action_noise", as_index=False)["episode_reward"].mean()
                   .sort_values("action_noise"))
            if arm.empty:
                continue
            ax.plot(arm["action_noise"], arm["episode_reward"],
                    color=color_for(STYLE_KEY[condition]), marker=MARKER[condition],
                    label=LABEL[condition])
        ax.set_xscale("symlog", linthresh=0.02, linscale=0.5)
        ax.set_xticks([0, 0.02, 0.05, 0.1, 0.25])
        ax.xaxis.set_major_formatter(FixedFormatter(["0", "0.02", "0.05", "0.1", "0.25"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Motor noise σ (action units)")
        ax.set_title(f"delay {delay} steps ({delay * 10} ms)", fontsize=10)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out episode reward")
    axes[0].set_ylim(0, None)
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("The forward model keeps less of its noise-free reward at every "
                 "delay but zero", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def c4_fm_mse(data) -> plt.Figure:
    noise = _noise(data)
    arm = noise[noise["condition"] == "expfm"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    line(ax, arm[arm["action_noise"] == 0.0], "fm_pred_mse", condition="expfm",
         dashed=True, alpha=0.5, label="No noise")
    line(ax, arm[arm["action_noise"] == SIGMA_SHOWN], "fm_pred_mse", condition="expfm",
         label=f"σ = {SIGMA_SHOWN}")
    log_delay_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Forward-model prediction MSE (held-out)")
    ax.set_title("Unobserved motor noise is an irreducible prediction error", pad=28)
    return finish(fig, ax, legend_loc="lower right")


# --------------------------------------------------------------------------------------
# Claim 5 -- the policy gradient does not learn a forward model
# --------------------------------------------------------------------------------------

MARKED_DELAY = 10


def c5_fm_mse_vs_delay(data) -> plt.Figure:
    delay = held_out(data["delay"])
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for condition in ("expfm", "pgfm"):
        line(ax, delay[delay["condition"] == condition], "fm_pred_mse",
             condition=condition)
    log_delay_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Forward-model prediction MSE (held-out)")
    ax.set_title("The policy gradient never drives the prediction error down\n"
                 "(same module, same inputs — only the loss differs)",
                 fontsize=10, pad=26)
    return finish(fig, ax, legend_loc="lower right")


def c5_scatter(data) -> plt.Figure:
    delay = held_out(data["delay"])
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for condition in ("expfm", "pgfm"):
        cell = delay[(delay["condition"] == condition)].dropna(
            subset=["fm_pred_mse", "episode_reward"])
        ax.scatter(cell["fm_pred_mse"], cell["episode_reward"],
                   c=color_for(STYLE_KEY[condition]), marker=MARKER[condition],
                   s=np.clip(14 + 1.6 * cell["delay_k"], 14, 90), alpha=0.85,
                   edgecolors="none", label=LABEL[condition])
        labelled = (cell[cell["delay_k"].isin((0, 10, 50, 100))]
                    .groupby("delay_k", as_index=False)[["fm_pred_mse",
                                                         "episode_reward"]].mean())
        for _, row in labelled.iterrows():
            ax.annotate(f"delay {int(row['delay_k'])}",
                        (row["fm_pred_mse"], row["episode_reward"]),
                        textcoords="offset points", xytext=(7, 4), fontsize=7.5,
                        color=color_for(STYLE_KEY[condition]))
    ax.set_xscale("log")
    ax.set_xlabel("Forward-model prediction MSE (held-out)")
    ax.set_ylabel("Held-out episode reward")
    ax.set_title("Two clouds, not one trend\n(marker size = delay; labels are delays)",
                 fontsize=10)
    return finish(fig, ax, legend_loc="lower left")


def c5_reward_vs_delay(data) -> plt.Figure:
    delay = held_out(data["delay"])
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for condition in ("expfm", "pgfm"):
        line(ax, delay[delay["condition"] == condition], "episode_reward",
             condition=condition)
    log_delay_axis(ax)
    ax.axvline(xpos([MARKED_DELAY])[0], color="0.4", lw=1.0, ls="-.", zorder=0)
    ax.annotate(f"delay {MARKED_DELAY}:\nreward ties,\nprediction does not",
                (xpos([MARKED_DELAY])[0], 0.06), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(8, 0), fontsize=8, color="0.35")
    ax.set_ylabel("Held-out episode reward")
    ax.set_ylim(0, None)
    ax.set_title("Same reward at short delay, different internals", pad=28)
    return finish(fig, ax, legend_loc="lower left")


PROBE_TARGET = "proprio"
#: Stage index spans of the two actor sub-networks, shaded so the x-axis reads as
#: anatomy rather than as eleven unrelated ticks.
ACTOR_SPANS = ((0.6, 5.4, "predictor"), (5.6, 11.4, "decoder"))


def _probe_panel(ax, probe, condition, delay, budget, *, dashed=False, alpha=1.0,
                 label=None, pathway="actor", prev=None):
    cell = probe[(probe["condition"] == condition) & (probe["delay_k"] == delay)
                 & (probe["budget"] == budget) & (probe["pathway"] == pathway)
                 & (probe["target"] == PROBE_TARGET)]
    if cell.empty:
        return prev
    stages = (cell.groupby(["stage_index", "stage_label"], as_index=False)
                  ["test_r2"].mean().sort_values("stage_index"))
    arm = "explicit" if condition.startswith("expfm") else "implicit"
    style = dict(color=ARM_COLOR[arm], marker=ARM_MARKER[arm], alpha=alpha,
                 label=label if label is not None else ARM_LABEL[arm])
    if dashed and label is None:
        style["label"] = "_nolegend_"    # the reference gets one shared legend entry
    if dashed:
        style.update(ls=(0, (5, 2)), lw=1.4, markersize=3, markerfacecolor="none")
    ax.plot(stages["stage_index"], stages["test_r2"], **style)
    if pathway == "actor" and not dashed:
        # Each arm's own linear input baseline: stage 0 extended across the panel. The
        # question is never "is R^2 high" but "does any layer beat its own inputs".
        baseline = stages[stages["stage_index"] == 0]["test_r2"]
        if len(baseline):
            ax.axhline(float(baseline.iloc[0]), color=ARM_COLOR[arm], lw=0.9,
                       ls=":", alpha=0.7)
    return stages if prev is None else prev


def _stage_axis(ax, stages, *, spans=()):
    labels = [lab.replace("\n", " ") for lab in stages["stage_label"]]
    ax.set_xticks(list(stages["stage_index"]))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Held-out R², current proprioception")
    for start, end, name in spans:
        ax.axvspan(start, end, color="0.92", zorder=0)
        ax.text((start + end) / 2, 0.965, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="0.45")


def c5_probe_delay10(data) -> plt.Figure:
    probe = data["probe"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    stages = None
    for condition in ("expfm", "pgfm"):
        stages = _probe_panel(ax, probe, condition, MARKED_DELAY, 600_000_000, prev=stages)
    _stage_axis(ax, stages, spans=ACTOR_SPANS)
    ax.set_ylim(0, 1.0)
    ax.set_title("Delay 10, 600 M steps: only the explicit predictor beats its own input\n"
                 "(dotted line = that arm's linear input baseline)", fontsize=10)
    return finish(fig, ax, legend_loc="lower left")


def c5_probe_delay20(data) -> plt.Figure:
    probe = data["probe"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    stages = None
    for condition in ("expfm", "pgfm"):
        _probe_panel(ax, probe, condition, MARKED_DELAY, 600_000_000, dashed=True,
                     alpha=0.45, label=None)
        stages = _probe_panel(ax, probe, condition, 20, 600_000_000, prev=stages)
    _stage_axis(ax, stages, spans=ACTOR_SPANS)
    ax.set_ylim(0, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.5", ls=(0, (5, 2)), lw=1.4,
                          marker="o", markersize=3, markerfacecolor="none"))
    labels.append("same arm at delay 10")
    ax.legend(handles, labels, loc="lower left", frameon=False)
    ax.set_title("Delay 20 (solid) against delay 10 (dashed): the same shape, lower down",
                 fontsize=10)
    fig.tight_layout()
    return fig


def c5_probe_2g(data) -> plt.Figure:
    probe = data["probe"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    stages = None
    for condition in ("expfm", "pgfm"):
        _probe_panel(ax, probe, condition, MARKED_DELAY, 600_000_000, dashed=True,
                     alpha=0.45, label=None)
        source = "expfm_2g" if condition == "expfm" else "pgfm_2g"
        stages = _probe_panel(ax, probe, source, MARKED_DELAY, 2_000_000_000, prev=stages)
    _stage_axis(ax, stages, spans=ACTOR_SPANS)
    ax.set_ylim(0, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.5", ls=(0, (5, 2)), lw=1.4,
                          marker="o", markersize=3, markerfacecolor="none"))
    labels.append("same arm at 600 M")
    ax.legend(handles, labels, loc="lower left", frameon=False)
    ax.set_title("Delay 10 at 2 B steps (solid) against 600 M (dashed): "
                 "3× the budget sharpens it, nothing more", fontsize=10)
    fig.tight_layout()
    return fig


def c5_probe_encoder(data) -> plt.Figure:
    probe = data["probe"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    stages = None
    for condition in ("expfm", "pgfm"):
        stages = _probe_panel(ax, probe, condition, MARKED_DELAY, 600_000_000,
                              pathway="encoder", prev=stages)
    _stage_axis(ax, stages)
    ax.set_ylim(0, 1.0)
    ax.set_title("The encoder carries no current-state information in either arm\n"
                 "(same axis as the actor figure — the leakage control)", fontsize=10)
    return finish(fig, ax, legend_loc="upper right")


GROUP_ORDER = ["input", "forward_model", "decoder", "fm_plus_decoder", "whole_network"]
GROUP_LABEL = {"input": "network input\n(delayed + efference)",
               "forward_model": "whole forward\nmodel",
               "decoder": "whole decoder",
               "fm_plus_decoder": "forward model\n+ decoder",
               "whole_network": "whole network\n(enc + fm + dec)"}


def c5_probe_groups(data) -> plt.Figure:
    """Concatenate whole sub-networks and ask the same question of each.

    A per-layer probe can only see what one layer holds; a sub-network may distribute the
    state across several. Concatenating removes that excuse, and the answer does not move.
    """
    groups = data["groups"]
    cell = groups[(groups["delay_k"] == MARKED_DELAY)
                  & (groups["budget"] == 600_000_000)
                  & (groups["target"] == PROBE_TARGET)]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    width = 0.38
    positions = np.arange(len(GROUP_ORDER))
    for offset, (condition, arm) in enumerate((("expfm", "explicit"),
                                               ("pgfm", "implicit"))):
        values, counts = [], []
        for group in GROUP_ORDER:
            rows = cell[(cell["condition"] == condition) & (cell["group"] == group)]
            values.append(float(rows["test_r2"].mean()) if len(rows) else np.nan)
            counts.append(len(rows))
        bars = ax.bar(positions + (offset - 0.5) * width, values, width,
                      color=ARM_COLOR[arm], alpha=0.9, label=ARM_LABEL[arm])
        for rect, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(rect.get_x() + rect.get_width() / 2, value + 0.012,
                        f"{value:.2f}", ha="center", fontsize=8,
                        color=ARM_COLOR[arm])
        # The input bar is each arm's own baseline; mark it across the panel.
        baseline = values[GROUP_ORDER.index("input")]
        if np.isfinite(baseline):
            ax.axhline(baseline, color=ARM_COLOR[arm], lw=0.9, ls=":", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER], fontsize=8)
    ax.set_ylabel("Held-out R², current proprioception")
    ax.set_ylim(0, 1.0)
    ax.set_title("Concatenating whole sub-networks does not rescue the implicit arm\n"
                 f"delay {MARKED_DELAY}, 600 M steps; dotted = that arm's input baseline",
                 fontsize=10)
    return finish(fig, ax, legend_loc="upper left")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

FIGURE_BUILDERS = [
    ("c1_reward_vs_delay", c1_reward, ("delay",)),
    ("c1_tracking_error_vs_delay", c1_tracking, ("delay",)),
    ("c1_lifetime_vs_delay", c1_lifetime, ("delay",)),
    ("c1_decoder_input_ablation", c1_decoder_inputs, ("curves",)),
    ("c2_reward_vs_delay", c2_three_arms, ("delay",)),
    ("c3_learning_curves", c3_curves, ("curves", "budget")),
    ("c3_reward_vs_delay_at_4g", c3_reward_at_4g, ("budget",)),
    ("c4_reward_vs_delay_at_noise", c4_reward_vs_delay, ("noise",)),
    ("c4_reward_vs_sigma", c4_reward_vs_sigma, ("noise",)),
    ("c4_prediction_mse_vs_delay", c4_fm_mse, ("noise",)),
    ("c5_prediction_mse_vs_delay", c5_fm_mse_vs_delay, ("delay",)),
    ("c5_mse_vs_reward", c5_scatter, ("delay",)),
    ("c5_reward_vs_delay", c5_reward_vs_delay, ("delay",)),
    ("c5_probe_delay10_600m", c5_probe_delay10, ("probe",)),
    ("c5_probe_delay20_vs_delay10", c5_probe_delay20, ("probe",)),
    ("c5_probe_2g_vs_600m", c5_probe_2g, ("probe",)),
    ("c5_probe_encoder", c5_probe_encoder, ("probe",)),
    ("c5_probe_concatenated_groups", c5_probe_groups, ("groups",)),
]

SOURCES = {"delay": DELAY, "budget": BUDGET, "curves": CURVES, "noise": NOISE,
           "probe": PROBE, "groups": GROUPS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="figure name prefixes to build, e.g. c3 c5_probe")
    args = parser.parse_args()

    apply_style()
    FIGURES.mkdir(exist_ok=True)
    data = {name: pd.read_csv(path) for name, path in SOURCES.items() if path.exists()}

    manifest_path = FIGURES / "manifest.json"
    manifest = {}
    if manifest_path.exists() and args.only:
        import json
        manifest = json.loads(manifest_path.read_text())

    for name, builder, needs in FIGURE_BUILDERS:
        if args.only and not any(name.startswith(prefix) for prefix in args.only):
            continue
        missing = [n for n in needs if n not in data]
        if missing:
            print(f"skip {name}: missing {missing}")
            continue
        fig = builder(data)
        manifest[f"{name}.png"] = provenance(fig, HERE, *[SOURCES[n] for n in needs])
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
