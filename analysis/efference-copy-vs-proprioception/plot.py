"""Figures for the efference-copy-vs-proprioception question.

Reads only the CSVs in this folder -- no WandB, no artifact store, no network -- so every
figure can be restyled or re-rendered from the committed data alone.

Conventions
-----------
**Control mode -> colour and line style**, matching ``../position-vs-torque-control/`` and
``../position-control-open-loop/`` so the arms read the same across all three questions:
solid blue = position actuators, dashed red = torque. The pair is
ColorBrewer RdBu 3/9 and was checked rather than assumed -- OKLab dE 28.7 for normal
vision, 20.3 / 23.7 / 26.8 under simulated protanopia / deuteranopia / tritanopia (target
>= 8), matched lightness (0.505 vs 0.492), 5.9:1 and 6.9:1 contrast on white -- and line
style carries the same distinction independently for print and forced-colour cases.

**Reference levels are drawn, not tabulated**, and labelled in place rather than in the
legend, which stays at two entries: the two swept lines are the measurement, and the
``intact`` / ``nointent`` levels are the scale to read them against.

The x axis is **symlog** with ``linthresh=1``: the sweep is 0, 1, 2, 3, 5, 10, 15, 20, 50,
100, which is logarithmic in spirit but contains 0, and a linear axis would collapse
everything below 20 into a fifth of the panel -- exactly the region where the position arm
does all of its recovering.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path

from vnl_experiments.wandb_utils.style import (
    CTRL_DT_MS,
    apply_style,
    provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

MODE_COLOR = {"position": "#2166ac", "torque": "#b2182b"}
MODE_LS = {"position": "-", "torque": "--"}
MODE_MARKER = {"position": "o", "torque": "s"}
MODE_LABEL = {"position": "Position actuators", "torque": "Torque actuators"}

#: The primary readout, and the earlier one kept for the late-training comparison. Named
#: rather than ordered: an earlier version of this folder read everything at 400 M because
#: the position sweep had crashed, and a swap of two same-shaped labels is exactly the edit
#: that silently inverts an axis.
PRIMARY_LABEL = "600M"
EARLY_LABEL = "400M"

#: Ticks for the symlog x axis: every efference length actually launched, so no plotted
#: point sits on an unlabelled position -- the sweep is irregular (3, 5, 15) and a reader
#: cannot infer the missing values from the labelled ones.
EFF_TICKS = [0, 1, 2, 3, 5, 10, 15, 20, 50, 100]

#: A sparser subset for the millisecond twin. Every EFF_TICK would collide there -- 100 ms
#: and 200 ms sit a few points apart -- and the twin is orientation, not a read-off axis.
MS_TICKS = [0, 1, 5, 10, 50, 100]


def load():
    data = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    return data, curves


def usable(data, condition, column=f"reward_{PRIMARY_LABEL}"):
    return data[(data["condition"] == condition) & data[column].notna()]


def sweep(data, condition, column=f"reward_{PRIMARY_LABEL}"):
    """Per-efference-length mean, spread and n for one condition."""
    arm = usable(data, condition, column)
    out = (arm.groupby("efference_length")[column]
           .agg(["mean", "min", "max", "count"]).reset_index())
    return out.sort_values("efference_length", ignore_index=True)


def level(data, condition, column=f"reward_{PRIMARY_LABEL}"):
    """The mean of a reference condition, or NaN when it has no usable run."""
    arm = usable(data, condition, column)
    return float(arm[column].mean()) if len(arm) else float("nan")


def _symlog_x(ax, ms_axis=True):
    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.set_xticks(EFF_TICKS)
    ax.set_xticklabels([str(t) for t in EFF_TICKS])
    ax.set_xlim(-0.15, 130)
    ax.set_xlabel("Efference-copy length (control steps)")
    # 10 / 15 / 20 sit close together on the log part of the axis.
    ax.tick_params(axis="x", labelsize=8)
    if not ms_axis:
        return None
    # `style.add_ms_axis` is not reused: it builds a *linear* twin, which cannot line up
    # with a symlog bottom axis. Mirroring the scale explicitly is what makes the top
    # ticks sit above their partners.
    top = ax.twiny()
    top.set_xscale("symlog", linthresh=1, linscale=0.5)
    top.set_xlim(ax.get_xlim())
    top.set_xticks(MS_TICKS)
    top.set_xticklabels([str(int(t * CTRL_DT_MS)) for t in MS_TICKS])
    top.set_xlabel("Efference-copy length (ms)", fontsize=9)
    top.tick_params(labelsize=8)
    for side in ("right", "left", "bottom"):
        top.spines[side].set_visible(False)
    return top


def _plot_arm(ax, points, mode, column="mean", runs=None):
    """One mode's swept line, with its individual runs as open markers behind it."""
    color = MODE_COLOR[mode]
    if runs is not None and len(runs):
        ax.scatter(runs["efference_length"], runs["y"], s=14, facecolors="none",
                   edgecolors=color, lw=0.8, alpha=0.55, zorder=2)
    if {"min", "max"} <= set(points.columns):
        ax.fill_between(points["efference_length"], points["min"], points["max"],
                        color=color, alpha=0.15, lw=0, zorder=1)
    ax.plot(points["efference_length"], points[column], color=color,
            ls=MODE_LS[mode], marker=MODE_MARKER[mode], ms=5,
            markeredgecolor="white", markeredgewidth=0.7,
            label=MODE_LABEL[mode], zorder=3)


def _reference(ax, y, color, text, ls, va="bottom", x=0.012, ha="left"):
    """A horizontal reference level, labelled in place instead of in the legend."""
    if not np.isfinite(y):
        return
    ax.axhline(y, color=color, lw=0.9, ls=ls, alpha=0.75, zorder=1)
    ax.annotate(text, xy=(x, y), xycoords=("axes fraction", "data"),
                ha=ha, va=va, fontsize=6.5, color=color,
                xytext=(0, 2 if va == "bottom" else -2), textcoords="offset points")


def _reference_pair(ax, entries, ls):
    """Two reference levels whose labels are pushed apart, higher above / lower below.

    The pairs are close -- the modes' intact baselines differ by 2.6 %, their blind floors
    by 12 % -- so a shared ``va`` puts both labels in the same place. Which of the two is
    on top is **not** fixed across panels: torque is the higher intact baseline but the
    lower blind floor on reward, and the higher blind floor on reward-per-step. Deriving
    the stagger from the values rather than hardcoding it is what stops one panel silently
    overlapping when the ordering flips. Labels sit at the left edge, where the swept lines
    have not started; they are kept short for the same reason.
    """
    ordered = sorted((e for e in entries if np.isfinite(e[0])), reverse=True)
    for i, (y, color, text) in enumerate(ordered):
        _reference(ax, y, color, text, ls, va="bottom" if i == 0 else "top")


# --------------------------------------------------------------------------------------
# figure 1 -- the question
# --------------------------------------------------------------------------------------

def figure_sweep(data):
    """Reward vs efference length, per control mode: absolute, then as a recovery fraction.

    Panel A is the plot the question asks for. Panel B divides each mode by *its own*
    proprioception-intact baseline, which is the fair comparison: the two modes' intact
    baselines differ by 2.6 %, so an absolute panel alone slightly flatters position.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    ax = axes[0]
    for mode, condition in (("position", "pos_noproprio"), ("torque", "torque_noproprio")):
        arm = usable(data, condition)
        runs = arm.assign(y=arm[f"reward_{PRIMARY_LABEL}"])
        _plot_arm(ax, sweep(data, condition), mode, runs=runs)
    _reference_pair(ax, [(level(data, "pos_intact"), MODE_COLOR["position"],
                          "position, intact"),
                         (level(data, "torque_intact"), MODE_COLOR["torque"],
                          "torque, intact")], ":")
    _reference_pair(ax, [(level(data, "pos_nointent"), MODE_COLOR["position"],
                          "position, no target"),
                         (level(data, "torque_nointent"), MODE_COLOR["torque"],
                          "torque, no target")], (0, (1, 3)))
    _symlog_x(ax)
    ax.set_ylim(520, 2075)
    ax.set_ylabel(f"Held-out episode reward at {PRIMARY_LABEL} steps")
    ax.set_title("A. Reward recovered by an efference copy", loc="left")
    ax.legend(loc="center right")

    ax = axes[1]
    for mode, condition, intact in (("position", "pos_noproprio", "pos_intact"),
                                    ("torque", "torque_noproprio", "torque_intact")):
        base = level(data, intact)
        points = sweep(data, condition)
        for col in ("mean", "min", "max"):
            points[col] = points[col] / base
        arm = usable(data, condition)
        runs = arm.assign(y=arm[f"reward_{PRIMARY_LABEL}"] / base)
        _plot_arm(ax, points, mode, runs=runs)
    _reference_pair(ax, [(level(data, "pos_nointent") / level(data, "pos_intact"),
                          MODE_COLOR["position"], "position, no target"),
                         (level(data, "torque_nointent") / level(data, "torque_intact"),
                          MODE_COLOR["torque"], "torque, no target")], (0, (1, 3)))
    _reference(ax, 1.0, "0.35", "proprioception intact (both modes)", ":", va="top")
    _symlog_x(ax)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Fraction of the same mode's intact baseline")
    ax.set_title("B. As a fraction of what was lost", loc="left")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# figure 2 -- survival vs tracking quality
# --------------------------------------------------------------------------------------

def figure_decomposition(data):
    """Is the remaining deficit falling over sooner, or tracking worse while upright?

    Episode reward is the product of the two and cannot separate them. Lifespan is how long
    the episode lasted before a termination; reward per surviving step is how well the
    reference was tracked while it did.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    panels = [("lifespan", f"lifespan_{PRIMARY_LABEL}", "Mean lifespan (control steps)",
               "A. Survival"),
              ("rps", f"reward_per_step_{PRIMARY_LABEL}", "Reward per surviving step",
               "B. Tracking quality while upright")]
    for ax, (_, column, ylabel, title) in zip(axes, panels):
        for mode, condition in (("position", "pos_noproprio"),
                                ("torque", "torque_noproprio")):
            arm = usable(data, condition, column)
            runs = arm.assign(y=arm[column])
            _plot_arm(ax, sweep(data, condition, column), mode, runs=runs)
        _reference_pair(ax, [(level(data, "pos_intact", column),
                              MODE_COLOR["position"], "position, intact"),
                             (level(data, "torque_intact", column),
                              MODE_COLOR["torque"], "torque, intact")], ":")
        _reference_pair(ax, [(level(data, "pos_nointent", column),
                              MODE_COLOR["position"], "position, no target"),
                             (level(data, "torque_nointent", column),
                              MODE_COLOR["torque"], "torque, no target")], (0, (1, 3)))
        _symlog_x(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    axes[0].legend(loc="center right")
    axes[1].legend(loc="center right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# figure 3 -- does the early readout change the answer?
# --------------------------------------------------------------------------------------

def figure_budget_check(data):
    """How much of each arm was still being learned in the last third of training.

    An earlier version of this folder had to read every arm at 400 M, because the position
    sweep had crashed; panel B is why that mattered and why the relaunch was worth waiting
    for. The pattern is not a clean monotone trend -- the position arm gains little at any
    length, and torque's gains scatter -- but torque's two longest queues gained ~14 % over
    the last 200 M while its short ones gained nothing, which is what a 3 832-input decoder
    layer against a 108-input one should look like. That is enough to have made the shape of
    torque's curve past its peak unreadable at 400 M. It is readable now.
    """
    both = data[data[f"reward_{PRIMARY_LABEL}"].notna()
                & data[f"reward_{EARLY_LABEL}"].notna()].copy()
    both["gain"] = 100 * (both[f"reward_{PRIMARY_LABEL}"]
                          / both[f"reward_{EARLY_LABEL}"] - 1)
    swept = both[both["condition"].str.endswith("noproprio")]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    ax = axes[0]
    lim = [0, 1.06 * both[[f"reward_{PRIMARY_LABEL}", f"reward_{EARLY_LABEL}"]].max().max()]
    ax.plot(lim, lim, color="0.6", lw=0.9, ls="--", zorder=1)
    ax.annotate("no change", xy=(0.62, 0.62), xycoords="axes fraction", fontsize=6.5,
                color="0.45", rotation=45, ha="center", va="bottom")
    for mode in MODE_COLOR:
        arm = both[both["mode"] == mode]
        ablated = arm[arm["condition"].str.endswith("noproprio")]
        other = arm[~arm["condition"].str.endswith("noproprio")]
        ax.scatter(ablated[f"reward_{EARLY_LABEL}"], ablated[f"reward_{PRIMARY_LABEL}"],
                   s=34, color=MODE_COLOR[mode], marker=MODE_MARKER[mode],
                   edgecolors="white", lw=0.6, zorder=3, label=MODE_LABEL[mode])
        ax.scatter(other[f"reward_{EARLY_LABEL}"], other[f"reward_{PRIMARY_LABEL}"],
                   s=34, facecolors="none", edgecolors=MODE_COLOR[mode], lw=1.0,
                   zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Reward at {EARLY_LABEL} steps")
    ax.set_ylabel(f"Reward at {PRIMARY_LABEL} steps")
    ax.set_title("A. What the last 200 M steps add", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="none", marker="o", ms=5, markerfacecolor="none",
                          markeredgecolor="0.4", label="reference conditions"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5)

    ax = axes[1]
    for mode in MODE_COLOR:
        arm = swept[swept["mode"] == mode].sort_values("efference_length")
        grouped = arm.groupby("efference_length")["gain"].mean().reset_index()
        ax.scatter(arm["efference_length"], arm["gain"], s=14, facecolors="none",
                   edgecolors=MODE_COLOR[mode], lw=0.8, alpha=0.55, zorder=2)
        ax.plot(grouped["efference_length"], grouped["gain"], color=MODE_COLOR[mode],
                ls=MODE_LS[mode], marker=MODE_MARKER[mode], ms=5,
                markeredgecolor="white", markeredgewidth=0.7,
                label=MODE_LABEL[mode], zorder=3)
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    _symlog_x(ax, ms_axis=False)
    ax.set_ylabel(f"Reward gain from {EARLY_LABEL} to {PRIMARY_LABEL} (%)")
    ax.set_title("B. Torque's long queues gained late", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# figure 4 -- the curves the readout is taken from
# --------------------------------------------------------------------------------------

def figure_curves(data, curves):
    """Every cohort run's learning curve, so the readout window can be seen in context.

    Two things to see: the readout window is flat for every arm, so the number is a
    plateau and not a snapshot of a still-rising curve; and the crashed runs simply stop
    where they stop, rather than diverging beforehand, which is what makes their 400 M
    values usable as the relaunch cross-check in ``pooling_check.txt``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    swept = {"position": "pos_noproprio", "torque": "torque_noproprio"}
    intact = {"position": "pos_intact", "torque": "torque_intact"}

    lengths = sorted(curves.loc[curves["condition"].str.endswith("noproprio"),
                                "efference_length"].dropna().unique())
    # Sequential, light -> dark, within each mode's hue: efference length is a magnitude,
    # so it gets a ramp, not categorical hues.
    ramps = {"position": plt.get_cmap("Blues"), "torque": plt.get_cmap("Reds")}

    for ax, mode in zip(axes, ("position", "torque")):
        for _, run in data[data["condition"] == intact[mode]].iterrows():
            curve = curves[curves["wandb_id"] == run["wandb_id"]]
            ax.plot(curve["step"] / 1e6, curve["reward"], color="0.55", lw=1.0,
                    ls=":", zorder=2)
        for i, eff in enumerate(lengths):
            shade = ramps[mode](0.32 + 0.62 * i / max(len(lengths) - 1, 1))
            arm = data[(data["condition"] == swept[mode])
                       & (data["efference_length"] == eff)]
            for _, run in arm.iterrows():
                curve = curves[curves["wandb_id"] == run["wandb_id"]]
                if curve.empty:
                    continue
                ax.plot(curve["step"] / 1e6, curve["reward"], color=shade, lw=1.2,
                        zorder=3)
        ax.axvspan(550, 600, color="0.85", lw=0, zorder=0)
        # Beside the band, not over it: at 400 M the intact curve is near the top of the
        # panel, so a centred label above the band would sit on it.
        ax.annotate(f"{PRIMARY_LABEL} readout", xy=(540, 0.995),
                    xycoords=("data", "axes fraction"), ha="right", va="top",
                    fontsize=6.5, color="0.4")
        ax.set_xlabel("Training steps (M)")
        ax.set_title(f"{MODE_LABEL[mode]}", loc="left")
        # One legend per panel, in that panel's own hue: a single shared blue key would
        # misdescribe the red ramp opposite it.
        handles = [Line2D([], [], color="0.55", ls=":", lw=1.0,
                          label="proprioception intact"),
                   Line2D([], [], color=ramps[mode](0.32), lw=1.4,
                          label=f"efference {int(min(lengths))} (light)"),
                   Line2D([], [], color=ramps[mode](0.94), lw=1.4,
                          label=f"efference {int(max(lengths))} (dark)")]
        ax.legend(handles=handles, loc="lower right", fontsize=7)
    axes[0].set_ylabel("Held-out episode reward")
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data, curves = load()

    built = {
        "efference_sweep.png": figure_sweep(data),
        "survival_vs_tracking.png": figure_decomposition(data),
        "budget_check.png": figure_budget_check(data),
        "training_curves.png": figure_curves(data, curves),
    }
    manifest = {}
    for name, fig in built.items():
        manifest[name] = provenance(fig, HERE, DATA, CURVES)
        fig.savefig(FIGURES / name)
        plt.close(fig)
        print(f"wrote figures/{name}")
    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
