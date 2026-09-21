"""Figures for the WalkerWalk joint-stiffness analysis. Reads only data.csv / curves.csv.

Four figures, in the order the report argues them:

``delay.png``
    The headline. Reward against delay, one line per stiffness. This is the hypothesis
    figure: if peripheral stiffness absorbs sensorimotor delay, the stiff lines should fall
    away more slowly than torque. Only the viable stiffnesses are drawn (``servo_kp >= 4``);
    the three soft ones never learn to walk at any delay and are in ``stiffness.png``.

``stiffness.png``
    The same data read along the other axis, one panel per delay, with torque as a
    horizontal reference line rather than a point at x=0 -- the servo family does not
    connect continuously to torque control at zero (``servo_control.md`` section 1.3).
    This is where the soft end and the viability threshold are visible.

``curves.png``
    Learning curves, which is what licenses or forbids reading any of the above as a
    converged level. At delay 10 it is the *torque* arm that is still climbing. Drawn on the
    same common grid as the criterion, so the 2026-09-09 runs appear at 100 points rather
    than their native 800 -- what is plotted is exactly what ``steps_to_900`` was read off.

``time_to_900_delay.png`` / ``time_to_900_stiffness.png``
    Learning *speed* rather than final level: the step at which the trailing 2.4e7-step mean
    of eval reward first reaches 900, read on the common 4.8e6-step grid that
    ``extract.on_common_grid`` builds. A run that never gets there is **censored, not
    dropped** -- drawn as a caret on the budget ceiling -- because at delay 10 the censored
    arm is torque, and a line that merely stopped would hide the result.

``convergence.png``
    The same two disqualifications as one number per run: reward gained over the last 1e8
    steps, and the largest peak-to-trough fall in the last 2e8.

Colour: ``torque`` takes its canonical condition colour from ``wandb_utils.style``.
Stiffness is an ordered magnitude rather than an identity, so it is carried by lightness
within a **single hue** (a sequential ramp, never a rainbow).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    add_ms_axis, apply_style, color_for, label_for, marker_for, plot_seeds, provenance,
    reward_label, write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

#: WalkerWalk's control step. Never the rodent's 10 ms (track README).
CTRL_DT_MS = 25

#: Where the `servo_kp = 0` baseline is drawn on the log stiffness axis. A drawing
#: position for a reference line, not a data point.
BASELINE_X = 0.13

#: Delays at which the servo arm was run. The torque arm reaches 20, and `delay.png` draws
#: its full extent: stopping it at 10 would hide how bad delay 15-20 is for torque.
SERVO_DELAYS = (0, 5, 10)

#: Stiffnesses drawn in the delay figure. Below 4 the servo never learns to walk at any
#: delay, so including them would spend most of the line slots on a floor.
VIABLE_KP = (4.0, 8.0, 16.0, 32.0, 48.0, 64.0)

#: A run whose end-of-training reward is not a converged level: still moving, or recovering
#: from a collapse. Thresholds are read off the torque arm at delay 0/5, which sits at
#: |late_gain| < 5 and drawdown < 5.
UNSETTLED = "(late_gain.abs() > 20) | (max_drawdown_late > 50)"


def _kp_colors(kps):
    """Single-hue sequential ramp over stiffness: an ordered magnitude, not an identity."""
    ramp = plt.get_cmap("PuRd")
    lo, hi = np.log2(min(kps)), np.log2(max(kps))
    return {kp: ramp(0.30 + 0.65 * (np.log2(kp) - lo) / (hi - lo)) for kp in kps}


def delay_figure(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    torque = df[df.condition == "torque"]
    servo = df[(df.condition == "servo") & df.servo_kp.isin(VIABLE_KP)]
    colors = _kp_colors(VIABLE_KP)

    plot_seeds(ax, torque, x="delay", y="reward_final", seed_col="seed",
               condition="torque")
    for kp in VIABLE_KP:
        g = servo[servo.servo_kp == kp].sort_values("delay")
        if not len(g):
            continue
        ax.plot(g.delay, g.reward_final, marker="D", ls="-", ms=6, lw=1.9,
                color=colors[kp], label=f"servo_kp = {kp:g}")
        un = g.query(UNSETTLED)
        ax.plot(un.delay, un.reward_final, "o", mfc="none", mec="0.25", ms=14, mew=1.3,
                lw=0, zorder=7)

    un_t = torque.query(UNSETTLED)
    ax.plot(un_t.delay, un_t.reward_final, "o", mfc="none", mec="0.25", ms=14, mew=1.3,
            lw=0, zorder=7, label="not converged / unstable")

    ax.set_xlabel("Delay (control steps)")
    ax.set_ylabel(reward_label("dmc_eval"))
    ax.set_ylim(0, 1060)
    ax.set_xticks([0, 5, 7, 10, 15, 20])
    ax.grid(alpha=0.25)
    add_ms_axis(ax, 20, ctrl_dt_ms=CTRL_DT_MS)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    ax.set_title("WalkerWalk at 4.8e8 steps: stiff joint servos hold up at delay 10\n"
                 "where torque control does not", fontsize=11)
    fig.tight_layout()
    provenance(fig, HERE, HERE / "data.csv")
    path = FIGURES / "delay.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def stiffness_figure(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(SERVO_DELAYS), figsize=(13.5, 4.4), sharey=True)
    servo, torque = df[df.condition == "servo"], df[df.condition == "torque"]

    for ax, delay in zip(axes, SERVO_DELAYS):
        s = servo[servo.delay == delay].sort_values("servo_kp")
        t = torque[torque.delay == delay]

        if len(t):
            lo, hi, mean = t.reward_final.min(), t.reward_final.max(), t.reward_final.mean()
            ax.axhspan(lo, hi, color=color_for("torque"), alpha=0.18, lw=0, zorder=1)
            ax.axhline(mean, color=color_for("torque"), lw=1.6, ls="--", zorder=2,
                       label=f"{label_for('torque')}, n={len(t)}")
            ax.plot([BASELINE_X] * len(t), t.reward_final, marker_for("torque"),
                    color=color_for("torque"), ms=6, zorder=4, clip_on=False)

        cell = s.groupby("servo_kp").reward_final.mean()
        ax.plot(cell.index, cell.values, marker=marker_for("servo"), ls="-",
                color=color_for("servo"), ms=7, lw=2.0, zorder=5,
                label=label_for("servo"))
        un = s.query(UNSETTLED)
        ax.plot(un.servo_kp, un.reward_final, "o", mfc="none", mec="0.25", ms=15, mew=1.4,
                lw=0, zorder=7,
                label="not converged / unstable" if delay == SERVO_DELAYS[0] else None)
        ax.plot(s.servo_kp, s.reward_final, "o", color=color_for("servo"), ms=3.5,
                alpha=0.55, lw=0, zorder=6,
                label="individual seed" if delay == SERVO_DELAYS[0] else None)

        ax.set_xscale("log")
        ax.set_xticks([BASELINE_X, 0.25, 1, 4, 16, 64])
        ax.set_xticklabels(["0\n(torque)", "0.25", "1", "4", "16", "64"])
        ax.set_xlim(0.1, 95)
        ax.axvline(0.18, color="0.75", lw=0.9, ls=":", zorder=0)
        ax.set_xlabel("servo_kp  (dimensionless, normalised per joint)")
        ax.set_title(f"delay {delay}  ({delay * CTRL_DT_MS} ms)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(reward_label("dmc_eval"))
    axes[0].set_ylim(0, 1060)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("Joint stiffness vs end-of-training reward, per delay "
                 "(mean of eval points in the last 5e7 of 4.8e8 steps)", fontsize=11)
    fig.tight_layout()
    provenance(fig, HERE, HERE / "data.csv")
    path = FIGURES / "stiffness.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def curves_figure(curves: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(SERVO_DELAYS), figsize=(13.5, 4.4), sharey=True)
    kps = sorted(curves[curves.servo_kp > 0].servo_kp.unique())
    colors = _kp_colors(kps)

    for ax, delay in zip(axes, SERVO_DELAYS):
        sub = curves[curves.delay == delay]
        first = True
        for _, g in sub[sub.servo_kp == 0].groupby("wandb_id"):
            g = g.sort_values("step")
            ax.plot(g.step / 1e6, g.reward_mean, color=color_for("torque"), lw=2.1,
                    alpha=0.95, zorder=6, label=label_for("torque") if first else None)
            first = False
        for kp in kps:
            labelled = False
            for _, g in sub[sub.servo_kp == kp].groupby("wandb_id"):
                g = g.sort_values("step")
                ax.plot(g.step / 1e6, g.reward_mean, color=colors[kp], lw=1.5,
                        label=f"kp = {kp:g}" if not labelled else None)
                labelled = True
        ax.set_xlabel("Environment steps (millions)")
        ax.set_title(f"delay {delay}  ({delay * CTRL_DT_MS} ms)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(reward_label("dmc_eval"))
    axes[0].set_ylim(0, 1060)
    axes[-1].legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.92)
    fig.suptitle("At delay 10 it is the torque arm that has not converged", fontsize=11)
    fig.tight_layout()
    provenance(fig, HERE, HERE / "curves.csv")
    path = FIGURES / "curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _censor_band(ax, budget_m: float) -> None:
    """Mark the region a censored run would have to be in, above the budget.

    A run that never reaches the criterion is censored, not missing. Drawing its line as
    simply stopping would read as absent data, which is how a real effect gets lost -- at
    delay 10 the *torque* arm is the censored one, so this is load-bearing here.
    """
    ax.axhspan(budget_m, budget_m * 1.16, color="0.85", lw=0, zorder=0)
    ax.axhline(budget_m, color="0.55", lw=1.0, ls="-", zorder=1)
    ax.text(0.985, budget_m * 1.07, "never reached criterion within budget",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7.5, color="0.35")


def _plot_with_censoring(ax, g, xcol, *, color, label, budget_m, marker="D",
                         ls="-", ms=6, lw=1.9, ycol="steps_to_900"):
    """Solved points as a line; censored ones as carets on the budget ceiling.

    The line is **broken across censored cells** rather than drawn through them. Joining
    two solved points over a neighbour that never reached the criterion would draw a smooth
    path exactly where the arm in fact failed -- which at delay 5 (``servo_kp`` 8 and 16
    both censored, 4 and 32 both solved) would be a straight line through the failure.
    """
    g = g.sort_values(xcol)
    ok = g[ycol].notna().to_numpy()
    # Contiguous stretches of solved cells, in x order.
    breaks = np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1
    labelled = False
    for chunk in np.split(np.arange(len(g)), breaks):
        sub = g.iloc[chunk]
        if not ok[chunk[0]]:
            continue
        ax.plot(sub[xcol], sub[ycol] / 1e6, marker=marker, ls=ls if len(sub) > 1 else "none",
                ms=ms, lw=lw, color=color, label=None if labelled else label)
        labelled = True
    censored = g[g[ycol].isna()]
    if len(censored):
        ax.plot(censored[xcol], [budget_m] * len(censored), marker=6, ls="none",
                ms=11, color=color, zorder=6, label=None if labelled else label)


def time_to_delay_figure(df: pd.DataFrame) -> Path:
    budget_m = 480.0
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    torque = df[df.condition == "torque"]
    servo = df[(df.condition == "servo") & df.servo_kp.isin(VIABLE_KP)]
    colors = _kp_colors(VIABLE_KP)

    # Every torque seed, not just their mean: at delay 5 one of the four takes 2.6x as long
    # as the other three, which is the noise floor any speed claim has to clear.
    for delay, g in torque.groupby("delay"):
        s = g[g.steps_to_900.notna()]
        ax.plot([delay] * len(s), s.steps_to_900 / 1e6, "o", color=color_for("torque"),
                ms=4.5, alpha=0.75, lw=0, zorder=5,
                label="torque, individual seed" if delay == 0 else None)
    tmean = torque.groupby("delay").agg(steps_to_900=("steps_to_900", "mean")).reset_index()
    tmean["n_solved"] = torque.groupby("delay").steps_to_900.count().values
    tmean.loc[tmean.n_solved == 0, "steps_to_900"] = np.nan
    _plot_with_censoring(ax, tmean, "delay", color=color_for("torque"),
                         label=label_for("torque"), budget_m=budget_m,
                         marker=marker_for("torque"), ms=7, lw=2.2)
    for kp in VIABLE_KP:
        g = servo[servo.servo_kp == kp]
        if len(g):
            _plot_with_censoring(ax, g, "delay", color=colors[kp],
                                 label=f"servo_kp = {kp:g}", budget_m=budget_m)

    _censor_band(ax, budget_m)
    ax.set_xlabel("Delay (control steps)")
    ax.set_ylabel("Steps to reach reward 900 (millions)")
    ax.set_ylim(0, budget_m * 1.16)
    ax.set_xticks([0, 5, 7, 10, 15, 20])
    ax.grid(alpha=0.25)
    add_ms_axis(ax, 20, ctrl_dt_ms=CTRL_DT_MS)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95, ncol=2)
    ax.set_title("Learning speed: torque control is fastest with no delay and never gets\n"
                 "there at delay 10, where stiff servos need ~1/4 of the budget",
                 fontsize=11)
    fig.tight_layout()
    provenance(fig, HERE, HERE / "data.csv")
    path = FIGURES / "time_to_900_delay.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def time_to_stiffness_figure(df: pd.DataFrame) -> Path:
    budget_m = 480.0
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    servo, torque = df[df.condition == "servo"], df[df.condition == "torque"]

    # Left: the headline criterion against stiffness, one line per delay.
    ax = axes[0]
    ramp = plt.get_cmap("PuRd")
    for delay, mk, shade in ((0, "o", 0.42), (5, "s", 0.66), (10, "^", 0.95)):
        g = servo[servo.delay == delay]
        if not len(g):
            continue
        _plot_with_censoring(ax, g, "servo_kp", color=ramp(shade),
                             label=f"servo, delay {delay}", budget_m=budget_m, marker=mk)
        # Torque at the same delay, as a reference line. A dashed line at the ceiling means
        # torque never reached the criterion at that delay.
        t = torque[torque.delay == delay]
        reached = t.steps_to_900.notna().any()
        y = t.steps_to_900.mean() / 1e6 if reached else budget_m
        ax.axhline(y, color=color_for("torque"), lw=1.3, ls="--", alpha=0.85, zorder=2)
        ax.text(90, y + 7, f"torque, delay {delay}" + ("" if reached else " (never)"),
                fontsize=7, color=color_for("torque"), ha="right", va="bottom")
    _censor_band(ax, budget_m)
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 16, 64])
    ax.set_xticklabels(["1", "4", "16", "64"])
    ax.set_xlim(0.8, 95)
    ax.set_ylim(0, budget_m * 1.16)
    ax.set_xlabel("servo_kp")
    ax.set_ylabel("Steps to reach reward 900 (millions)")
    # Monotone at delays 0 and 10; delay 5 is not (kp 8/16 censored, kp 64 slow), which is
    # why the title does not claim "at every delay".
    ax.set_title("Stiffer servos reach the criterion sooner (delay 5 is non-monotone)",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.5)

    # Right: the same at delay 10 for three criteria, so the conclusion is visibly not an
    # artefact of choosing 900 (which is this project's convention, not a standard).
    ax = axes[1]
    ramp = plt.get_cmap("PuRd")
    for thr, shade, mk in ((800, 0.45, "o"), (900, 0.65, "s"), (950, 0.88, "^")):
        g = servo[servo.delay == 10].sort_values("servo_kp")
        solved = g[g[f"steps_to_{thr}"].notna()]
        censored = g[g[f"steps_to_{thr}"].isna()]
        ax.plot(solved.servo_kp, solved[f"steps_to_{thr}"] / 1e6, marker=mk, ls="-",
                ms=6, lw=1.8, color=ramp(shade), label=f"reward {thr}")
        ax.plot(censored.servo_kp, [budget_m] * len(censored), marker=6, ls="none",
                ms=11, color=ramp(shade), zorder=6)
    # Torque at delay 10 reaches none of the three, so it is a caret at every stiffness.
    ax.axhline(budget_m, color=color_for("torque"), lw=1.3, ls="--", alpha=0.9)
    ax.text(70, budget_m * 0.965, "torque, delay 10: never reaches 800 either",
            fontsize=7, color=color_for("torque"), ha="right", va="top")
    _censor_band(ax, budget_m)
    ax.set_xscale("log")
    ax.set_xticks([4, 16, 64])
    ax.set_xticklabels(["4", "16", "64"])
    ax.set_xlim(3, 95)
    ax.set_ylim(0, budget_m * 1.16)
    ax.set_xlabel("servo_kp")
    ax.set_ylabel("Steps to reach criterion (millions)")
    ax.set_title("Delay 10: the ordering does not depend on the criterion", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.5)

    fig.suptitle("Time to criterion, on a common 4.8e6-step eval grid "
                 "(trailing 2.4e7-step mean)", fontsize=11)
    fig.tight_layout()
    provenance(fig, HERE, HERE / "data.csv")
    path = FIGURES / "time_to_900_stiffness.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def convergence_figure(df: pd.DataFrame) -> Path:
    """Why a point on the headline figures may not be a converged level."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    specs = [("late_gain", "Reward gained over the last 1e8 steps",
              "Still improving?  (> 0 = not converged)", False),
             ("max_drawdown_late", "Largest peak-to-trough fall, last 2e8 steps",
              "Stable?  (large = training collapse)", True)]
    marks = {0: "o", 5: "s", 10: "^"}

    for ax, (col, ylabel, title, logy) in zip(axes, specs):
        for delay, mk in marks.items():
            sub = df[df.delay == delay]
            t = sub[sub.condition == "torque"]
            ax.plot([BASELINE_X] * len(t), t[col], mk, color=color_for("torque"), ms=6,
                    ls="none", label=f"torque, delay {delay}")
            s = sub[sub.condition == "servo"].groupby("servo_kp")[col].mean()
            ax.plot(s.index, s.values, marker=mk, ls="-", color=color_for("servo"), ms=6,
                    lw=1.5, alpha=0.45 + 0.055 * delay, label=f"servo, delay {delay}")
        if logy:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color="0.4", lw=1.0)
        ax.set_xscale("log")
        ax.set_xticks([BASELINE_X, 0.25, 1, 4, 16, 64])
        ax.set_xticklabels(["0\n(torque)", "0.25", "1", "4", "16", "64"])
        ax.set_xlim(0.1, 95)
        ax.set_xlabel("servo_kp")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle("At delay 10, torque is still climbing (+45) while servo_kp 48-64 "
                 "have settled", fontsize=11)
    fig.tight_layout()
    provenance(fig, HERE, HERE / "data.csv")
    path = FIGURES / "convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    curves = pd.read_csv(HERE / "curves.csv")

    entries = {
        delay_figure(df).name: "Reward vs delay, per stiffness (headline)",
        stiffness_figure(df).name: "Reward vs servo_kp, per delay",
        curves_figure(curves).name: "Eval-reward learning curves, per delay",
        time_to_delay_figure(df).name: "Steps to reward 900 vs delay, per stiffness",
        time_to_stiffness_figure(df).name:
            "Steps to criterion vs stiffness, and criterion sensitivity at delay 10",
        convergence_figure(df).name: "Late gain and max drawdown vs servo_kp",
    }
    write_figure_manifest(HERE, entries)
    for name in entries:
        print(f"wrote figures/{name}")


if __name__ == "__main__":
    main()
