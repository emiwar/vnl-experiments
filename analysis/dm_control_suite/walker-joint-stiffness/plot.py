"""Figures for the WalkerWalk joint-stiffness pilot. Reads only data.csv / curves.csv.

Three figures, in the order the report argues them:

``stiffness.png``
    The headline: end-of-training reward against ``servo_kp``, one panel per delay, with
    the torque arm as a horizontal reference band rather than a point at x=0. ``servo_kp``
    is on a log axis and the baseline is a *band*, because the servo family does not
    connect continuously to torque control at zero: ``servo_kp = 0`` is torque control at
    full authority, while ``servo_kp = 0+`` is an essentially free joint (see
    ``envs/servo_control.md`` section 1.3). Drawing the baseline as the x=0 point of a
    continuous curve would assert a limit that does not exist.

``curves.png``
    Why the headline panel cannot be read as a converged comparison: the servo arm is
    still climbing at the 4.8e8 budget while the torque arm plateaued long before.

``convergence.png``
    The same fact as a single number per run -- reward gained over the last 1e8 steps --
    so the "not converged" claim is checkable rather than eyeballed off a curve.

Colour follows ``wandb_utils.style``: one condition, one colour, everywhere. In
``curves.png`` stiffness is carried by lightness within a **single hue** (a sequential
ramp, not a rainbow), because ``servo_kp`` there is an ordered magnitude rather than an
identity.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    apply_style, color_for, label_for, marker_for, provenance, reward_label,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

#: WalkerWalk's control step. Never the rodent's 10 ms (track README).
CTRL_DT_MS = 25

#: Where the `servo_kp = 0` baseline is drawn on a log axis. It is a reference *line*, not
#: a data point, so this is a drawing position and nothing more.
BASELINE_X = 0.13

DELAYS = (0, 5)


def _kp_colors(kps):
    """Single-hue sequential ramp over stiffness: an ordered magnitude, not an identity."""
    ramp = plt.get_cmap("PuRd")
    lo, hi = np.log10(min(kps)), np.log10(max(kps))
    return {kp: ramp(0.35 + 0.6 * (np.log10(kp) - lo) / (hi - lo)) for kp in kps}


def stiffness_figure(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(DELAYS), figsize=(10.5, 4.3), sharey=True)
    servo, torque = df[df.condition == "servo"], df[df.condition == "torque"]

    for ax, delay in zip(axes, DELAYS):
        s = servo[servo.delay == delay].sort_values("servo_kp")
        t = torque[torque.delay == delay]

        if len(t):
            lo, hi, mean = t.reward_final.min(), t.reward_final.max(), t.reward_final.mean()
            ax.axhspan(lo, hi, color=color_for("torque"), alpha=0.18, lw=0, zorder=1)
            ax.axhline(mean, color=color_for("torque"), lw=1.6, ls="--", zorder=2,
                       label=f"{label_for('torque')}, n={len(t)}")
            ax.plot([BASELINE_X] * len(t), t.reward_final, marker_for("torque"),
                    color=color_for("torque"), ms=6, zorder=4, clip_on=False)

        cell = s.groupby("servo_kp").reward_final
        ax.plot(cell.mean().index, cell.mean().values, marker=marker_for("servo"),
                ls="-", color=color_for("servo"), ms=7, lw=2.0, zorder=5,
                label=label_for("servo"))
        # A run still climbing steeply, or one caught in a collapse, has no converged
        # level to put on this axis. Ring those points rather than let the marker imply a
        # plateau -- at delay 5 that is most of the servo arm. Thresholds are read off
        # data.csv: the torque runs sit at late_gain < 5 and drawdown < 5.
        unsettled = s[(s.late_gain.abs() > 20) | (s.max_drawdown_late > 50)]
        if len(unsettled):
            ax.plot(unsettled.servo_kp, unsettled.reward_final, "o", mfc="none",
                    mec="0.25", ms=15, mew=1.4, lw=0, zorder=7,
                    label="not converged / unstable" if delay == DELAYS[0] else None)
        # Every seed, so the reader sees the spread the mean was taken over (README §7).
        ax.plot(s.servo_kp, s.reward_final, "o", color=color_for("servo"), ms=3.5,
                alpha=0.55, lw=0, zorder=6,
                label="individual seed" if delay == DELAYS[0] else None)

        ax.set_xscale("log")
        ax.set_xticks([BASELINE_X, 0.25, 0.5, 1, 4, 8, 16])
        ax.set_xticklabels(["0\n(torque)", "0.25", "0.5", "1", "4", "8", "16"])
        ax.set_xlim(0.1, 22)
        ax.axvline(0.18, color="0.75", lw=0.9, ls=":", zorder=0)
        ax.set_xlabel("servo_kp  (dimensionless, normalised per joint)")
        ax.set_title(f"delay {delay}  ({delay * CTRL_DT_MS} ms)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(reward_label("dmc_eval"))
    axes[0].set_ylim(0, 1050)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("WalkerWalk: joint stiffness vs end-of-training reward "
                 "(mean of eval points in the last 5e7 of 4.8e8 steps)", fontsize=11)
    fig.tight_layout()
    provenance(fig, HERE, HERE / "data.csv")
    path = FIGURES / "stiffness.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def curves_figure(curves: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(DELAYS), figsize=(10.5, 4.3), sharey=True)
    kps = sorted(curves[curves.servo_kp > 0].servo_kp.unique())
    colors = _kp_colors(kps)

    for ax, delay in zip(axes, DELAYS):
        sub = curves[curves.delay == delay]
        for seed, g in sub[sub.servo_kp == 0].groupby("seed"):
            g = g.sort_values("step")
            ax.plot(g.step / 1e6, g.reward_mean, color=color_for("torque"), lw=1.8,
                    alpha=0.9, label=label_for("torque") if seed == sub.seed.min() else None)
        for kp in kps:
            for seed, g in sub[sub.servo_kp == kp].groupby("seed"):
                g = g.sort_values("step")
                ax.plot(g.step / 1e6, g.reward_mean, color=colors[kp], lw=1.6,
                        label=f"servo_kp = {kp:g}" if seed == g.seed.min() else None)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_title(f"delay {delay}  ({delay * CTRL_DT_MS} ms)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(reward_label("dmc_eval"))
    axes[0].set_ylim(0, 1050)
    axes[1].legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.92,
                   bbox_to_anchor=(0.02, 0.98))
    fig.suptitle("The servo arm has not converged at 4.8e8 steps; the torque arm has",
                 fontsize=11)
    fig.tight_layout()
    provenance(fig, HERE, HERE / "curves.csv")
    path = FIGURES / "curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def convergence_figure(df: pd.DataFrame) -> Path:
    """Why no stiffness ordering can be read off the headline panel.

    Two things disqualify a run's end-of-training reward from being a *level*: it was still
    climbing, or it fell off a cliff just before the readout. The torque arm does neither,
    which is what makes it the only converged arm here.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    specs = [("late_gain", "Reward gained over the last 1e8 steps",
              "Still improving?  (> 0 = not converged)", False),
             ("max_drawdown_late", "Largest peak-to-trough fall, last 2e8 steps",
              "Stable?  (large = training collapse)", True)]

    for ax, (col, ylabel, title, logy) in zip(axes, specs):
        for delay, ls, mk in ((0, "-", "o"), (5, "--", "s")):
            sub = df[df.delay == delay]
            t = sub[sub.condition == "torque"]
            ax.plot([BASELINE_X] * len(t), t[col], mk, color=color_for("torque"), ms=6,
                    ls="none", label=f"torque, delay {delay}")
            s = sub[sub.condition == "servo"].groupby("servo_kp")[col].mean()
            ax.plot(s.index, s.values, marker=mk, ls=ls, color=color_for("servo"), ms=6,
                    lw=1.8, label=f"servo, delay {delay}")
        if not logy:
            ax.axhline(0, color="0.4", lw=1.0)
        else:
            ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks([BASELINE_X, 0.25, 0.5, 1, 4, 8, 16])
        ax.set_xticklabels(["0\n(torque)", "0.25", "0.5", "1", "4", "8", "16"])
        ax.set_xlim(0.1, 22)
        ax.set_xlabel("servo_kp")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Neither arm's delay-5 reward is a converged level, except torque",
                 fontsize=11)
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
        stiffness_figure(df).name: "End-of-training reward vs servo_kp, per delay",
        curves_figure(curves).name: "Eval-reward learning curves, per delay",
        convergence_figure(df).name: "Reward gained in the last 1e8 steps",
    }
    write_figure_manifest(HERE, entries)
    for name in entries:
        print(f"wrote figures/{name}")


if __name__ == "__main__":
    main()
