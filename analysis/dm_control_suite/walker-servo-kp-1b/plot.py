"""Figures for walker-servo-kp-1b. Reads only data.csv and curves.csv.

    ../.venv/bin/python analysis/dm_control_suite/walker-servo-kp-1b/plot.py

Conventions: raw reward on every reward axis (README §7), named by ``reward_label``;
torque is C1 and servo C3 (``CONDITION_STYLE``); runs without an efference copy are
drawn hollow and never pooled with those that have one; a run that never reached the
criterion is a caret at its own budget, never a missing point.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from vnl_experiments.wandb_utils.style import (
    add_ms_axis,
    apply_style,
    color_for,
    provenance,
    reward_label,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

CTRL_DT_MS = 25          # WalkerWalk; env_params.ctrl_dt = 0.025, asserted in main
THR = 900
T_COL = f"steps_to_{THR}"
NEW_SWEEP_DELAYS = (0, 5, 7)

#: The predecessor's settling-time account: the hip servo (slowest joint) settles in
#: 195.655 ms at servo_kp = 16 and t_settle ~ kp**-0.5, so it beats a d-step delay above
#: kp* = 16 * (195.655 / (25 d))**2. Recomputed from data.csv in `crossover_kp`.
TORQUE = color_for("torque")
SERVO = color_for("servo")
KP_CMAP = plt.get_cmap("Reds")


def crossover_kp(df: pd.DataFrame, delay: int) -> float | None:
    """The servo_kp at which the slowest joint settles in exactly ``delay`` control steps,
    from the settling times recorded in the runs (t_settle * sqrt(kp) is constant)."""
    if delay == 0:
        return None
    s = df.dropna(subset=["settling_max_s"])
    const = (s["settling_max_s"] * np.sqrt(s["servo_kp"])).median()
    return float((const / (delay * CTRL_DT_MS / 1000)) ** 2)


def kp_color(kp: float):
    # Start at 0.45: the light end of Reds is too close to torque's orange.
    return KP_CMAP(0.45 + 0.55 * LogNorm(0.25, 512)(kp))


def _torque_band(ax, vals: pd.Series, label: str):
    vals = vals.dropna()
    if vals.empty:
        return
    ax.axhspan(vals.min(), vals.max(), color=TORQUE, alpha=0.15, lw=0)
    ax.axhline(vals.median(), color=TORQUE, lw=1.5, ls="--",
               label=f"{label} (n={len(vals)}: band = range, dashed = median)")


def _kp_axis(ax):
    ax.set_xscale("log", base=2)
    ticks = [0.25, 1, 4, 16, 64, 256]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xlim(0.18, 720)
    ax.set_xlabel("servo_kp (dimensionless stiffness)")


# --------------------------------------------------------------------------------------

def fig_reward_vs_kp(df: pd.DataFrame) -> plt.Figure:
    """1e9 reward_final against servo_kp at the two delays of the new sweep; full range
    on top, the solved band zoomed below."""
    d = df[(df.budget == 1_000_000_000)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    for col, delay in enumerate(NEW_SWEEP_DELAYS):
        sub = d[d.delay == delay]
        for row, ax in enumerate(axes[:, col]):
            _torque_band(ax, sub[(sub.condition == "torque") & (sub.efference == "copy")]
                         ["reward_final"], "torque, 1e9")
            servo = sub[(sub.condition == "servo") & (sub.efference == "copy")]
            med = servo.groupby("servo_kp")["reward_final"].median()
            if row == 0:   # in the zoom a connecting line only dives off-axis
                ax.plot(med.index, med.values, color=SERVO, lw=1.2, alpha=0.6, zorder=2)
            new = servo[servo.is_new_sweep]
            old = servo[~servo.is_new_sweep]
            ax.scatter(new.servo_kp, new.reward_final, color=SERVO, marker="D", s=34,
                       edgecolor="white", lw=0.8, zorder=4, label="servo, new sweep (i45/p45)")
            ax.scatter(old.servo_kp, old.reward_final, color=SERVO, marker="o", s=26,
                       alpha=0.6, edgecolor="white", lw=0.6, zorder=3,
                       label="servo, other seeds (i51-i54)")
            none = sub[sub.efference == "none"]
            for cond, g in none.groupby("condition"):
                x = g.servo_kp.where(g.servo_kp > 0, 0.25)
                ax.scatter(x if cond == "servo" else np.full(len(g), 0.21), g.reward_final,
                           facecolor="none", edgecolor=color_for(cond), s=30, zorder=3,
                           label=f"{cond}, no efference copy" if row == 0 else None)
            if row == 1:
                ax.set_ylim(935, 985)
                _kp_axis(ax)
            else:
                ax.set_ylim(0, 1020)
                ax.set_title(f"delay {delay} ({delay * CTRL_DT_MS} ms)", fontsize=10)
            kstar = crossover_kp(df, delay)
            if kstar:
                ax.axvline(kstar, color="0.4", ls=":", lw=1)
                if row == 0:
                    ax.text(kstar * 1.08, 30, f"hip servo settles\nin {delay * CTRL_DT_MS} ms "
                            f"(kp≈{kstar:.0f})", fontsize=7, color="0.3")
            if col == 0:
                ax.set_ylabel(reward_label("dmc_eval") + ("\n(zoom)" if row else ""),
                              fontsize=8)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0, -1].legend(uniq.values(), uniq.keys(), fontsize=6.5, loc="center right",
                      frameon=False)
    fig.suptitle("1e9 steps: final reward (mean of last 5e7 steps) vs stiffness",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_time_vs_kp(df: pd.DataFrame) -> plt.Figure:
    """steps_to_900 against servo_kp, both budgets; censored runs as carets."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, delay in zip(axes, NEW_SWEEP_DELAYS):
        sub = df[(df.delay == delay) & (df.efference == "copy")]
        _torque_band(ax, sub[sub.condition == "torque"][T_COL] / 1e6,
                     "torque, both budgets")
        servo = sub[sub.condition == "servo"]
        for budget, marker, fill, lab in [
                (1_000_000_000, "D", True, "servo, 1e9"),
                (480_000_000, "s", False, "servo, 4.8e8 (walker-joint-stiffness)")]:
            g = servo[servo.budget == budget]
            hit = g[g[T_COL].notna()]
            cen = g[g[T_COL].isna()]
            kw = dict(color=SERVO) if fill else dict(facecolor="none", edgecolor=SERVO)
            ax.scatter(hit.servo_kp, hit[T_COL] / 1e6, marker=marker, s=34, zorder=4,
                       label=lab, **kw)
            ax.scatter(cen.servo_kp, np.full(len(cen), budget / 1e6), marker="^", s=40,
                       zorder=4, **kw)
        # Censored torque runs, if any, at their budget on the left edge.
        tc = sub[(sub.condition == "torque") & sub[T_COL].isna()]
        ax.scatter(np.full(len(tc), 0.21), tc.budget / 1e6, marker="^", color=TORQUE, s=40)
        for b in (480, 1000):
            ax.axhline(b, color="0.75", lw=0.7, ls="-", zorder=0)
        kstar = crossover_kp(df, delay)
        if kstar:
            ax.axvline(kstar, color="0.4", ls=":", lw=1)
            ax.text(kstar * 1.08, 22, f"hip servo settles in\n{delay * CTRL_DT_MS} ms "
                    f"(kp≈{kstar:.0f})", fontsize=7, color="0.3")
        ax.set_yscale("log")
        ax.set_ylim(18, 1300)
        ax.set_yticks([25, 50, 100, 200, 480, 1000], ["25", "50", "100", "200", "480", "1000"])
        _kp_axis(ax)
        ax.set_title(f"delay {delay} ({delay * CTRL_DT_MS} ms)", fontsize=10)
    axes[0].set_ylabel(f"Steps to reward {THR} (M)\n(trailing 2.4e7 mean, eval episodes)",
                       fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="^", color="0.4", ls="", ms=6))
    labels.append("never reached (drawn at its budget)")
    axes[0].legend(handles, labels, fontsize=6.5, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_delay_summary(df: pd.DataFrame) -> plt.Figure:
    """Torque vs servo_kp = 64 across delays 0/5/7/10 in all three readouts."""
    sub = df[df.servo_kp.isin([0.0, 64.0])]
    panels = [
        ("reward_480", "Reward at matched 4.8e8 budget\n(all runs, both budgets)", None),
        ("reward_final", "Reward at 1e9 (final 5e7 window)", 1_000_000_000),
        (T_COL, f"Steps to reward {THR} (M)", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    rng = np.random.default_rng(0)
    for ax, (col, title, budget) in zip(axes, panels):
        p = sub if budget is None else sub[sub.budget == budget]
        for cond, offset in (("torque", -0.18), ("servo", 0.18)):
            for eff, filled in (("copy", True), ("none", False)):
                g = p[(p.condition == cond) & (p.efference == eff)]
                if g.empty:
                    continue
                x = g.delay + offset + (0.0 if filled else 0.12 * np.sign(offset)) \
                    + rng.uniform(-0.05, 0.05, len(g))
                y = g[col] / (1e6 if col == T_COL else 1)
                if col == T_COL:
                    y = y.fillna(g.budget / 1e6)
                    cen = g[col].isna()
                    kw = (dict(color=color_for(cond)) if filled
                          else dict(facecolor="none", edgecolor=color_for(cond)))
                    ax.scatter(x[~cen], y[~cen], s=22, zorder=3, **kw)
                    ax.scatter(x[cen], y[cen], s=38, marker="^", zorder=3, **kw)
                else:
                    kw = (dict(color=color_for(cond)) if filled
                          else dict(facecolor="none", edgecolor=color_for(cond)))
                    ax.scatter(x, y, s=22, zorder=3, **kw)
                if filled:
                    # Median of the efference-copy runs; censored counted at +inf, so the
                    # median is honest when fewer than half are censored.
                    vals = g.assign(v=(g[col].fillna(np.inf) / 1e6) if col == T_COL else g[col])
                    med = vals.groupby("delay")["v"].median()
                    med = med[np.isfinite(med)]
                    ax.plot(med.index + offset, med.values, color=color_for(cond), lw=1.8,
                            label=("torque (servo_kp = 0)" if cond == "torque"
                                   else "servo_kp = 64"))
        ax.set_xticks([0, 5, 7, 10])
        ax.set_xlabel("Observation delay (control steps)")
        ax.set_title(title, fontsize=9)
        add_ms_axis(ax, 10.5, ctrl_dt_ms=CTRL_DT_MS)
        if col == T_COL:
            ax.set_yscale("log")
            ax.set_yticks([25, 50, 100, 200, 480, 1000],
                          ["25", "50", "100", "200", "480", "1000"])
            ax.set_ylabel(f"Steps to {THR} (M)", fontsize=8)
        else:
            ax.set_ylabel(reward_label("dmc_eval"), fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", color="0.4", ls="", ms=5),
                Line2D([], [], marker="o", mfc="none", mec="0.4", ls="", ms=5),
                Line2D([], [], marker="^", color="0.4", ls="", ms=6)]
    labels += ["efference copy (line = median)", "no efference copy",
               "never reached (at its budget)"]
    axes[0].legend(handles, labels, fontsize=6.5, loc="lower left", frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_curves(curves: pd.DataFrame) -> plt.Figure:
    """Trailing-mean eval curves at 1e9: the new sweep by stiffness, torque seeds in C1."""
    c = curves[(curves.budget == 1_000_000_000) & (curves.efference == "copy")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, delay in zip(axes, NEW_SWEEP_DELAYS):
        sub = c[c.delay == delay]
        for wid, g in sub[sub.condition == "torque"].groupby("wandb_id"):
            ax.plot(g.step / 1e6, g.reward_smooth, color=TORQUE, lw=1.3, ls="--")
        servo = sub[(sub.condition == "servo") & (sub.seed_pair == "i45/p45")]
        for kp, g in servo.groupby("servo_kp"):
            ax.plot(g.step / 1e6, g.reward_smooth, color=kp_color(kp), lw=1.4)
            last = g.dropna(subset=["reward_smooth"]).iloc[-1]
            if last.reward_smooth < THR:   # the solved ones overlap; named once below
                ax.text(1008, last.reward_smooth, f"{kp:g}", fontsize=6.5, va="center",
                        color="0.25")
        solved = servo.groupby("servo_kp")["reward_smooth"].last()
        solved = solved[solved >= THR].index
        if len(solved):
            ax.text(1008, 1000, "kp " + ", ".join(f"{k:g}" for k in solved),
                    fontsize=5.5, va="bottom", ha="right", color="0.25")
        ax.axhline(THR, color="0.6", lw=0.8, ls=":")
        ax.set_xlim(0, 1060)
        ax.set_xlabel("Environment steps (M)")
        ax.set_title(f"delay {delay} ({delay * CTRL_DT_MS} ms)", fontsize=10)
    axes[0].set_ylabel(reward_label("dmc_eval") + "\ntrailing 2.4e7-step mean", fontsize=8)
    handles = [Line2D([], [], color=TORQUE, lw=1.3, ls="--", label="torque, one line per seed"),
               Line2D([], [], color=kp_color(64), lw=1.4,
                      label="servo, new sweep (i45/p45);\nlabel at right = servo_kp, "
                            "darker = stiffer"),
               Line2D([], [], color="0.6", lw=0.8, ls=":", label=f"criterion ({THR})")]
    axes[0].legend(handles=handles, fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    if set(df.reward_source) != {"dmc_eval"}:
        raise SystemExit("data.csv mixes reward sources")
    if not np.allclose(df.ctrl_dt, CTRL_DT_MS / 1000):
        raise SystemExit("ctrl_dt is not 25 ms on every run")

    manifest = {}
    for name, builder, inputs in [
            ("reward_vs_kp", lambda: fig_reward_vs_kp(df), (DATA,)),
            ("time_vs_kp", lambda: fig_time_vs_kp(df), (DATA,)),
            ("delay_summary", lambda: fig_delay_summary(df), (DATA,)),
            ("curves", lambda: fig_curves(curves), (CURVES,))]:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")
    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
