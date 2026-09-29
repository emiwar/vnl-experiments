"""Figures for efference-copy-across-tasks (dm_control_suite).

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/efference-copy-across-tasks/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/efference-copy-across-tasks/plot.py

Figures
-------
  reward_vs_delay.png       main: end-of-training reward vs delay, one panel per env
  steps_to_criterion.png    main: time to that panel's 90 %-of-undelayed criterion
  difference_vs_delay.png   the cross-task summary: (no copy - with copy) against delay
  mode_rate.png             how often a run reached the criterion at all (bimodal cells)
  training_curves.png       supp: the eval series, at each panel's largest-effect delay

Conventions this file is careful about
--------------------------------------
**One panel per environment, raw reward on every y-axis, each panel on its own scale.**
That is the track README's resolution of the raw-reward preference for a cohort spanning
tasks whose returns are different quantities: it answers "which arm is better, and by how
much" without dividing through by a per-task ceiling. WalkerWalk appears twice, torque and
servo, because the servo is a different plant and not a parameter of the same one.

**The x axis is delay in steps, with each panel's own millisecond twin.** ``ctrl_dt`` is
10 / 20 / 25 ms across these six panels, so a shared ms axis would be wrong and the
``add_ms_axis`` default (the rodent's 10) would be wrong five times out of six. The value
is read from ``data.csv`` per panel, never assumed.

**The criterion is per panel and is printed in the panel.** 900 is a WalkerWalk convention;
CartpoleSwingup never reaches it at any delay. Each panel's criterion is 90 % of its own
delay-0 anchor, and the panel title carries the absolute number so no reader has to take
"the criterion" on trust.

**The delay-0 runs are a level, not a point.** At ``delay_k = 0`` an efference-matched run
*is* an ``efference_length = 0`` run, so they are drawn as a horizontal reference.

**Cells that are not measured are marked, not omitted.** A delay where an arm has a run
that never finished, or whose eval series stopped early, is flagged with a caret on the
axis; a delay where an arm was never launched is simply a gap in a broken line. Those are
different statements and the figure keeps them different.

**Censoring is drawn, never dropped**, and a cell where only *some* runs reached the
criterion is excluded from the mean line -- a mean over the subset that happened to succeed
is the one number these panels must not show.

**Colour is the arm**, reusing ``CONDITION_STYLE``'s shared ``efference`` / ``no_efference``
entries, so the manipulation is the same colour here as in the rodent's
``../../rodent/proprioceptive-delay-efference/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from vnl_experiments.wandb_utils.style import (
    add_ms_axis,
    apply_style,
    color_for,
    label_for,
    marker_for,
    plot_seeds,
    provenance,
    reward_label,
    seed_legend_handles,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

#: Panel order, matching `extract.PANELS`. Asserted against `data.csv` in `main` so a panel
#: added to the extract cannot be silently missing from every figure.
PANEL_ORDER = ("cartpole_swingup", "cheetah_run", "reacher_hard", "humanoid_walk",
               "walker_walk_torque", "walker_walk_servo")
NCOLS = 3

EFF, NONE, REF = "efference", "no_efference", "undelayed"

#: The headline criterion, as a percentage of the panel's delay-0 anchor. Must match
#: `extract.HEADLINE_FRACTION`; asserted via the column name existing in `data.csv`.
HEADLINE_PCT = 90
FRACTION_PCTS = (80, 90, 95)

#: dm_control returns are bounded above by episode length x a per-step reward in [0, 1].
TASK_MAX = 1000

STEP_M = 1e6

#: `reward_label("dmc_eval")` on one line is ~40 characters and collides with the ylabel of
#: the row above once the grid is three wide and two tall. Same string, wrapped.
_REWARD_YLABEL = "Episode reward\n(eval episodes, unscaled)"

#: Linear region of the difference figure's symlog axis, in reward. Set at the median
#: multi-seed within-cell spread of the two panels that have one at every delay
#: (WalkerWalk torque 4.4, servo 6.6) rounded up an order of magnitude to cover the
#: transitioning cells: a difference inside this band is not worth reading, so the linear
#: region is exactly the "no effect" band.
DIFF_LINTHRESH = 25.0


def _panels(df: pd.DataFrame) -> list[str]:
    return [p for p in PANEL_ORDER if p in set(df["panel"])]


def _label(df: pd.DataFrame, panel: str) -> str:
    return str(df[df["panel"] == panel]["panel_label"].iloc[0])


def _ctrl_dt(df: pd.DataFrame, panel: str) -> float:
    return float(df[df["panel"] == panel]["ctrl_dt_ms"].iloc[0])


def _budget(df: pd.DataFrame, panel: str) -> int:
    return int(df[df["panel"] == panel]["budget"].iloc[0])


def _anchor(df: pd.DataFrame, panel: str) -> float | None:
    v = df[df["panel"] == panel]["anchor"].dropna()
    return float(v.iloc[0]) if len(v) else None


def _thr(df: pd.DataFrame, panel: str, pct: int) -> float | None:
    v = df[df["panel"] == panel][f"thr_{pct}"].dropna()
    return float(v.iloc[0]) if len(v) else None


def _grid(fig_w: float, fig_h: float, n: int, *, sharey: bool = False):
    nrows = -(-n // NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(fig_w * NCOLS, fig_h * nrows),
                             squeeze=False, sharey=sharey)
    flat = [a for row in axes for a in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def _ylabel_left_only(axes, label: str) -> None:
    """Label the leftmost panel of each row only.

    A `reward_label(...)` string is ~40 characters and, repeated on all three columns of a
    three-wide grid, runs into the neighbouring panel's tick labels.
    """
    for i, ax in enumerate(axes):
        ax.set_ylabel(label if i % NCOLS == 0 else "")


def _delay_axis(ax, delays, ctrl_dt_ms: float, min_gap: float = 4.0) -> None:
    """Tick the delays that were actually run, plus a thinned millisecond twin.

    The delay grid is uneven (0/2/3/5/7/10/12/15/20/25), so matplotlib's default locator
    would name delays that do not exist, and the derived ms labels of the 2-and-3 and
    10-and-12 pairs overprint at this panel width. Only the ms labels are thinned; every
    delay that was run keeps its tick.
    """
    ticks = sorted(set(int(d) for d in delays))
    ax.set_xticks(ticks)
    ax2 = add_ms_axis(ax, max(ticks), ctrl_dt_ms=ctrl_dt_ms)
    kept, last = [], -min_gap
    for t in ax2.get_xticks():
        keep = t - last >= min_gap
        kept.append(f"{int(t * ctrl_dt_ms)}" if keep else "")
        if keep:
            last = t
    ax2.set_xticklabels(kept)


def _plot_gapped(ax, sub: pd.DataFrame, arm: str, *, y: str, grid, scale: float = 1.0,
                 mask: pd.Series | None = None):
    """Cell means over ``grid``, with a **break** at every delay this arm does not cover.

    The no-copy arm is one run per delay and is measured at a subset of the delays the
    efference arm covers. A line joining its measured points would draw a curve through
    delays at which nothing was measured, which is the impression these figures must not
    give; reindexing onto the panel's union grid and letting matplotlib break at NaN makes
    every such delay a visible gap instead.
    """
    s = sub[sub["arm"] == arm]
    if mask is not None:
        s = s[mask.reindex(s.index, fill_value=False)]
    means = (s.dropna(subset=[y]).groupby("delay")[y].mean().mul(scale)
             .reindex(sorted(grid)))
    ax.plot(means.index, means.to_numpy(), color=color_for(arm), marker=marker_for(arm),
            ms=5.5, lw=2.2, alpha=0.95, label=label_for(arm))
    return means


def _mark_unmeasured(ax, panel_df: pd.DataFrame) -> bool:
    """Caret on the axis at every delay where an arm has a run without a usable readout.

    Derived from ``data.csv``, so the marks disappear by themselves when the runs finish.
    A delay where an arm was never *launched* gets no caret -- that is a gap in the line,
    and the two facts must not look alike.
    """
    pending = panel_df[~panel_df["usable"].astype(bool)
                       & panel_df["arm"].isin((EFF, NONE))]
    for arm, g in pending.groupby("arm"):
        for d in sorted(set(int(x) for x in g["delay"])):
            # Below the axis, not inside it: at y = 0.02 of the data area these markers sat
            # among real points in the panels where a run scored near zero.
            ax.plot([d], [-0.06], marker="^", ms=5.5, color=color_for(arm), mfc="none",
                    mew=1.2, transform=ax.get_xaxis_transform(), clip_on=False, zorder=7)
    return bool(len(pending))


def fig_reward(df: pd.DataFrame):
    """End-of-training reward vs delay, one panel per environment."""
    panels = _panels(df)
    # Shared y. Every dm_control return here is 1000 steps x a per-step reward in [0, 1],
    # so the six panels are already on one scale and pretending otherwise costs the reader
    # six axis readings. This is not normalisation -- nothing is divided by anything.
    fig, axes = _grid(4.6, 3.9, len(panels), sharey=True)
    any_pending = False
    for ax, panel in zip(axes, panels):
        sub = df[(df["panel"] == panel)]
        usable = sub[sub["usable"].astype(bool)]
        sweep = usable[usable["arm"].isin((EFF, NONE))]
        grid = sorted(sweep["delay"].unique())

        eff = sweep[sweep["arm"] == EFF]
        if not eff.empty:
            plot_seeds(ax, eff, x="delay", y="reward_final", seed_col="seed_pair",
                       condition=EFF, marker_size=5.5)
        _plot_gapped(ax, sweep, NONE, y="reward_final", grid=grid)

        anchor = _anchor(df, panel)
        if anchor is not None:
            n_ref = int((usable["arm"] == REF).sum())
            ax.axhline(anchor, color=color_for(REF), lw=1.1, ls="-.", alpha=0.7, zorder=0)
            ax.text(0.985, anchor, f"no delay (n={n_ref}) ", color=color_for(REF),
                    fontsize=7.0, va="bottom", ha="right",
                    transform=ax.get_yaxis_transform())
        any_pending |= _mark_unmeasured(ax, sub)

        ax.set_title(f"{_label(df, panel)}\nbudget {_budget(df, panel) / 1e6:.0f}e6 steps",
                     fontsize=9.5)
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylim(0, TASK_MAX * 1.05)
        _delay_axis(ax, sweep["delay"], _ctrl_dt(df, panel))
    _ylabel_left_only(axes[:len(panels)], _REWARD_YLABEL)

    handles = [Line2D([], [], color=color_for(a), marker=marker_for(a), lw=2.2,
                      label=label_for(a)) for a in (EFF, NONE)]
    handles += [Line2D([], [], color=color_for(REF), ls="-.", lw=1.1,
                       label="Undelayed reference (delay 0)")]
    handles += seed_legend_handles()
    if any_pending:
        handles += [Line2D([], [], color="0.4", ls="none", marker="^", mfc="none",
                           mew=1.2, ms=6,
                           label="a run at this delay has no usable readout yet")]
    fig.legend(handles=handles, fontsize=8.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("Does a delayed policy need an efference copy? One panel per environment\n"
                 "Raw reward throughout; the axis is shared because every return here is "
                 "1000 steps of a per-step reward in [0, 1]", fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    return fig


def fig_steps(df: pd.DataFrame):
    """Steps to each panel's own criterion, log y, censored runs on the budget line."""
    panels = _panels(df)
    fig, axes = _grid(4.6, 3.9, len(panels))
    col, solved = f"steps_to_{HEADLINE_PCT}", f"solved_{HEADLINE_PCT}"
    censored_any = False
    for ax, panel in zip(axes, panels):
        sub = df[df["panel"] == panel]
        usable = sub[sub["usable"].astype(bool)]
        sweep = usable[usable["arm"].isin((EFF, NONE))]
        grid = sorted(sweep["delay"].unique())
        budget = _budget(df, panel)

        for index, arm in enumerate((EFF, NONE)):
            s = sweep[sweep["arm"] == arm]
            color = color_for(arm)
            if s.empty:
                ax.plot([], [], color=color, marker=marker_for(arm), label=label_for(arm))
                continue
            status = s.groupby("delay")[solved].agg(n_solved="sum", n="count")
            full = status.index[status["n_solved"] == status["n"]]
            part = status.index[(status["n_solved"] > 0)
                                & (status["n_solved"] < status["n"])]
            if arm == NONE:
                _plot_gapped(ax, sweep, arm, y=col, grid=grid, scale=1 / STEP_M,
                             mask=sweep["delay"].isin(full))
            elif len(full):
                plot_seeds(ax, s[s["delay"].isin(full)], x="delay", y=col,
                           seed_col="seed_pair", condition=arm, scale=1 / STEP_M,
                           marker_size=5.5)
            else:
                ax.plot([], [], color=color, marker=marker_for(arm), label=label_for(arm))
            # A partially censored cell: the runs that did solve, individually, never a
            # mean over them.
            for d in part:
                got = s[(s["delay"] == d) & s[solved].astype(bool)]
                ax.scatter(np.full(len(got), d + (index - 0.5) * 0.5),
                           got[col].astype(float) / STEP_M, facecolors=color,
                           edgecolors=color, marker=marker_for(arm), s=24, zorder=5)
            for d, g in s[~s[solved].astype("boolean").fillna(False)].groupby("delay"):
                ax.scatter(np.full(len(g), d + (index - 0.5) * 0.5),
                           np.full(len(g), budget / STEP_M), facecolors="none",
                           edgecolors=color, marker=marker_for(arm), s=34, lw=1.3,
                           zorder=6)
                censored_any = True

        ax.axhline(budget / STEP_M, color="0.55", lw=0.8, ls="--", zorder=0)
        ax.text(0.015, budget / STEP_M, f" {budget / 1e6:.0f}e6 budget", color="0.45",
                fontsize=7.0, va="bottom", ha="left", transform=ax.get_yaxis_transform())
        ref = df[(df["panel"] == panel) & (df["arm"] == REF) & df["usable"]][col].dropna()
        if len(ref):
            ax.axhline(float(ref.astype(float).mean()) / STEP_M, color=color_for(REF),
                       lw=1.1, ls="-.", alpha=0.7, zorder=0)
        _mark_unmeasured(ax, sub)

        thr = _thr(df, panel, HEADLINE_PCT)
        ax.set_yscale("log")
        ax.set_ylim(15, budget / STEP_M * 2.6)
        ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000, 2000]))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_minor_formatter(lambda v, _: "")
        ax.set_title(f"{_label(df, panel)}\ncriterion = {HEADLINE_PCT} % of its own "
                     f"delay-0 anchor" + (f" ({thr:.0f} reward)" if thr else ""),
                     fontsize=9.5)
        ax.set_xlabel("Observation delay (steps)")
        _delay_axis(ax, sweep["delay"], _ctrl_dt(df, panel))
    _ylabel_left_only(axes[:len(panels)], "Steps to criterion ($\\times 10^6$, log)")

    handles = [Line2D([], [], color=color_for(a), marker=marker_for(a), lw=2.2,
                      label=label_for(a)) for a in (EFF, NONE)]
    handles += [Line2D([], [], color=color_for(REF), ls="-.", lw=1.1,
                       label="Undelayed reference (delay 0)")] + seed_legend_handles()
    if censored_any:
        handles += [Line2D([], [], color="0.4", ls="none", marker="o", mfc="none",
                           mew=1.3, ms=6,
                           label="never reached criterion within the budget")]
    fig.legend(handles=handles, fontsize=8.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("Time to reach each environment's own criterion\n"
                 "The budgets differ by panel, so a slow panel is not slower than a fast "
                 "one -- read within a panel", fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    return fig


def _diff_rows(df: pd.DataFrame, panel: str, *, paired: bool):
    """(ms, difference) per delay for one panel, from cell means or from matched seeds.

    ``paired=True`` keeps only delays where one seed pair appears in *both* arms and
    differences that pair alone. Where it exists it is strictly better evidence: the
    efference cells of ReacherHard span 1.1e2-1.8e2 reward across seeds, which is larger
    than the cell-mean differences there, so an unpaired contrast at that panel cannot
    resolve its own effect and a paired one can.
    """
    sub = df[df["usable"].astype(bool) & (df["panel"] == panel)
             & df["arm"].isin((EFF, NONE))]
    ctrl = _ctrl_dt(df, panel)
    rows = []
    if paired:
        for (delay, seed), g in sub.groupby(["delay", "seed_pair"]):
            if set(g["arm"]) != {EFF, NONE}:
                continue
            e = g[g["arm"] == EFF]["reward_final"]
            n = g[g["arm"] == NONE]["reward_final"]
            if e.isna().any() or n.isna().any():
                continue
            rows.append((delay * ctrl, float(n.iloc[0]) - float(e.iloc[0])))
    else:
        for delay, g in sub.groupby("delay"):
            e = g[g["arm"] == EFF]["reward_final"]
            n = g[g["arm"] == NONE]["reward_final"]
            if e.empty or n.empty or e.isna().any() or n.isna().any():
                continue
            rows.append((delay * ctrl, n.mean() - e.mean()))
    return sorted(rows)


def _diff_panel(ax, df: pd.DataFrame, *, paired: bool, linthresh: float):
    markers = ("o", "s", "^", "v", "D", "P")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, panel in enumerate(_panels(df)):
        rows = _diff_rows(df, panel, paired=paired)
        if not rows:
            continue
        ms, diff = zip(*rows)
        ax.plot(ms, diff, marker=markers[i % len(markers)], ms=6,
                color=colors[i % len(colors)], lw=1.9, label=_label(df, panel))
    ax.axhline(0, color="0.5", lw=1.0, zorder=0)
    # Symmetric log. One point -- HumanoidWalk at delay 7 -- is -7.9e2 where most of the
    # cohort lives inside +-1e2, and on a linear axis it flattens every other curve onto
    # the zero line.
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.8)
    ax.set_yticks([-800, -400, -200, -100, -50, 0, 50, 100, 200, 400])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.axhspan(-linthresh, linthresh, color="0.5", alpha=0.10, zorder=0, lw=0)
    ax.set_xlabel("Observation delay (ms)")
    ax.set_ylim(-1600, 900)


def fig_difference(df: pd.DataFrame):
    """The cross-task summary: (no copy - with copy), by cell mean and by matched seed.

    A **derived** y axis, which README §7 asks to be justified rather than assumed. Two
    reasons it is the right one here and the small multiples are not enough. First, the
    question this folder asks is whether the answer is a property of *delay* or of the
    *task*, and that is a statement about six curves at once, which six panels cannot make.
    Second, a difference of two returns is still in return units, and these six tasks share
    the same return scale by construction -- 1000 steps x a per-step reward in [0, 1] -- so
    100 reward means the same fraction of an episode in every panel. That is *not* true of
    a ratio, which is why this is a difference and why the small multiples keep raw reward.

    Two panels, because the left one is the weaker evidence. Cell means average over seeds
    that the two arms do not share; the right panel keeps only the delays where one seed
    pair ran both arms, and differences that pair alone.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True)
    _diff_panel(axes[0], df, paired=False, linthresh=DIFF_LINTHRESH)
    _diff_panel(axes[1], df, paired=True, linthresh=DIFF_LINTHRESH)
    axes[0].set_title("Cell means\n(the two arms need not share a seed)", fontsize=10)
    axes[1].set_title("Matched seed only\n(same initialisation and PPO stream, both arms)",
                      fontsize=10)
    # A panel absent from the right-hand axes has no shared seed, which is a fact about
    # how the launches were seeded and not a gap in the data. Derived, so it corrects
    # itself as seeds are added.
    unpaired = [_label(df, k) for k in _panels(df) if not _diff_rows(df, k, paired=True)]
    if unpaired:
        axes[1].text(0.985, 0.025, "no shared seed, so absent here:\n"
                     + "\n".join(unpaired), transform=axes[1].transAxes, fontsize=7.5,
                     va="bottom", ha="right", color="0.35")
    axes[0].set_ylabel("(no copy) − (with copy), end-of-training reward  (symlog)")
    for ax in axes:
        ax.text(0.015, 0.975, "above 0: the copy HURT", transform=ax.transAxes,
                fontsize=8.5, va="top", ha="left", color="0.3")
        ax.text(0.015, 0.025, "below 0: the copy HELPED", transform=ax.transAxes,
                fontsize=8.5, va="bottom", ha="left", color="0.3")
    axes[0].text(0.985, DIFF_LINTHRESH, f"|diff| < {DIFF_LINTHRESH:.0f} ", color="0.45",
                 fontsize=7.5, va="bottom", ha="right",
                 transform=axes[0].get_yaxis_transform())
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, fontsize=8.5, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Only ReacherHard shows a consistent effect — and there the copy "
                 "consistently HURTS\nElsewhere the differences are inside noise at short "
                 "delay and sign-inconsistent at long delay", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.92))
    return fig


def _largest_effect_delay(df: pd.DataFrame, panel: str) -> int | None:
    """The delay in ``panel`` where the two arms differ most in reward.

    Chosen from the data rather than written down, so the supplementary figure keeps
    pointing at the interesting delay as runs are added.
    """
    usable = df[df["usable"].astype(bool) & (df["panel"] == panel)]
    best, best_gap = None, -1.0
    for delay, g in usable.groupby("delay"):
        eff = g[g["arm"] == EFF]["reward_final"]
        non = g[g["arm"] == NONE]["reward_final"]
        if eff.empty or non.empty or eff.isna().any() or non.isna().any():
            continue
        gap = abs(non.mean() - eff.mean())
        if gap > best_gap:
            best, best_gap = int(delay), gap
    return best


def fig_training_curves(curves: pd.DataFrame, df: pd.DataFrame):
    """The eval series, at each panel's largest-effect delay."""
    panels = _panels(df)
    fig, axes = _grid(4.6, 3.6, len(panels), sharey=True)
    for ax, panel in zip(axes, panels):
        delay = _largest_effect_delay(df, panel)
        budget = _budget(df, panel)
        if delay is None:
            ax.text(0.5, 0.5, "no delay has both arms", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="0.45")
        else:
            cell = curves[(curves["panel"] == panel) & (curves["delay"] == delay)]
            for arm in (EFF, NONE):
                for _, run in cell[cell["arm"] == arm].groupby("wandb_id"):
                    ax.plot(run["step"] / budget, run["reward_mean"],
                            color=color_for(arm), lw=0.9, alpha=0.85)
            thr = _thr(df, panel, HEADLINE_PCT)
            if thr:
                ax.axhline(thr, color="0.55", lw=0.8, ls="--", zorder=0)
            n = df[(df["panel"] == panel) & (df["delay"] == delay)
                   & df["usable"]].groupby("arm").size()
            ax.set_title(f"{_label(df, panel)}\ndelay {delay} "
                         f"({delay * _ctrl_dt(df, panel):.0f} ms) — "
                         f"copy n={n.get(EFF, 0)}, none n={n.get(NONE, 0)}", fontsize=9)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, TASK_MAX * 1.05)
        ax.set_xlabel("Fraction of this panel's budget")

    _ylabel_left_only(axes[:len(panels)], _REWARD_YLABEL)
    handles = [Line2D([], [], color=color_for(a), lw=1.8, label=label_for(a))
               for a in (EFF, NONE)]
    handles += [Line2D([], [], color="0.55", lw=0.8, ls="--",
                       label=f"that panel's {HEADLINE_PCT} % criterion")]
    fig.legend(handles=handles, fontsize=8.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("Every run's eval series at the delay where the two arms differ most\n"
                 "x is the fraction of each panel's own budget, because the budgets differ",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    return fig


def fig_mode_rate(df: pd.DataFrame):
    """Fraction of each cell's runs that reached the criterion, one panel per environment.

    Added 2026-09-29, because HumanoidWalk made the mean unusable there. From delay 5 that
    panel's cells are bimodal in **both** arms -- individual runs at 4.4e1 beside others at
    9.4e2 -- so a cell mean is a weighted average of "solved" and "never got going" and
    describes no run in the cell. The rate is a quantity every run in the cell has.

    Drawn for all six panels rather than only the bimodal one, because the comparison is
    the point: in ReacherHard and both WalkerWalk panels nearly every marker sits at 0 or 1
    and the mean is the right readout, and the figure shows *which* panels the distinction
    matters for rather than asserting it. The n behind each marker is annotated, because a
    rate of 2/3 and a rate of 20/30 are the same height and not the same evidence.
    """
    panels = _panels(df)
    fig, axes = _grid(4.6, 3.5, len(panels), sharey=True)
    key = f"solved_{HEADLINE_PCT}"
    for ax, panel in zip(axes, panels):
        sub = df[(df["panel"] == panel) & df["usable"].astype(bool)
                 & df["arm"].isin((EFF, NONE))]
        for arm in (EFF, NONE):
            s = sub[sub["arm"] == arm]
            if s.empty:
                continue
            g = s.groupby("delay")[key].agg(
                rate=lambda v: v.astype("boolean").fillna(False).mean(),
                n="size")
            ax.plot(g.index, g["rate"], color=color_for(arm), marker=marker_for(arm),
                    ms=5.5, lw=2.0, alpha=0.95, label=label_for(arm))
            for delay, row in g.iterrows():
                ax.annotate(f"{int(round(row['rate'] * row['n']))}/{int(row['n'])}",
                            (delay, row["rate"]), textcoords="offset points",
                            xytext=(0, 7 if arm == EFF else -12), ha="center",
                            fontsize=6.2, color=color_for(arm))
        ax.set_ylim(-0.12, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{_label(df, panel)}\ncriterion = {HEADLINE_PCT} % of its delay-0 "
                     f"anchor", fontsize=9.5)
        ax.set_xlabel("Observation delay (steps)")
        if not sub.empty:
            _delay_axis(ax, sub["delay"], _ctrl_dt(df, panel))
    _ylabel_left_only(axes[:len(panels)],
                      "Fraction of the cell's runs\nreaching the criterion")
    handles = [Line2D([], [], color=color_for(a), marker=marker_for(a), lw=2.0,
                      label=label_for(a)) for a in (EFF, NONE)]
    fig.legend(handles=handles, fontsize=8.5, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("How often did a run get there at all?\n"
                 "The readout to prefer wherever a cell is bimodal — which in this cohort "
                 "is HumanoidWalk from delay 5 on", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)

    sources = df["reward_source"].dropna().unique()
    if len(sources) > 1:
        raise SystemExit(
            f"data.csv mixes reward sources {list(sources)}; before the `971ab99` "
            "eval-env fix the eval series was scaled x10 and truncated by the training "
            "EpisodeWrapper, so the two are in different units and different episode "
            "lengths. Split the figures or restrict the cohort.")

    missing = set(df["panel"]) - set(PANEL_ORDER)
    if missing:
        raise SystemExit(
            f"data.csv has panels {sorted(missing)} that PANEL_ORDER does not list, so "
            f"they would be silently absent from every figure. Add them to PANEL_ORDER.")

    if f"thr_{HEADLINE_PCT}" not in df.columns:
        raise SystemExit(
            f"data.csv has no thr_{HEADLINE_PCT} column: extract.HEADLINE_FRACTION and "
            f"plot.HEADLINE_PCT have drifted apart, and every criterion label here would "
            f"name a level the numbers were not read at.")

    # One budget per panel, or the budget line and the x-normalised curves are both lies.
    for panel, g in df.groupby("panel"):
        if g["budget"].nunique() != 1:
            raise SystemExit(f"panel {panel} spans budgets "
                             f"{sorted(g['budget'].unique())}; it must be exactly one.")
        if g["ctrl_dt_ms"].nunique() != 1:
            raise SystemExit(f"panel {panel} spans ctrl_dt_ms "
                             f"{sorted(g['ctrl_dt_ms'].unique())}; it must be exactly one.")

    manifest = {}
    for name, builder, inputs in [
        ("reward_vs_delay", lambda: fig_reward(df), (DATA,)),
        ("steps_to_criterion", lambda: fig_steps(df), (DATA,)),
        ("difference_vs_delay", lambda: fig_difference(df), (DATA,)),
        ("mode_rate", lambda: fig_mode_rate(df), (DATA,)),
        ("training_curves", lambda: fig_training_curves(curves, df), (CURVES, DATA)),
    ]:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
