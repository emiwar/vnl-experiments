"""Figures for walker-servo-x-forward-model (dm_control_suite, WalkerWalk).

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/walker-servo-x-forward-model/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/walker-servo-x-forward-model/plot.py

Figures
-------
  reward_and_steps_vs_delay.png  main: both readouts vs delay, all four arms
  interaction.png                main: the 2x2 as simple effects + their difference
  training_curves.png            supp: the eval series both readouts come from
  threshold_sensitivity.png      supp: time-to-criterion at 800 / 950 as well as 900

Conventions this file is careful about
--------------------------------------
**Raw reward on the reward y-axis** (README §7). One task, one reward function, four arms
-- nothing to normalise. The task maximum (1e3, by construction) is drawn as a reference
line rather than divided through.

**The interaction panels are differences of raw reward, not ratios.** A difference of two
rewards is still in reward units and inherits whatever nonlinearity reward has, which is
exactly what §7's objection to ratios is about -- so the raw four-arm panel is kept beside
it and the report quotes cell values, not only the difference.

**The time-to-criterion axis is logarithmic**, and that is not the same concession. Steps
are a physical count, so ratios are meaningful, and the quantity spans 2.9e1 to 8.2e2
million. Note the consequence for the mean line: an arithmetic mean drawn on a log axis
sits above the visual midpoint of its seeds.

**Censoring is drawn, never dropped.** A run that never reaches the criterion has no
time-to-criterion. Left out, the two MLP lines would simply stop at delay 10 -- which
reads as missing data when it is in fact the strongest thing in the figure. Censored runs
are parked on the budget line as open markers. A cell where only *some* runs miss the
criterion is excluded from the mean line, which therefore breaks: a mean over the subset
that happened to succeed is the one number this panel must not show.

**Colour is the arm, not the factor.** Each single-manipulation corner keeps the colour it
has as a condition in the sibling folders (``mlp_torque`` C1, ``fm_torque`` C2,
``mlp_servo`` C3) and only the combination is new -- see ``CONDITION_STYLE``.
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

#: Drawing order, chosen so the two baselines are underneath and the 2x2 reads
#: baseline -> one manipulation -> one manipulation -> both.
ARMS = ["mlp_torque", "mlp_servo", "fm_torque", "fm_servo"]

BUDGET = 1_000_000_000
SERVO_KP = 64

#: WalkerWalk runs at ctrl_dt = 0.025 s, so one delay step is 25 ms -- **not** the
#: rodent's 10, which is what `add_ms_axis` defaults to. Asserted against `data.csv`.
CTRL_DT_MS = 25

#: The headline criterion and the two it is checked against. None is a published
#: standard; see `extract.py`.
HEADLINE_THRESHOLD = 900
THRESHOLDS = (800, 900, 950)

#: dm_control returns are bounded above by episode length x a per-step reward in [0, 1].
#: Drawn as a reference line, never used as a denominator (README §7).
TASK_MAX = 1000

#: Delays for the training-curve panels: where the arms separate (12), where the MLP arms
#: fail (15, 20), and where the *forward-model* arm starts to (25).
CURVE_DELAYS = (12, 15, 20, 25)

STEP_M = 1e6  # steps -> "millions of steps"


def _steps_col(thr: int) -> str:
    return f"steps_to_{thr}"


def _solved_col(thr: int) -> str:
    return f"solved_{thr}"


#: Two ms labels closer together than this (in delay steps) collide, so the later one is
#: blanked. The delay axis below still ticks every delay. Panel width decides it: the
#: two-panel figures get 2.0, the three-panel one 3.5.
MS_LABEL_MIN_GAP = 2.0
MS_LABEL_MIN_GAP_NARROW = 3.5


def _ms_axis(ax, delays, min_gap: float = MS_LABEL_MIN_GAP) -> None:
    """The ms twin axis, with colliding labels thinned rather than overprinted.

    The delay grid is uneven -- 0/2/3/5/7/10/12/15/20/25 -- so the 2-and-3 and 10-and-12
    pairs put their millisecond labels 25 ms and 50 ms apart on an axis that spans 625,
    and they overprint. The *delay* axis keeps a tick at every delay that was run; only
    the derived ms labels are thinned.
    """
    ax2 = add_ms_axis(ax, max(delays), ctrl_dt_ms=CTRL_DT_MS)
    kept, last = [], -min_gap
    for t in ax2.get_xticks():
        keep = t - last >= min_gap
        kept.append(f"{int(t * CTRL_DT_MS)}" if keep else "")
        if keep:
            last = t
    ax2.set_xticklabels(kept)


def _integer_delay_ticks(ax, df: pd.DataFrame) -> None:
    """Tick the delays that were actually run.

    The grid is 0/2/5/7/10/15/20, which matplotlib's default locator renders on a round
    2.5-step grid -- an axis whose labels name delays that do not exist.
    """
    ax.set_xticks(sorted(df["delay"].unique()))


def _spread(delay: float, n: int, arm: str, width: float = 0.5) -> np.ndarray:
    """``n`` x-positions near ``delay``, so co-located markers stay countable.

    Censored runs at one delay sit at the same point, so without this two censored seeds
    are one marker and the panel understates how many runs failed.
    """
    centre = delay + (ARMS.index(arm) - (len(ARMS) - 1) / 2) * width
    if n <= 1:
        return np.full(n, centre)
    return centre + np.linspace(-width / 2, width / 2, n)


def _reward_panel(ax, df: pd.DataFrame) -> None:
    """Raw reward at 1e9 vs delay, one line per arm."""
    for arm in ARMS:
        sub = df[df["condition"] == arm]
        if sub.empty:
            continue
        plot_seeds(ax, sub, x="delay", y="reward_final", seed_col="seed_pair",
                   condition=arm, marker_size=5)

    ax.axhline(TASK_MAX, color="0.55", lw=0.8, ls=":", zorder=0)
    ax.text(0.985, TASK_MAX, "task maximum ", color="0.45", fontsize=7.5,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(HEADLINE_THRESHOLD, color="0.55", lw=0.8, ls="--", zorder=0)
    ax.text(0.015, HEADLINE_THRESHOLD, f" {HEADLINE_THRESHOLD} criterion", color="0.45",
            fontsize=7.5, va="top", ha="left", transform=ax.get_yaxis_transform())

    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(reward_label("dmc_eval"))
    ax.set_ylim(500, TASK_MAX * 1.03)
    _integer_delay_ticks(ax, df)
    _ms_axis(ax, df["delay"])


def _steps_panel(ax, df: pd.DataFrame, thr: int = HEADLINE_THRESHOLD,
                 title: bool = True,
                 ms_gap: float = MS_LABEL_MIN_GAP) -> list:
    """Steps to criterion vs delay, log y, censored runs on the budget line.

    Returns the legend proxy for censoring if any occurred, so a panel with none does not
    carry an explanation of it.
    """
    censored = False
    for arm in ARMS:
        sub = df[df["condition"] == arm]
        if sub.empty:
            continue
        status = sub.groupby("delay")[_solved_col(thr)].agg(n_solved="sum", n="count")
        full = status.index[status["n_solved"] == status["n"]]
        part = status.index[(status["n_solved"] > 0) & (status["n_solved"] < status["n"])]
        color = color_for(arm)

        # Mean + per-seed lines, over the cells where *every* run reached the criterion.
        if len(full):
            plot_seeds(ax, sub[sub["delay"].isin(full)], x="delay", y=_steps_col(thr),
                       seed_col="seed_pair", condition=arm, scale=1 / STEP_M,
                       marker_size=5)
        else:  # keep the arm in the legend even if it never solved anything
            ax.plot([], [], color=color, marker=marker_for(arm), label=label_for(arm))

        # Cells where some runs solved and some did not: the solved runs individually,
        # never a mean over them.
        for d in part:
            solved = sub[(sub["delay"] == d) & sub[_solved_col(thr)]]
            ax.scatter(_spread(d, len(solved), arm), solved[_steps_col(thr)] / STEP_M,
                       facecolors=color, edgecolors=color, marker=marker_for(arm),
                       s=26, zorder=5)

        for d, g in sub[~sub[_solved_col(thr)].astype(bool)].groupby("delay"):
            ax.scatter(_spread(d, len(g), arm), np.full(len(g), BUDGET / STEP_M),
                       facecolors="none", edgecolors=color, marker=marker_for(arm),
                       s=38, lw=1.4, zorder=6)
            censored = True

    ax.axhline(BUDGET / STEP_M, color="0.55", lw=0.8, ls="--", zorder=0)
    ax.text(0.015, BUDGET / STEP_M, " 1e9 budget", color="0.45", fontsize=7.5,
            va="bottom", ha="left", transform=ax.get_yaxis_transform())

    ax.set_yscale("log")
    ax.set_ylim(20, BUDGET / STEP_M * 2.4)
    ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000, 2000]))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_minor_formatter(lambda v, _: "")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(f"Steps to reach {thr} reward ($\\times 10^6$, log)")
    if title:
        ax.set_title(f"Time to solve\n(first step whose trailing 2.4e7-step mean "
                     f"reaches {thr})", fontsize=9.5)
    _integer_delay_ticks(ax, df)
    _ms_axis(ax, df["delay"], ms_gap)
    # Deliberately not "never reached {thr}": this panel is reused across three criteria,
    # and a label naming one of them was silently wrong on the other two.
    return [Line2D([], [], color="0.4", ls="none", marker="o", mfc="none", mew=1.4, ms=6,
                   label="never reached criterion within 1e9\n(one open marker per run)")
            ] if censored else []


def fig_main(df: pd.DataFrame):
    """The two readouts side by side, all four arms."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    _reward_panel(axes[0], df)
    axes[0].set_title("End-of-training reward\n(mean of eval points in the last 5e7 "
                      "steps)", fontsize=9.5)
    proxies = _steps_panel(axes[1], df)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles + seed_legend_handles() + proxies, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(f"WalkerWalk at 1e9 steps: joint servo (servo_kp = {SERVO_KP}) x "
                 f"explicit forward model", fontsize=11)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    return fig


def _cell_table(df: pd.DataFrame, readout: str) -> pd.DataFrame:
    """Cell means per (delay, arm), blank wherever any run in the cell is censored."""
    out = {}
    for arm in ARMS:
        sub = df[df["condition"] == arm]
        # A mean over only the runs that reached the criterion is the one number these
        # panels must not show, so a partially censored cell is dropped entirely.
        agg = sub.groupby("delay")[readout].agg(
            lambda s: np.nan if s.isna().any() else s.mean())
        out[arm] = agg
    return pd.DataFrame(out)


def _missing_delays(table: pd.DataFrame) -> list[int]:
    """Delays present in the cohort where the interaction cannot be formed.

    Derived rather than written into the caption: the design has changed twice already,
    and a hardcoded "nothing past delay 10" was silently wrong the moment delay 12
    completed.
    """
    inter = ((table["fm_servo"] - table["fm_torque"])
             - (table["mlp_servo"] - table["mlp_torque"]))
    return [int(d) for d in inter.index[inter.isna()]]


def _effect_panel(ax, table: pd.DataFrame, *, scale: float, ylabel: str,
                  ytext: str, note: str, note_xy: tuple[float, float],
                  note_align: tuple[str, str]) -> None:
    """The two simple effects of the servo, and their difference.

    ``servo | MLP`` = ``mlp_servo - mlp_torque``, drawn in the servo colour;
    ``servo | FM`` = ``fm_servo - fm_torque``, drawn in the combination colour;
    the interaction is their difference, in grey. Only delays where all four corners
    exist contribute, so a missing corner leaves a gap rather than a fabricated point.
    """
    s_mlp = (table["mlp_servo"] - table["mlp_torque"]) * scale
    s_fm = (table["fm_servo"] - table["fm_torque"]) * scale
    inter = s_fm - s_mlp

    for series, color, marker, label in [
            (s_mlp, color_for("mlp_servo"), marker_for("mlp_servo"),
             "Effect of the servo, without a forward model"),
            (s_fm, color_for("fm_servo"), marker_for("fm_servo"),
             "Effect of the servo, with a forward model"),
            (inter, "0.35", "x", "Interaction (the difference of the two)")]:
        ok = series.dropna()
        ax.plot(ok.index, ok.to_numpy(), color=color, marker=marker, ms=5.5,
                lw=1.9 if label.startswith("Effect") else 1.3,
                ls="-" if label.startswith("Effect") else "--", label=label)

    ax.axhline(0, color="0.6", lw=0.9, zorder=0)
    ax.text(0.985, 0, "no effect ", color="0.45", fontsize=7.5, va="top", ha="right",
            transform=ax.get_yaxis_transform())
    ax.text(*note_xy, note, transform=ax.transAxes, fontsize=7.4, color="0.4",
            ha=note_align[0], va=note_align[1])
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(ytext, fontsize=9.5)
    ax.set_xticks([int(d) for d in table.index])
    _ms_axis(ax, table.index)


def fig_interaction(df: pd.DataFrame):
    """Does the servo still buy anything once a forward model is present?"""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    reward = _cell_table(df, "reward_final")
    steps = _cell_table(df, _steps_col(HEADLINE_THRESHOLD))
    _effect_panel(axes[0], reward, scale=1.0,
                  ylabel="Change in end-of-training reward",
                  ytext="Reward at 1e9 steps\n(positive = the servo helps)",
                  # Delays 3 and 25 have no `mlp_servo` run, so the interaction cannot
                  # be formed there. Left unsaid, the gaps read as failures rather than
                  # as holes in the design.
                  note="gaps at delays "
                       + ", ".join(str(d) for d in _missing_delays(reward))
                       + ": no MLP + servo\nrun was launched there, so the two simple\n"
                         "effects have no common delay to differ at",
                  note_xy=(0.02, 0.03), note_align=("left", "bottom"))
    # The high delays are absent from the right-hand panel because the MLP arms are
    # censored there -- and that is the whole reason the left-hand panel exists. Saying
    # so on the figure keeps the two panels from reading as disagreeing.
    _effect_panel(axes[1], steps, scale=1 / STEP_M,
                  ylabel=f"Change in steps to {HEADLINE_THRESHOLD} "
                         f"($\\times 10^6$)",
                  ytext=f"Time to reach {HEADLINE_THRESHOLD}\n"
                        f"(negative = the servo helps)",
                  note="nothing past delay "
                       + str(max(d for d in steps.index
                                 if d not in _missing_delays(steps)))
                       + ": an MLP arm is censored at every\n"
                         "higher delay, so the MLP-side effect the interaction\n"
                         "needs does not exist there",
                  note_xy=(0.98, 0.97), note_align=("right", "top"))

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    # Two lines, because the one-line version was true only out to delay 20 and the
    # 2026-09-22 delay-25 run is the whole reason to look at this panel again.
    fig.suptitle("Out to delay 20 the servo adds almost nothing on top of an explicit "
                 "forward model\nAt delay 25, where the forward model alone becomes "
                 "seed-unstable, a single run suggests otherwise", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    return fig


def fig_training_curves(curves: pd.DataFrame, df: pd.DataFrame):
    """The eval series both readouts come from, at the delays where the arms separate."""
    fig, axes = plt.subplots(1, len(CURVE_DELAYS),
                             figsize=(3.5 * len(CURVE_DELAYS), 3.8), sharey=True)
    for ax, delay in zip(axes, CURVE_DELAYS):
        cell = curves[curves["delay"] == delay]
        for arm in ARMS:
            for _, run in cell[cell["condition"] == arm].groupby("wandb_id"):
                ax.plot(run["step"] / 1e9, run["reward_mean"], color=color_for(arm),
                        lw=0.8, alpha=0.8)
        ax.axhline(HEADLINE_THRESHOLD, color="0.55", lw=0.8, ls="--", zorder=0)
        n = df[df["delay"] == delay].groupby("condition").size()
        # Per-panel n, compact: the spelled-out arm names ran into the neighbouring
        # panel's title at this figure width.
        ax.set_title(f"delay {delay} ({delay * CTRL_DT_MS} ms)\n"
                     f"MLP  torque n={n.get('mlp_torque', 0)}, "
                     f"servo n={n.get('mlp_servo', 0)}\n"
                     f"FM   torque n={n.get('fm_torque', 0)}, "
                     f"servo n={n.get('fm_servo', 0)}", fontsize=8.2)
        ax.set_xlabel("Environment steps ($\\times 10^9$)")
        ax.set_ylim(0, TASK_MAX * 1.06)
        ax.set_xlim(0, BUDGET / 1e9 * 1.02)
    axes[0].set_ylabel(reward_label("dmc_eval"))
    axes[-1].legend(handles=[Line2D([], [], color=color_for(a), lw=1.6,
                                    label=label_for(a)) for a in ARMS]
                    + [Line2D([], [], color="0.55", lw=0.8, ls="--",
                              label=f"{HEADLINE_THRESHOLD} criterion")],
                    fontsize=7.2, loc="lower right", framealpha=0.9)
    fig.suptitle("Every run's eval series, on the common 4.8e6 grid the criterion is "
                 "read from (one line per run)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return fig


def fig_threshold_sensitivity(df: pd.DataFrame):
    """Does the picture depend on where the criterion is put?"""
    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(3.7 * len(THRESHOLDS), 4.2),
                             sharey=True)
    proxies = []
    for ax, thr in zip(axes, THRESHOLDS):
        proxies = _steps_panel(ax, df, thr=thr, title=False,
                               ms_gap=MS_LABEL_MIN_GAP_NARROW) or proxies
        ax.set_title(f"criterion = {thr}", fontsize=9.5)
        ax.set_ylabel("")
    axes[0].set_ylabel("Steps to criterion ($\\times 10^6$, log)")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles + proxies, fontsize=8, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    # Deliberately not "the picture does not change": it does. Dropping the bar to 800
    # un-censors MLP + servo at delays 15 and 20 while leaving MLP + torque censored,
    # which is the clearest single statement of the servo's effect in the whole folder.
    fig.suptitle("The censoring pattern, not the numbers, is what survives the choice "
                 "of criterion", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
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
            "lengths. Split the figure or restrict the cohort.")

    # The ms axis is silently wrong if this drifts: `add_ms_axis`'s default is the
    # rodent's 10 ms, and a 20-step delay would be labelled 200 ms instead of 500.
    ctrl_dt = df["ctrl_dt"].dropna().unique()
    if list(ctrl_dt) != [CTRL_DT_MS / 1000]:
        raise SystemExit(
            f"ctrl_dt in data.csv is {ctrl_dt}, but the ms axis is drawn with "
            f"CTRL_DT_MS = {CTRL_DT_MS}. Fix CTRL_DT_MS rather than the axis label.")

    if sorted(df["condition"].unique()) != sorted(ARMS):
        raise SystemExit(
            f"data.csv has arms {sorted(df['condition'].unique())} but plot.py draws "
            f"{sorted(ARMS)}; an arm would be silently missing from every panel.")

    if list(df["budget"].unique()) != [BUDGET]:
        raise SystemExit(
            f"data.csv spans budgets {sorted(df['budget'].unique())}, but the censoring "
            f"line and the 'within 1e9' legend assume one. Facet the panels or restrict "
            f"the cohort.")

    manifest = {}
    for name, builder, inputs in [
        ("reward_and_steps_vs_delay", lambda: fig_main(df), (DATA,)),
        ("interaction", lambda: fig_interaction(df), (DATA,)),
        ("training_curves", lambda: fig_training_curves(curves, df), (CURVES, DATA)),
        ("threshold_sensitivity", lambda: fig_threshold_sensitivity(df), (DATA,)),
    ]:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
