"""Figures for walker-forward-model-1b (dm_control_suite, WalkerWalk).

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/plot.py

Figures
-------
  reward_and_steps_vs_delay.png  main: both readouts vs delay, every run >= 1e9 steps
  budget_480m_replication.png    supp: the same two panels on the 4.8e8 cohort (seed 42)
  training_curves.png            supp: the eval series behind both readouts
  threshold_sensitivity.png      supp: time-to-criterion at 800 / 900 / 950

Conventions this file is careful about
--------------------------------------
**Raw reward on the reward y-axis** (README §7). Both arms are on one task with one reward
function, so there is nothing to normalise away and no excuse for a ratio. The task
maximum (1e3, by construction: 1e3 steps x a reward in [0, 1]) is drawn as a reference
line instead of divided through.

**The time-to-criterion axis is logarithmic, and that is not the same concession.** §7's
rule is about *reward*, which does not map linearly onto competence, so a reward ratio
asserts a linearity the task does not have. Steps are a physical count: ratios are exactly
what is meaningful ("the MLP takes 5.7x longer"), and the quantity spans 2.1e1 to 2.3e3
million steps -- two decades. On a linear axis everything below delay 12 would be pressed
into the bottom 2 % of the panel. Note the consequence for reading the *mean* line: it is
an arithmetic mean drawn on a log axis, so it sits above the visual midpoint of its seeds.
The report quotes per-run values and ranges for this panel, not the mean line.

**Every seed as a thin line**, via ``plot_seeds``. Not decoration here: the two MLP runs at
delay 20 with a 4e9 budget disagree about whether the task is solvable at all (one crosses
at 2.3e9, the other never does), and the two forward-model runs at delay 25 land 1.9e2
reward apart.

**Censoring is drawn, never dropped, and at each run's own budget.** A run that never
reaches the criterion has no time-to-criterion. If such runs are simply absent, the MLP's
line stops at delay 12 and reads as missing data rather than as failure. And since the
cohort spans 1e9 to 4e9, *how much* budget a censored run had is part of the claim --
"did not solve it in 4e9" is much stronger than "did not solve it in 1e9" -- so each
censored marker is parked at its own budget rather than on a shared line. A cell where
only *some* runs miss the criterion is **excluded from the mean line**, which therefore
breaks: a mean over the subset that happened to succeed is the one number this panel must
not show.

**The 4.8e8 cohort is never averaged with the primary one**, only faceted -- see
``extract.py``. The MLP is the slower arm, so a shared short readout flatters the forward
model.
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

ARMS = ["delayed_mlp", "flat_forward_model"]

#: The step every primary-cohort reward is read at. Must match ``extract.READOUT_STEP``;
#: asserted against ``data.csv`` in ``main``.
READOUT_STEP = 1_000_000_000
SUPPORT_BUDGET = 480_000_000

#: WalkerWalk runs at ctrl_dt = 0.025 s, so one delay step is 25 ms -- **not** the
#: rodent's 10, which is what `add_ms_axis` defaults to. Asserted against `data.csv`.
CTRL_DT_MS = 25

#: The headline criterion, and the two it is checked against. None of the three is a
#: published standard; see `extract.py`.
HEADLINE_THRESHOLD = 900
THRESHOLDS = (800, 900, 950)

#: dm_control returns are bounded above by the episode length x a per-step reward in
#: [0, 1]. Drawn as a reference line rather than used as a denominator (README §7).
TASK_MAX = 1000

#: Delays for the training-curve panels: where the arms separate, and where the extended
#: runs live.
CURVE_DELAYS = (12, 15, 20, 25)

STEP_M = 1e6  # steps -> "millions of steps"


def _solved_col(thr: int) -> str:
    return f"solved_{thr}"


def _steps_col(thr: int) -> str:
    return f"steps_to_{thr}"


def _cell_status(df: pd.DataFrame, arm: str, thr: int) -> pd.DataFrame:
    """Per delay: how many runs reached the criterion, out of how many."""
    sub = df[df["condition"] == arm]
    return sub.groupby("delay")[_solved_col(thr)].agg(n_solved="sum", n="count")


def _spread(delay: float, n: int, arm: str, width: float = 0.42) -> np.ndarray:
    """``n`` x-positions near ``delay``, so co-located markers stay countable.

    Censored runs that share a delay *and* a budget sit at the same point, so without
    this two censored seeds are one marker and the panel understates how many runs
    failed. The per-arm offset keeps the two arms apart where both fail at one delay.
    """
    centre = delay + (ARMS.index(arm) - (len(ARMS) - 1) / 2) * width
    if n <= 1:
        return np.full(n, centre)
    return centre + np.linspace(-width / 2, width / 2, n)


def _integer_delay_ticks(ax, df: pd.DataFrame) -> None:
    """Tick the delays that were actually run.

    The grids are 0/3/5/7/10/12/15/20/25 and 0/2/5/7/10/15/20, which matplotlib's default
    locator renders on a round 2.5-step grid -- an axis whose labels name delays that do
    not exist.
    """
    ax.set_xticks(sorted(df["delay"].unique()))


def _reward_panel(ax, df: pd.DataFrame, y: str, *, extend: bool) -> None:
    """Raw reward vs delay, one line per arm.

    ``extend``: additionally show where the runs with more than ``READOUT_STEP`` steps
    *ended up*, as an open marker joined to their 1e9 value by a vertical line. Without
    it the panel silently discards the whole point of those runs -- the MLP at delay 15
    reads 7.8e2 at 1e9 and 9.6e2 at 2e9, and only the second number says the MLP gets
    there at all.
    """
    for arm in ARMS:
        sub = df[df["condition"] == arm]
        if sub.empty:
            continue
        plot_seeds(ax, sub, x="delay", y=y, condition=arm, marker_size=5)

    if extend:
        ext = df[df["budget"] > READOUT_STEP]
        for _, r in ext.iterrows():
            color = color_for(r["condition"])
            ax.annotate(
                "", xy=(r["delay"], r["reward_final"]), xytext=(r["delay"], r[y]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, alpha=0.75,
                                shrinkA=0, shrinkB=0))
            ax.plot(r["delay"], r["reward_final"], marker=marker_for(r["condition"]),
                    mfc="none", mec=color, mew=1.3, ms=7, ls="none", zorder=5)

    ax.axhline(TASK_MAX, color="0.55", lw=0.8, ls=":", zorder=0)
    ax.text(0.985, TASK_MAX, "task maximum ", color="0.45", fontsize=7.5,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(HEADLINE_THRESHOLD, color="0.55", lw=0.8, ls="--", zorder=0)
    ax.text(0.015, HEADLINE_THRESHOLD, f" {HEADLINE_THRESHOLD} criterion", color="0.45",
            fontsize=7.5, va="bottom", ha="left", transform=ax.get_yaxis_transform())

    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(reward_label("dmc_eval"))
    ax.set_ylim(0, TASK_MAX * 1.06)
    _integer_delay_ticks(ax, df)
    add_ms_axis(ax, df["delay"].max(), ctrl_dt_ms=CTRL_DT_MS)


def _steps_panel(ax, df: pd.DataFrame, thr: int = HEADLINE_THRESHOLD,
                 title: bool = True) -> list:
    """Steps to reach the criterion vs delay, log y, with censored runs at their budget.

    Returns the legend proxies for whichever censoring cases occur, so a panel with no
    censoring does not carry an explanation of censoring.
    """
    proxies, censored = [], False

    for arm in ARMS:
        status = _cell_status(df, arm, thr)
        sub = df[df["condition"] == arm]
        full = status.index[status["n_solved"] == status["n"]]
        part = status.index[(status["n_solved"] > 0) & (status["n_solved"] < status["n"])]
        color = color_for(arm)

        # Mean + per-seed lines, over the cells where *every* run reached the criterion.
        if len(full):
            plot_seeds(ax, sub[sub["delay"].isin(full)], x="delay", y=_steps_col(thr),
                       condition=arm, scale=1 / STEP_M, marker_size=5)
        else:  # keep the arm in the legend even if it never solved anything
            ax.plot([], [], color=color, marker=marker_for(arm), label=label_for(arm))

        # Cells where some runs solved and some did not: the solved runs individually,
        # never a mean over them.
        for d in part:
            solved = sub[(sub["delay"] == d) & sub[_solved_col(thr)]]
            ax.scatter(_spread(d, len(solved), arm),
                       solved[_steps_col(thr)] / STEP_M, facecolors=color,
                       edgecolors=color, marker=marker_for(arm), s=26, zorder=5)

        # Every censored run, at its own budget, grouped so that runs sharing a delay and
        # a budget are spread apart and therefore countable.
        for (d, budget), g in sub[~sub[_solved_col(thr)].astype(bool)].groupby(
                ["delay", "budget"]):
            ax.scatter(_spread(d, len(g), arm), np.full(len(g), budget / STEP_M),
                       facecolors="none", edgecolors=color, marker=marker_for(arm),
                       s=34, lw=1.3, zorder=5)
            censored = True

    for budget in sorted(df["budget"].unique()):
        ax.axhline(budget / STEP_M, color="0.55", lw=0.8, ls="--", zorder=0)
        ax.text(0.015, budget / STEP_M, f" {budget / 1e9:g}e9 budget", color="0.45",
                fontsize=7.5, va="bottom", ha="left",
                transform=ax.get_yaxis_transform())
    if censored:
        # Deliberately not "never reached {thr}": this panel is reused across three
        # criteria, and a label naming one of them was silently wrong on the other two.
        proxies.append(Line2D([], [], color="0.4", ls="none", marker="o", mfc="none",
                              mew=1.3, ms=6,
                              label="never reached criterion (censored at the\n"
                                    "budget it is drawn at; one marker per run)"))
    ax.set_yscale("log")
    ax.set_ylim(15, df["budget"].max() / STEP_M * 2.2)
    ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000, 2000, 4000]))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_minor_formatter(lambda v, _: "")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(f"Steps to reach {thr} reward ($\\times 10^6$, log)")
    if title:
        ax.set_title(f"Time to solve\n(first step whose trailing 1e7-step mean "
                     f"reaches {thr})", fontsize=9.5)
    _integer_delay_ticks(ax, df)
    add_ms_axis(ax, df["delay"].max(), ctrl_dt_ms=CTRL_DT_MS)
    return proxies


def fig_primary(df: pd.DataFrame):
    """The two readouts side by side, every run with at least READOUT_STEP steps."""
    sub = df[df["cohort"] == "primary"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    _reward_panel(axes[0], sub, "reward_at_1b", extend=True)
    axes[0].set_title("Reward at a common 1e9 steps\n(arrows: where the 2e9/4e9 runs "
                      "ended up)", fontsize=9.5)
    proxies = _steps_panel(axes[1], sub)

    handles, _ = axes[0].get_legend_handles_labels()
    extra = seed_legend_handles() if sub.groupby("condition")["seed"].nunique().max() > 1 \
        else []
    budget_proxy = [Line2D([], [], color="0.4", ls="none", marker="o", mfc="none",
                           mew=1.3, ms=6, label="endpoint of a 2e9 / 4e9 run")]
    fig.legend(handles=handles + extra + budget_proxy + proxies, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("WalkerWalk: explicit forward model vs delayed MLP  --  "
                 "all runs of 1e9 steps or more (seeds 43, 46-49)", fontsize=11)
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    return fig


def fig_support(df: pd.DataFrame):
    """The 4.8e8 cohort: a third seed, on a budget too short to read as a level."""
    sub = df[df["cohort"] == "support"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    _reward_panel(axes[0], sub, "reward_final", extend=False)
    axes[0].set_title("End-of-training reward at 4.8e8\n(mean of eval points in the "
                      "last 5e7 steps)", fontsize=9.5)
    proxies = _steps_panel(axes[1], sub)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles + proxies, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("WalkerWalk at 4.8e8 steps, seed 42  --  a third seed on a budget "
                 "shorter than the MLP needs (not pooled with the figure above)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    return fig


def fig_training_curves(curves: pd.DataFrame, df: pd.DataFrame):
    """The eval series both readouts come from, at the delays that separate the arms.

    Each panel keeps its own x-range (the budgets differ by 4x across panels), so the
    1e9-step detail is not squashed to make room for a 4e9 run in a neighbouring panel.
    """
    sub = curves[curves["cohort"] == "primary"]
    fig, axes = plt.subplots(1, len(CURVE_DELAYS),
                             figsize=(3.6 * len(CURVE_DELAYS), 3.7), sharey=True)
    for ax, delay in zip(axes, CURVE_DELAYS):
        cell = sub[sub["delay"] == delay]
        for arm in ARMS:
            for _, run in cell[cell["condition"] == arm].groupby("wandb_id"):
                ax.plot(run["step"] / 1e9, run["reward_mean"], color=color_for(arm),
                        lw=0.7, alpha=0.75)
        ax.axhline(HEADLINE_THRESHOLD, color="0.55", lw=0.8, ls="--", zorder=0)
        ax.axvline(READOUT_STEP / 1e9, color="0.35", lw=0.9, ls=":", zorder=0)
        n = df[(df["cohort"] == "primary") & (df["delay"] == delay)]
        ax.set_title(f"delay {delay} ({delay * CTRL_DT_MS} ms)\n"
                     f"MLP n={(n.condition == 'delayed_mlp').sum()}, "
                     f"FM n={(n.condition == 'flat_forward_model').sum()}", fontsize=9.5)
        ax.set_xlabel("Environment steps ($\\times 10^9$)")
        ax.set_ylim(0, TASK_MAX * 1.06)
        ax.set_xlim(0, cell["step"].max() / 1e9 * 1.02)
    axes[0].set_ylabel(reward_label("dmc_eval"))
    axes[0].text(READOUT_STEP / 1e9, 30, " 1e9\n readout", fontsize=7, color="0.35",
                 va="bottom", ha="left")
    axes[-1].legend(handles=[Line2D([], [], color=color_for(a), lw=1.4,
                                    label=label_for(a)) for a in ARMS]
                    + [Line2D([], [], color="0.55", lw=0.8, ls="--",
                              label=f"{HEADLINE_THRESHOLD} criterion")],
                    fontsize=7.6, loc="lower right", framealpha=0.9)
    fig.suptitle("WalkerWalk: every primary-cohort run's eval series (one line per run; "
                 "note the per-panel x-range)", fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    return fig


def fig_threshold_sensitivity(df: pd.DataFrame):
    """Does the arm ordering depend on where the criterion is put? (No.)"""
    sub = df[df["cohort"] == "primary"]
    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(3.7 * len(THRESHOLDS), 4.0),
                             sharey=True)
    proxies = []
    for ax, thr in zip(axes, THRESHOLDS):
        proxies = _steps_panel(ax, sub, thr=thr, title=False) or proxies
        ax.set_title(f"criterion = {thr}", fontsize=9.5)
        ax.set_ylabel("")
    axes[0].set_ylabel("Steps to criterion ($\\times 10^6$, log)")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles + proxies, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    # Deliberately not "ordered the same": at delay 0-5 the two arms swap places between
    # the 800 and 950 panels, which is within the seed spread and not worth a claim. What
    # is stable across all three bars is the crossover and the censoring pattern.
    fig.suptitle("WalkerWalk, runs >= 1e9 steps: where the criterion is put changes the "
                 "numbers, not the crossover or who gets censored", fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
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
    # rodent's 10 ms, and a 25-step delay would be labelled 250 ms instead of 625.
    ctrl_dt = df["ctrl_dt"].dropna().unique()
    if list(ctrl_dt) != [CTRL_DT_MS / 1000]:
        raise SystemExit(
            f"ctrl_dt in data.csv is {ctrl_dt}, but the ms axis is drawn with "
            f"CTRL_DT_MS = {CTRL_DT_MS}. Fix CTRL_DT_MS rather than the axis label.")

    # `reward_at_1b` is only a like-for-like readout if every primary run has one; an
    # all-NaN column would make `plot_seeds` silently draw nothing.
    primary = df[df["cohort"] == "primary"]
    if primary["reward_at_1b"].isna().any():
        raise SystemExit(
            "primary-cohort rows with no reward_at_1b: "
            f"{primary.loc[primary['reward_at_1b'].isna(), 'wandb_id'].tolist()}. "
            f"Re-run extract.py; the main panel would otherwise drop them without saying "
            f"so.")
    if not (primary["budget"] >= READOUT_STEP).all():
        raise SystemExit(
            f"a primary-cohort run has a budget below READOUT_STEP ({READOUT_STEP:,}); "
            f"plot.py and extract.py disagree about what 'primary' means.")

    manifest = {}
    for name, builder, inputs in [
        ("reward_and_steps_vs_delay", lambda: fig_primary(df), (DATA,)),
        ("budget_480m_replication", lambda: fig_support(df), (DATA,)),
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
