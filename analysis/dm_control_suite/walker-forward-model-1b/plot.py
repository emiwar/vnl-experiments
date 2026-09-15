"""Figures for walker-forward-model-1b (dm_control_suite, WalkerWalk).

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/plot.py

Figures
-------
  reward_and_steps_vs_delay.png  main: both readouts vs delay, 1e9 cohort
  budget_480m_replication.png    supp: the same two panels on the 4.8e8 cohort (3rd seed,
                                 and the only forward-model run at delay 5)
  training_curves.png            supp: the eval series behind both readouts, delays 10-25
  threshold_sensitivity.png      supp: time-to-criterion at 800 / 900 / 950

Conventions this file is careful about
--------------------------------------
**Raw reward on the y-axis** (README §7). Both arms are on one task with one reward
function, so there is nothing to normalise away and no excuse for a ratio. The task
maximum (1e3, by construction: 1e3 steps x a reward in [0, 1]) is drawn as a reference
line instead of divided through.

**Every seed as a thin line**, via ``plot_seeds``. This is not decoration here: the two
forward-model runs at delay 25 land 1.9e2 apart (9.5e2 vs 7.5e2), and the mean of those
two is a number that describes neither run.

**Censoring is drawn, never dropped.** A run that never reaches the criterion has no
time-to-criterion, and if such runs are simply absent then the MLP's line stops at delay
10 and reads as missing data rather than as failure -- while the surviving mean reads as
"the MLP solves it in 6e8 steps", which is true only of the delays where it solves it at
all. So: a cell whose runs *all* miss the criterion is plotted as open markers on the
training-budget line; a cell where only *some* miss it (only the forward model at delay
25) has its solved seeds drawn individually and is **excluded from the mean line**, which
therefore breaks. A mean over the subset that happened to succeed is the one number this
panel must not show.

**The two budgets are never averaged together**, only faceted -- see ``extract.py``. The
MLP is the slower arm, so a shared 4.8e8 readout flatters the forward model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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

PRIMARY_BUDGET = 1_000_000_000
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

#: Delays for the training-curve panels: the range over which the two arms separate.
CURVE_DELAYS = (10, 15, 20, 25)

STEP_M = 1e6  # steps -> "millions of steps" for the y-axis


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

    Censored runs all sit at the same y (the budget line), so without this two censored
    seeds are one marker and the panel understates how many runs failed. The per-arm
    offset keeps the two arms' censored markers from landing on top of each other where
    both fail at the same delay.
    """
    centre = delay + (ARMS.index(arm) - (len(ARMS) - 1) / 2) * width
    if n <= 1:
        return np.full(n, centre)
    return centre + np.linspace(-width / 2, width / 2, n)


def _integer_delay_ticks(ax, df: pd.DataFrame) -> None:
    """Tick the delays that were actually run.

    The 4.8e8 grid is 0/2/5/7/10/15/20, which matplotlib's default locator renders as
    2.5-step ticks -- an axis whose labels name delays that do not exist.
    """
    ax.set_xticks(sorted(df["delay"].unique()))


def _reward_panel(ax, df: pd.DataFrame, budget: int) -> None:
    """Raw end-of-training reward vs delay, one line per arm."""
    for arm in ARMS:
        sub = df[df["condition"] == arm]
        if sub.empty:
            continue
        plot_seeds(ax, sub, x="delay", y="reward_final", condition=arm, marker_size=5)

    # Reference lines labelled on the *right*: both arms start near the task maximum at
    # delay 0, so a left-hand label sits on top of them.
    ax.axhline(TASK_MAX, color="0.55", lw=0.8, ls=":", zorder=0)
    ax.text(0.985, TASK_MAX, "task maximum ", color="0.45", fontsize=7.5,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(HEADLINE_THRESHOLD, color="0.55", lw=0.8, ls="--", zorder=0)
    ax.text(0.985, HEADLINE_THRESHOLD, f"{HEADLINE_THRESHOLD} criterion ", color="0.45",
            fontsize=7.5, va="bottom", ha="right", transform=ax.get_yaxis_transform())

    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(reward_label("dmc_eval"))
    ax.set_title(f"End-of-training reward\n(mean of eval points in the last 5e7 steps)",
                 fontsize=9.5)
    ax.set_ylim(0, TASK_MAX * 1.06)
    _integer_delay_ticks(ax, df)
    add_ms_axis(ax, df["delay"].max(), ctrl_dt_ms=CTRL_DT_MS)


def _steps_panel(ax, df: pd.DataFrame, budget: int, thr: int = HEADLINE_THRESHOLD,
                 title: bool = True) -> list:
    """Steps to reach the criterion vs delay, with censored cells drawn.

    Returns the legend proxies for whichever censoring cases actually occur, so a panel
    with no censoring does not carry an explanation of censoring.
    """
    proxies, saw_censored, saw_partial = [], False, False
    budget_m = budget / STEP_M

    for arm in ARMS:
        status = _cell_status(df, arm, thr)
        sub = df[df["condition"] == arm]
        full = status.index[status["n_solved"] == status["n"]]
        part = status.index[(status["n_solved"] > 0) & (status["n_solved"] < status["n"])]
        none = status.index[status["n_solved"] == 0]
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
            cell = sub[sub["delay"] == d]
            solved = cell[cell[_solved_col(thr)]]
            ax.scatter(_spread(d, len(solved), arm), solved[_steps_col(thr)] / STEP_M,
                       facecolors=color, edgecolors=color, marker=marker_for(arm),
                       s=26, zorder=5)
            n_cens = int((~cell[_solved_col(thr)]).sum())
            ax.scatter(_spread(d, n_cens, arm), np.full(n_cens, budget_m),
                       facecolors="none", edgecolors=color, marker=marker_for(arm),
                       s=34, lw=1.3, zorder=5)
            saw_partial = True

        # Cells where nothing reached the criterion, parked on the budget line.
        for d in none:
            n_cens = int(status.loc[d, "n"])
            ax.scatter(_spread(d, n_cens, arm), np.full(n_cens, budget_m),
                       facecolors="none", edgecolors=color, marker=marker_for(arm),
                       s=34, lw=1.3, zorder=5)
            saw_censored = True

    ax.axhline(budget_m, color="0.55", lw=0.8, ls="--", zorder=0)
    # Labelled on the *left*: the censored markers pile up at the high delays on the
    # right, which is exactly where this line is most crowded.
    ax.text(0.015, budget_m, " training budget", color="0.45", fontsize=7.5,
            va="bottom", ha="left", transform=ax.get_yaxis_transform())
    if saw_censored or saw_partial:
        # Deliberately not "never reached {thr}": this panel is reused across three
        # criteria, and a label naming one of them was silently wrong on the other two.
        proxies.append(Line2D([], [], color="0.4", ls="none", marker="o",
                              mfc="none", mew=1.3, ms=6,
                              label="never reached criterion (censored,\n"
                                    "one marker per run)"))
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(f"Steps to reach {thr} reward ($\\times 10^6$)")
    if title:
        ax.set_title(f"Time to solve\n(first step whose trailing 1e7-step mean "
                     f"reaches {thr})", fontsize=9.5)
    ax.set_ylim(0, budget_m * 1.13)
    _integer_delay_ticks(ax, df)
    add_ms_axis(ax, df["delay"].max(), ctrl_dt_ms=CTRL_DT_MS)
    return proxies


def fig_vs_delay(df: pd.DataFrame, budget: int, subtitle: str):
    """The two readouts side by side, for one budget."""
    sub = df[df["budget"] == budget]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    _reward_panel(axes[0], sub, budget)
    proxies = _steps_panel(axes[1], sub, budget)

    # One figure-level legend below both panels rather than two in-axes ones. Both
    # panels are full in every corner at some delay -- the reward panel's curves cross
    # the lower left and the time panel's budget line owns the top -- and an in-axes
    # legend ended up over the MLP's curve in whichever one it was put.
    handles, _ = axes[0].get_legend_handles_labels()
    extra = seed_legend_handles() if sub.groupby("condition")["seed"].nunique().max() > 1 \
        else []
    fig.legend(handles=handles + extra + proxies, fontsize=8, ncol=len(handles + extra
               + proxies), loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False)

    fig.suptitle(f"WalkerWalk: explicit forward model vs delayed MLP  --  {subtitle}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    return fig


def fig_training_curves(curves: pd.DataFrame):
    """The eval series both readouts are computed from, at the delays that separate."""
    sub = curves[curves["budget"] == PRIMARY_BUDGET]
    fig, axes = plt.subplots(1, len(CURVE_DELAYS), figsize=(3.5 * len(CURVE_DELAYS), 3.6),
                             sharey=True)
    for ax, delay in zip(axes, CURVE_DELAYS):
        cell = sub[sub["delay"] == delay]
        for arm in ARMS:
            for _, run in cell[cell["condition"] == arm].groupby("wandb_id"):
                ax.plot(run["step"] / 1e9, run["reward_mean"], color=color_for(arm),
                        lw=0.7, alpha=0.75)
        ax.axhline(HEADLINE_THRESHOLD, color="0.55", lw=0.8, ls="--", zorder=0)
        ax.axvline(SUPPORT_BUDGET / 1e9, color="0.55", lw=0.8, ls=":", zorder=0)
        ax.set_title(f"delay {delay} ({delay * CTRL_DT_MS} ms)", fontsize=9.5)
        ax.set_xlabel("Environment steps ($\\times 10^9$)")
        ax.set_ylim(0, TASK_MAX * 1.06)
    axes[0].set_ylabel(reward_label("dmc_eval"))
    axes[0].text(SUPPORT_BUDGET / 1e9, 40, " 4.8e8: the\n sibling\n analysis'\n budget",
                 fontsize=7, color="0.4", va="bottom", ha="left")
    axes[-1].legend(handles=[Line2D([], [], color=color_for(a), lw=1.4,
                                    label=label_for(a)) for a in ARMS]
                    + [Line2D([], [], color="0.55", lw=0.8, ls="--",
                              label=f"{HEADLINE_THRESHOLD} criterion")],
                    fontsize=7.8, loc="lower right", framealpha=0.9)
    fig.suptitle("WalkerWalk at 1e9 steps: every run's eval series (one line per run)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    return fig


def fig_threshold_sensitivity(df: pd.DataFrame):
    """Does the arm ordering depend on where the criterion is put? (No.)"""
    sub = df[df["budget"] == PRIMARY_BUDGET]
    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(3.6 * len(THRESHOLDS), 3.8),
                             sharey=True)
    proxies = []
    for ax, thr in zip(axes, THRESHOLDS):
        proxies = _steps_panel(ax, sub, PRIMARY_BUDGET, thr=thr, title=False) or proxies
        ax.set_title(f"criterion = {thr}", fontsize=9.5)
        ax.set_ylabel("")
    axes[0].set_ylabel(f"Steps to criterion ($\\times 10^6$)")
    handles, _ = axes[0].get_legend_handles_labels()
    # Figure-level and below the axes: three panels of the same quantity leave no
    # in-axes corner free at the 800 criterion, and one legend serves all three anyway.
    fig.legend(handles=handles + proxies, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("WalkerWalk at 1e9 steps: time-to-criterion is ordered the same "
                 "wherever the bar is put", fontsize=11)
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
    # rodent's 10 ms, and a 25-step delay would be labelled 250 ms instead of 625.
    ctrl_dt = df["ctrl_dt"].dropna().unique()
    if list(ctrl_dt) != [CTRL_DT_MS / 1000]:
        raise SystemExit(
            f"ctrl_dt in data.csv is {ctrl_dt}, but the ms axis is drawn with "
            f"CTRL_DT_MS = {CTRL_DT_MS}. Fix CTRL_DT_MS rather than the axis label.")

    manifest = {}
    for name, builder, inputs in [
        ("reward_and_steps_vs_delay",
         lambda: fig_vs_delay(df, PRIMARY_BUDGET,
                              "1e9-step budget, seeds 43 & 46"), (DATA,)),
        ("budget_480m_replication",
         lambda: fig_vs_delay(df, SUPPORT_BUDGET,
                              "4.8e8-step budget, seed 42 (replication)"), (DATA,)),
        ("training_curves", lambda: fig_training_curves(curves), (CURVES,)),
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
