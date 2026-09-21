"""Figures for reacher-hard-first-look (dm_control_suite).

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/reacher-hard-first-look/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/reacher-hard-first-look/plot.py

Figures
-------
  reward_vs_delay.png      Q1 + Q3: end-of-training and best-ever reward vs delay
  seed_spread.png          Q2: the three baseline seeds, and their spread vs delay
  training_curves.png      Q4 (shape): the eval series, one panel per delay
  steps_to_threshold.png   Q4 (number): steps to reach 800 / 900 eval reward

Raw reward on every y-axis (README §7): this is one task, so there is no cross-task scale
problem to solve and no reason to divide by anything. The only non-reward axis is
`steps_to_threshold`, which is the question asked.

Censoring is drawn, not dropped. Five baseline runs never reach 900, and a mean over the
ones that did would report the delay sweep's failures as successes that happened to be
fast. Those runs are plotted as open upward triangles on the 480 M ceiling line, and the
threshold panels deliberately show individual runs rather than a mean.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

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

#: The budget every run was given. A run that never crossed a threshold is drawn here.
TOTAL_STEPS = 480_000_000

#: Thresholds with a panel in steps_to_threshold.png. 900 is the question; 800 is the
#: control, because at 900 the baseline is censored often enough that the panel is as
#: much about censoring as about speed.
THRESHOLD_PANELS = [800, 900]


def _ctrl_dt_ms(df: pd.DataFrame) -> float:
    """Milliseconds per delay step, off the data rather than assumed.

    ReacherHard is `ctrl_dt = 0.02` -> 20 ms, which is neither `style.CTRL_DT_MS`'s
    rodent 10 nor the 25 of the Walker/Humanoid tasks this track mostly plots.
    """
    values = df["ctrl_dt"].dropna().unique()
    if len(values) != 1:
        raise SystemExit(f"data.csv mixes ctrl_dt {list(values)}; the ms axis would be "
                         f"wrong for some of the points.")
    return float(values[0]) * 1000


def _delay_axis(ax, df, *, label=True):
    ax.set_xlabel("Observation delay (control steps)" if label else "")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.figure.canvas.draw()
    add_ms_axis(ax, df["delay"].max(), ctrl_dt_ms=_ctrl_dt_ms(df))


# --------------------------------------------------------------------------------------


def fig_reward_vs_delay(df: pd.DataFrame) -> plt.Figure:
    """Q1 and Q3: how far the delay can be pushed, and whether the FM pushes it further.

    Two panels of the same runs. End-of-training reward (left) is the competence the run
    was left with; best-ever (right) is the competence it ever had. They come apart for
    the forward model at delay >= 12, which is the difference between "never learned it"
    and "learned it and lost it" -- a distinction the left panel alone cannot make.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, column, title in [
        (axes[0], "reward_final", "End of training (mean of last 50 M steps)"),
        (axes[1], "reward_max", "Best eval point during training"),
    ]:
        for arm in ARMS:
            group = df[df.condition == arm]
            if not group.empty:
                plot_seeds(ax, group, x="delay", y=column, condition=arm)
        ax.set_title(title, fontsize=9, pad=40)
        ax.set_ylabel(reward_label("dmc_eval"))
        ax.set_ylim(600, 1000)
        _delay_axis(ax, df)
    axes[0].add_artist(axes[0].legend(frameon=False, fontsize=7, loc="lower left"))
    axes[0].legend(handles=seed_legend_handles(), fontsize=6, loc="lower right",
                   frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_seed_spread(df: pd.DataFrame) -> plt.Figure:
    """Q2: how much of the delay curve is seed noise.

    Left, the evidence: every baseline run, one thin line per seed setting under the
    mean. Right, the summary: the within-delay standard deviation and full range across
    the three seeds. The point of the right panel is that the spread is not constant --
    it is ~5 reward up to delay 7 and ~100 beyond it, so a single-seed curve is a fair
    description of this task at short delay and not at long delay.
    """
    base = df[(df.condition == "delayed_mlp") & df.reward_final.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    plot_seeds(axes[0], base, x="delay", y="reward_final", condition="delayed_mlp")
    axes[0].set_ylabel(reward_label("dmc_eval"))
    axes[0].set_ylim(600, 1000)
    axes[0].set_title(f"{base.seed.nunique()} seed settings, {len(base)} runs",
                      fontsize=9, pad=40)
    axes[0].add_artist(axes[0].legend(frameon=False, fontsize=7, loc="lower left"))
    axes[0].legend(handles=seed_legend_handles(), fontsize=6, loc="lower right",
                   frameon=False)
    _delay_axis(axes[0], base)

    stats = base.groupby("delay")["reward_final"].agg(
        sd=lambda s: s.std(ddof=1), spread=lambda s: s.max() - s.min())
    colour = color_for("delayed_mlp")
    axes[1].plot(stats.index, stats["spread"], color=colour, marker="o", ms=4, lw=1.4,
                 ls="--", mfc="none", label="range (max $-$ min)")
    axes[1].plot(stats.index, stats["sd"], color=colour,
                 marker=marker_for("delayed_mlp"), ms=4, lw=2.2,
                 label="standard deviation")
    pooled = float(np.sqrt(np.nanmean(np.square(stats["sd"]))))
    axes[1].axhline(pooled, color="0.5", lw=1, ls=":")
    axes[1].annotate(f"pooled sd = {pooled:.0f}", xy=(0.4, pooled), xytext=(0, 4),
                     textcoords="offset points", fontsize=7, color="0.4")
    axes[1].set_ylabel("Spread across seeds (reward)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Seed-to-seed spread of the same number", fontsize=9, pad=40)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    _delay_axis(axes[1], base)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_training_curves(curves: pd.DataFrame, df: pd.DataFrame) -> plt.Figure:
    """Q4, as a shape: the whole eval series, one panel per delay.

    The 900 line is drawn in every panel so the crossing the next figure quantifies can
    be read off directly. Delay 8 exists only in the forward-model arm (it was added when
    that arm was launched), so its panel has one curve by design, not by attrition.
    """
    delays = sorted(curves["delay"].unique())
    ncol = 5
    nrow = int(np.ceil(len(delays) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.65 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    flat = [ax for row in axes for ax in row]
    ms_per_step = _ctrl_dt_ms(df)

    for ax, delay in zip(flat, delays):
        sub = curves[curves.delay == delay]
        for arm in ARMS:
            group = sub[sub.condition == arm]
            if not group.empty:
                plot_seeds(ax, group, x="step", y="reward_mean", condition=arm,
                           marker_size=0)
        ax.axhline(900, color="0.6", lw=0.8, ls=":")
        ax.set_title(f"delay {delay}  ({delay * ms_per_step:.0f} ms)", fontsize=8)
        ax.set_ylim(0, 1000)
    for ax in flat[len(delays):]:
        ax.set_visible(False)

    # One shared pair of axis labels. Per-axes labels collide: the reward label is
    # longer than a panel is tall, so two rows of it overlap in the middle.
    flat[0].add_artist(flat[0].legend(frameon=False, fontsize=6, loc="lower right"))
    flat[0].legend(handles=seed_legend_handles(), fontsize=5, loc="center right",
                   frameon=False)
    fig.tight_layout(rect=(0.018, 0.055, 1, 1))
    # After tight_layout: matplotlib reserves room for these *and* honours the rect,
    # which pushes them a quarter of the figure below the axes if set beforehand.
    fig.supxlabel("Environment steps", fontsize=10, y=0.025)
    fig.supylabel(reward_label("dmc_eval"), fontsize=10, x=0.008)
    return fig


def fig_steps_to_threshold(df: pd.DataFrame) -> plt.Figure:
    """Q4, as a number: steps to reach the threshold, one panel per threshold.

    Individual runs, not a mean. Five baseline runs never reach 900 inside the 480 M
    budget, and any summary statistic over the rest would silently convert "failed" into
    "fast". Those runs are the open upward triangles on the ceiling line; read them as
    "> 480 M", not as a value.

    A small horizontal jitter separates the three baseline seeds at one delay; it is
    cosmetic and carries no information.
    """
    fig, axes = plt.subplots(1, len(THRESHOLD_PANELS), figsize=(9.6, 4.2), sharey=True)
    for ax, threshold in zip(np.atleast_1d(axes), THRESHOLD_PANELS):
        column = f"steps_to_{threshold}"
        for sign, arm in zip((-1, 1), ARMS):
            group = df[df.condition == arm]
            if group.empty:
                continue
            colour, marker = color_for(arm), marker_for(arm)
            jitter = sign * 0.18
            reached = group[group[column].notna()]
            ax.scatter(reached["delay"] + jitter, reached[column] / 1e6,
                       color=colour, marker=marker, s=26, zorder=3, alpha=0.9)
            censored = group[group[column].isna()]
            ax.scatter(censored["delay"] + jitter,
                       np.full(len(censored), TOTAL_STEPS / 1e6),
                       facecolors="none", edgecolors=colour, marker="^", s=46,
                       linewidths=1.2, zorder=3)
            # The median as a guide. Censored runs enter it as +inf rather than being
            # dropped: the median of {221, 274, >480} is 274, not the 247 that averaging
            # the two finite values gives, and dropping them biases the baseline fast at
            # exactly the delays where it is struggling. Where more than half the cell is
            # censored the median is itself censored and no point is drawn.
            filled = group.assign(**{column: group[column].fillna(np.inf)})
            median = filled.groupby("delay")[column].median()
            censored_cells = set(median.index[np.isinf(median)])
            median = (median[np.isfinite(median)] / 1e6).sort_index()
            n_censored = group.groupby("delay")[column].apply(lambda s: s.isna().sum())
            for x0, x1 in zip(median.index[:-1], median.index[1:]):
                # Dashed where either endpoint's cell contains a censored run: the
                # median is then a real median but the cell is not fully observed.
                partial = bool(n_censored.get(x0, 0) or n_censored.get(x1, 0))
                ax.plot([x0, x1], [median[x0], median[x1]], color=colour, lw=1.8,
                        alpha=0.85, ls="--" if partial else "-", zorder=2)
            for x in censored_cells:
                ax.annotate("median\n> budget", xy=(x, TOTAL_STEPS / 1e6),
                            xytext=(-4, -30), textcoords="offset points", fontsize=5.5,
                            color=colour, ha="right", va="top")

        ax.axhline(TOTAL_STEPS / 1e6, color="0.6", lw=0.9, ls=":")
        # Only where there is something censored to explain: at 800 every run crossed,
        # and the note would sit on the legend saying nothing.
        if df[column].isna().any():
            ax.annotate("training budget (480 M) —\nopen markers never crossed",
                        xy=(0.02, TOTAL_STEPS / 1e6), xycoords=("axes fraction", "data"),
                        xytext=(0, -16), textcoords="offset points", fontsize=6.5,
                        color="0.4", va="top")
        ax.set_title(f"Steps to reach {threshold} eval reward", fontsize=9, pad=40)
        ax.set_xlabel("Observation delay (control steps)")
        ax.set_ylim(0, TOTAL_STEPS / 1e6 * 1.08)
        _delay_axis(ax, df)
    axes[0].set_ylabel("Environment steps to threshold (millions)")

    handles = [Line2D([], [], color=color_for(a), marker=marker_for(a), lw=1.8,
                      ls="-", ms=5, label=label_for(a)) for a in ARMS]
    handles += [
        Line2D([], [], color="0.4", lw=1.8, ls="--", label="median (some runs censored)"),
        Line2D([], [], color="none", marker="^", mfc="none", mec="0.4", ms=7,
               label=f"never reached in {TOTAL_STEPS / 1e6:.0f} M"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# --------------------------------------------------------------------------------------


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)

    sources = df["reward_source"].dropna().unique()
    if len(sources) > 1:
        raise SystemExit(
            f"data.csv mixes reward sources {list(sources)}; before the `971ab99` "
            "eval-env fix the eval series was scaled x10 and truncated, so the two are "
            "in different units. Split the figure or restrict the cohort.")

    manifest = {}
    for name, builder, inputs in [
        ("reward_vs_delay", lambda: fig_reward_vs_delay(df), (DATA,)),
        ("seed_spread", lambda: fig_seed_spread(df), (DATA,)),
        ("training_curves", lambda: fig_training_curves(curves, df), (CURVES, DATA)),
        ("steps_to_threshold", lambda: fig_steps_to_threshold(df), (DATA,)),
    ]:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
