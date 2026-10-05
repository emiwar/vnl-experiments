"""Figures for ball-in-cup-first-look (dm_control_suite).

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/ball-in-cup-first-look/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/ball-in-cup-first-look/plot.py

Figures
-------
  training_curves.png   every eval series, one row per arm, one column per delay
  seed_medians.png      time to solve and reliability per arm x seed -- the unit of
                        replication (see extract.py: delays at one seed share 96-100 %
                        of their initial weights)
  steps_to_solve.png    steps to first catch vs delay, one panel per seed
  solve_rates.png       fraction of runs that ever solve / never lose it / end solved
  solved_over_time.png  cumulative fraction ever solved, and fraction solved *now*
  reward_vs_delay.png   end-of-training and whole-run mean reward vs delay

Raw reward on every reward axis (README §7). The rate and time-to-event axes are not a
substitute for reward but the readout the bimodality forces: a cell mean of {0, 950, 950}
describes no run in the cell, so the reward panel is kept and the rates sit beside it.

Uncertainty is drawn per seed, never as a binomial interval over runs: runs at different
delays from one seed are near-replicates, so an interval over 27 runs would claim far
more precision than three initialisations can give.
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

ARMS = ["no_efference", "efference", "flat_forward_model"]
REFERENCE = "undelayed"
REFERENCE_FM = "undelayed_fm"
#: The delay-0 reference each arm is drawn against: at delay 0 the two `DelayedMLP` arms
#: are one network, while the forward model has its own (predictor with nothing to do).
REFERENCE_FOR = {"no_efference": REFERENCE, "efference": REFERENCE,
                 "flat_forward_model": REFERENCE_FM}
SEEDS = ["i42/p1234", "i43/p43", "i44/p44"]
SEED_MARKERS = dict(zip(SEEDS, ("o", "s", "D")))
TOTAL_STEPS = 480_000_000
SOLVE_LEVEL = 500  # must match extract.py
#: Eval cadence; used to put every run's series on one step grid for the "solved now"
#: curve (requeued runs log slightly shifted steps).
EVAL_EVERY = 4_800_000


def _ctrl_dt_ms(df: pd.DataFrame) -> float:
    values = df["ctrl_dt"].dropna().unique()
    if len(values) != 1:
        raise SystemExit(f"data.csv mixes ctrl_dt {list(values)}")
    return float(values[0]) * 1000


def _delay_axis(ax, df):
    ax.set_xlabel("Observation delay (control steps)")
    ax.set_xticks(sorted(df["delay"].unique()))
    ax.tick_params(axis="x", labelsize=7)
    ax.figure.canvas.draw()
    top = add_ms_axis(ax, df["delay"].max(), ctrl_dt_ms=_ctrl_dt_ms(df))
    top.tick_params(axis="x", labelsize=6)
    top.xaxis.label.set_size(8)


def _log_steps_axis(ax, label=True):
    ax.set_yscale("log")
    ax.set_ylim(3, TOTAL_STEPS / 1e6 * 1.6)
    ax.set_yticks([5, 10, 30, 100, 300, 480], ["5", "10", "30", "100", "300", "480"])
    ax.minorticks_off()
    if label:
        ax.set_ylabel(f"Steps to first eval ≥ {SOLVE_LEVEL}\n(millions, log)")
    ax.axhline(TOTAL_STEPS / 1e6, color="0.6", lw=0.9, ls=":")


def _median_steps(s: pd.Series) -> float:
    """Median with never-solved counted as slowest; clipped to the budget for drawing."""
    return min(float(np.median(s.astype(float).fillna(np.inf))), TOTAL_STEPS) / 1e6


# --------------------------------------------------------------------------------------


def fig_training_curves(curves: pd.DataFrame, df: pd.DataFrame) -> plt.Figure:
    """Every run. Rows are arms, columns delays; the first column holds each arm's own
    delay-0 reference (`DelayedMLP` in grey for the two MLP rows, `FlatForwardModel`
    for the forward-model row).

    One line per seed setting, shaded by seed. No mean is drawn: the mean of a 0-or-950
    process is a line no run follows. Read along a row at one shade: that is one
    initialisation tried at ten delays, and it tends to behave the same way throughout.
    """
    delays = sorted(df["delay"].unique())
    alphas = dict(zip(SEEDS, (1.0, 0.6, 0.32)))
    ms = _ctrl_dt_ms(df)
    fig, axes = plt.subplots(len(ARMS), len(delays), figsize=(1.6 * len(delays), 5.6),
                             sharex=True, sharey=True, squeeze=False)
    for i, arm in enumerate(ARMS):
        for j, delay in enumerate(delays):
            ax = axes[i, j]
            cond = REFERENCE_FOR[arm] if delay == 0 else arm
            sub = curves[(curves.condition == cond) & (curves.delay == delay)]
            for seed, g in sub.groupby("seed"):
                ax.plot(g.step / 1e6, g.reward_mean, color=color_for(cond),
                        alpha=alphas[seed], lw=0.9)
            ax.axhline(SOLVE_LEVEL, color="0.75", lw=0.6, ls=":")
            if i == 0:
                ax.set_title(f"delay {delay}\n({delay * ms:.0f} ms)", fontsize=7)
            if j == 0:
                ax.set_ylabel(label_for(arm).replace(" ", "\n", 1), fontsize=7)
            if sub.empty:
                ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center",
                        va="center", fontsize=6, color="0.5")
            ax.set_ylim(-30, 1030)
            ax.set_xticks([0, 240, 480])
            ax.tick_params(labelsize=6)
    handles = [Line2D([], [], color="0.3", alpha=alphas[s], lw=1.2, label=s)
               for s in SEEDS]
    handles.append(Line2D([], [], color=color_for(REFERENCE), lw=1.2,
                          label="delay-0 reference (column 1)"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=6.5,
               frameon=False, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(rect=(0.03, 0.06, 1, 0.95))
    fig.supxlabel("Environment steps (millions)", fontsize=9, y=0.03)
    fig.supylabel(reward_label("dmc_eval"), fontsize=9, x=0.005)
    return fig


def fig_seed_medians(df: pd.DataFrame) -> plt.Figure:
    """The arm comparison at the level it is actually replicated: one point per
    (arm, seed), lines joining the same seed across arms. A seed shares 84-99 % of its
    initial weights across the three arms, so each line is close to a paired
    comparison. The question each panel asks is whether the lines agree in slope; where
    they cross, the arm effect is not distinguishable from which initialisation a run
    happened to get."""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
    xs = np.arange(len(ARMS))
    panels = [
        ("steps", "Median steps to first catch\n(over each seed's delays)"),
        ("never_lost", "Fraction solved and never lost it"),
        ("solved_at_end", "Fraction solved at end (last 50 M)"),
    ]
    for ax, (what, title) in zip(axes, panels):
        for seed in SEEDS:
            ys = []
            for arm in ARMS:
                g = df[(df.condition == arm) & (df.seed == seed)]
                ys.append(_median_steps(g.steps_to_solve) if what == "steps"
                          else g[what].mean())
            ax.plot(xs, ys, color="0.55", lw=1.0, zorder=1)
            for x, y, arm in zip(xs, ys, ARMS):
                ax.scatter(x, y, color=color_for(arm), marker=SEED_MARKERS[seed], s=42,
                           zorder=3, edgecolors="white", linewidths=0.6)
        if what == "steps":
            _log_steps_axis(ax, label=False)
        else:
            ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(xs, [label_for(a).replace(" ", "\n", 1) for a in ARMS], fontsize=7)
        ax.set_xlim(-0.4, len(ARMS) - 0.6)
        ax.set_title(title, fontsize=8.5)
    handles = [Line2D([], [], color="0.4", marker=SEED_MARKERS[s], ls="none", ms=6,
                      label=f"seed {s}") for s in SEEDS]
    fig.legend(handles=handles, frameon=False, fontsize=6.5, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def fig_steps_to_solve(df: pd.DataFrame) -> plt.Figure:
    """Steps to the first caught eval, one panel per seed. Never-solved runs are open
    triangles on the budget line -- failures, not slow successes. The delay-0 references
    sit at x = 0 (grey star: `DelayedMLP`; green star: `FlatForwardModel`).

    The first eval is at 4.9 M steps, which is the floor. Within a panel the arms keep
    roughly their order across delays; between panels the order changes -- that is the
    seed effect the pooled view hides."""
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(12.0, 4.0), sharey=True)
    offsets = {REFERENCE: -0.3, REFERENCE_FM: 0.3,
               "no_efference": -0.3, "efference": 0.0, "flat_forward_model": 0.3}
    for ax, seed in zip(axes, SEEDS):
        sub = df[df.seed == seed]
        for cond in [REFERENCE, REFERENCE_FM] + ARMS:
            g = sub[sub.condition == cond].sort_values("delay")
            if g.empty:
                continue
            colour, marker = color_for(cond), marker_for(cond)
            x = g["delay"] + offsets[cond]
            reached = g["steps_to_solve"].notna()
            ax.scatter(x[reached], g.loc[reached, "steps_to_solve"] / 1e6, color=colour,
                       marker=marker, s=60 if marker == "*" else 26, alpha=0.9, zorder=3,
                       label=label_for(cond))
            ax.scatter(x[~reached], np.full((~reached).sum(), TOTAL_STEPS / 1e6),
                       facecolors="none", edgecolors=colour, marker="^", s=46,
                       linewidths=1.2, zorder=3)
            if cond in ARMS:
                y = g.steps_to_solve.fillna(TOTAL_STEPS) / 1e6
                ax.plot(x, y, color=colour, lw=0.9, alpha=0.45, zorder=2)
        _log_steps_axis(ax, label=ax is axes[0])
        ax.set_title(f"seed {seed}", fontsize=9, pad=36)
        _delay_axis(ax, df)
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="none", marker="^", mfc="none", mec="0.4", ms=7,
                       label="never solved in 480 M")]
    fig.legend(handles=handles, frameon=False, fontsize=6.5, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def fig_solve_rates(df: pd.DataFrame) -> plt.Figure:
    """The fraction-of-runs readouts, pooled over delays > 0. Bars are the pooled
    fraction; the three markers on each bar are the three seeds' own fractions (nine
    delays each). The spread of the markers, not a binomial interval, is the uncertainty
    -- see the module docstring. 'Never lost' = solved, and never then below the level
    for >= 2 consecutive evals; 'at end' = solved over the last 50 M steps."""
    measures = [("solved_ever", "ever solved"),
                ("never_lost", "solved, and never lost it\n(no collapse >= 2 evals)"),
                ("solved_at_end", "solved at end\n(last 50 M steps)")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.8 / len(ARMS)
    for k, cond in enumerate(ARMS):
        g = df[df.condition == cond]
        for m, (col, _) in enumerate(measures):
            x = m + (k - (len(ARMS) - 1) / 2) * width
            ax.bar(x, g[col].mean(), width * 0.92, color=color_for(cond), alpha=0.75,
                   label=f"{label_for(cond)} (n = {len(g)})" if m == 0 else None)
            for j, seed in enumerate(SEEDS):
                gs = g[g.seed == seed]
                ax.scatter(x + (j - 1) * width * 0.22, gs[col].mean(), color="0.15",
                           marker=SEED_MARKERS[seed], s=12, zorder=3)
    ax.set_xticks(range(len(measures)), [m[1] for m in measures], fontsize=8)
    ax.set_ylabel("Fraction of runs")
    ax.set_ylim(0, 1.05)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.15", marker=SEED_MARKERS[s], ls="none", ms=4,
                       label=f"seed {s}") for s in SEEDS]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.28), ncol=3)
    ax.set_title("Pooled over delays 2-25; markers = each seed's own fraction",
                 fontsize=7, color="0.4", loc="left", y=-0.3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_solved_over_time(curves: pd.DataFrame, df: pd.DataFrame) -> plt.Figure:
    """Left: cumulative fraction of runs that have caught the ball at least once -- a
    survival curve for 'not yet solved', with no censoring other than the budget.
    Right: fraction of runs in the caught state at each eval, which is the left curve
    minus whatever has been forgotten. Thick lines pool the arm's runs; thin lines are
    each seed's nine delays on their own."""
    grid = curves.assign(t=(curves.step / EVAL_EVERY).round().astype(int))
    grid = grid.groupby(["wandb_id", "condition", "seed", "t"],
                        as_index=False)["reward_mean"].mean()
    # Requeued runs skip or shift the odd eval, so put every run on one full grid and
    # carry its last state forward; otherwise the denominator changes from point to
    # point and the cumulative curve is not monotone.
    wide = (grid.assign(solved=grid.reward_mean >= SOLVE_LEVEL)
            .pivot(index="t", columns="wandb_id", values="solved")
            .astype(float).reindex(range(1, int(grid.t.max()) + 1)).ffill().fillna(0)
            .astype(bool))
    meta = grid.drop_duplicates("wandb_id").set_index("wandb_id")[["condition", "seed"]]
    grid = wide.stack().rename("solved").reset_index().join(meta, on="wandb_id")
    grid["ever"] = grid.sort_values("t").groupby("wandb_id")["solved"].cummax()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for cond in ARMS:
        g = grid[grid.condition == cond]
        n = g.wandb_id.nunique()
        for ax, col in zip(axes, ("ever", "solved")):
            frac = g.groupby("t")[col].mean()
            ax.step(frac.index * EVAL_EVERY / 1e6, frac.values, where="post",
                    color=color_for(cond), lw=1.9, label=f"{label_for(cond)} (n = {n})")
            for _, gs in g.groupby("seed"):
                frac = gs.groupby("t")[col].mean()
                ax.step(frac.index * EVAL_EVERY / 1e6, frac.values, where="post",
                        color=color_for(cond), lw=0.7, alpha=0.35)
    axes[0].set_title("Has caught the ball at least once", fontsize=9)
    axes[1].set_title(f"Catching it at this eval (reward ≥ {SOLVE_LEVEL})", fontsize=9)
    axes[0].set_ylabel("Fraction of runs")
    for ax in axes:
        ax.set_xlabel("Environment steps (millions)")
        ax.set_xlim(0, TOTAL_STEPS / 1e6)
        ax.set_ylim(0, 1.03)
    handles, _ = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", lw=0.7, alpha=0.5, label="one seed"))
    axes[1].legend(handles=handles, frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_reward_vs_delay(df: pd.DataFrame) -> plt.Figure:
    """The conventional readout, kept because it is the one every other folder shows.
    Left: mean of the last 50 M steps. Right: mean over the whole run, which folds in
    how long the run took to find the catch and how much of the time it lost it. Thin
    lines are seeds. Read the means with figure solve_rates beside them: a cell mean of
    ~600 here is one dropped seed out of three, not a mediocre policy."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, column, title in [
        (axes[0], "reward_final", "End of training (mean of last 50 M steps)"),
        (axes[1], "reward_mean_all", "Whole run (mean over all eval points)"),
    ]:
        for arm in ARMS:
            g = df[df.condition == arm]
            plot_seeds(ax, g, x="delay", y=column, condition=arm)
        for ref, dx in ((REFERENCE, -0.25), (REFERENCE_FM, 0.25)):
            g = df[df.condition == ref]
            ax.scatter(g["delay"] + dx, g[column], color=color_for(ref),
                       marker=marker_for(ref), s=50, zorder=4,
                       label=f"{label_for(ref)}, each run")
        ax.set_title(title, fontsize=9, pad=40)
        ax.set_ylim(-30, 1030)
        _delay_axis(ax, df)
    axes[0].set_ylabel(reward_label("dmc_eval"))
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles + seed_legend_handles(), frameon=False, fontsize=6.5,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


# --------------------------------------------------------------------------------------


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)

    sources = df["reward_source"].dropna().unique()
    if len(sources) > 1:
        raise SystemExit(f"data.csv mixes reward sources {list(sources)}")
    seeds = set(df.loc[df.condition.isin(ARMS), "seed"])
    if seeds != set(SEEDS):
        raise SystemExit(f"data.csv has seeds {sorted(seeds)}; SEEDS in plot.py is "
                         f"{SEEDS}. Update it deliberately.")

    manifest = {}
    for name, builder, inputs in [
        ("training_curves", lambda: fig_training_curves(curves, df), (CURVES, DATA)),
        ("seed_medians", lambda: fig_seed_medians(df), (DATA,)),
        ("steps_to_solve", lambda: fig_steps_to_solve(df), (DATA,)),
        ("solve_rates", lambda: fig_solve_rates(df), (DATA,)),
        ("solved_over_time", lambda: fig_solved_over_time(curves, df), (CURVES, DATA)),
        ("reward_vs_delay", lambda: fig_reward_vs_delay(df), (DATA,)),
    ]:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
