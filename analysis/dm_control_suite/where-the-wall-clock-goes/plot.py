"""Figures for where-the-wall-clock-goes.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/where-the-wall-clock-goes/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/where-the-wall-clock-goes/plot.py

Four figures, in the order the question is asked:

1. ``time_budget`` -- where the hours go, per condition, absolutely and as a share.
2. ``cost_per_event`` -- what one of each thing costs, and how many of them a run does.
   This is the figure the cadence argument is made from: a bucket is a count times a unit
   cost, and only the count is a free parameter.
3. ``outage`` -- the stalled iterations on the clock, which is what separates "these runs
   are slow" from "the filesystem went away on the evening of 2026-09-14".
4. ``cadence_savings`` -- what the wall clock would have been at slower cadences.

Seconds, not reward, so the raw-reward conventions in ``../README.md`` §7 do not apply;
the one that carries over is the reason behind them. A CartpoleSwingup run is 0.5 h and a
CheetahRun 10 h, so absolute hours stay the primary panel and the share panel sits beside
it rather than replacing it -- a percentage alone would say a Cartpole run and a Cheetah
run have the same problem.

Unit costs are plotted as points, not bars: the x-axes span two orders of magnitude and
have to be logarithmic, and the length of a bar on a log axis encodes nothing.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    apply_style,
    bucket_color,
    bucket_label,
    provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
STALLS = HERE / "stalls.csv"

#: Stack order: the periodic work the question is about sits together, above training,
#: with the stall on the outside so the healthy part of a bar stays comparable.
STACK = ["train_s", "eval_s", "video_s", "checkpoint_s", "overhead_s",
         "startup_tail_s", "stall_s"]

#: What one run does of each thing, and what one of them costs. Names match the columns.
EVENTS = [("iteration", "PPO step", "o"), ("eval", "eval rollout", "s"),
          ("video", "video", "^"), ("checkpoint", "checkpoint", "D")]
EVENT_BUCKET = {"iteration": "train_s", "eval": "eval_s",
                "video": "video_s", "checkpoint": "checkpoint_s"}

#: Which bucket a hang of each kind was charged to -- for colouring only; see
#: `timing.STALL_BUCKET`, which is the same mapping used to take it back out.
STALL_BUCKET = {"checkpoint": "checkpoint_s", "video": "video_s", "eval": "eval_s",
                "other": "overhead_s"}
STALL_MARKER = {"checkpoint": "s", "video": "^", "eval": "o", "other": "D"}

SCENARIO_LABEL = {
    "save_eval_4p8M_s": "eval every 4.8 M (100 per run, as intended)",
    "save_eval_10M_s": "eval every 10 M",
    "save_video_30M_s": "video every 30 M",
    "save_checkpoint_50M_s": "checkpoint every 50 M",
    "save_intended_s": "eval 4.8 M + video 30 M",
}

#: Window the outage zoom panel covers. Set from the data, not hard-coded, so it follows
#: if the cohort grows -- but padded, because the interesting thing is how *narrow* the
#: cluster is against the hours around it.
ZOOM_PAD = pd.Timedelta(minutes=75)


def _order(df: pd.DataFrame) -> list[str]:
    """Conditions, slowest run first, so the figure opens on the expensive cases."""
    return list(df.groupby("condition")["wall_clock_s"].median()
                .sort_values(ascending=False).index)


def _cond_label(df: pd.DataFrame, condition: str) -> str:
    sub = df[df["condition"] == condition]
    return f"{condition}  (n={len(sub)}, {sub['actual_step'].median() / 1e6:.0f} M)"


def _rows(ax, y: np.ndarray, labels: list[str]) -> None:
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()


def fig_time_budget(df: pd.DataFrame) -> plt.Figure:
    """Median hours per run, by bucket -- absolute and as a share of the wall clock.

    The bucket values are the *healthy* ones, with time lost to hangs pulled out into its
    own segment rather than left inside whichever bucket happened to be running when the
    filesystem stopped answering. Leaving it in reads as "checkpointing is 12 % of a
    CheetahRun", when checkpointing is 0.1 % of it and an outage was the other 12 %.
    """
    order = _order(df)
    med = df.groupby("condition").median(numeric_only=True)
    hours = pd.DataFrame(
        {b: med.loc[order, f"healthy_{b}"] if b != "stall_s" else med.loc[order, b]
         for b in STACK}) / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(order))
    for ax, frame, xlabel in (
        (axes[0], hours, "Median hours per run"),
        (axes[1], 100 * hours.div(hours.sum(axis=1), axis=0), "Share of wall clock (%)"),
    ):
        left = np.zeros(len(order))
        for bucket in STACK:
            values = frame[bucket].to_numpy()
            ax.barh(y, values, left=left, color=bucket_color(bucket),
                    label=bucket_label(bucket), height=0.72,
                    edgecolor="white", linewidth=0.4)
            left += values
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[1].set_xlim(0, 100)

    _rows(axes[0], y, [_cond_label(df, c) for c in order])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Wall clock of a dm_control run, by what it was spent on", fontsize=10)
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    return fig


def fig_cost_per_event(df: pd.DataFrame) -> plt.Figure:
    """Seconds for one of each thing (left) and how many a run does (right).

    A bucket is a count times a unit cost. The unit costs are physics and hardware and
    are not going to move; the counts are three integers in ``conf/train/dmc.yaml``.
    Plotting them apart is the whole argument for changing the cadences rather than
    anything else.

    The inversion that does most of the damage is visible on the left: on the Walker and
    Humanoid tasks one *eval* costs several times one *training step*, because an eval is
    1 000 sequential steps of 256 environments -- latency-bound -- while a training step
    is 30 steps of 8 192.
    """
    order = _order(df)
    med = df.groupby("condition").median(numeric_only=True).loc[order]
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in axes:
        for yi in y:
            ax.axhline(yi, color="0.9", lw=6, zorder=0)
    for name, label, marker in EVENTS:
        color = bucket_color(EVENT_BUCKET[name])
        axes[0].scatter(med[f"{name}_s_median"], y, s=46, marker=marker, color=color,
                        label=f"one {label}", zorder=3, edgecolor="white", linewidth=0.5)
        axes[1].scatter(med[f"n_{name}"], y, s=46, marker=marker, color=color, zorder=3,
                        edgecolor="white", linewidth=0.5)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Seconds for one event (median over runs)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Events per run")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(False)
    _rows(axes[0], y, [_cond_label(df, c) for c in order])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("A bucket is a count times a unit cost; only the count is a setting",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    return fig


def fig_outage(stalls: pd.DataFrame, df: pd.DataFrame) -> plt.Figure:
    """Every stalled iteration on the UTC clock, and the hours each run lost.

    The point of the left panels is coincidence, not magnitude: runs on two GPU models,
    on different nodes, running four different tasks, all stopped inside the same few
    minutes. Nothing about a task or a network does that.

    Top-left is the whole cohort's span, so the reader can see that the rest of the
    fortnight is quiet; bottom-left zooms on the evening, where the cluster is resolvable.
    """
    fig = plt.figure(figsize=(11.5, 5.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1],
                            hspace=0.62, wspace=0.28)
    ax_all, ax_zoom, ax_runs = (fig.add_subplot(grid[0, 0]),
                                fig.add_subplot(grid[1, 0]),
                                fig.add_subplot(grid[:, 1]))

    if stalls.empty:
        for ax in (ax_all, ax_zoom, ax_runs):
            ax.text(0.5, 0.5, "no stalled iterations", ha="center", transform=ax.transAxes)
        return fig

    at = pd.to_datetime(stalls["at_utc"])
    worst = at[stalls["excess_s"].idxmax()]
    window = (worst - ZOOM_PAD, worst + ZOOM_PAD)

    # Twenty runs stalled within seconds of each other, so in the zoom they would draw
    # as one marker. Jitter the zoom panel only, deterministically, and say so: the
    # spread is drawn, not measured.
    rng = np.random.default_rng(0)
    jitter = pd.to_timedelta(rng.uniform(-150, 150, len(stalls)), unit="s")

    for ax, title, offset in (
        (ax_all, "whole cohort", pd.to_timedelta(np.zeros(len(stalls)), unit="s")),
        (ax_zoom, f"zoom: {worst.strftime('%Y-%m-%d')} evening (x jittered +/- 2.5 min)",
         jitter),
    ):
        for kind in stalls["kind"].unique():
            m = (stalls["kind"] == kind).to_numpy()
            ax.scatter((at + offset)[m], stalls.loc[m, "excess_s"] / 60.0, s=30,
                       alpha=0.85, marker=STALL_MARKER.get(kind, "o"),
                       color=bucket_color(STALL_BUCKET.get(kind, "overhead_s")),
                       edgecolor="white", linewidth=0.4, label=f"stalled in {kind}")
        ax.set_yscale("log")
        ax.set_ylabel("Minutes lost", fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, loc="left", color="0.35")
        ax.tick_params(axis="x", labelsize=7)
    ax_zoom.set_xlim(*window)
    ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=at.dt.tz))
    ax_zoom.set_xlabel("UTC", fontsize=8)
    ax_all.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=at.dt.tz))
    ax_all.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)

    n_in_window = int(((at >= window[0]) & (at <= window[1])).sum())
    lost_in_window = float(stalls.loc[(at >= window[0]) & (at <= window[1]),
                                      "excess_s"].sum()) / 3600.0

    # Right: how much each affected run lost. Runs that lost under a minute are the
    # ordinary video-upload hiccups and would draw as invisible bars.
    lost = (df[df["stall_s"] > 60].sort_values("stall_s", ascending=False)
            .head(24).iloc[::-1])
    ax_runs.barh(np.arange(len(lost)), lost["stall_s"] / 3600.0,
                 color=bucket_color("stall_s"), height=0.7)
    ax_runs.set_yticks(np.arange(len(lost)))
    ax_runs.set_yticklabels([f"{r.task} {r.wandb_id}" for r in lost.itertuples()],
                            fontsize=6)
    ax_runs.set_xlabel("Hours lost by that run")
    ax_runs.grid(axis="x", alpha=0.25)
    ax_runs.set_axisbelow(True)
    ax_runs.set_title(f"{len(lost)} runs lost more than a minute", fontsize=8,
                      loc="left", color="0.35")

    fig.suptitle(
        f"{len(stalls)} stalled iterations over the fortnight; {n_in_window} of them "
        f"({lost_in_window:.0f} of {stalls['excess_s'].sum() / 3600:.0f} h lost) "
        f"within {2 * ZOOM_PAD.seconds // 3600} h on {worst.strftime('%Y-%m-%d')}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig


def fig_cadence_savings(df: pd.DataFrame) -> plt.Figure:
    """Hours a run would have saved at each slower cadence, and what share that is.

    Priced at the *median* cost of each event, not the total, so a run that lost an hour
    to the outage is not credited with saving that hour again by evaluating less often.
    The bars are what a healthy run saves, which is the number to plan with.

    The share panel is here because the hours panel alone understates the small tasks:
    0.18 h off a CartpoleSwingup run is the same 30 % it is off a Humanoid run.
    """
    order = _order(df)
    med = df.groupby("condition").median(numeric_only=True).loc[order]
    y = np.arange(len(order))
    scenarios = [c for c in SCENARIO_LABEL if c in med]
    offsets = np.linspace(0.32, -0.32, len(scenarios))
    shades = plt.cm.viridis(np.linspace(0.12, 0.82, len(scenarios)))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    for column, dy, color in zip(scenarios, offsets, shades):
        axes[0].barh(y + dy, med[column] / 3600.0, height=0.16, color=color,
                     label=SCENARIO_LABEL[column])
        axes[1].barh(y + dy, 100 * med[column] / med["healthy_wall_s"], height=0.16,
                     color=color)
    axes[0].set_xlabel("Hours saved per run")
    axes[1].set_xlabel("Share of a healthy run's wall clock saved (%)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    _rows(axes[0], y, [f"{c}  ({med.loc[c, 'healthy_wall_s'] / 3600:.1f} h now)"
                       for c in order])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("What a slower cadence would have saved, per run", fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    stalls = pd.read_csv(STALLS) if STALLS.exists() else pd.DataFrame()

    missing = int((~df["has_timing"]).sum())
    if missing:
        raise SystemExit(
            f"{missing} of {len(df)} runs in data.csv have no timing artifact, so their "
            f"buckets are blank and every median here would be over a silently different "
            f"subset. Close the gap (see coverage.txt) and re-run extract.py first.")

    builders = [
        ("time_budget", lambda: fig_time_budget(df)),
        ("cost_per_event", lambda: fig_cost_per_event(df)),
        ("outage", lambda: fig_outage(stalls, df)),
        ("cadence_savings", lambda: fig_cadence_savings(df)),
    ]
    manifest = {}
    for name, builder in builders:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, DATA, STALLS)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
