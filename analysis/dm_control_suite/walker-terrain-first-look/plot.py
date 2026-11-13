"""Figures for walker-terrain-first-look.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/plot.py

Every endpoint reward is the mean of the eval points in the 5e7 steps up to 4.8e8, raw and
unscaled. Swept continuous values (height, delay) use single-hue sequential ramps; the
four conditions use their registered CONDITION_STYLE colours.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize

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
LONG = HERE / "long_curves.csv"

YLABEL = reward_label("dmc_eval")
DELAY_XLABEL = "Observation delay (control steps)"
HEIGHT_XLABEL = "Mean block height (m)"
CTRL_MS = 25
STEP_SCALE = 1e6
READ_M = 480
#: The seed pair the terrain sweeps were launched at; flat-ground runs at this pair are
#: the paired reference in the curve panels.
SWEEP_SEEDS = "i42/p1234"
#: Truncated so the lightest step stays visible on white.
HEIGHT_CMAP = colormaps["Oranges"]
DELAY_CMAP = colormaps["Blues"]


def _ramp(cmap, values):
    norm = Normalize(min(values) - 0.8 * (max(values) - min(values) or 1), max(values))
    return {v: cmap(norm(v)) for v in values}


def _curve(ax, c: pd.DataFrame, **kw):
    c = c.sort_values("step")
    ax.plot(c.step / STEP_SCALE, c.reward, **kw)


def _flat_mean_by_delay(df: pd.DataFrame) -> pd.Series:
    flat = df[df.condition == "flat_ground"]
    return flat.groupby(["delay_k", "seed_pair"]).reward.mean().groupby("delay_k").mean()


def fig_overview(df: pd.DataFrame) -> plt.Figure:
    """Every measured (delay, height) cell, coloured by endpoint reward."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ter = (df[df.condition == "terrain_blind"]
           .groupby(["delay_k", "height", "seed_pair"]).reward.mean()
           .groupby(["delay_k", "height"]).agg(["mean", "size"]).reset_index())
    flat = _flat_mean_by_delay(df).reset_index().assign(height=0.0)
    cells = pd.concat([flat.rename(columns={"reward": "mean"}).assign(size=np.nan), ter])
    # Delays on evenly spaced columns (a categorical axis), so 2 and 3 do not collide.
    delays = sorted(cells.delay_k.unique())
    col = {d: i for i, d in enumerate(delays)}
    x = cells.delay_k.map(col)
    sc = ax.scatter(x, cells.height, c=cells["mean"], cmap="viridis",
                    vmin=250, vmax=1000, s=330, marker="s", edgecolors="white",
                    linewidths=2)
    for xi, r in zip(x, cells.itertuples()):
        ax.annotate(f"{r.mean:.0f}", (xi, r.height), ha="center", va="center",
                    fontsize=6.5, color="white" if r.mean < 750 else "black")
    ax.set_xlabel(DELAY_XLABEL + ", ms in brackets (columns evenly spaced)")
    ax.set_ylabel(HEIGHT_XLABEL)
    ax.set_yticks(sorted(cells.height.unique()))
    ax.set_xticks(range(len(delays)), [f"{d}\n({d * CTRL_MS})" for d in delays],
                  fontsize=7.5)
    ax.set_xlim(-0.6, len(delays) - 0.4)
    ax.set_ylim(-0.04, cells.height.max() + 0.04)
    fig.colorbar(sc, ax=ax, label=YLABEL, pad=0.02)
    ax.set_title("Measured cells: height 0 is flat-ground WalkerWalk (mean over seeds),\n"
                 "the rest random blocks, unseen (1 seed pair per cell; 2 at 0.3 m, "
                 "delays 0 and 5, averaged)", fontsize=9)
    return fig


def _delay_sweep(ax, blind, flat, height, tag):
    """Delay sweep at one height on the sweep seed pair, flat-ground twins dashed.

    A delay with no sweep-pair run at this height falls back to the other pair's run,
    labelled with its seeds (its dashed twin is then absent)."""
    sub = blind[blind.height == height]
    colors = _ramp(DELAY_CMAP, sorted(sub.delay_k.unique()))
    for d, cd in sub.groupby("delay_k"):
        main = cd[cd.seed_pair == SWEEP_SEEDS]
        if main.empty:
            seed = sorted(cd.seed_pair.unique())[0]
            main, label = cd[cd.seed_pair == seed], f"delay {d} ({seed})"
        else:
            label = f"delay {d}"
        for k, (_, c) in enumerate(main.groupby("wandb_id")):
            _curve(ax, c, color=colors[d], lw=1.6, label=label if k == 0 else None)
        for _, fc in flat[flat.delay_k == d].groupby("wandb_id"):
            _curve(ax, fc, color=colors[d], lw=1.0, ls="--", alpha=0.8)
    ax.plot([], [], color="0.4", ls="--", lw=1.0, label="flat ground, same seeds")
    ax.set_title(f"({tag}) Delay sweep, random blocks {height:g} m, unseen (solid)\n"
                 "vs flat ground at the same seed pair (dashed)", fontsize=9)
    if tag == "a":
        ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    else:
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def fig_curves(df: pd.DataFrame, curves: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharey=True)
    blind = curves[curves.condition == "terrain_blind"]
    flat = curves[(curves.condition == "flat_ground") & (curves.seed_pair == SWEEP_SEEDS)]
    _delay_sweep(axes[0, 0], blind, flat, 0.1, "a")
    _delay_sweep(axes[0, 1], blind, flat, 0.3, "b")

    # (c, d) height sweeps at delay 5 and delay 0, flat ground at height 0.
    for ax, delay, tag in ((axes[1, 0], 5, "c"), (axes[1, 1], 0, "d")):
        sub = blind[blind.delay_k == delay]
        heights = sorted(sub.height.unique())
        colors = _ramp(HEIGHT_CMAP, heights)
        primary = set(sub[sub.seed_pair == SWEEP_SEEDS].height)
        seen = set()
        for (h, seed, _), c in sub.groupby(["height", "seed_pair", "wandb_id"]):
            # Thin only for a *second* seed pair at a height the sweep pair also covers;
            # a height measured only at another pair is that height's main curve.
            replicate = seed != SWEEP_SEEDS and h in primary
            first = not replicate and h not in seen
            seen.add(h)
            _curve(ax, c, color=colors[h], lw=1.0 if replicate else 1.6,
                   alpha=0.6 if replicate else 1.0,
                   label=f"{h:g} m" if first else None)
        fl = flat[flat.delay_k == delay]
        for k, (_, fc) in enumerate(fl.groupby("wandb_id")):
            _curve(ax, fc, color=color_for("flat_ground"), lw=1.2, ls="--",
                   label="flat ground" if k == 0 else None)
        ax.set_title(f"({tag}) Height sweep at delay {delay} ({delay * CTRL_MS} ms), "
                     "random blocks unseen\n(thin: a second seed pair)", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right", title="block height", title_fontsize=7)

    for ax in axes.flat:
        ax.set_ylim(0, 1000)
    for ax in axes[1]:
        ax.set_xlabel("Environment steps (M)")
    for ax in axes[:, 0]:
        ax.set_ylabel(YLABEL)
    fig.tight_layout()
    return fig


def fig_delay(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    flat = df[df.condition == "flat_ground"]
    plot_seeds(ax, flat, x="delay_k", y="reward", seed_col="seed_pair",
               condition="flat_ground")
    blind = df[df.condition == "terrain_blind"]
    for cond, h, ls in (("terrain_blind", 0.1, "-"), ("terrain_blind", 0.3, ":"),
                        ("terrain_observed", 0.3, ":")):
        sub = df[(df.condition == cond) & (df.height == h)]
        mean = sub.groupby(["delay_k", "seed_pair"]).reward.mean().groupby("delay_k").mean()
        ax.plot(mean.index, mean.values, ls=ls, color=color_for(cond),
                marker=marker_for(cond), ms=4, lw=1.8,
                label=f"{label_for(cond)}, {h:g} m")
    ax.set_xlabel(DELAY_XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_ylim(200, 1000)
    add_ms_axis(ax, 25, ctrl_dt_ms=CTRL_MS)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + seed_legend_handles(), fontsize=7, loc="lower left")
    ax.set_title("Endpoint reward vs delay. Terrain: one seed pair per point (two at "
                 "0.3 m delays 0 and 5,\naveraged; delay 10 at 0.3 m is i43/p43); "
                 "flat ground 2-7 seed pairs per delay", fontsize=9)
    return fig


def fig_height(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    flat = _flat_mean_by_delay(df)
    blind = df[df.condition == "terrain_blind"]
    delays = (0, 5, 10)
    colors = _ramp(DELAY_CMAP, delays)
    for d in delays:
        sub = blind[blind.delay_k == d]
        per = sub.groupby(["height", "seed_pair"]).reward.mean().reset_index()
        pts = pd.concat([pd.DataFrame({"height": [0.0], "reward": [flat[d]],
                                       "seed_pair": ["flat"]}), per])
        mean = pts.groupby("height").reward.mean()
        ax.plot(mean.index, mean.values, color=colors[d], marker="o", ms=4, lw=1.8,
                label=f"delay {d} ({d * CTRL_MS} ms)")
        rep = per[per.duplicated("height", keep=False)]
        ax.scatter(rep.height, rep.reward, color=colors[d], s=14, alpha=0.6, zorder=3)
    ax.set_xlabel(HEIGHT_XLABEL + "   (0 = flat-ground WalkerWalk, mean over seeds)")
    ax.set_ylabel(YLABEL)
    ax.set_ylim(300, 1000)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Endpoint reward vs block height (random blocks, unseen).\n"
                 "Small dots: individual seed pairs where a cell has two", fontsize=9)
    return fig


def fig_plateau(df: pd.DataFrame) -> plt.Figure:
    """Plateau or slow learning, in raw reward and in time-to-reach a fixed reward.

    No differences between rewards: the score is bounded, so a gap's size means
    different things at different starting levels. Time to reach a fixed level keeps
    its ordering under any monotone rescaling of the score."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    blind = df[df.condition == "terrain_blind"]
    flat = df[df.condition == "flat_ground"]

    # (a) raw reward at 100/200/300/480 M vs height at delay 5.
    ax = axes[0]
    sub = blind[blind.delay_k == 5]
    f5 = flat[flat.delay_k == 5]
    cols = ["reward_at_100M", "reward_at_200M", "reward_at_300M", "reward"]
    budget_colors = _ramp(colormaps["Greys"], list(range(len(cols))))
    for i, col in enumerate(cols):
        per = pd.concat([f5.assign(height=0.0), sub]).groupby(
            ["height", "seed_pair"])[col].mean().groupby("height").mean()
        ax.plot(per.index, per.values, marker="o", ms=4, lw=1.6,
                color=budget_colors[i],
                label=col.replace("reward_at_", "at ").replace("reward", "at 480M"))
    ax.set_xlabel(HEIGHT_XLABEL + " (0 = flat ground)")
    ax.set_ylabel(YLABEL)
    ax.set_ylim(0, 1000)
    ax.legend(fontsize=7, loc="lower left", title="read at", title_fontsize=7)
    ax.set_title("(a) Delay 5: reward read at four budgets vs height", fontsize=9)

    # (b, c) steps to reach 500 / 800 / 900 vs height; never reached drawn above 480 M.
    levels = (500, 800, 900)
    level_colors = _ramp(colormaps["Greys"], list(range(len(levels))))
    ceiling = READ_M + 40
    for ax, d, tag in ((axes[1], 5, "b"), (axes[2], 0, "c")):
        pts = pd.concat([flat[flat.delay_k == d].assign(height=0.0),
                         blind[blind.delay_k == d]])
        for i, lv in enumerate(levels):
            col = f"steps_to_{lv}"
            per = pts.groupby(["height", "seed_pair"])[col].mean().reset_index()
            per["y"] = per[col] / STEP_SCALE
            reached = per.groupby("height").y.mean()  # NaN if any pair never reached it
            missing = per[per.y.isna()].height.unique()
            ok = reached.dropna()
            ax.plot(ok.index, ok.values, marker="o", ms=4, lw=1.6, color=level_colors[i],
                    label=f"reward {lv}")
            ax.scatter(per.height, per.y, s=10, color=level_colors[i], alpha=0.5, zorder=3)
            ax.scatter(missing + (i - 1) * 0.008, np.full(len(missing), ceiling),
                       marker="o", s=30,
                       facecolors="white", edgecolors=level_colors[i], zorder=4)
        ax.axhline(READ_M, color="0.7", lw=0.8)
        ax.text(0.0, ceiling + 12, "open: not reached by 480 M", fontsize=7, color="0.3")
        ax.set_ylim(0, ceiling + 40)
        ax.set_xlabel(HEIGHT_XLABEL + " (0 = flat ground)")
        ax.set_ylabel("Environment steps to first reach the level (M)")
        ax.legend(fontsize=7, loc="center left")
        ax.set_title(f"({tag}) Delay {d} ({d * CTRL_MS} ms): steps to first reach a fixed\n"
                     "reward (lines: mean over seed pairs; dots: seed pairs)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_observed(df: pd.DataFrame, curves: pd.DataFrame) -> plt.Figure:
    obs = df[(df.condition == "terrain_observed") & (df.height == 0.3)]
    delays = sorted(obs.delay_k.unique())
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, len(delays), height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    flat = _flat_mean_by_delay(df)
    ax.plot(flat.index, flat.values, color=color_for("flat_ground"), ls="--", lw=1.2,
            marker=marker_for("flat_ground"), ms=3, label="flat ground (mean over seeds)")
    for cond in ("terrain_blind", "terrain_observed"):
        sub = df[(df.condition == cond) & (df.height == 0.3)]
        mean = sub.groupby(["delay_k", "seed_pair"]).reward.mean().groupby("delay_k").mean()
        ax.plot(mean.index, mean.values, color=color_for(cond), marker=marker_for(cond),
                ms=5, lw=1.8, label=f"{label_for(cond)}, 0.3 m")
        per = sub.groupby(["delay_k", "seed_pair"]).reward.mean().reset_index()
        rep = per[per.duplicated("delay_k", keep=False)]
        ax.scatter(rep.delay_k, rep.reward, color=color_for(cond), s=14, alpha=0.6,
                   zorder=3)
    ax.set_xlabel(DELAY_XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_ylim(200, 1000)
    add_ms_axis(ax, 25, ctrl_dt_ms=CTRL_MS)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(a) 0.3 m blocks, reward at 480 M. One observed run per delay; "
                 "small dots: individual blind seed pairs",
                 fontsize=9)
    for i, d in enumerate(delays):
        ax = fig.add_subplot(gs[1, i])
        for cond in ("terrain_blind", "terrain_observed"):
            sub = curves[(curves.condition == cond) & (curves.delay_k == d)
                         & (curves.height == 0.3)]
            for k, (_, c) in enumerate(sub.groupby("wandb_id")):
                _curve(ax, c, color=color_for(cond), lw=1.3,
                       label=label_for(cond) if k == 0 else None)
        ax.set_ylim(0, 1000)
        ax.set_xlabel("Environment steps (M)")
        if i == 0:
            ax.set_ylabel(YLABEL)
            ax.legend(fontsize=6.5, loc="lower right")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"({chr(ord('b') + i)}) delay {d} ({d * CTRL_MS} ms)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_long(df: pd.DataFrame, long: pd.DataFrame, curves: pd.DataFrame) -> plt.Figure:
    """The one terrain run trained past 480 M against its extrapolation and flat ground."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ter = long[long.condition != "flat_ground"]
    for wid, c in long[long.condition == "flat_ground"].groupby("wandb_id"):
        _curve(ax, c, color=color_for("flat_ground"), lw=1.0, alpha=0.8,
               label=None)
    ax.plot([], [], color=color_for("flat_ground"),
            label=f"flat ground, delay 5, 1 G runs ({long[long.condition == 'flat_ground'].wandb_id.nunique()} seed pairs)")
    for wid, c in ter.groupby("wandb_id"):
        row = df[df.wandb_id == wid].iloc[0]
        _curve(ax, c, color=color_for(row.condition), lw=1.6,
               label=f"{label_for(row.condition)}, {row.height:g} m, delay "
                     f"{row.delay_k}, {row.seed_pair}, 2 G run")
        # The twin(s) at the same cell trained to 480 M.
        twins = df[(df.condition == row.condition) & (df.delay_k == row.delay_k)
                   & (df.height == row.height) & (df.wandb_id != wid)]
        for k, tw in enumerate(twins.itertuples()):
            _curve(ax, curves[curves.wandb_id == tw.wandb_id], color="0.35", lw=0.9,
                   alpha=0.8, label="same cell, 480 M runs" if k == 0 else None)
        t = np.linspace(480e6, c.step.max(), 200)
        ax.plot(t / STEP_SCALE, row.fit_rinf - row.fit_a * np.exp(-t / row.fit_tau),
                color=color_for(row.condition), ls="--", lw=1.2,
                label=f"its saturating fit to <= 480 M, extrapolated (ceiling "
                      f"{row.fit_rinf:.0f})")
        for at in (480, 1000, 2000):
            col = "reward" if at == 480 else f"reward_at_{at}M"
            ax.annotate(f"{row[col]:.0f}", (at, row[col]), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7.5, color="0.2")
    ax.axvline(480, color="0.7", lw=0.8)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel(YLABEL)
    ax.set_ylim(0, 1000)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Delay 5, 0.3 m blocks trained to 2 G: still rising after 480 M, past the "
                 "ceiling its own\n480 M fit predicted. Numbers: 50 M-window reward at "
                 "480 M / 1 G / 2 G", fontsize=9)
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    long = pd.read_csv(LONG)
    figs = {
        "overview_cells.png": fig_overview(df),
        "curves.png": fig_curves(df, curves),
        "reward_vs_delay.png": fig_delay(df),
        "reward_vs_height.png": fig_height(df),
        "plateau_or_slow.png": fig_plateau(df),
        "observed_vs_blind.png": fig_observed(df, curves),
        "long_run.png": fig_long(df, long, curves),
    }
    manifest = {}
    for name, fig in figs.items():
        manifest[name] = provenance(fig, HERE, DATA, CURVES, LONG)
        fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", FIGURES / name)
    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
