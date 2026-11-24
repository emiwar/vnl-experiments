"""Figures for walker-terrain-network-size.

Reads only ``data.csv`` and ``curves.csv`` in this folder.

    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-network-size/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/walker-terrain-network-size/plot.py

Most sweep shapes are one run, so every run is drawn as itself. The best seed spread
available is the 256x4 default's (7 runs), shown as a grey band (min-max) on each
panel: a sweep point outside that band is outside one shape's seed range, not
"significant".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import apply_style, provenance, write_figure_manifest

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

#: Depth is the categorical identity; fixed order, plus a marker per depth so colour
#: is never the only cue.
DEPTH_COLOR = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}
DEPTH_MARKER = {1: "o", 2: "s", 4: "D", 8: "^"}
BAND = "0.85"
READOUTS = [("reward_100M", "100 M"), ("reward_200M", "200 M"), ("reward_480M", "480 M")]


def _band(ax, vals, horizontal=True):
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    ax.axhspan(lo, hi, color=BAND, zorder=0, lw=0)


WIDTH_COLOR = {64: "#86b6ef", 128: "#5598e7", 256: "#2a78d6", 512: "#1c5cab",
               1024: "#104281"}
WIDTH_MARKER = {64: "v", 128: "s", 256: "o", 512: "D", 1024: "^"}
#: Long-run shapes, fixed categorical order.
LONG_COLOR = {"4x128": "#2a78d6", "8x512": "#eb6834", "8x1024": "#1baf7a",
              "4x512": "#eda100"}


def _sweep(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df.cohort == "sweep"].copy()
    s["shape"] = s.depth.astype(str) + "x" + s.width.astype(str)
    return s


def fig_reward_vs_params(df: pd.DataFrame) -> plt.Figure:
    """Reward at three budgets vs actor parameter count, coloured by depth.

    Every run that reached the budget is a point; the line joins per-shape means.
    """
    s = _sweep(df)
    base = s[s["shape"] == "4x256"]
    other = s.drop(base.index)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (col, label) in zip(axes, READOUTS):
        _band(ax, base[col])
        ax.scatter(base.actor_params, base[col], s=14, color="0.45", zorder=2,
                   label=f"4x256 ({base[col].notna().sum()} seeds)")
        for depth, g in other.groupby("depth"):
            g = g[g[col].notna()]
            means = g.groupby("actor_params")[col].mean()
            ax.plot(means.index, means.values, color=DEPTH_COLOR[depth], lw=1, alpha=0.6)
            ax.scatter(g.actor_params, g[col], color=DEPTH_COLOR[depth],
                       marker=DEPTH_MARKER[depth], s=36, zorder=3,
                       edgecolor="white", linewidth=0.8, label=f"depth {depth}")
            if col == "reward_480M":
                for p, sub in g.groupby("actor_params"):
                    ax.annotate(str(int(sub.width.iloc[0])), (p, sub[col].min()),
                                textcoords="offset points", xytext=(4, -9), fontsize=6,
                                color="0.35")
        ax.set_xscale("log")
        ax.set_title(f"Eval reward at {label} steps", fontsize=9)
        ax.set_xlabel("Actor parameters")
    axes[0].set_ylabel("Episode reward (mean of preceding 50 M)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Actor-size sweep, eff+priv cell (one point per run; line = shape mean; "
                 "number = width; grey band = 4x256 seed range)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def fig_curves_by_depth(df: pd.DataFrame, curves: pd.DataFrame) -> plt.Figure:
    """Eval curves to 480 M, one panel per depth; darker = wider, dashed = 2nd seed."""
    s = _sweep(df)
    s = s[s.complete]
    base_ids = s[s["shape"] == "4x256"].wandb_id
    cv = curves[curves.step <= 4.85e8]
    piv = cv[cv.wandb_id.isin(base_ids)].pivot_table(index="step", columns="wandb_id",
                                                      values="value")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True, sharex=True)
    for ax, depth in zip(axes, (1, 2, 4, 8)):
        x = piv.index / 1e6
        ax.fill_between(x, piv.min(axis=1), piv.max(axis=1), color=BAND, lw=0,
                        label=f"4x256 seed range ({len(base_ids)})")
        g = s[(s.depth == depth) & ~s.wandb_id.isin(base_ids)]
        widths = sorted(g.width.unique())
        shades = dict(zip(widths, np.linspace(0.35, 1.0, len(widths))))
        base_rgb = np.array(plt.matplotlib.colors.to_rgb(DEPTH_COLOR[depth]))
        for w, sub in g.sort_values(["width", "seed_pair"]).groupby("width"):
            col = 1 - shades[w] * (1 - base_rgb)  # blend toward white for narrow nets
            for k, (_, r) in enumerate(sub.iterrows()):
                c = cv[cv.wandb_id == r.wandb_id]
                ax.plot(c.step / 1e6, c.value.rolling(3, min_periods=1).mean(),
                        color=col, lw=1.4, ls="-" if k == 0 else "--",
                        label=f"{depth}x{int(w)}" + (f" ({len(sub)} seeds)"
                                                       if len(sub) > 1 and k == 0
                                                       else "") if k == 0 else None)
        ax.set_title(f"depth {depth}", fontsize=9)
        ax.set_xlabel("Training steps (M)")
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    axes[0].set_ylabel("Eval reward (3-point rolling mean)")
    fig.suptitle("Learning curves by actor depth; darker = wider, dashed = second seed",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def fig_long_budget(df: pd.DataFrame, curves: pd.DataFrame) -> plt.Figure:
    """Every sweep run trained well past 480 M."""
    s = _sweep(df)
    long = s[s.steps_trained > 6e8]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    base = long[long["shape"] == "4x256"]
    for i, w in enumerate(base.wandb_id):
        c = curves[curves.wandb_id == w]
        ax.plot(c.step / 1e6, c.value.rolling(5, min_periods=1).mean(), color="0.55",
                lw=1, label=f"4x256 ({len(base)} seeds)" if i == 0 else None)
    for _, r in long[long["shape"] != "4x256"].iterrows():
        c = curves[curves.wandb_id == r.wandb_id]
        end = "" if r.steps_trained > 1.99e9 else f", crashed at {r.steps_trained / 1e6:.0f} M"
        ax.plot(c.step / 1e6, c.value.rolling(5, min_periods=1).mean(),
                color=LONG_COLOR[r["shape"]], lw=1.6,
                label=f"{r['shape']} ({r.actor_params / 1e6:.2f} M params, "
                      f"{r.seed_pair}{end})")
    ax.axvline(480, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Eval reward (5-point rolling mean)")
    ax.set_title("Every eff+priv run trained past 480 M", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_factors_by_width(df: pd.DataFrame) -> plt.Figure:
    """Depth-4 actors of width 64-1024 in every privileged-critic factor cell."""
    d = df[(df.depth == 4) & df.width.isin(list(WIDTH_COLOR)) & df.complete]
    cells = (d[d.width == 256].groupby("cell").reward_480M.mean().sort_values().index)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    offsets = dict(zip(WIDTH_COLOR, np.linspace(-0.24, 0.24, len(WIDTH_COLOR))))
    seen = set()
    for i, cell in enumerate(cells):
        g = d[d.cell == cell]
        for w, sub in g.groupby("width"):
            ax.scatter(sub.reward_480M, np.full(len(sub), i) + offsets[w],
                       color=WIDTH_COLOR[w], marker=WIDTH_MARKER[w], s=26,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=f"4x{w}" if w not in seen else None)
            seen.add(w)
    ax.set_yticks(range(len(cells)), cells, fontsize=8)
    for i in range(len(cells)):
        ax.axhline(i + 0.5, color="0.92", lw=0.6, zorder=0)
    ax.set_xlabel("Eval reward, (430 M, 480 M]  (one marker per run)")
    ax.set_title("Actor width (depth 4) in each factor cell, sorted by the 4x256 mean\n"
                 "FM: decoder only; predictor stays 4x256", fontsize=9)
    h, l = ax.get_legend_handles_labels()
    order = np.argsort([int(x.split("x")[1]) for x in l])
    ax.legend([h[k] for k in order], [l[k] for k in order], frameon=False, fontsize=7,
              loc="lower right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    manifest = {}
    for name, builder, inputs in [
        ("reward_vs_params", lambda: fig_reward_vs_params(df), (DATA,)),
        ("curves_by_depth", lambda: fig_curves_by_depth(df, curves), (DATA, CURVES)),
        ("long_budget", lambda: fig_long_budget(df, curves), (DATA, CURVES)),
        ("factors_by_width", lambda: fig_factors_by_width(df), (DATA,)),
    ]:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")
    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
