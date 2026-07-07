"""Plot reference-root vs current-root frame, and FM advantage under reference-root.

Run from the repo root::

    ../.venv/bin/python analysis/reference-root-frame/plot.py

Reads ONLY data.csv (never the WandB API) and writes figures/. See analysis/README.md §2.

Visual encoding: **network** -> colour + marker (reused ``color_for``/``marker_for`` so the
efference/forward-model colours match every other FM report); **frame** -> line style
(solid = reference_root, dashed = current_root). The shared style helper has no line-style
channel, so that one convention is defined locally.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from vnl_experiments.wandb_utils import (
    add_ms_axis,
    apply_style,
    color_for,
    label_for,
    marker_for,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

FRAME_LS = {"reference_root": "-", "current_root": "--"}
FRAME_LABEL = {"reference_root": "reference-root frame", "current_root": "current-root frame"}
NETWORKS = ["efference", "forward_model"]
MAX_DELAY = 50


def dedup(df: pd.DataFrame, by, metric: str) -> pd.DataFrame:
    """Keep the highest-``metric`` row per ``by`` group (collapses the duplicate delay-0 run)."""
    return (
        df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .drop_duplicates(by)
        .sort_values("delay_k")
    )


def series(df, frame, network):
    sub = df[(df.frame == frame) & (df.network == network)]
    return dedup(sub, ["delay_k"], "reward_mean")


def fig_frame_comparison(df):
    """Q1: within each network, reference-root vs current-root across the delay sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, net in zip(axes, NETWORKS):
        for frame in ("current_root", "reference_root"):
            s = series(df, frame, net)
            ax.plot(
                s["delay_k"], s["reward_mean"],
                color=color_for(net), marker=marker_for(net),
                linestyle=FRAME_LS[frame], label=FRAME_LABEL[frame],
            )
        ax.set_title(label_for(net))
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylim(bottom=0)
        add_ms_axis(ax, MAX_DELAY)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean eval episode reward")
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    out = FIGURES / "frame_comparison.png"
    fig.savefig(out)
    print(f"Saved {out}")


def fig_fm_advantage(df):
    """Q2: FM vs efference under reference_root (left) + FM-minus-efference for both frames (right)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: reference_root, efference vs forward_model reward curves.
    for net in NETWORKS:
        s = series(df, "reference_root", net)
        axL.plot(s["delay_k"], s["reward_mean"],
                 color=color_for(net), marker=marker_for(net), label=label_for(net))
    axL.set_title("reference-root frame")
    axL.set_xlabel("Observation delay (steps)")
    axL.set_ylabel("Mean eval episode reward")
    axL.set_ylim(bottom=0)
    add_ms_axis(axL, MAX_DELAY)
    axL.legend(loc="lower left", frameon=False)
    sns.despine(ax=axL)

    # Right: forward-model advantage (FM - efference) vs delay, one line per frame.
    for frame in ("current_root", "reference_root"):
        eff = series(df, frame, "efference").set_index("delay_k")["reward_mean"]
        fm = series(df, frame, "forward_model").set_index("delay_k")["reward_mean"]
        adv = (fm - eff).dropna().sort_index()
        axR.plot(adv.index, adv.values, color=color_for("forward_model"),
                 marker="^", linestyle=FRAME_LS[frame], label=FRAME_LABEL[frame])
    axR.axhline(0, color="0.6", lw=0.8, zorder=0)
    axR.set_title("Forward-model advantage")
    axR.set_xlabel("Observation delay (steps)")
    axR.set_ylabel("Reward gain: forward model − efference")
    add_ms_axis(axR, MAX_DELAY)
    axR.legend(loc="upper left", frameon=False)
    sns.despine(ax=axR)

    fig.tight_layout()
    out = FIGURES / "fm_advantage.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    fig_frame_comparison(df)
    fig_fm_advantage(df)


if __name__ == "__main__":
    main()
