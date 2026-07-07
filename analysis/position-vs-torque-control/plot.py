"""Plot position vs torque control, and the forward-model advantage under each.

Run from the repo root::

    ../.venv/bin/python analysis/position-vs-torque-control/plot.py

Reads ONLY data.csv (never the WandB API) and writes figures/. See analysis/README.md §2.

Visual encoding: **network** -> colour + marker (reused ``color_for``/``marker_for`` so the
efference/forward-model colours match every other FM report); **control mode** -> line style
(solid = position, dashed = torque). The shared style helper has no line-style channel, so
that convention is defined locally. In the advantage panel (no network split) control mode is
additionally given a distinct colour from a small local palette for readability.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import (
    add_ms_axis,
    apply_style,
    color_for,
    label_for,
    marker_for,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

CONTROL_LS = {"position": "-", "torque": "--"}
CONTROL_LABEL = {"position": "position control", "torque": "torque control"}
CONTROL_COLOR = {"position": "#2166ac", "torque": "#b2182b"}  # advantage panel only
NETWORKS = ["efference", "forward_model"]
MAX_DELAY = 100


def dedup(df: pd.DataFrame, by, metric: str) -> pd.DataFrame:
    """Keep the highest-``metric`` row per ``by`` group (collapses the duplicate delay-0 run)."""
    return (
        df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .drop_duplicates(by)
        .sort_values("delay_k")
    )


def series(df, control_mode, network):
    sub = df[(df.control_mode == control_mode) & (df.network == network)]
    return dedup(sub, ["delay_k"], "reward_mean")


def fig_control_comparison(df):
    """Q1: within each network, position vs torque reward across the delay sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, net in zip(axes, NETWORKS):
        for control in ("torque", "position"):
            s = series(df, control, net)
            ax.plot(
                s["delay_k"], s["reward_mean"],
                color=color_for(net), marker=marker_for(net),
                linestyle=CONTROL_LS[control], label=CONTROL_LABEL[control],
            )
        ax.set_title(label_for(net))
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylim(bottom=0)
        add_ms_axis(ax, MAX_DELAY)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean eval episode reward")
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    out = FIGURES / "control_comparison.png"
    fig.savefig(out)
    print(f"Saved {out}")


def fig_fm_advantage(df):
    """Q2: FM vs efference under position control (left) + FM-minus-efference for both modes (right)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: position control, efference vs forward_model reward curves (full delay sweep).
    for net in NETWORKS:
        s = series(df, "position", net)
        axL.plot(s["delay_k"], s["reward_mean"],
                 color=color_for(net), marker=marker_for(net), label=label_for(net))
    axL.set_title("position control")
    axL.set_xlabel("Observation delay (steps)")
    axL.set_ylabel("Mean eval episode reward")
    axL.set_ylim(bottom=0)
    add_ms_axis(axL, MAX_DELAY)
    axL.legend(loc="lower left", frameon=False)
    sns.despine(ax=axL)

    # Right: forward-model advantage (FM - efference) vs delay, one line per control mode.
    for control in ("torque", "position"):
        eff = series(df, control, "efference").set_index("delay_k")["reward_mean"]
        fm = series(df, control, "forward_model").set_index("delay_k")["reward_mean"]
        adv = (fm - eff).dropna().sort_index()
        axR.plot(adv.index, adv.values, color=CONTROL_COLOR[control],
                 marker="^", linestyle=CONTROL_LS[control], label=CONTROL_LABEL[control])
    axR.axhline(0, color="0.6", lw=0.8, zorder=0)
    axR.set_title("Forward-model advantage")
    axR.set_xlabel("Observation delay (steps)")
    axR.set_ylabel("Reward gain: forward model - efference")
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
    fig_control_comparison(df)
    fig_fm_advantage(df)


if __name__ == "__main__":
    main()
