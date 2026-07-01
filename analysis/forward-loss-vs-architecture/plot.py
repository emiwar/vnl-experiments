"""Plot forward-model loss vs architecture, from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/forward-loss-vs-architecture/plot.py

Reads ONLY data.csv (never WandB). See analysis/README.md §2. Produces:
  - figures/reward_vs_delay.png   performance: explicit FM vs policy-gradient FM (loss=0)
  - figures/fm_l2_vs_delay.png    forward-prediction L2 error for both — does the PG model
                                  implicitly learn to predict? (log y)

Both figures overlay the 3 bridge runs (regular FM at the new commit) as hollow check markers to
show the code-version difference between the two main conditions is benign.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import (
    add_ms_axis, apply_style, color_for, label_for, marker_for,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"


def curve(df, cond, col):
    sub = df[df.condition == cond]
    return sub.groupby("delay_k")[col].mean().sort_index()


def _bridge_overlay(ax, df, col):
    """Regular FM re-run at the new commit (detach=True, fm_w=1): consistency check markers."""
    s = curve(df, "forward_model_nnxupdate", col)
    if len(s):
        ax.plot(s.index, s.values, ls="none", marker="o", ms=8, mfc="none",
                mec=color_for("forward_model"), mew=1.3, zorder=6,
                label="Regular FM (new nnx-ppo, ×3 check)")


# --------------------------------------------------------------------------- #
# Figure 1: performance
# --------------------------------------------------------------------------- #
def plot_reward(df):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    max_delay = df["delay_k"].max()
    for cond in ("forward_model", "pg_forward_model"):
        s = curve(df, cond, "episode_reward_mean")
        ax.plot(s.index, s.values, color=color_for(cond), marker=marker_for(cond),
                ms=5, label=label_for(cond))
    # Untrained predictor (loss=0, detached) as the "no forward learning" reference.
    s = curve(df, "fm0_untrained", "episode_reward_mean")
    ax.plot(s.index, s.values, color=color_for("fm0_untrained"), marker=marker_for("fm0_untrained"),
            ms=8, ls=":", label=label_for("fm0_untrained"))
    _bridge_overlay(ax, df, "episode_reward_mean")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Eval episode reward")
    ax.set_xlim(0, max_delay * 1.05)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Explicit forward-model loss vs policy-gradient-only forward model",
                 fontsize=9.5)
    sns.despine(ax=ax)
    add_ms_axis(ax, max_delay)
    fig.tight_layout()
    out = FIGURES / "reward_vs_delay.png"
    fig.savefig(out)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Figure 2: forward-prediction L2 error
# --------------------------------------------------------------------------- #
def plot_fm_l2(df):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    max_delay = df["delay_k"].max()
    for cond in ("forward_model", "pg_forward_model", "fm0_untrained"):
        s = curve(df, cond, "fm_mse_p50")
        p25 = curve(df, cond, "fm_mse_p25")
        p75 = curve(df, cond, "fm_mse_p75")
        ls = ":" if cond == "fm0_untrained" else "-"
        ax.plot(s.index, s.values, color=color_for(cond), marker=marker_for(cond),
                ms=5, ls=ls, label=label_for(cond))
        # p25-p75 spread band (batch spread within the run).
        idx = s.index
        if len(p25) and len(p75):
            ax.fill_between(idx, p25.reindex(idx).values, p75.reindex(idx).values,
                            color=color_for(cond), alpha=0.15, lw=0)
    _bridge_overlay(ax, df, "fm_mse_p50")
    ax.set_yscale("log")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Forward-prediction L2 error  (fm_pred_mse, median)")
    ax.set_xlim(0, max_delay * 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Does the policy-gradient FM implicitly learn to predict? (no)", fontsize=9.5)
    sns.despine(ax=ax)
    add_ms_axis(ax, max_delay)
    fig.tight_layout()
    out = FIGURES / "fm_l2_vs_delay.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_reward(df)
    plot_fm_l2(df)


if __name__ == "__main__":
    main()
