"""Plot the nnx-ppo-update / seed reproducibility check from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/nnx-ppo-update-reproducibility/plot.py

Reads ONLY data.csv (never WandB). See analysis/README.md §2. Produces:
  - figures/curve_reproducibility.png
      left  : episode-reward delay sweep, original baseline vs new (updated nnx-ppo + new seed),
              with the single new-code/old-seed delay-0 test point overlaid.
      right : per-delay difference (new_seed - baseline) at shared delays, to quantify how far
              the curves deviate relative to zero.
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


def _curve(df, cond):
    """Mean reward per delay (averages the 2 baseline seeds at delay 0), sorted."""
    sub = df[df.condition == cond]
    return sub.groupby("delay_k")["episode_reward_mean"].mean().sort_index()


def plot_reproducibility(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    max_delay = df["delay_k"].max()

    b = _curve(df, "baseline")
    n = _curve(df, "new_seed")
    t = _curve(df, "new_code_old_seed")

    # ---- Left: overlaid sweeps ----
    for cond, s in (("baseline", b), ("new_seed", n)):
        ax1.plot(s.index, s.values, color=color_for(cond), marker=marker_for(cond),
                 ms=4, label=label_for(cond))
    # Single new-code/old-seed test point (delay 0): isolates the code change from the seed.
    ax1.plot(t.index, t.values, color=color_for("new_code_old_seed"),
             marker=marker_for("new_code_old_seed"), ms=11, ls="none",
             label=label_for("new_code_old_seed"), zorder=5)
    ax1.set_xlabel("Observation delay (steps)")
    ax1.set_ylabel("Eval episode reward")
    ax1.set_xlim(0, max_delay * 1.05)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("Delay sweep: original vs updated nnx-ppo", fontsize=10)
    sns.despine(ax=ax1)
    add_ms_axis(ax1, max_delay)

    # ---- Right: per-delay difference at shared delays ----
    shared = sorted(set(b.index) & set(n.index))
    diff = np.array([n[d] - b[d] for d in shared])
    ax2.axhline(0, color="0.6", lw=1)
    mean_diff = diff.mean()
    ax2.axhline(mean_diff, color="C1", lw=1, ls="--",
                label=f"mean {mean_diff:+.0f} ({100 * mean_diff / b.loc[shared].mean():+.1f}%)")
    ax2.plot(shared, diff, color="C1", marker="s", ms=4, ls="-")
    ax2.set_xlabel("Observation delay (steps)")
    ax2.set_ylabel("Reward difference\n(new seed 43  −  baseline)")
    ax2.set_xlim(0, max_delay * 1.05)
    # Symmetric y so the deviation is read against zero.
    ymax = np.abs(diff).max() * 1.25
    ax2.set_ylim(-ymax, ymax)
    r = np.corrcoef([b[d] for d in shared], [n[d] for d in shared])[0, 1]
    ax2.set_title(f"Curve-to-curve deviation (Pearson r = {r:.4f})", fontsize=10)
    ax2.legend(fontsize=7, loc="lower left")
    sns.despine(ax=ax2)
    add_ms_axis(ax2, max_delay)

    fig.suptitle("Delay-sweep reproducibility under updated nnx-ppo + new seed", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "curve_reproducibility.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_reproducibility(df)


if __name__ == "__main__":
    main()
