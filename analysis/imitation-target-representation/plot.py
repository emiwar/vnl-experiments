"""Plot the imitation-target-representation comparison from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/imitation-target-representation/plot.py

Reads ONLY data.csv (never WandB / eval_results). See analysis/README.md §2. With only six
runs (3 conditions × 2 delays, single seed each) we use **bar plots**, not trend lines. Produces:
  - figures/performance.png         raw episode reward + survival fraction, per dataset
  - figures/quality_vs_risk.png     reward-per-step vs termination hazard on the new eval set
  - figures/failure_modes.png       stacked termination reasons on the new eval set

Error bars are clip-level SEM (reward: std/sqrt(n_clips); fractions: binomial sqrt(p(1-p)/n)).
They show across-clip spread, NOT seed variance — each (condition, delay) cell is a single seed,
so read all differences cautiously.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import apply_style, color_for, label_for

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
CTRL_DT_S = 0.01  # 100 control steps / s; lifespan_steps -> seconds

# Baseline -> the two absolute variants (Q2 pair adjacent).
COND_ORDER = ["relative", "absolute_current", "absolute_reference"]
DELAYS = (0, 10)
DATASETS = [("train", "Training clips"), ("old_eval", "Held-out test (old eval)"),
            ("new_eval", "New eval (30 s clips)")]


def _cell(df, dataset, cond, delay):
    r = df[(df.dataset == dataset) & (df.condition == cond) & (df.delay_k == delay)]
    return r.iloc[0] if len(r) else None


def _grouped_bars(ax, df, dataset, value, err_kind=None):
    """Grouped bars: x = delay groups, one bar per condition. err_kind in {None,'reward','frac'}."""
    n = len(COND_ORDER)
    width = 0.8 / n
    x = np.arange(len(DELAYS))
    for i, cond in enumerate(COND_ORDER):
        ys, es = [], []
        for d in DELAYS:
            row = _cell(df, dataset, cond, d)
            if row is None:
                ys.append(np.nan); es.append(0.0); continue
            ys.append(row[value])
            if err_kind == "reward":
                es.append(row["episode_reward_std"] / np.sqrt(row["n_clips"]))
            elif err_kind == "frac":
                p, nn = row[value], row["n_clips"]
                es.append(np.sqrt(max(p * (1 - p), 0) / nn))
            else:
                es.append(0.0)
        ax.bar(x + (i - (n - 1) / 2) * width, ys, width, color=color_for(cond),
               label=label_for(cond), yerr=es, capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([f"delay {d}" for d in DELAYS])
    ax.set_xlim(-0.5, len(DELAYS) - 0.5)
    sns.despine(ax=ax)


# --------------------------------------------------------------------------- #
# Figure 1: headline performance across the three datasets
# --------------------------------------------------------------------------- #
def plot_performance(df):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for j, (ds, title) in enumerate(DATASETS):
        _grouped_bars(axes[0, j], df, ds, "episode_reward_mean", err_kind="reward")
        axes[0, j].set_title(title, fontsize=10)
        _grouped_bars(axes[1, j], df, ds, "survival_frac", err_kind="frac")
        axes[1, j].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Episode reward\n(raw sum, per dataset)")
    axes[1, 0].set_ylabel("Mean lifetime fraction\n(lifespan / clip rollout)")
    # Reward y-scale differs by dataset (clip length) -> independent y per column; that is the
    # point of faceting. The lifetime row is already a 0-1 fraction, shared.
    # Legend goes in the new_eval/survival panel, which has headroom above the short delay-10 bars.
    axes[1, 2].legend(fontsize=7, loc="upper right")
    fig.suptitle("Imitation-target representation: performance on train / test / new eval",
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES / "performance.png"
    fig.savefig(out)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Figure 2: quality-while-alive vs risk-of-failing, on the discriminating new eval set
# --------------------------------------------------------------------------- #
def plot_quality_vs_risk(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    _grouped_bars(ax1, df, "new_eval", "reward_per_step")
    ax1.set_ylabel("Reward per step (while alive)")
    ax1.set_title("Tracking quality while alive", fontsize=10)
    ax1.set_ylim(0, None)
    ax1.legend(fontsize=7, loc="lower left")
    _grouped_bars(ax2, df, "new_eval", "hazard_rate")
    ax2.set_ylabel("Termination hazard (per s)\nexcl. end-of-clip truncation")
    ax2.set_title("Risk of failing", fontsize=10)
    ax2.set_ylim(0, None)
    fig.suptitle("New eval set: equal quality while alive, very different failure risk",
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES / "quality_vs_risk.png"
    fig.savefig(out)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Figure 3: how they fail (termination reasons) on the new eval set
# --------------------------------------------------------------------------- #
def plot_failure_modes(df):
    """Stacked termination-reason fractions on new_eval, per condition × delay."""
    reasons = [("survived", "Survived (no failure)", "0.8"),
               ("term_root_too_rotated", "Root over-rotated", "C3"),
               ("term_root_too_far", "Root too far", "C4"),
               ("term_pose_error", "Pose error", "C5")]
    sub = df[df.dataset == "new_eval"]
    labels, bottoms = [], []
    bars = {key: [] for key, _, _ in reasons}
    for d in DELAYS:
        for cond in COND_ORDER:
            row = _cell(sub, "new_eval", cond, d)
            labels.append(f"{label_for(cond)}\ndelay {d}")
            for key, _, _ in reasons:
                bars[key].append(float(row[key]) if row is not None and pd.notna(row[key]) else 0.0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for key, lab, col in reasons:
        ax.bar(x, bars[key], bottom=bottom, color=col, label=lab, width=0.7)
        bottom += np.array(bars[key])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("Episode outcome fraction")
    ax.set_ylim(0, 1)
    ax.set_title("New eval set: how episodes end (stacked outcome fractions)", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    sns.despine(ax=ax)
    fig.tight_layout()
    out = FIGURES / "failure_modes.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_performance(df)
    plot_quality_vs_risk(df)
    plot_failure_modes(df)


if __name__ == "__main__":
    main()
