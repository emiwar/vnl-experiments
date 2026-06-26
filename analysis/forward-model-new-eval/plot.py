"""Plot the forward-model vs efference comparison on the new eval set, from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-new-eval/plot.py

Reads ONLY data.csv (never WandB / eval_results). See analysis/README.md §2. Produces:
  - figures/reward_new_eval.png    raw episode reward vs delay, FM / efference / no-eff (new set)
  - figures/lifetime_new_eval.png  raw lifetime (s) vs delay, FM / efference / no-eff (new set)
  - figures/old_vs_new_normalized.png  length-fair old-vs-new-eval comparison (one summary figure)

The first two figures stay on the new eval set, so reward and lifetime are directly comparable
across conditions and are left NON-normalised. Only the final figure compares across datasets,
where the differing clip lengths require length-fair metrics: reward-per-step (alive) and the
per-second termination hazard (failures per unit alive-time, excluding end-of-clip truncations),
which is invariant to clip length unlike the raw survival fraction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import (
    add_ms_axis, apply_style, color_for, label_for, marker_for,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
CTRL_DT_S = 0.01  # 100 control steps / s; lifespan_steps -> seconds

COND_ORDER = ["forward_model", "efference", "no_efference"]


def agg(df, condition, dataset, value):
    """Mean over replicate seeds per delay, sorted by delay."""
    sub = df[(df.condition == condition) & (df.dataset == dataset)]
    return sub.groupby("delay_k")[value].mean().sort_index()


# --------------------------------------------------------------------------- #
# Figures 1 & 2: raw FM vs efference comparison on the new eval set
# --------------------------------------------------------------------------- #
def _new_eval_raw(df, value, scale, ylabel, fname, title):
    fig, ax = plt.subplots()
    max_delay = df["delay_k"].max()
    for cond in COND_ORDER:
        s = agg(df, cond, "new_eval", value) * scale
        ax.plot(s.index, s.values, color=color_for(cond), marker=marker_for(cond),
                label=label_for(cond), ms=4)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max_delay * 1.05)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    sns.despine(ax=ax)
    add_ms_axis(ax, max_delay)
    fig.tight_layout()
    out = FIGURES / fname
    fig.savefig(out)
    print(f"Saved {out}")


def plot_reward_new_eval(df):
    _new_eval_raw(df, "episode_reward_mean", 1.0, "Episode reward (raw, sum)",
                  "reward_new_eval.png", "New eval set: episode reward")


def plot_lifetime_new_eval(df):
    _new_eval_raw(df, "lifespan_steps", CTRL_DT_S, "Lifetime (s)",
                  "lifetime_new_eval.png", "New eval set: lifetime")


# --------------------------------------------------------------------------- #
# Figure 3: normalised old-vs-new-eval summary (the only cross-dataset figure)
# --------------------------------------------------------------------------- #
def plot_old_vs_new_normalized(df):
    """Length-fair reward-per-step & termination hazard, FM vs efference, old_eval vs new_eval."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    max_delay = df["delay_k"].max()
    ds_style = {"old_eval": dict(ls="--", alpha=0.9), "new_eval": dict(ls="-")}
    conds = ["forward_model", "efference"]
    for ax, metric, ylabel, ylim in (
        (ax1, "reward_per_step", "Reward per step (alive)", (0, None)),
        (ax2, "hazard_rate", "Termination hazard (per s)\nexcl. end-of-clip truncation", (0, None)),
    ):
        for cond in conds:
            for ds in ("old_eval", "new_eval"):
                s = agg(df, cond, ds, metric)
                ax.plot(s.index, s.values, color=color_for(cond), marker=marker_for(cond),
                        ms=3, **ds_style[ds])
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max_delay * 1.05)
        if ylim:
            ax.set_ylim(*ylim)
        sns.despine(ax=ax)
        add_ms_axis(ax, max_delay)

    # Two-part legend: colour = condition, line style = dataset.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color_for(c), marker=marker_for(c), label=label_for(c))
               for c in conds]
    handles += [
        Line2D([], [], color="0.4", ls="--", label="Held-out test (old eval)"),
        Line2D([], [], color="0.4", ls="-", label="New eval set"),
    ]
    ax1.legend(handles=handles, fontsize=7, loc="lower left")
    fig.suptitle("Old vs new eval set (length-fair metrics)", fontsize=10)
    fig.tight_layout()
    out = FIGURES / "old_vs_new_normalized.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_reward_new_eval(df)
    plot_lifetime_new_eval(df)
    plot_old_vs_new_normalized(df)


if __name__ == "__main__":
    main()
