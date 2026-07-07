"""Plot forward model vs regular encoder-decoder (multi-seed), from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-vs-encdec-seeds/plot.py

Reads ONLY data.csv (never WandB / eval_results). See analysis/README.md sec 2.

Style: each seed is a thin, transparent line; the mean across seeds is a solid line.
Within a (condition, seed, delay) cell any replicate runs are averaged first, so each
seed contributes one curve and the mean weights each seed equally.

Figures (all show the regular `old_eval` set and the new 30 s `new_eval` set side by side):
  - reward_vs_delay.png          raw episode reward vs delay
  - lifetime_vs_delay.png        lifetime (s) vs delay
  - tracking_error_vs_delay.png  body tracking error (mm) vs delay
  - failure_modes.png            per-reason termination rate vs delay        (bonus: failures)
  - reward_composition.png       per-alive-step reward by term vs delay      (bonus: rewards)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from vnl_experiments.wandb_utils import (
    add_ms_axis, apply_style, color_for, label_for, marker_for,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
CTRL_DT_S = 0.01           # 1 control step = 10 ms
COND_ORDER = ["forward_model", "efference"]
DS_TITLE = {"old_eval": "Regular eval (held-out, 5 s)", "new_eval": "New eval (32 x 30 s)"}
DS_ORDER = ["old_eval", "new_eval"]

SEED_LINE = dict(lw=0.9, alpha=0.35)
MEAN_LINE = dict(lw=2.2, alpha=0.95)


def seed_curves(df, cond, dataset, value, scale=1.0):
    """Return (per-seed {seed: Series}, mean-over-seeds Series) indexed by delay.

    Replicates within a (seed, delay) cell are averaged first; the mean curve is the
    mean across seeds (equal weight per seed, not per run)."""
    sub = df[(df.condition == cond) & (df.dataset == dataset)].dropna(subset=[value])
    if sub.empty:
        return {}, pd.Series(dtype=float)
    per_seed = sub.groupby(["seed", "delay_k"])[value].mean().mul(scale)
    curves = {s: per_seed.loc[s].sort_index() for s in per_seed.index.get_level_values(0).unique()}
    mean = per_seed.groupby("delay_k").mean().sort_index()
    return curves, mean


# --------------------------------------------------------------------------- #
# Generic 2-panel (old_eval | new_eval) seed-lines + mean figure.
# --------------------------------------------------------------------------- #
def two_panel(df, value, scale, ylabel, title, fname, ymin=0):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharex=True)
    max_delay = df["delay_k"].max()
    for ax, ds in zip(axes, DS_ORDER):
        for cond in COND_ORDER:
            col = color_for(cond)
            curves, mean = seed_curves(df, cond, ds, value, scale)
            for s in curves.values():
                ax.plot(s.index, s.values, color=col, **SEED_LINE)
            if not mean.empty:
                ax.plot(mean.index, mean.values, color=col, marker=marker_for(cond),
                        ms=3.5, label=label_for(cond), **MEAN_LINE)
        ax.set_title(DS_TITLE[ds], fontsize=9)
        ax.set_xlabel("Observation delay (steps)")
        ax.set_xlim(0, max_delay * 1.05)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        sns.despine(ax=ax)
        add_ms_axis(ax, max_delay)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7, loc="best")
    # A thin-line proxy so the reader knows what the faint lines are.
    axes[1].legend(handles=[Line2D([], [], color="0.4", **SEED_LINE, label="individual seed"),
                            Line2D([], [], color="0.4", **MEAN_LINE, label="mean across seeds")],
                   fontsize=7, loc="best")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / fname)
    print(f"Saved {FIGURES / fname}")
    plt.close(fig)


def plot_reward(df):
    two_panel(df, "episode_reward_mean", 1.0, "Episode reward (raw, sum)",
              "Forward model vs encoder-decoder: episode reward", "reward_vs_delay.png")


def plot_lifetime(df):
    two_panel(df, "lifespan_steps", CTRL_DT_S, "Lifetime (s)",
              "Forward model vs encoder-decoder: lifetime", "lifetime_vs_delay.png")


def plot_tracking_error(df):
    # body_errors/total is in metres -> mm.
    two_panel(df, "err_body_total_m", 1000.0, "Body tracking error (mm)",
              "Forward model vs encoder-decoder: body tracking error", "tracking_error_vs_delay.png")


# --------------------------------------------------------------------------- #
# Bonus 1: failure modes -- per-reason termination rate vs delay.
# --------------------------------------------------------------------------- #
def plot_failure_modes(df):
    """Fraction of clips ending for each reason vs delay, FM (solid) vs efference (dashed)."""
    reasons = [("term_root_too_far", "root too far", "C3"),
               ("term_root_too_rotated", "root too rotated", "C4")]
    cond_ls = {"forward_model": "-", "efference": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharex=True, sharey=True)
    max_delay = df["delay_k"].max()
    for ax, ds in zip(axes, DS_ORDER):
        for col, _, rc in reasons:
            for cond in COND_ORDER:
                _, mean = seed_curves(df, cond, ds, col)
                if not mean.empty:
                    ax.plot(mean.index, mean.values, color=rc, ls=cond_ls[cond], lw=1.8,
                            marker=marker_for(cond), ms=3)
        ax.set_title(DS_TITLE[ds], fontsize=9)
        ax.set_xlabel("Observation delay (steps)")
        ax.set_xlim(0, max_delay * 1.05)
        ax.set_ylim(0, 1)
        sns.despine(ax=ax)
        add_ms_axis(ax, max_delay)
    axes[0].set_ylabel("Termination rate (fraction of clips)")
    handles = [Line2D([], [], color=rc, lw=2, label=name) for _, name, rc in reasons]
    handles += [Line2D([], [], color="0.4", ls=cond_ls[c], marker=marker_for(c),
                       label=label_for(c)) for c in COND_ORDER]
    axes[1].legend(handles=handles, fontsize=6.5, loc="best")
    fig.suptitle("Failure modes: termination reason vs delay (mean across seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "failure_modes.png")
    print(f"Saved {FIGURES / 'failure_modes.png'}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Bonus 2: reward composition -- per-alive-step reward by term vs delay.
# --------------------------------------------------------------------------- #
def plot_reward_composition(df, dataset="new_eval"):
    """Per-alive-step reward for each term vs delay, FM vs efference.

    Dividing by lifetime removes the "survives longer -> earns more of everything"
    confound, so differences here reflect a genuinely different reward *mix*."""
    terms = [("rtps_root_pos", "root position"), ("rtps_root_quat", "root orientation"),
             ("rtps_joints", "joint pose"), ("rtps_end_eff", "end effectors"),
             ("rtps_torso_z_range", "torso height"), ("rtps_control_cost", "control cost"),
             ("rtps_control_diff_cost", "control-diff cost"), ("rtps_energy_cost", "energy cost")]
    ncol = 4
    nrow = int(np.ceil(len(terms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.7 * nrow), sharex=True)
    max_delay = df["delay_k"].max()
    for ax, (col, name) in zip(axes.flat, terms):
        for cond in COND_ORDER:
            curves, mean = seed_curves(df, cond, dataset, col)
            for s in curves.values():
                ax.plot(s.index, s.values, color=color_for(cond), **SEED_LINE)
            if not mean.empty:
                ax.plot(mean.index, mean.values, color=color_for(cond), lw=2, label=label_for(cond))
        ax.set_title(name, fontsize=8)
        ax.set_xlim(0, max_delay * 1.05)
        sns.despine(ax=ax)
    for ax in axes.flat[len(terms):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Delay (steps)")
    axes[0, 0].set_ylabel("Reward / alive-step")
    axes[0, 0].legend(fontsize=6.5, loc="best")
    fig.suptitle(f"Reward composition (per alive-step) vs delay - {DS_TITLE[dataset]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "reward_composition.png")
    print(f"Saved {FIGURES / 'reward_composition.png'}")
    plt.close(fig)


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_reward(df)
    plot_lifetime(df)
    plot_tracking_error(df)
    plot_failure_modes(df)
    plot_reward_composition(df)


if __name__ == "__main__":
    main()
