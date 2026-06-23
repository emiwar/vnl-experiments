"""Plot the forward-model loss-weight analysis from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-loss-weight/plot.py

Reads ONLY data.csv (never the WandB API). See analysis/README.md §2. Produces:
  - figures/loss_weight_sweep.png   reward vs fm_loss_weight at delay 10
  - figures/untrained_vs_trained.png  weight=0 vs weight=1 vs references, across delays
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import apply_style, color_for, label_for, marker_for

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
METRIC = "episode_reward/mean"

SWEEP_DELAY = 10
UNTRAINED_DELAYS = [5, 10, 20, 50]  # delays where fm_loss_weight == 0 was run


def best(df, **eq):
    sub = df
    for k, v in eq.items():
        sub = sub[sub[k] == v]
    sub = sub.dropna(subset=[METRIC])
    return sub[METRIC].max() if len(sub) else np.nan


def plot_loss_weight_sweep(df):
    fm = df[df["condition"] == "forward_model"]
    sweep = fm[fm["delay_k"] == SWEEP_DELAY].dropna(subset=[METRIC])
    nonzero = sweep[sweep["fm_loss_weight"] > 0].sort_values("fm_loss_weight")
    zero = sweep[sweep["fm_loss_weight"] == 0]

    # Place weight=0 one decade left of the smallest nonzero weight on the log axis.
    wmin = nonzero["fm_loss_weight"].min()
    zero_x = wmin / 10.0

    fig, ax = plt.subplots()
    ax.plot(nonzero["fm_loss_weight"], nonzero[METRIC],
            color=color_for("forward_model"), marker="^", label="Forward model")
    if len(zero):
        ax.scatter([zero_x], [zero[METRIC].max()], color=color_for("forward_model"),
                   marker="x", s=60, zorder=3, label="Forward model (weight = 0)")

    # Reference levels at the same delay.
    eff = best(df, condition="efference", delay_k=SWEEP_DELAY)
    noeff = best(df, condition="no_efference", delay_k=SWEEP_DELAY)
    ax.axhline(eff, ls="--", lw=1.2, color=color_for("efference"), label="Plain efference copy")
    ax.axhline(noeff, ls="--", lw=1.2, color=color_for("no_efference"), label="No efference copy")

    ax.set_xscale("log")
    # Relabel the sentinel tick as "0".
    ticks = [zero_x] + sorted(nonzero["fm_loss_weight"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [f"{w:g}" for w in sorted(nonzero["fm_loss_weight"].unique())],
                       rotation=45, fontsize=7)
    ax.set_xlabel("Forward-model loss weight")
    ax.set_ylabel("Mean episode reward")
    ax.set_ylim(bottom=0)
    ax.set_title(f"delay = {SWEEP_DELAY} steps", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    sns.despine(ax=ax)
    out = FIGURES / "loss_weight_sweep.png"
    fig.savefig(out)
    print(f"Saved {out}")


def plot_untrained_vs_trained(df):
    delays = UNTRAINED_DELAYS
    series = {
        "fm_trained": dict(label="Forward model (weight = 1)", color=color_for("forward_model"),
                           marker="^", ls="-",
                           y=[best(df, condition="forward_model", delay_k=d, fm_loss_weight=1.0) for d in delays]),
        "fm_untrained": dict(label="Forward model (weight = 0)", color=color_for("forward_model"),
                             marker="x", ls="--",
                             y=[best(df, condition="forward_model", delay_k=d, fm_loss_weight=0.0) for d in delays]),
        "efference": dict(label=label_for("efference"), color=color_for("efference"),
                          marker=marker_for("efference"), ls="-",
                          y=[best(df, condition="efference", delay_k=d) for d in delays]),
        "no_efference": dict(label=label_for("no_efference"), color=color_for("no_efference"),
                             marker=marker_for("no_efference"), ls="-",
                             y=[best(df, condition="no_efference", delay_k=d) for d in delays]),
    }
    fig, ax = plt.subplots()
    for s in series.values():
        ax.plot(delays, s["y"], color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"])
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode reward")
    ax.set_ylim(bottom=0)
    ax.set_xticks(delays)
    ax.legend(fontsize=7, loc="upper right")
    sns.despine(ax=ax)
    out = FIGURES / "untrained_vs_trained.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_loss_weight_sweep(df)
    plot_untrained_vs_trained(df)


if __name__ == "__main__":
    main()
