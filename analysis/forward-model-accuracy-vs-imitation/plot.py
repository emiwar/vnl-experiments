"""Plot forward-model prediction accuracy vs imitation performance from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-accuracy-vs-imitation/plot.py

Reads ONLY data.csv (never the WandB API). See analysis/README.md §2. Produces:
  - figures/mse_vs_reward.png        all FM runs, coloured by delay
  - figures/mse_vs_reward_delay10.png  fixed-delay (10) loss-weight sweep
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import apply_style

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
MSE = "fm_pred_mse"
REWARD = "episode_reward/mean"


def spearman(df):
    return df[MSE].corr(df[REWARD], method="spearman")


def plot_all(df):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    trained = df[df["fm_trained"]]
    untrained = df[~df["fm_trained"]]
    sc = ax.scatter(trained[MSE], trained[REWARD], c=trained["delay_k"],
                    cmap="viridis", s=45, marker="o", label="trained (weight > 0)")
    if len(untrained):
        ax.scatter(untrained[MSE], untrained[REWARD], c=untrained["delay_k"],
                   cmap="viridis", s=70, marker="X", edgecolor="r", linewidth=1.0,
                   label="untrained (weight = 0)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Observation delay (steps)")
    ax.set_xscale("log")
    ax.set_xlabel("Forward-model prediction MSE (eval, median)")
    ax.set_ylabel("Mean episode reward")
    ax.set_ylim(bottom=0)
    ax.set_title(f"All forward-model runs (Spearman r = {spearman(df):.2f})", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    sns.despine(ax=ax)
    fig.tight_layout()
    out = FIGURES / "mse_vs_reward.png"
    fig.savefig(out)
    print(f"Saved {out}")


def plot_delay10(df):
    d10 = df[df["delay_k"] == 10].sort_values(MSE)
    fig, ax = plt.subplots()
    ax.plot(d10[MSE], d10[REWARD], color="0.6", lw=1.0, zorder=1)
    sc = ax.scatter(d10[MSE], d10[REWARD], c=d10["fm_loss_weight"].clip(lower=1e-6),
                    cmap="plasma", norm=plt.matplotlib.colors.LogNorm(), s=60, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Forward-model loss weight")
    ax.set_xscale("log")
    ax.set_xlabel("Forward-model prediction MSE (eval, median)")
    ax.set_ylabel("Mean episode reward")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Fixed delay = 10 steps (Spearman r = {spearman(d10):.2f})", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    out = FIGURES / "mse_vs_reward_delay10.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_all(df)
    plot_delay10(df)


if __name__ == "__main__":
    main()
