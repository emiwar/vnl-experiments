"""Plot the "is the forward-model benefit just more weights?" comparison.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-vs-bigger-decoder/plot.py

Reads ONLY data.csv (never the WandB API). See analysis/README.md §2. Produces:
  - figures/decoder_size_sweep.png   reward vs delay, all four conditions
  - figures/reward_vs_params.png     reward vs decoder weights, at large delays
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
METRIC = "episode_reward/mean"

# Efference variants ordered by parameter count (for the params figure).
EFF_BY_PARAMS = ["efference", "efference_deeper", "efference_larger"]
LINE_ORDER = ["forward_model", "efference", "efference_deeper", "efference_larger"]
PARAM_DELAYS = [20, 50, 80]  # large delays where all four conditions have runs


def dedup(sub: pd.DataFrame) -> pd.DataFrame:
    return (
        sub.dropna(subset=[METRIC, "delay_k"])
        .sort_values(METRIC, ascending=False)
        .drop_duplicates("delay_k")
        .sort_values("delay_k")
    )


def reward_at(df, cond, delay):
    s = dedup(df[df["condition"] == cond]).set_index("delay_k")[METRIC]
    return s.get(delay)


def plot_vs_delay(df):
    max_delay = df["delay_k"].max()
    fig, ax = plt.subplots()
    for cond in LINE_ORDER:
        sub = dedup(df[df["condition"] == cond])
        if sub.empty:
            continue
        ax.plot(sub["delay_k"], sub[METRIC], color=color_for(cond),
                marker=marker_for(cond), label=label_for(cond))
    ax.set_xlim(0, max_delay * 1.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode reward")
    ax.legend(fontsize=7, loc="upper right")
    sns.despine(ax=ax)
    add_ms_axis(ax, max_delay)
    out = FIGURES / "decoder_size_sweep.png"
    fig.savefig(out)
    print(f"Saved {out}")


def plot_vs_params(df):
    params = {c: df.loc[df["condition"] == c, "extra_hidden_params"].iloc[0]
              for c in LINE_ORDER}
    fig, axes = plt.subplots(1, len(PARAM_DELAYS), figsize=(9, 3.2), sharey=True)
    for ax, delay in zip(axes, PARAM_DELAYS):
        # Efference variants: connected line showing the parameter trend.
        xs = [params[c] / 1e6 for c in EFF_BY_PARAMS]
        ys = [reward_at(df, c, delay) for c in EFF_BY_PARAMS]
        ax.plot(xs, ys, color="0.6", lw=1.2, zorder=1)
        for c in EFF_BY_PARAMS:
            ax.scatter(params[c] / 1e6, reward_at(df, c, delay), s=45,
                       color=color_for(c), marker=marker_for(c),
                       label=label_for(c), zorder=2)
        # Forward model: distinct star, placed at its own parameter count.
        ax.scatter(params["forward_model"] / 1e6, reward_at(df, "forward_model", delay),
                   s=130, color=color_for("forward_model"), marker="*",
                   edgecolor="k", linewidth=0.4, label=label_for("forward_model"), zorder=3)
        ax.set_title(f"delay = {delay} steps", fontsize=9)
        ax.set_xlabel("Decoder hidden\nweights (millions)")
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean episode reward")
    axes[-1].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    out = FIGURES / "reward_vs_params.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_vs_delay(df)
    plot_vs_params(df)


if __name__ == "__main__":
    main()
