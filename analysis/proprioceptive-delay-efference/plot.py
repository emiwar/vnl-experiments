"""Plot the proprioceptive delay sweep (efference copy vs none) from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/proprioceptive-delay-efference/plot.py

Reads ONLY data.csv (never the WandB API). See analysis/README.md §2.
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
# Plot order so the legend reads efference first, then no-efference.
ORDER = ["efference", "no_efference"]


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-reward run per delay_k."""
    return (
        df.dropna(subset=[METRIC, "delay_k"])
        .sort_values(METRIC, ascending=False)
        .drop_duplicates("delay_k")
        .sort_values("delay_k")
    )


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")

    max_delay = df["delay_k"].max()
    fig, ax = plt.subplots()
    for cond in ORDER:
        sub = dedup(df[df["condition"] == cond])
        if sub.empty:
            continue
        ax.plot(
            sub["delay_k"], sub[METRIC],
            color=color_for(cond), marker=marker_for(cond), label=label_for(cond),
        )

    ax.set_xlim(0, max_delay * 1.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode reward")
    ax.legend(loc="upper right")
    sns.despine(ax=ax)
    add_ms_axis(ax, max_delay)

    out = FIGURES / "delay_sweep.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
