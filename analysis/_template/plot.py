"""Plot <QUESTION> from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/<question-slug>/plot.py

Reads ONLY data.csv (never the WandB API) and writes figures/. See analysis/README.md §2.
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


def dedup(df: pd.DataFrame, by, metric: str) -> pd.DataFrame:
    """Keep the highest-``metric`` row per ``by`` group."""
    return (
        df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .drop_duplicates(by)
    )


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")

    fig, ax = plt.subplots()
    for cond, sub in df.groupby("condition"):
        sub = dedup(sub, ["delay_k"], "episode_reward/mean").sort_values("delay_k")
        ax.plot(
            sub["delay_k"], sub["episode_reward/mean"],
            color=color_for(cond), marker=marker_for(cond), label=label_for(cond),
        )

    ax.set_xlabel("Observation delay (steps)")  # TODO
    ax.set_ylabel("Mean episode reward")        # TODO
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    sns.despine(ax=ax)

    out = FIGURES / "figure.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
