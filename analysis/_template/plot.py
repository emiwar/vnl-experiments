"""Figures for <question-slug>.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.
That separation is what lets a figure be restyled or re-rendered years later, and it is
why ``data.csv`` is committed rather than regenerated on demand.

    ../.venv/bin/python analysis/<question-slug>/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/<question-slug>/plot.py   # for slides
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    apply_style,
    color_for,
    label_for,
    marker_for,
    provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"


def dedup(df: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    """Mean over seeds/repeats so each cell contributes one point."""
    return (df.groupby(keys, as_index=False)
              .agg(**{metric: (metric, "mean"), "n": (metric, "size")}))


def fig_overview(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for condition, group in df.groupby("condition"):
        points = dedup(group, ["delay_k"], "reward_mean").sort_values("delay_k")
        ax.plot(points["delay_k"], points["reward_mean"],
                color=color_for(condition), marker=marker_for(condition),
                label=label_for(condition))
    ax.set_xlabel("Observation delay (control steps)")
    ax.set_ylabel("Episode reward")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)

    manifest = {}
    for name, builder in [("overview", fig_overview)]:
        fig = builder(df)
        manifest[f"{name}.png"] = provenance(fig, HERE, DATA)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
