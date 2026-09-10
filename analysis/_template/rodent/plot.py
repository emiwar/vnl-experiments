"""Figures for <question-slug>.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.
That separation is what lets a figure be restyled or re-rendered years later, and it is
why ``data.csv`` is committed rather than regenerated on demand.

    ../.venv/bin/python analysis/rodent/<question-slug>/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/rodent/<question-slug>/plot.py   # for slides
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    add_ms_axis,
    apply_style,
    plot_seeds,
    provenance,
    reward_label,
    seed_legend_handles,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"

#: Which reward this folder plots -- see style.REWARD_SOURCES. It goes in the axis label,
#: because "reward" without its source is what hid the two worst bugs in the project.
REWARD_SOURCE = "old_eval"


def fig_overview(df: pd.DataFrame) -> plt.Figure:
    """Raw reward vs delay, one line per condition, every seed drawn.

    Raw reward on the y-axis by default: a ratio or a fraction-of-baseline asserts that
    reward is linear in competence, which it is not (README §7). If this figure has to
    show a derived measure, keep a raw panel beside it and justify it in report.md.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for condition, group in df.groupby("condition"):
        plot_seeds(ax, group, x="delay_k", y="reward_mean", condition=condition)
    ax.set_xlabel("Observation delay (control steps)")
    ax.set_ylabel(reward_label(REWARD_SOURCE))
    add_ms_axis(ax, df["delay_k"].max())        # rodent: 10 ms/step, the default
    # Two legends: the conditions, plus one pair of proxies saying what the thin lines
    # are. add_artist keeps the first from being replaced by the second.
    ax.add_artist(ax.legend(frameon=False, loc="upper right"))
    ax.legend(handles=seed_legend_handles(), fontsize=7, loc="lower left", frameon=False)
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
