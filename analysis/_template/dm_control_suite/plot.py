"""Figures for <question-slug>.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.
That separation is what lets a figure be restyled or re-rendered years later, and it is
why ``data.csv`` is committed rather than regenerated on demand.

    ../.venv/bin/python analysis/dm_control_suite/<question-slug>/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/<question-slug>/plot.py

The default figure is **small multiples: one panel per task, raw reward on each y-axis**.
That is the resolution of the raw-reward preference (README §7) for a track with nine
tasks whose returns are not the same quantity: it answers "which is better, and by how
much" without dividing through by a per-task ceiling and asserting that reward is linear
in competence. Normalise only for a genuine across-task summary, and say so in the
caption.
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

#: Control-step duration per task -- dm_control sets it per task, so this is NOT one
#: number for the track: CartpoleSwingup runs at 100 Hz and the locomotion tasks at 40 Hz.
#: The `add_ms_axis` default is the rodent's 10 and would mislabel silently.
CTRL_DT_MS = {          # from EnvSpec.default_config().ctrl_dt -- verified, not assumed
    "CartpoleBalance": 10, "CartpoleSwingup": 10, "CheetahRun": 10,
    "BallInCup": 20,
    "WalkerStand": 25, "WalkerWalk": 25, "WalkerRun": 25,
    "HumanoidStand": 25, "HumanoidWalk": 25,
}
DEFAULT_CTRL_DT_MS = 25


def fig_per_task(df: pd.DataFrame) -> plt.Figure:
    """Raw reward vs delay, one panel per task, every seed drawn."""
    tasks = [t for t in df["task"].dropna().unique()]
    ncols = min(3, len(tasks)) or 1
    nrows = -(-len(tasks) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                             squeeze=False)
    flat = [a for row in axes for a in row]

    # One reward source for the whole figure (main() refuses a mixed cohort), so the
    # y-label goes on the leftmost panel of each row rather than on all of them.
    ylabel = reward_label(str(df["reward_source"].iloc[0]))

    for i, (ax, task) in enumerate(zip(flat, tasks)):
        sub = df[df["task"] == task]
        for condition, group in sub.groupby("condition"):
            plot_seeds(ax, group, x="delay_k", y="reward_mean", condition=condition)
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("Observation delay (control steps)")
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        # Each panel keeps its own y-scale: a WalkerWalk return and a CheetahRun return
        # are different quantities, which is the whole reason for small multiples.
        if sub["delay_k"].notna().any():
            add_ms_axis(ax, sub["delay_k"].max(),
                        ctrl_dt_ms=CTRL_DT_MS.get(task, DEFAULT_CTRL_DT_MS))
    for ax in flat[len(tasks):]:
        ax.set_visible(False)

    flat[0].add_artist(flat[0].legend(frameon=False, fontsize=7, loc="upper right"))
    flat[0].legend(handles=seed_legend_handles(), fontsize=6, loc="lower left",
                   frameon=False)
    # Leave a strip at the bottom for the provenance footer, which `provenance()` adds
    # in figure coordinates after this returns and would otherwise sit on the y-label.
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)

    sources = df["reward_source"].dropna().unique()
    if len(sources) > 1:
        raise SystemExit(
            f"data.csv mixes reward sources {list(sources)}; they are in different units "
            "and episode lengths (see the dm_control_suite README). Split the figure, or "
            "restrict the cohort, rather than plotting one axis for both.")

    manifest = {}
    for name, builder in [("reward_per_task", fig_per_task)]:
        fig = builder(df)
        manifest[f"{name}.png"] = provenance(fig, HERE, DATA)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
