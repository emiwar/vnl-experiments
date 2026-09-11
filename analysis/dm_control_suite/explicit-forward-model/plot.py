"""Figures for explicit-forward-model (dm_control_suite).

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/explicit-forward-model/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/explicit-forward-model/plot.py

Figures
-------
  reward_vs_delay.png       main: raw end-of-training reward vs delay, one panel per task
  training_curves.png       supp: the eval series at delay 10, one panel per task
  cartpole_best_vs_final.png  supp: best-ever vs end-of-training reward on CartpoleSwingup

Raw reward on every y-axis, one panel per task rather than a fraction-of-maximum
(README §7): a CartpoleSwingup return and a HumanoidWalk return are different quantities,
and dividing by a per-task ceiling would assert that reward is linear in competence.
Each panel keeps its own y-scale for the same reason.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from vnl_experiments.wandb_utils.style import (
    add_ms_axis,
    apply_style,
    color_for,
    label_for,
    marker_for,
    plot_seeds,
    provenance,
    reward_label,
    seed_legend_handles,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

#: Panel order: increasing difficulty, which is also the order the result changes sign in.
TASKS = ["CartpoleSwingup", "WalkerWalk", "HumanoidWalk"]

#: Control-step duration per task. dm_control sets this per task -- CartpoleSwingup runs
#: at 100 Hz and the two locomotion tasks at 40 Hz -- so the ms axis is NOT one number
#: for this track. Read off `ctrl_dt` in data.csv rather than assumed.
CTRL_DT_MS = {"CartpoleSwingup": 10, "WalkerWalk": 25, "HumanoidWalk": 25}

#: The delay at which the supplementary training curves are drawn. Present for both arms
#: in all three tasks, and past the point where WalkerWalk separates.
CURVE_DELAY = 10

ARMS = ["delayed_mlp", "flat_forward_model"]


def _panels(n, width=4.2, height=3.4):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False)
    return fig, list(axes[0])


def _finish(ax, task, df_task, ylabel, xlabel="Observation delay (control steps)"):
    ax.set_title(task, fontsize=10)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel.startswith("Observation delay") and df_task["delay"].notna().any():
        # Delays are whole control steps; the default locator offers 2.5 and 7.5.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.figure.canvas.draw()
        add_ms_axis(ax, df_task["delay"].max(), ctrl_dt_ms=CTRL_DT_MS[task])


def fig_reward_vs_delay(df: pd.DataFrame) -> plt.Figure:
    """Main result: raw end-of-training reward against delay, per task."""
    fig, axes = _panels(len(TASKS))
    ylabel = reward_label(str(df["reward_source"].iloc[0]))
    for i, (ax, task) in enumerate(zip(axes, TASKS)):
        sub = df[df.task == task]
        for arm in ARMS:
            group = sub[sub.condition == arm]
            if not group.empty:
                plot_seeds(ax, group, x="delay", y="reward_final", condition=arm)
        _finish(ax, task, sub, ylabel if i == 0 else None)
        ax.set_ylim(bottom=0)
    axes[0].add_artist(axes[0].legend(frameon=False, fontsize=7, loc="lower left"))
    # Only CartpoleSwingup has more than one seed, so that is where the proxies belong.
    axes[0].legend(handles=seed_legend_handles(), fontsize=6, loc="lower right",
                   frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def fig_training_curves(curves: pd.DataFrame) -> plt.Figure:
    """Supplementary 1: the eval series at one representative delay."""
    fig, axes = _panels(len(TASKS))
    sel = curves[curves.delay == CURVE_DELAY]
    for i, (ax, task) in enumerate(zip(axes, TASKS)):
        sub = sel[sel.task == task]
        for arm in ARMS:
            group = sub[sub.condition == arm]
            if not group.empty:
                plot_seeds(ax, group, x="step", y="reward_mean", condition=arm,
                           marker_size=0)
        ax.set_title(f"{task}, delay {CURVE_DELAY}", fontsize=10)
        ax.set_xlabel("Environment steps")
        ax.set_ylim(bottom=0)
        if i == 0:
            ax.set_ylabel(reward_label("dmc_eval"))
    axes[0].add_artist(axes[0].legend(frameon=False, fontsize=7, loc="lower right"))
    axes[-1].legend(handles=seed_legend_handles(), fontsize=6, loc="upper right",
                    frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def fig_cartpole_best_vs_final(df: pd.DataFrame) -> plt.Figure:
    """Supplementary 2: on CartpoleSwingup, is a low score failure or forgetting?

    Best-ever eval reward (solid) against end-of-training reward (dashed, open markers).
    A flat best-ever line under a ragged final line means the solution was found and then
    lost, which is a different claim from "the delay made the task unsolvable".
    """
    sub = df[df.task == "CartpoleSwingup"]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for arm in ARMS:
        group = sub[sub.condition == arm]
        if group.empty:
            continue
        colour = color_for(arm)
        best = group.groupby(["seed", "delay"])["reward_max"].mean().groupby("delay").mean()
        final = group.groupby(["seed", "delay"])["reward_final"].mean().groupby("delay").mean()
        ax.plot(best.index, best.values, color=colour, marker=marker_for(arm),
                ms=4, lw=2.2, label=f"{label_for(arm)} — best during training")
        ax.plot(final.index, final.values, color=colour, marker=marker_for(arm),
                ms=4, lw=1.4, ls="--", mfc="none",
                label=f"{label_for(arm)} — end of training")
    ax.set_xlabel("Observation delay (control steps)")
    ax.set_ylabel(reward_label("dmc_eval"))
    ax.set_ylim(bottom=0)
    # pad clears the secondary ms axis, whose label sits where a title normally goes.
    ax.set_title("CartpoleSwingup: best-ever vs end-of-training reward",
                 fontsize=10, pad=34)
    add_ms_axis(ax, sub["delay"].max(), ctrl_dt_ms=CTRL_DT_MS["CartpoleSwingup"])
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)

    sources = df["reward_source"].dropna().unique()
    if len(sources) > 1:
        raise SystemExit(
            f"data.csv mixes reward sources {list(sources)}; before the `971ab99` "
            "eval-env fix the series was scaled x10 and truncated, so the two are in "
            "different units. Split the figure or restrict the cohort.")

    manifest = {}
    for name, builder, inputs in [
        ("reward_vs_delay", lambda: fig_reward_vs_delay(df), (DATA,)),
        ("training_curves", lambda: fig_training_curves(curves), (CURVES,)),
        ("cartpole_best_vs_final", lambda: fig_cartpole_best_vs_final(df), (DATA,)),
    ]:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
