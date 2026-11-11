"""Figures for simulator-stack.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/simulator-stack/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/simulator-stack/plot.py

Three figures:

* ``reward_curves`` -- one panel per reward cell (task x delay, every other config field
  equal). Old-stack seed pairs are thin grey lines; the old-stack run with the *same*
  seed pair as a post-update run is a thicker grey line, since those two share their
  network initialisation exactly.
* ``reward_480M`` -- the readout, one point per run, per cell. A post-update run is
  joined to its same-seed old-stack twin.
* ``speed`` -- median per-iteration throughput by GPU model, one panel per task x gauge.
  GPUs without runs on both sides are left out: speed is only compared within a model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    SEED_LINE,
    apply_style,
    color_for,
    label_for,
    marker_for,
    provenance,
    reward_label,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA, CURVES, CELLS = HERE / "data.csv", HERE / "curves.csv", HERE / "cells.csv"

OLD = "mjx_jax_mujoco36"
ORDER = ["mjx_jax_mujoco36", "warp_mujoco40"]


def _cells(df: pd.DataFrame) -> list[tuple[str, str]]:
    cells = (df[df["in_reward_cell"]].drop_duplicates("config_id")
             .sort_values(["task", "delay_k"]))
    return [(r.config_id, f"{r.task}, delay {r.delay_k}") for r in cells.itertuples()]


def _panels(n: int, w: float = 3.8, h: float = 3.0):
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), squeeze=False)
    return fig, list(axes[0])


def fig_reward_curves(df: pd.DataFrame, curves: pd.DataFrame) -> plt.Figure:
    cells = _cells(df)
    fig, axes = _panels(len(cells))
    for ax, (cid, title) in zip(axes, cells):
        runs = df[df["config_id"] == cid]
        new_seeds = set(runs.loc[runs["condition"] != OLD, "seed_pair"])
        n_new = 0
        for r in runs.itertuples():
            c = curves[curves["wandb_id"] == r.wandb_id].sort_values("step")
            if c.empty:
                continue
            x = c["step"] / 1e6
            if r.condition == OLD and r.seed_pair not in new_seeds:
                ax.plot(x, c["reward"], color=color_for(OLD), **SEED_LINE)
            elif r.condition == OLD:
                ax.plot(x, c["reward"], color=color_for(OLD), lw=1.6,
                        label=f"{label_for(OLD)}, same seeds ({r.seed_pair})")
            else:
                ax.plot(x, c["reward"], color=color_for(r.condition), lw=1.6,
                        ls=("-", "--", ":")[n_new % 3],
                        label=f"{label_for(r.condition)} ({r.seed_pair})")
                n_new += 1
        ax.plot([], [], color=color_for(OLD), **SEED_LINE,
                label=f"{label_for(OLD)}, other seeds")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Environment steps (M)")
    fig.legend(*_merged_legend(axes), fontsize=7, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.05),
               ncol=3)
    axes[0].set_ylabel(reward_label("dmc_eval"))
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    return fig


def _merged_legend(axes):
    seen = {}
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, h)
    return list(seen.values()), list(seen.keys())


def fig_reward_480m(df: pd.DataFrame) -> plt.Figure:
    cells = _cells(df)
    fig, axes = _panels(len(cells), w=3.0)
    rng = np.random.default_rng(0)
    for ax, (cid, title) in zip(axes, cells):
        runs = df[df["config_id"] == cid]
        present = [c for c in ORDER if c in set(runs["condition"])]
        xpos = {c: i for i, c in enumerate(present)}
        old = runs[runs["condition"] == OLD]
        for r in runs.itertuples():
            jitter = 0.0 if r.condition != OLD else rng.uniform(-0.08, 0.08)
            ax.plot(xpos[r.condition] + jitter, r.reward_480M, ls="none",
                    marker=marker_for(r.condition), color=color_for(r.condition), ms=6)
            twin = old[old["seed_pair"] == r.seed_pair]
            if r.condition != OLD and len(twin):
                ax.plot([xpos[OLD], xpos[r.condition]],
                        [twin["reward_480M"].iloc[0], r.reward_480M],
                        color="0.3", lw=0.8, ls="--", zorder=0)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([label_for(c).replace(", MuJoCo", "\nMuJoCo")
                            .replace(" (old default)", "").replace(" (new default)", "")
                            for c in present], fontsize=7)
        ax.set_xlim(-0.5, len(present) - 0.5)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(reward_label("dmc_eval") + "\nmean over 430-480 M steps")
    fig.text(0.01, 0.06, "Dashed: same seed pair (identical network init) on both stacks.",
             fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def fig_speed(df: pd.DataFrame) -> plt.Figure:
    gauges = [("train_sps", "Training throughput (env steps/s, PPO step)"),
              ("eval_sps", "Eval throughput (env steps/s, 256 envs)")]
    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(len(gauges), len(tasks), figsize=(4.2 * len(tasks), 6.0),
                             squeeze=False)
    for j, task in enumerate(tasks):
        sub = df[df["task"] == task]
        # Only GPU models that hold both a pre- and a post-update run: speed is
        # compared within a model, so a model with one side only has nothing to show.
        gpus = sorted(set(sub.loc[sub["condition"] != OLD, "gpu"])
                      & set(sub.loc[sub["condition"] == OLD, "gpu"]))
        sub = sub[sub["gpu"].isin(gpus)]
        for i, (col, ylabel) in enumerate(gauges):
            ax = axes[i][j]
            for g, gpu in enumerate(gpus):
                present = [c for c in ORDER if c in set(sub.loc[sub["gpu"] == gpu, "condition"])]
                for k, cond in enumerate(present):
                    vals = sub[(sub["gpu"] == gpu) & (sub["condition"] == cond)][col].dropna()
                    x = g + (k - (len(present) - 1) / 2) * 0.25
                    ax.plot(np.full(len(vals), x), vals / 1e3, ls="none", alpha=0.5,
                            marker=marker_for(cond), color=color_for(cond), ms=4)
                    if len(vals):
                        ax.plot([x - 0.1, x + 0.1], [vals.median() / 1e3] * 2,
                                color=color_for(cond), lw=2.5)
            ax.set_xticks(range(len(gpus)))
            ax.set_xticklabels([f"{gpu}\n(n={', '.join(str(int(((sub['gpu'] == gpu) & (sub['condition'] == c)).sum())) for c in ORDER if ((sub['gpu'] == gpu) & (sub['condition'] == c)).any())})"
                                for gpu in gpus], fontsize=7)
            ax.set_ylabel(ylabel.replace("env steps/s", "k env steps/s"), fontsize=8)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(task, fontsize=9)
    handles = [plt.Line2D([], [], ls="none", marker=marker_for(c), color=color_for(c),
                          label=label_for(c)) for c in ORDER]
    axes[0][0].legend(handles=handles, fontsize=7, frameon=False, loc="lower left")
    fig.text(0.01, 0.02, "Points: runs (median gauge after 10 % warm-up). Bars: median "
             "across runs. Within one GPU model only.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    if set(df["reward_source"].dropna()) != {"dmc_eval"}:
        raise SystemExit("expected a single reward source, dmc_eval")

    manifest = {}
    for name, fig, inputs in [
        ("reward_curves", fig_reward_curves(df, curves), (DATA, CURVES)),
        ("reward_480M", fig_reward_480m(df), (DATA,)),
        ("speed", fig_speed(df), (DATA,)),
    ]:
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")
    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
