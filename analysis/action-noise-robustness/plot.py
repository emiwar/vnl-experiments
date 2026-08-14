"""Figures for action-noise-robustness.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/action-noise-robustness/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/action-noise-robustness/plot.py
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

DATASETS = ("train", "old_eval", "new_eval")
PRIMARY = ("expfm", "encdec")
#: The delays at which the min_std = 0.25 tranche exists, so the exploration-width
#: comparison is made on matched delays rather than on differently-composed cohorts.
STD25_DELAYS = (0, 5, 10, 20, 50)


def add_relative(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """``metric`` divided by the same run's own sigma = 0 value on the same dataset.

    The arms sit at different absolute levels and every delay sits at a different level
    again, so *fractional* degradation is the only thing that can be pooled across
    delays. Normalising per (run, dataset) also cancels the ~1 % GPU-physics
    irreproducibility in the baseline, since numerator and denominator come from the same
    checkpoint.
    """
    baseline = (df[df["action_noise"] == 0.0]
                .set_index(["wandb_id", "dataset"])[metric])
    keys = pd.MultiIndex.from_frame(df[["wandb_id", "dataset"]])
    out = df.copy()
    out[f"rel_{metric}"] = df[metric].to_numpy() / baseline.reindex(keys).to_numpy()
    return out


def pooled(df: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    """Mean and spread over whatever is left after grouping (here: over delays)."""
    return (df.groupby(keys, as_index=False)
              .agg(**{metric: (metric, "mean"),
                      f"{metric}_sd": (metric, "std"),
                      "n": (metric, "size")}))


def _lines(ax, df: pd.DataFrame, metric: str, conditions=None) -> None:
    for condition, group in df.groupby("condition"):
        if conditions is not None and condition not in conditions:
            continue
        points = pooled(group, ["action_noise"], metric).sort_values("action_noise")
        ax.errorbar(points["action_noise"], points[metric],
                    yerr=points[f"{metric}_sd"], capsize=2,
                    color=color_for(condition), marker=marker_for(condition),
                    label=label_for(condition))
    ax.set_xlabel(r"Action noise $\sigma$")


def fig_degradation(df: pd.DataFrame) -> plt.Figure:
    """The headline. Top row absolute, bottom row relative to each run's own sigma = 0;
    columns are the three datasets. Pooled over the 23 delays, so the error bars are
    the spread *across delays*, not a seed spread."""
    rel = add_relative(df, "reward_per_step")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True)
    for col, dataset in enumerate(DATASETS):
        sub = rel[rel["dataset"] == dataset]
        _lines(axes[0][col], sub, "reward_per_step", PRIMARY)
        _lines(axes[1][col], sub, "rel_reward_per_step", PRIMARY)
        axes[0][col].set_title(dataset)
        axes[1][col].axhline(1.0, color="0.7", lw=0.8, zorder=0)
        axes[0][col].set_xlabel("")
    axes[0][0].set_ylabel("Reward per step")
    axes[1][0].set_ylabel(r"Reward per step, rel. to $\sigma=0$")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_survival(df: pd.DataFrame) -> plt.Figure:
    """Lifespan and hazard rate: whether noise costs reward per step or costs episodes."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True)
    for col, dataset in enumerate(DATASETS):
        sub = df[df["dataset"] == dataset]
        _lines(axes[0][col], sub, "lifespan_s", PRIMARY)
        _lines(axes[1][col], sub, "hazard_rate", PRIMARY)
        axes[0][col].set_title(dataset)
        axes[0][col].set_xlabel("")
        axes[1][col].set_yscale("log")
    axes[0][0].set_ylabel("Lifespan (s)")
    axes[1][0].set_ylabel("Hazard rate (1/step)")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_by_delay(df: pd.DataFrame, dataset: str = "old_eval") -> plt.Figure:
    """Does the noise penalty grow with the delay the predictor has to bridge? One panel
    per sigma > 0; x is the delay, so each point is a single run."""
    rel = add_relative(df, "reward_per_step")
    rel = rel[(rel["dataset"] == dataset) & rel["condition"].isin(PRIMARY)]
    noises = sorted(n for n in rel["action_noise"].unique() if n > 0)
    fig, axes = plt.subplots(1, len(noises), figsize=(3.6 * len(noises), 3.8),
                             sharey=True, squeeze=False)
    for ax, sigma in zip(axes[0], noises):
        sub = rel[rel["action_noise"] == sigma]
        for condition, group in sub.groupby("condition"):
            points = group.sort_values("delay_k")
            ax.plot(points["delay_k"], points["rel_reward_per_step"],
                    color=color_for(condition), marker=marker_for(condition),
                    label=label_for(condition))
        ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)
        ax.set_title(rf"$\sigma = {sigma:g}$")
        ax.set_xlabel("Observation delay (control steps)")
    axes[0][0].set_ylabel(rf"Reward/step on {dataset}, rel. to $\sigma=0$")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_prediction_error(df: pd.DataFrame, dataset: str = "old_eval") -> plt.Figure:
    """The mechanism: the predictor's own L2 error against the true current
    proprioception, per delay. Only the forward-model arms log it."""
    sub = df[(df["dataset"] == dataset)].dropna(subset=["fm_pred_mse"])
    fig, ax = plt.subplots(figsize=(5.8, 4))
    for delay, group in sub[sub["condition"] == "expfm"].groupby("delay_k"):
        points = group.sort_values("action_noise")
        ax.plot(points["action_noise"], points["fm_pred_mse"],
                marker="o", lw=1.0, label=f"delay {int(delay)}")
    ax.set_xlabel(r"Action noise $\sigma$")
    ax.set_ylabel(f"Forward-model prediction MSE ({dataset})")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=2, fontsize="small")
    fig.tight_layout()
    return fig


def fig_exploration_width(df: pd.DataFrame, dataset: str = "old_eval") -> plt.Figure:
    """Does training with wider exploration (min_std 0.25 vs 0.1) buy robustness to
    evaluation noise? Restricted to the delays where the 0.25 tranche exists. There is no
    enc-dec run at min_std 0.25, so this axis is internal to the forward model."""
    rel = add_relative(df, "reward_per_step")
    rel = rel[(rel["dataset"] == dataset) & rel["delay_k"].isin(STD25_DELAYS)
              & rel["condition"].isin(("expfm", "expfm_std25", "pgfm_std25"))]
    fig, ax = plt.subplots(figsize=(5.8, 4))
    _lines(ax, rel, "rel_reward_per_step")
    ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)
    ax.set_ylabel(rf"Reward/step on {dataset}, rel. to $\sigma=0$")
    ax.set_title(f"Delays {', '.join(str(d) for d in STD25_DELAYS)} only")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    have = df[df["have_artifact"]]
    missing = len(df) - len(have)
    if have.empty:
        raise SystemExit(
            f"No eval artifacts present for any of the {len(df)} "
            f"(run, sigma, dataset) cells. Produce or pull them first (see coverage.txt "
            f"and the extract docstring).")
    if missing:
        print(f"WARNING: {missing}/{len(df)} cells have no artifact; the figures below "
              f"cover {len(have)}. See coverage.txt.")

    manifest = {}
    builders = [("degradation", fig_degradation), ("survival", fig_survival),
                ("by_delay", fig_by_delay), ("prediction_error", fig_prediction_error),
                ("exploration_width", fig_exploration_width)]
    for name, builder in builders:
        fig = builder(have)
        manifest[f"{name}.png"] = provenance(fig, HERE, DATA)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
