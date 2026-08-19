"""Figures for explicit-vs-implicit-fm-probe.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.
The stage indices and labels were resolved into ``data.csv`` by ``extract.py``, so nothing
here needs to know a layer path.

Encoding, once: **colour is the arm** (explicit = the forward-model green, implicit = the
policy-gradient purple, from the shared CONDITION_STYLE so an arm keeps its colour across
every question), **line style is the budget** (2 G solid and filled, 600 M dashed and
hollow). Raw values throughout -- no deltas, no ratios -- and every figure carries the
reward of both arms.

    ../.venv/bin/python analysis/explicit-vs-implicit-fm-probe/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/explicit-vs-implicit-fm-probe/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    add_ms_axis,
    apply_style,
    color_for,
    provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
REWARD = HERE / "data_reward.csv"
CURVES = HERE / "curves.csv"

DELAYS = (0, 10, 20, 50)
BUDGETS = ("600M", "2G")
ARMS = ("explicit", "implicit")
#: The arm -> shared-style condition mapping, so colours match the sibling analyses.
NETWORK_OF = {"explicit": "forward_model", "implicit": "pg_forward_model"}
ARM_LABEL = {"explicit": "explicit FM (L2 loss)", "implicit": "implicit FM (policy gradient)"}
TARGETS = [("proprio", "Decode current proprioception"),
           ("delta", "Decode delta (current − delayed)")]
#: Run-to-run spread measured from the one duplicated cell (implicit/600M/delay-10); the
#: independent estimate from xml-ceiling-vs-convergence is ±2.9 %, which agrees.
NOISE_PCT = 2.9


def style_for(arm: str, budget: str) -> dict:
    return dict(color=color_for(NETWORK_OF[arm]),
                linestyle="-" if budget == "2G" else "--",
                marker="o" if arm == "explicit" else "s",
                markerfacecolor=color_for(NETWORK_OF[arm]) if budget == "2G" else "none",
                markersize=4.5, linewidth=1.8)


# --------------------------------------------------------------------------------------
# Reward annotation, used by every figure
# --------------------------------------------------------------------------------------

def reward_note(reward: pd.DataFrame, budget: str, delay: int) -> str:
    """``reward 1635 vs 1399 · fm MSE 0.074 / 0.467`` for one (budget, delay) cell.

    Cells with no offline eval print ``n/a`` rather than a blank, so a missing artifact
    cannot be misread as a missing effect.
    """
    parts = []
    for label, column, fmt in [("reward", "reward_window", "{:.0f}"),
                               ("fm MSE", "fm_mse_old_eval", "{:.3f}")]:
        values = []
        for arm in ARMS:
            sub = reward[(reward.budget_label == budget) & (reward.arm == arm)
                         & (reward.delay_k == delay)][column].dropna()
            values.append(fmt.format(sub.mean()) if len(sub) else "n/a")
        parts.append(f"{label} {values[0]} vs {values[1]}")
    return "\n".join(parts)


def stage_ticks(data: pd.DataFrame, pathway: str) -> tuple[list[int], list[str]]:
    """Axis ticks straight from the CSV, so the labels are part of the committed data."""
    sub = (data[data.pathway == pathway][["stage_index", "stage_label"]]
           .dropna().drop_duplicates().sort_values("stage_index"))
    return [int(i) for i in sub.stage_index], list(sub.stage_label)


# --------------------------------------------------------------------------------------
# F0: the opener
# --------------------------------------------------------------------------------------

def fig_reward_vs_delay(reward: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [("reward_window", "Episode reward (trailing-50 M window)",
               "In-training reward"),
              ("survived_old_eval", "Fraction of clips surviving 502 steps",
               "Held-out survival (offline eval)")]
    for ax, (column, ylabel, title) in zip(axes, panels):
        for budget in BUDGETS:
            for arm in ARMS:
                sub = reward[(reward.budget_label == budget) & (reward.arm == arm)]
                # Reindex on the full delay grid so a missing artifact leaves a NaN and the
                # line *breaks*. Dropping the row instead would join 10 to 50 straight
                # through the gap, drawing a measurement that was never made.
                values = (sub.groupby("delay_k")[column].mean()
                          .reindex(DELAYS))
                if values.isna().all():
                    continue
                ax.plot(list(DELAYS), values.to_numpy(),
                        label=f"{ARM_LABEL[arm]} — {budget}", **style_for(arm, budget))
                # The duplicated cell: show both runs, not just their mean.
                dup = sub[sub.delay_k.duplicated(keep=False)]
                if len(dup):
                    ax.plot(dup.delay_k, dup[column], linestyle="none", marker="_",
                            color=color_for(NETWORK_OF[arm]), markersize=9, zorder=5)
        ax.set_xlabel("Observation delay (control steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        add_ms_axis(ax, max(DELAYS))
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# F1-F4: R^2 along a pathway
# --------------------------------------------------------------------------------------

def fig_pathway(data: pd.DataFrame, reward: pd.DataFrame, pathway: str,
                budget: str) -> plt.Figure:
    ticks, labels = stage_ticks(data, pathway)
    fig, axes = plt.subplots(len(TARGETS), len(DELAYS), sharex=True, sharey="row",
                             figsize=(3.6 * len(DELAYS), 7.0), squeeze=False)
    sub_all = data[(data.pathway == pathway) & (data.budget_label == budget)]

    for row, (target, row_title) in enumerate(TARGETS):
        for col, delay in enumerate(DELAYS):
            ax = axes[row][col]
            cell = sub_all[(sub_all.target == target) & (sub_all.delay_k == delay)]

            if len(cell) and cell.target_degenerate.all():
                ax.text(0.5, 0.5, "delta ≡ 0 at delay 0\n(degenerate)", ha="center",
                        va="center", color="0.5", fontsize=9, transform=ax.transAxes)
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels, rotation=90, fontsize=7)
                continue

            for arm in ARMS:
                points = (cell[cell.arm == arm]
                          .groupby(["stage_index"], as_index=False)["test_r2"].mean()
                          .sort_values("stage_index"))
                if not len(points):
                    continue
                ax.plot(points.stage_index, points.test_r2,
                        label=ARM_LABEL[arm] if (row, col) == (0, 0) else None,
                        **style_for(arm, budget))
                # The network's own linear input baseline, extended across the panel so
                # "does any layer beat its inputs?" is readable without a ruler.
                base = points[points.stage_index == 0]["test_r2"]
                if len(base) and pathway == "actor":
                    ax.axhline(float(base.iloc[0]), color=color_for(NETWORK_OF[arm]),
                               linestyle=":", linewidth=0.9, alpha=0.55)

            ceiling = data[(data.budget_label == budget) & (data.delay_k == delay)
                           & (data.target == target)
                           & (data.probe == "input::current_proprio")]["test_r2"]
            if len(ceiling):
                ax.axhline(float(ceiling.mean()), color="0.7", linestyle=":", lw=0.9,
                           label="current input (ceiling)" if (row, col) == (0, 0) else None)
            ax.axhline(0, color="k", lw=0.6)
            if pathway == "actor":
                ax.axvline(5.5, color="0.88", lw=0.9, zorder=0)   # predictor | decoder
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            if row == 0:
                ax.set_title(f"delay = {delay} steps ({delay * 10} ms)\n"
                             f"{reward_note(reward, budget, delay)}", fontsize=8)
        axes[row][0].set_ylabel(f"{row_title}\nheld-out R²")
        axes[row][0].set_ylim(top=1.05)

    name = {"actor": "actor pathway", "encoder": "encoder pathway"}[pathway]
    fig.suptitle(f"Linear decodability along the {name} — {budget} steps", y=0.995)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig


# --------------------------------------------------------------------------------------
# F5: the forward-model error over training
# --------------------------------------------------------------------------------------

#: Delay 0 is excluded: with no delay the prediction task is the identity, so the explicit
#: arm's error sits at ~1e-4 and compresses the log axis for every other cell. Its value as a
#: capacity check is recorded in the probe figures and checks.txt instead.
FM_DELAYS = (10, 20, 50)


def fig_fm_prediction(curves: pd.DataFrame, reward: pd.DataFrame) -> plt.Figure:
    """Forward-model error and reward over training, from the *same* eval passes.

    Both rows are columns of the same ``hist2000-fc46b078`` rows, so a step-by-step
    comparison is exact: no difference in clip population, episode start or checkpoint can
    separate them. That is what makes "reward still climbing while the prediction error has
    flattened" a readable claim rather than an artefact of two measurements.
    """
    rows = [("fm_mse_eval", "Forward-model error\n(eval fm_pred_mse, log)", True),
            ("reward_mean", "Episode reward\n(same eval passes)", False)]
    fig, axes = plt.subplots(len(rows), len(FM_DELAYS), squeeze=False, sharex="col",
                             sharey="row", figsize=(3.4 * len(FM_DELAYS), 5.8))

    for row, (column, ylabel, log) in enumerate(rows):
        lo, hi = np.inf, -np.inf
        for col, delay in enumerate(FM_DELAYS):
            ax = axes[row][col]
            for budget in BUDGETS:
                for arm in ARMS:
                    sub = curves[(curves.budget_label == budget) & (curves.arm == arm)
                                 & (curves.delay_k == delay)].dropna(subset=[column])
                    if not len(sub):
                        continue
                    # Average the duplicated cell at each logged step.
                    series = sub.groupby("step", as_index=False)[column].mean()
                    style = style_for(arm, budget)
                    style["marker"] = None
                    ax.plot(series.step / 1e9, series[column],
                            label=f"{ARM_LABEL[arm]} — {budget}"
                            if (row, col) == (0, 0) else None, **style)
                    lo, hi = min(lo, series[column].min()), max(hi, series[column].max())
            ax.axvline(0.6, color="0.88", lw=0.9, zorder=0)
            if row == 0:
                ax.set_title(f"delay = {delay} ({delay * 10} ms)\n"
                             f"{reward_note(reward, '2G', delay)}", fontsize=8)
            if row == len(rows) - 1:
                ax.set_xlabel("Training steps (G)")

        # Limits from the plotted data only, so the panels are not squeezed by cells that
        # are no longer shown.
        if np.isfinite(lo) and np.isfinite(hi):
            if log:
                axes[row][0].set_yscale("log")
                axes[row][0].set_ylim(lo / 1.6, hi * 1.6)
            else:
                pad = 0.06 * (hi - lo)
                axes[row][0].set_ylim(lo - pad, hi + pad)
        axes[row][0].set_ylabel(ylabel)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Forward-model error and reward over training, from the same eval passes",
                 y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


# --------------------------------------------------------------------------------------
# F6: the comparability cross-check
# --------------------------------------------------------------------------------------

def fig_budget_crosscheck(reward: pd.DataFrame) -> plt.Figure:
    """600 M-vs-2 G is an across-runs comparison; this is the evidence it is fair.

    Both axes are the *same* measurement -- the trailing-50 M window of the training-curve
    reward at step 600 M -- one from the standalone 600 M runs, one from the 2 G runs passing
    through that step. Mixing an offline eval into one axis would prove nothing.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    pairs = []
    for arm in ARMS:
        for delay in DELAYS:
            short = reward[(reward.budget_label == "600M") & (reward.arm == arm)
                           & (reward.delay_k == delay)]["reward_curve_at_600M"].mean()
            long = reward[(reward.budget_label == "2G") & (reward.arm == arm)
                          & (reward.delay_k == delay)]["reward_curve_at_600M"].mean()
            if np.isfinite(short) and np.isfinite(long):
                pairs.append((short, long, arm, delay))
                ax.plot(short, long, **{**style_for(arm, "2G"), "linestyle": "none"})
                ax.annotate(f"d{delay}", (short, long), textcoords="offset points",
                            xytext=(6, -3), fontsize=7,
                            color=color_for(NETWORK_OF[arm]))
    if pairs:
        values = [v for pair in pairs for v in pair[:2]]
        lo, hi = min(values) * 0.95, max(values) * 1.05
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8)
        ax.fill_between([lo, hi], [lo * (1 - NOISE_PCT / 100), hi * (1 - NOISE_PCT / 100)],
                        [lo * (1 + NOISE_PCT / 100), hi * (1 + NOISE_PCT / 100)],
                        color="0.85", alpha=0.5, zorder=0,
                        label=f"±{NOISE_PCT} % noise floor")
        worst = max(abs(100 * (b / a - 1)) for a, b, _, _ in pairs)
        ax.set_title(f"Standalone 600 M runs vs the 2 G runs at step 600 M\n"
                     f"worst disagreement {worst:.1f} %", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Reward at 600 M — standalone 600 M run")
    ax.set_ylabel("Reward at 600 M — the 2 G run, same step")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------

def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    data = pd.read_csv(DATA)
    reward = pd.read_csv(REWARD)
    curves = pd.read_csv(CURVES)

    manifest: dict[str, str] = {}

    def emit(name: str, fig: plt.Figure, *inputs: Path) -> None:
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    emit("reward_vs_delay", fig_reward_vs_delay(reward), REWARD)
    for budget, tag in [("600M", "600m"), ("2G", "2g")]:
        emit(f"probe_{tag}", fig_pathway(data, reward, "actor", budget), DATA, REWARD)
        emit(f"encoder_{tag}", fig_pathway(data, reward, "encoder", budget), DATA, REWARD)
    emit("fm_prediction_curves", fig_fm_prediction(curves, reward), CURVES, REWARD)
    emit("budget_crosscheck", fig_budget_crosscheck(reward), REWARD)

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
