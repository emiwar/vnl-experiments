"""Plot position vs torque control, and the forward-model advantage under each.

Run from the repo root::

    ../.venv/bin/python analysis/position-vs-torque-control/plot.py

Reads ONLY data.csv (never the WandB API) and writes figures/. See analysis/README.md §2.

Visual encoding: **network** -> colour + marker (reused ``color_for``/``marker_for`` so the
efference/forward-model colours match every other FM report); **control mode** -> line style
(solid = position, dashed = torque). The shared style helper has no line-style channel, so
that convention is defined locally. In the advantage panel (no network split) control mode is
additionally given a distinct colour from a small local palette for readability.
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

CONTROL_LS = {"position": "-", "torque": "--"}
CONTROL_LABEL = {"position": "position control", "torque": "torque control"}
CONTROL_COLOR = {"position": "#2166ac", "torque": "#b2182b"}  # advantage panel only
NETWORKS = ["efference", "forward_model"]
MAX_DELAY = 100


def dedup(df: pd.DataFrame, by, metric: str) -> pd.DataFrame:
    """Keep the highest-``metric`` row per ``by`` group (collapses the duplicate delay-0 run)."""
    return (
        df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .drop_duplicates(by)
        .sort_values("delay_k")
    )


def series(df, control_mode, network):
    sub = df[(df.control_mode == control_mode) & (df.network == network)]
    return dedup(sub, ["delay_k"], "reward_mean")


def eval_series(dfe, control_mode, network, metric):
    """One delay-sorted series of ``metric`` for a (control, network) cell on a given dataset.

    ``dfe`` is expected to be already filtered to a single dataset. The duplicate delay-0
    torque_efference run is collapsed by keeping its higher-reward_per_step row.
    """
    sub = dfe[(dfe.control_mode == control_mode) & (dfe.network == network)]
    keep = dedup(sub, ["delay_k"], "reward_per_step")
    return keep.set_index("delay_k")[metric]


def fig_control_comparison(df):
    """Q1: within each network, position vs torque reward across the delay sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, net in zip(axes, NETWORKS):
        for control in ("torque", "position"):
            s = series(df, control, net)
            ax.plot(
                s["delay_k"], s["reward_mean"],
                color=color_for(net), marker=marker_for(net),
                linestyle=CONTROL_LS[control], label=CONTROL_LABEL[control],
            )
        ax.set_title(label_for(net))
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylim(bottom=0)
        add_ms_axis(ax, MAX_DELAY)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean eval episode reward")
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    out = FIGURES / "control_comparison.png"
    fig.savefig(out)
    print(f"Saved {out}")


def fig_fm_advantage(df):
    """Q2: FM vs efference under position control (left) + FM-minus-efference for both modes (right)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: position control, efference vs forward_model reward curves (full delay sweep).
    for net in NETWORKS:
        s = series(df, "position", net)
        axL.plot(s["delay_k"], s["reward_mean"],
                 color=color_for(net), marker=marker_for(net), label=label_for(net))
    axL.set_title("position control")
    axL.set_xlabel("Observation delay (steps)")
    axL.set_ylabel("Mean eval episode reward")
    axL.set_ylim(bottom=0)
    add_ms_axis(axL, MAX_DELAY)
    axL.legend(loc="lower left", frameon=False)
    sns.despine(ax=axL)

    # Right: forward-model advantage (FM - efference) vs delay, one line per control mode.
    for control in ("torque", "position"):
        eff = series(df, control, "efference").set_index("delay_k")["reward_mean"]
        fm = series(df, control, "forward_model").set_index("delay_k")["reward_mean"]
        adv = (fm - eff).dropna().sort_index()
        axR.plot(adv.index, adv.values, color=CONTROL_COLOR[control],
                 marker="^", linestyle=CONTROL_LS[control], label=CONTROL_LABEL[control])
    axR.axhline(0, color="0.6", lw=0.8, zorder=0)
    axR.set_title("Forward-model advantage")
    axR.set_xlabel("Observation delay (steps)")
    axR.set_ylabel("Reward gain: forward model - efference")
    add_ms_axis(axR, MAX_DELAY)
    axR.legend(loc="upper left", frameon=False)
    sns.despine(ax=axR)

    fig.tight_layout()
    out = FIGURES / "fm_advantage.png"
    fig.savefig(out)
    print(f"Saved {out}")


def fig_eval_raw_reward(dfe):
    """Raw cumulative episode reward on new_eval (30 s clips), faceted by network.

    Not comparable to the 5 s training-clip reward (6× longer rollout) — but directly
    interpretable within new_eval, and it folds in survival (failed clips stop accruing
    reward), so it drops faster with delay than the length-normalised reward_per_step.
    """
    ne = dfe[dfe.dataset == "new_eval"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, net in zip(axes, NETWORKS):
        for control in ("torque", "position"):
            s = eval_series(ne, control, net, "episode_reward_mean").dropna().sort_index()
            ax.plot(s.index, s.values, color=color_for(net), marker=marker_for(net),
                    linestyle=CONTROL_LS[control], label=CONTROL_LABEL[control])
        ax.set_title(label_for(net))
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylim(bottom=0)
        add_ms_axis(ax, MAX_DELAY)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Raw episode reward (new_eval, 30 s clips)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out = FIGURES / "eval_raw_reward.png"
    fig.savefig(out)
    print(f"Saved {out}")


def fig_eval_robustness(dfe):
    """Held-out (new_eval, 30 s clips) length-fair metrics vs delay: reward_per_step + hazard.

    Four lines per panel = 2 control modes x 2 networks (colour = network, line style =
    control mode), the same encoding as the training-clip figures.
    """
    ne = dfe[dfe.dataset == "new_eval"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, ylab in [
        (axL, "reward_per_step", "Reward per step (new_eval)"),
        (axR, "hazard_rate", "Failure hazard (per s, new_eval)"),
    ]:
        for control in ("torque", "position"):
            for net in NETWORKS:
                s = eval_series(ne, control, net, metric).dropna().sort_index()
                ax.plot(s.index, s.values, color=color_for(net), marker=marker_for(net),
                        linestyle=CONTROL_LS[control],
                        label=f"{CONTROL_LABEL[control]}, {label_for(net)}")
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylabel(ylab)
        add_ms_axis(ax, MAX_DELAY)
        sns.despine(ax=ax)
    axL.set_ylim(bottom=0)
    axR.set_ylim(bottom=0)
    axR.legend(loc="upper left", frameon=False, fontsize=7)
    fig.suptitle("Held-out generalization (32 unseen 30 s clips)", y=1.02)
    fig.tight_layout()
    out = FIGURES / "eval_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def fig_eval_fm_advantage(dfe):
    """Does the forward model still help on held-out clips? FM - efference on new_eval.

    Left: reward-per-step gain (FM - efference). Right: hazard reduction (efference - FM,
    positive = FM fails less often). One line per control mode.
    """
    ne = dfe[dfe.dataset == "new_eval"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4))
    for control in ("torque", "position"):
        rps_eff = eval_series(ne, control, "efference", "reward_per_step")
        rps_fm = eval_series(ne, control, "forward_model", "reward_per_step")
        rps_adv = (rps_fm - rps_eff).dropna().sort_index()
        axL.plot(rps_adv.index, rps_adv.values, color=CONTROL_COLOR[control],
                 marker="^", linestyle=CONTROL_LS[control], label=CONTROL_LABEL[control])

        hz_eff = eval_series(ne, control, "efference", "hazard_rate")
        hz_fm = eval_series(ne, control, "forward_model", "hazard_rate")
        hz_red = (hz_eff - hz_fm).dropna().sort_index()
        axR.plot(hz_red.index, hz_red.values, color=CONTROL_COLOR[control],
                 marker="^", linestyle=CONTROL_LS[control], label=CONTROL_LABEL[control])
    for ax in (axL, axR):
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        ax.set_xlabel("Observation delay (steps)")
        add_ms_axis(ax, MAX_DELAY)
        sns.despine(ax=ax)
    axL.set_title("Reward-per-step gain: FM - efference")
    axL.set_ylabel("Reward-per-step gain (new_eval)")
    axR.set_title("Hazard reduction: efference - FM")
    axR.set_ylabel("Failure-hazard reduction (per s, new_eval)")
    axL.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out = FIGURES / "eval_fm_advantage.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    fig_control_comparison(df)
    fig_fm_advantage(df)
    dfe = pd.read_csv(HERE / "data_eval.csv")
    fig_eval_raw_reward(dfe)
    fig_eval_robustness(dfe)
    fig_eval_fm_advantage(dfe)


if __name__ == "__main__":
    main()
