"""Figures for walker-delayed-critic (dm_control_suite).

Reads only the committed CSVs in this folder.

    ../.venv/bin/python analysis/dm_control_suite/walker-delayed-critic/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/walker-delayed-critic/plot.py

Figures
-------
  reward_vs_delay.png     main: reward at 480 M vs delay, one panel per efference arm
  training_curves.png     eval curves at delays 7 / 10 / 12 / 15 / 20 / 25, both arms
  mechanism.png           critic R^2, policy log-likelihood and clipping fraction vs delay
  credit_horizon.png      the structural argument in report.md: how much of a rollout
                          window's advantages can carry critic-mediated credit for a_t

Encoding
--------
**Colour is the efference arm** (``CONDITION_STYLE``'s shared ``efference`` C1 /
``no_efference`` C0), so the arm reads the same colour as in every other delay figure.
**The critic is the line style and marker**: the privileged arm, which has several seeds per
delay, is each run as a small semi-transparent circle with a dashed median line through
them; the delayed critic is each seed as a thin semi-transparent line with diamonds and the
mean across seeds as a solid line (README §7). A privileged run at the delayed arm's first
seed (i42/p1234) is drawn as an open black-edged circle, so those matched pairs can be read
directly; the second seed (i43/p43) has no privileged run at the same PPO seed. The delay-0 runs are a grey reference band: there both critics see
the same input.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from vnl_experiments.wandb_utils.style import (
    add_ms_axis,
    apply_style,
    color_for,
    label_for,
    marker_for,
    provenance,
    reward_label,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"
TRAIN = HERE / "train_curves.csv"

CTRL_DT_MS = 25
ARMS = ("no_efference", "efference")
ARM_TITLE = {"no_efference": "No efference copy (efference = 0)",
             "efference": "With efference copy (efference = delay)"}
CURVE_DELAYS = (7, 10, 12, 15, 20, 25)

# GAE settings of every run here (asserted from data.csv's comparability, see report.md).
GAMMA, LAMBDA, ROLLOUT = 0.99, 0.95, 30


def _undelayed_band(ax, d, col):
    und = d[d["arm"] == "undelayed"][col].dropna()
    if len(und):
        ax.axhspan(und.min(), und.max(), color=color_for("undelayed"), alpha=0.15, lw=0)
        ax.axhline(und.median(), color=color_for("undelayed"), lw=1, ls=":")


def _privileged(ax, sub, col, color, *, median=True):
    priv = sub[sub["critic"] == "privileged"].dropna(subset=[col])
    jitter = (priv["seed"] % 7 - 3) * 0.06
    ax.scatter(priv["delay"] + jitter, priv[col], s=16, color=color, alpha=0.35, lw=0)
    m = priv[priv["matched_seed"]]
    ax.scatter(m["delay"], m[col], s=34, facecolor="none", edgecolor="k", lw=0.9, zorder=4)
    if median:
        med = priv.groupby("delay")[col].median()
        ax.plot(med.index, med.values, color=color, ls="--", lw=1.4, alpha=0.9)


def _delayed(ax, sub, col, color):
    """Each delayed-critic seed thin, the mean across seeds solid (README §7)."""
    dl = sub[sub["critic"] == "delayed"].dropna(subset=[col]).sort_values("delay")
    for _, g in dl.groupby("seed_pair"):
        ax.plot(g["delay"], g[col], color=color, lw=0.9, alpha=0.55,
                marker=marker_for("delayed_critic"), ms=4, mec="k", mew=0.4, zorder=5)
    mean = dl.groupby("delay")[col].mean()
    ax.plot(mean.index, mean.values, color=color, lw=2.4, zorder=6)


def legend_handles(color="0.3"):
    return [
        Line2D([], [], color=color, lw=2.4,
               label=label_for("delayed_critic") + ": mean of 2 seeds"),
        Line2D([], [], color=color, lw=0.9, alpha=0.55, marker=marker_for("delayed_critic"),
               ms=4, mec="k", label="Delayed critic: each seed (i42/p1234, i43/p43)"),
        Line2D([], [], color=color, lw=1.4, ls="--", marker="o", alpha=0.6, ms=4,
               label=label_for("privileged_critic") + ": each run, median dashed"),
        Line2D([], [], ls="", marker="o", mfc="none", mec="k",
               label="Privileged run at the same seed (i42/p1234)"),
        plt.Rectangle((0, 0), 1, 1, color=color_for("undelayed"), alpha=0.25,
                      label="Delay 0: both critics, range"),
    ]


def fig_reward(d: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        sub = d[d["arm"] == arm]
        c = color_for(arm)
        _undelayed_band(ax, d, "reward_480")
        _privileged(ax, sub, "reward_480", c)
        _delayed(ax, sub, "reward_480", c)
        ax.set_title(ARM_TITLE[arm], color=c)
        ax.set_xlabel("Observation delay (steps)")
        ax.set_xlim(-0.5, 26)
        add_ms_axis(ax, 26, ctrl_dt_ms=CTRL_DT_MS)
    axes[0].set_ylabel(reward_label("dmc_eval") + "\nmean over 430-480 M steps")
    axes[1].legend(handles=legend_handles(), loc="lower left", fontsize=8)
    fig.suptitle("WalkerWalk (torque): privileged vs delayed critic, at 480 M steps", y=1.02)
    fig.tight_layout()
    return fig


def fig_curves(d: pd.DataFrame, curves: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, len(CURVE_DELAYS), figsize=(16, 6), sharex=True,
                             sharey=True)
    meta = d.set_index("wandb_id")
    curves = curves.join(meta[["critic", "arm", "delay", "matched_seed"]], on="wandb_id")
    for i, arm in enumerate(ARMS):
        c = color_for(arm)
        for j, delay in enumerate(CURVE_DELAYS):
            ax = axes[i, j]
            cell = curves[(curves["arm"] == arm) & (curves["delay"] == delay)]
            for wid, g in cell.groupby("wandb_id"):
                crit = g["critic"].iloc[0]
                if crit == "delayed":
                    ax.plot(g["step"] / 1e6, g["reward"], color=c, lw=2.0, zorder=5)
                else:
                    ax.plot(g["step"] / 1e6, g["reward"], color=c, lw=0.9, alpha=0.4,
                            ls="--" if not g["matched_seed"].iloc[0] else "-")
                    if g["matched_seed"].iloc[0]:
                        ax.plot(g["step"] / 1e6, g["reward"], color="k", lw=0.8, alpha=0.6)
            ax.set_title(f"delay {delay} ({delay * CTRL_DT_MS} ms)", fontsize=9)
            if i == 1:
                ax.set_xlabel("Steps (M)")
        axes[i, 0].set_ylabel(f"{'With' if arm == 'efference' else 'No'} efference copy\n"
                              + reward_label("dmc_eval"), fontsize=8)
    handles = [
        Line2D([], [], color="0.3", lw=2.0,
               label="Delayed critic (i42/p1234, i43/p43)"),
        Line2D([], [], color="0.3", lw=0.9, ls="--", alpha=0.6,
               label="Privileged critic, other seeds"),
        Line2D([], [], color="k", lw=0.8, label="Privileged critic, i42/p1234"),
    ]
    axes[0, -1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle("Eval reward during training, to 480 M steps (longer privileged runs "
                 "truncated there)", y=1.0)
    fig.tight_layout()
    return fig


def fig_mechanism(d: pd.DataFrame) -> plt.Figure:
    panels = [
        ("critic_r2_480", "Critic $R^2$ (430-480 M)\n1 - MSE / Var(GAE target)"),
        ("loglik_480", "Policy log-likelihood of its own actions\n(higher = narrower policy)"),
        ("clip_frac_480", "PPO clipping fraction\n(how far each update moves the policy)"),
        ("train_reward_480", "Training reward per step (x10 scaled)\nstochastic policy"),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(16, 7), sharex=True)
    for i, arm in enumerate(ARMS):
        sub = d[d["arm"] == arm]
        c = color_for(arm)
        for j, (col, label) in enumerate(panels):
            ax = axes[i, j]
            _undelayed_band(ax, d, col)
            _privileged(ax, sub, col, c)
            _delayed(ax, sub, col, c)
            if i == 0:
                ax.set_title(label, fontsize=9)
            if i == 1:
                ax.set_xlabel("Observation delay (steps)")
        axes[i, 0].set_ylabel(ARM_TITLE[arm], color=c, fontsize=9)
    axes[0, 0].legend(handles=legend_handles(), fontsize=7, loc="lower left")
    fig.suptitle("Training-side diagnostics, mean over 430-480 M steps", y=1.0)
    fig.tight_layout()
    return fig


def credit_share(k: int) -> tuple[float, float]:
    """For a critic that never sees a_t's effect until k steps later (no efference copy):

    * the share of steps in a ROLLOUT-step window whose GAE advantage contains *any* value
      term that depends on a_t -- a value V(o_{t+l-k}) depends on a_t only if l >= k + 1,
      and l is capped by the window end;
    * the GAE weight that critic-mediated credit gets even then: (gamma * lambda)^(k+1).
    """
    share = np.mean([1.0 if (ROLLOUT - 1 - t) >= k + 1 else 0.0 for t in range(ROLLOUT)])
    return share, (GAMMA * LAMBDA) ** (k + 1)


def fig_credit(d: pd.DataFrame) -> plt.Figure:
    ks = np.arange(0, 31)
    share, weight = zip(*(credit_share(int(k)) for k in ks))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, share, color="k", lw=1.8,
            label="share of a 30-step window whose advantage\ncan see $a_t$ through the critic")
    ax.plot(ks, weight, color="k", lw=1.2, ls=":",
            label=r"GAE weight on that credit, $(\gamma\lambda)^{k+1}$")
    priv = d[(d["arm"] == "no_efference") & (d["critic"] == "privileged")] \
        .groupby("delay")["reward_480"].median()
    dl = d[(d["arm"] == "no_efference") & (d["critic"] == "delayed")] \
        .groupby("delay")["reward_480"].mean()
    ratio = (dl / priv).dropna()
    ax.plot(ratio.index, ratio.values, color=color_for("no_efference"), lw=2,
            marker=marker_for("delayed_critic"), mec="k",
            label="no efference copy: delayed (mean of 2 seeds) /\nprivileged (median) reward at 480 M")
    ax.set_xlabel("Observation delay k (steps)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Without an efference copy, a delayed critic cannot see $a_t$'s effect\n"
                 "for k steps: structural limit vs measured ratio", fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    FIG.mkdir(exist_ok=True)
    d = pd.read_csv(DATA)
    d = d[d["usable"]]
    curves = pd.read_csv(CURVES)
    entries = {}
    for name, fig, inputs in (
        ("reward_vs_delay.png", fig_reward(d), (DATA,)),
        ("training_curves.png", fig_curves(d, curves), (DATA, CURVES)),
        ("mechanism.png", fig_mechanism(d), (DATA,)),
        ("credit_horizon.png", fig_credit(d), (DATA,)),
    ):
        entries[name] = provenance(fig, HERE, *inputs)
        fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    write_figure_manifest(HERE, entries)
    print("wrote", ", ".join(entries))


if __name__ == "__main__":
    main()
