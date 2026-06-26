"""Plot the train/eval generalization analysis from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/train-eval-generalization/plot.py

Reads ONLY data.csv (never WandB / eval_results). See analysis/README.md §2. Produces:
  - figures/delay_tolerance.png   reward-per-step & survival-fraction vs delay, per dataset
  - figures/generalization_gap.png  train-vs-test gap vs delay, per network size
  - figures/failure_modes.png     how the policy fails on new_eval as delay grows

Cumulative episode reward is NOT comparable across datasets (the rollout window is 1500
control steps for new_eval vs 250 for train/old), so we plot length-normalised metrics:
reward_per_step (reward rate while alive) and survival_frac (lifespan / rollout length).

Note on time units: clip_length is in mocap frames @ 50 Hz, so the clips are 5 s (250 frames,
train/old) and 30 s (1500 frames, new). The eval rollout scans the FULL clip — 502 control
steps (train/old) and 3002 (new) at ctrl_dt = 0.01 s (100 Hz) — so survival_frac is normalised
by those scan lengths. See report.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import add_ms_axis, apply_style, color_for, label_for, marker_for

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

# Datasets get their own palette (conditions already own CONDITION_STYLE colours).
DATASET_STYLE = {
    "train":    {"color": "0.25", "marker": "o", "label": "Train (80% split, 5 s clips)"},
    "old_eval": {"color": "C0",   "marker": "s", "label": "Held-out test (same 5 s clips)"},
    "new_eval": {"color": "C3",   "marker": "^", "label": "New eval set (30 s clips)"},
}
DATASET_ORDER = ["train", "old_eval", "new_eval"]
SIZE_ORDER = ["efference", "efference_deeper", "efference_larger"]
CTRL_DT_S = 0.01  # 100 control steps / s; lifespan_steps -> seconds


def agg(df, condition, dataset, value):
    """Mean over replicate seeds per delay (only delay 0 has >1), sorted by delay."""
    sub = df[(df.condition == condition) & (df.dataset == dataset)]
    return sub.groupby("delay_k")[value].mean().sort_index()


# --------------------------------------------------------------------------- #
# Figure 0: raw (non-normalised) reward & lifetime (efference)
# --------------------------------------------------------------------------- #
def plot_raw(df):
    """Raw episode reward and lifetime vs delay. train/old (5 s clips, 502-step rollout) and
    new_eval (30 s clips, 3002-step rollout) are on different panels because their scales differ
    purely by clip length."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
    max_delay = df[df.condition == "efference"]["delay_k"].max()
    groups = [("train", "old_eval"), ("new_eval",)]
    titles = ["Train / held-out test (5 s clips)", "New eval set (30 s clips)"]
    for col, (grp, title) in enumerate(zip(groups, titles)):
        for ds in grp:
            st = DATASET_STYLE[ds]
            rew = agg(df, "efference", ds, "episode_reward_mean")
            axes[0, col].plot(rew.index, rew.values, color=st["color"], marker=st["marker"],
                              label=st["label"], ms=4)
            life = agg(df, "efference", ds, "lifespan_steps") * CTRL_DT_S
            axes[1, col].plot(life.index, life.values, color=st["color"], marker=st["marker"],
                              label=st["label"], ms=4)
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].legend(fontsize=7, loc="upper right")
    axes[0, 0].set_ylabel("Episode reward (raw, sum)")
    axes[1, 0].set_ylabel("Lifetime (s)")
    for col in (0, 1):
        axes[0, col].set_xticklabels([])
        add_ms_axis(axes[0, col], max_delay)
        axes[1, col].set_xlabel("Observation delay (steps)")
    for ax in axes.flat:
        ax.set_xlim(0, max_delay * 1.05)
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)
    fig.tight_layout()
    out = FIGURES / "raw_reward_lifetime.png"
    fig.savefig(out)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Figure 1: delay tolerance on each dataset (efference)
# --------------------------------------------------------------------------- #
def plot_delay_tolerance(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    max_delay = df[df.condition == "efference"]["delay_k"].max()
    for ds in DATASET_ORDER:
        st = DATASET_STYLE[ds]
        for ax, metric in ((ax1, "reward_per_step"), (ax2, "survival_frac")):
            s = agg(df, "efference", ds, metric)
            ax.plot(s.index, s.values, color=st["color"], marker=st["marker"],
                    label=st["label"], ms=4)
    ax1.set_ylabel("Reward per step (alive)")
    ax1.set_ylim(bottom=0)
    ax1.set_title("Imitation quality while tracking", fontsize=9)
    ax2.set_ylabel("Survival fraction (lifespan / rollout)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("How long it stays on the clip", fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xlabel("Observation delay (steps)")
        ax.set_xlim(0, max_delay * 1.05)
        sns.despine(ax=ax)
        add_ms_axis(ax, max_delay)
    ax1.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    out = FIGURES / "delay_tolerance.png"
    fig.savefig(out)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Figure 2: train-vs-test generalization gap by network size
# --------------------------------------------------------------------------- #
def plot_generalization_gap(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    max_delay = 100
    for cond in SIZE_ORDER:
        params = df[df.condition == cond]["total_params"].iloc[0]
        lab = f"{label_for(cond)} ({params/1e6:.1f} M)"
        for ax, metric in ((ax1, "reward_per_step"), (ax2, "survival_frac")):
            tr = agg(df, cond, "train", metric)
            te = agg(df, cond, "old_eval", metric)
            delays = tr.index.intersection(te.index)
            gap = (tr[delays] - te[delays])
            ax.plot(delays, gap.values, color=color_for(cond), marker=marker_for(cond),
                    label=lab, ms=4)
    ax1.set_ylabel("train − held-out test\n(reward per step)")
    ax1.set_title("Overfitting in imitation quality", fontsize=9)
    ax2.set_ylabel("train − held-out test\n(survival fraction)")
    ax2.set_title("Overfitting in robustness", fontsize=9)
    for ax in (ax1, ax2):
        ax.axhline(0, color="0.7", lw=0.8, ls="--")
        ax.set_xlabel("Observation delay (steps)")
        ax.set_xlim(0, max_delay * 1.05)
        sns.despine(ax=ax)
        add_ms_axis(ax, max_delay)
    ax1.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = FIGURES / "generalization_gap.png"
    fig.savefig(out)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Figure 3: how the policy fails on new_eval as delay grows (efference)
# --------------------------------------------------------------------------- #
def plot_failure_modes(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    max_delay = 100

    # (a) termination reason rates on the new eval set
    reasons = [
        ("survived", "Survived to clip end", "C2", "o"),
        ("term_root_too_rotated", "Terminated: root too rotated", "C3", "^"),
        ("term_root_too_far", "Terminated: root too far", "C1", "s"),
        ("term_pose_error", "Terminated: pose error", "C4", "v"),
    ]
    for col, lab, color, marker in reasons:
        s = agg(df, "efference", "new_eval", col)
        ax1.plot(s.index, s.values, color=color, marker=marker, label=lab, ms=4)
    ax1.set_ylabel("Fraction of clips")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("Termination reason (new eval)", fontsize=9)
    ax1.legend(fontsize=7, loc="center right")

    # (b) error components on new_eval, as fold-change vs the zero-delay baseline,
    # so components on very different scales can be compared on one axis.
    errors = [
        ("err_root_angular_error", "Root angular error", "C3", "^"),
        ("err_root_pos_distance", "Root position dist.", "C1", "s"),
        ("err_joint_l2_error", "Joint L2 error", "C0", "o"),
        ("err_body_total", "Body error (total)", "C5", "D"),
    ]
    for col, lab, color, marker in errors:
        s = agg(df, "efference", "new_eval", col)
        base = s.iloc[0]
        ax2.plot(s.index, s.values / base, color=color, marker=marker, label=lab, ms=4)
    ax2.set_ylabel("Error ÷ zero-delay error")
    ax2.set_title("Which error grows with delay (new eval)", fontsize=9)
    ax2.legend(fontsize=7, loc="upper left")

    for ax in (ax1, ax2):
        ax.set_xlabel("Observation delay (steps)")
        ax.set_xlim(0, max_delay * 1.05)
        sns.despine(ax=ax)
        add_ms_axis(ax, max_delay)
    fig.tight_layout()
    out = FIGURES / "failure_modes.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_raw(df)
    plot_delay_tolerance(df)
    plot_generalization_gap(df)
    plot_failure_modes(df)


if __name__ == "__main__":
    main()
