"""Figures for the refactor regression check. Reads only the CSVs in this folder."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    add_ms_axis, apply_style, color_for, label_for, marker_for, provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"


def load() -> pd.DataFrame:
    return pd.read_csv(DATA)


def fig_reward_vs_delay(df: pd.DataFrame) -> Path:
    """The pre-refactor sweep as the reference curve, the new runs as points on it."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    base = df[df.condition == "pre_refactor"].sort_values("delay_k")
    ax.plot(base.delay_k, base.old_eval_reward, color=color_for("pre_refactor"),
            marker=marker_for("pre_refactor"), markersize=4, linewidth=1.6,
            label=f"{label_for('pre_refactor')}, 4096 envs (n={len(base)})")

    for condition in ("unregularized", "fixed"):
        sub = df[df.condition == condition].sort_values("delay_k")
        for n_envs, grp in sub.groupby("n_envs"):
            filled = n_envs == 4096
            ax.plot(grp.delay_k, grp.old_eval_reward, linestyle="none",
                    marker=marker_for(condition), markersize=11,
                    color=color_for(condition),
                    markerfacecolor=color_for(condition) if filled else "none",
                    markeredgewidth=2.0,
                    label=f"{label_for(condition)}, {int(n_envs)} envs")

    ax.set_xlabel("Proprioception delay (control steps)")
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_title("Refactor regression: the fixed path lands back on the baseline curve")
    ax.set_xlim(-2, 32)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    add_ms_axis(ax, 32)
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "reward_vs_delay.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_matched(df: pd.DataFrame) -> Path:
    """The controlled comparison: same delay, same n_envs, same seed, three code epochs."""
    matched = df[(df.n_envs == 4096) & (df.seed == 42) & (df.delay_k.isin([0, 10]))]
    order = ["pre_refactor", "unregularized", "fixed"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=False)
    for ax, delay in zip(axes, (0, 10)):
        sub = matched[matched.delay_k == delay]
        present = [c for c in order if c in set(sub.condition)]
        values = [float(sub[sub.condition == c].old_eval_reward.iloc[0]) for c in present]
        ax.bar(range(len(present)), values,
               color=[color_for(c) for c in present], width=0.6)
        baseline = sub[sub.condition == "pre_refactor"].old_eval_reward
        if len(baseline):
            ax.axhline(float(baseline.iloc[0]), color="k", linestyle=":", linewidth=1)
        for i, (c, v) in enumerate(zip(present, values)):
            delta = ""
            if len(baseline) and c != "pre_refactor":
                delta = f"\n{100 * (v - float(baseline.iloc[0])) / float(baseline.iloc[0]):+.1f}%"
            ax.text(i, v, f"{v:.0f}{delta}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([label_for(c).replace(", ", ",\n") for c in present],
                           fontsize=7)
        ax.set_title(f"delay {delay}, 4096 envs, seed 42")
        ax.set_ylim(0, max(values) * 1.25)
        if ax is axes[0]:
            ax.set_ylabel("Held-out episode reward")
    fig.suptitle("Matched comparison across the three code epochs", fontsize=10)
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "matched_epochs.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = load()
    entries = {p.name: "" for p in (fig_reward_vs_delay(df), fig_matched(df))}
    write_figure_manifest(HERE, entries)
    print("wrote", ", ".join(entries))


if __name__ == "__main__":
    main()
