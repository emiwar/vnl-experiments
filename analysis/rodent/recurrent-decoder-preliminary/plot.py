"""Figures for the preliminary recurrent-decoder question. Reads only the CSVs here."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    apply_style, color_for, label_for, marker_for, provenance, write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"

#: Cells where a feedforward and a recurrent run share delay, efference, n_envs, seed
#: *and* rollout_length. Grouped: the efference copy intact, then truncated.
MATCHED_FULL = [(0, 0, 1024, 52, 20), (10, 10, 1024, 52, 20), (10, 10, 4096, 42, 20)]
MATCHED_TRUNC = [(10, 1, 1024, 43, 20), (5, 1, 4096, 42, 20), (10, 1, 4096, 42, 60)]


def _cell(df, delay, eff, n_envs, seed, roll):
    return df[(df.delay_k == delay) & (df.efference_length == eff)
              & (df.n_envs == n_envs) & (df.seed == seed)
              & (df.rollout_length == roll)]


def fig_matched(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels, ff_vals, rnn_vals, group = [], [], [], []
    for tag, cells in (("full", MATCHED_FULL), ("trunc", MATCHED_TRUNC)):
        for delay, eff, n_envs, seed, roll in cells:
            cell = _cell(df, delay, eff, n_envs, seed, roll)
            ff = cell[cell.condition == "feedforward"].old_eval_reward
            rnn = cell[cell.condition == "recurrent"].old_eval_reward
            if not len(ff) or not len(rnn):
                continue
            labels.append(f"delay {delay}, eff {eff}\n{n_envs} envs, seed {seed}\nrollout {roll}")
            ff_vals.append(float(ff.iloc[0]))
            rnn_vals.append(float(rnn.iloc[0]))
            group.append(tag)

    x = np.arange(len(labels))
    ax.bar(x - 0.19, ff_vals, width=0.36, color=color_for("feedforward"),
           label=label_for("feedforward"))
    ax.bar(x + 0.19, rnn_vals, width=0.36, color=color_for("recurrent"),
           label=label_for("recurrent"))
    for i, (a, b) in enumerate(zip(ff_vals, rnn_vals)):
        pct = 100 * (b - a) / a
        ax.text(i + 0.19, b, f"{pct:+.1f}%", ha="center", va="bottom", fontsize=8,
                fontweight="bold" if abs(pct) > 20 else "normal")

    peak = max(ff_vals + rnn_vals)
    split = group.index("trunc") - 0.5 if "trunc" in group else None
    if split is not None:
        ax.axvline(split, color="0.6", linestyle="--", linewidth=1)
        top = peak * 1.07
        ax.text((split - 1) / 2, top, "efference copy intact", ha="center", fontsize=9,
                color="0.35")
        ax.text((split + len(labels) - 0.5) / 2, top, "efference copy truncated",
                ha="center", fontsize=9, color="0.35")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_ylim(0, peak * 1.15)
    # Title and legend both live in the figure margin, so neither can collide with a
    # bar, the percentage labels or the group annotations.
    fig.suptitle("Recurrence ties when the action queue is available,\n"
                 "and wins by ~65 % when it is truncated", fontsize=11, y=0.995)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False, fontsize=9)
    provenance(fig, HERE, DATA)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIGURES / "matched_pairs.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_efference(df: pd.DataFrame) -> Path:
    """How each architecture degrades as the explicit action memory is removed."""
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    sub = df[(df.delay_k == 10) & (df.n_envs == 4096) & (df.seed == 42)]

    for condition in ("feedforward", "recurrent"):
        rows = sub[sub.condition == condition]
        points = []
        for eff in sorted(rows.efference_length.unique()):
            cand = rows[rows.efference_length == eff]
            # Prefer the rollout=20 run so the series is internally consistent.
            pick = cand[cand.rollout_length == 20]
            pick = pick if len(pick) else cand.sort_values("rollout_length").head(1)
            r = pick.iloc[0]
            points.append((eff, r.old_eval_reward, int(r.rollout_length)))
        if not points:
            continue
        xs, ys, rolls = zip(*points)
        ax.plot(xs, ys, marker=marker_for(condition), markersize=9,
                color=color_for(condition), linewidth=1.8, label=label_for(condition))
        for xv, yv, rv in points:
            if rv != 20:
                ax.annotate(f"rollout {rv}", (xv, yv), textcoords="offset points",
                            xytext=(6, -12), fontsize=7.5, color=color_for(condition))

    ax.set_xlabel("Efference-copy queue length (delay = 10)")
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_title("Removing the explicit action memory costs the feedforward net\n"
                 "~50 %, the recurrent net ~14 %", fontsize=10.5)
    ax.set_xticks([0, 1, 10])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "efference_dependence.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_rollout(df: pd.DataFrame) -> Path:
    """BPTT horizon sweep for the recurrent decoder."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sub = df[(df.condition == "recurrent") & (df.delay_k == 10)
             & (df.n_envs == 4096) & (df.seed == 42)]
    for eff, style in ((10, "-"), (1, "--")):
        rows = sub[sub.efference_length == eff].sort_values("rollout_length")
        if not len(rows):
            continue
        ax.plot(rows.rollout_length, rows.old_eval_reward, style,
                marker=marker_for("recurrent"), markersize=9,
                color=color_for("recurrent"),
                markerfacecolor=color_for("recurrent") if eff == 10 else "none",
                markeredgewidth=2.0, label=f"efference queue = {eff}")
    ff = df[(df.condition == "feedforward") & (df.delay_k == 10) & (df.n_envs == 4096)
            & (df.seed == 42) & (df.efference_length == 10)]
    if len(ff):
        ax.axhline(float(ff.old_eval_reward.iloc[0]), color=color_for("feedforward"),
                   linestyle=":", linewidth=1.6,
                   label="Feedforward, eff 10, rollout 20")
    secax = ax.secondary_xaxis("top", functions=(lambda v: v * 10, lambda v: v / 10))
    secax.set_xlabel("BPTT horizon (ms)")
    ax.set_xlabel("rollout_length (control steps)")
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_title("Longer BPTT helps the recurrent decoder, saturating by 40", fontsize=10.5)
    ax.set_xticks([20, 40, 60])
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "rollout_length.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    figs = (fig_matched(df), fig_efference(df), fig_rollout(df))
    write_figure_manifest(HERE, {p.name: "" for p in figs})
    print("wrote", ", ".join(p.name for p in figs))


if __name__ == "__main__":
    main()
