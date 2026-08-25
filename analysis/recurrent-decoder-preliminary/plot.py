"""Figures for the preliminary recurrent-decoder question. Reads only the CSVs here."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    add_ms_axis, apply_style, color_for, label_for, marker_for, provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"

#: Cells where a feedforward and a recurrent run share delay, efference, n_envs and seed.
MATCHED = [(0, 0, 1024, 52), (10, 10, 1024, 52), (10, 1, 1024, 43), (10, 10, 4096, 42)]


def fig_matched(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels, ff_vals, rnn_vals = [], [], []
    for delay, eff, n_envs, seed in MATCHED:
        cell = df[(df.delay_k == delay) & (df.efference_length == eff)
                  & (df.n_envs == n_envs) & (df.seed == seed)]
        ff = cell[cell.condition == "feedforward"].old_eval_reward
        rnn = cell[cell.condition == "recurrent"].old_eval_reward
        if not len(ff) or not len(rnn):
            continue
        labels.append(f"delay {delay}, eff {eff}\n{n_envs} envs, seed {seed}")
        ff_vals.append(float(ff.iloc[0]))
        rnn_vals.append(float(rnn.iloc[0]))

    x = np.arange(len(labels))
    ax.bar(x - 0.19, ff_vals, width=0.36, color=color_for("feedforward"),
           label=label_for("feedforward"))
    ax.bar(x + 0.19, rnn_vals, width=0.36, color=color_for("recurrent"),
           label=label_for("recurrent"))
    for i, (a, b) in enumerate(zip(ff_vals, rnn_vals)):
        ax.text(i + 0.19, b, f"{100 * (b - a) / a:+.1f}%", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_title("Matched pairs: recurrence ties at delay 0 and at 4096 envs,\n"
                 "loses at 1024 envs under delay", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(ff_vals + rnn_vals) * 1.2)
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "matched_pairs.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_efference(df: pd.DataFrame) -> Path:
    """Where a short-efference recurrent run lands on the feedforward delay curve."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    curve = df[(df.condition == "feedforward") & (df.n_envs == 4096)
               & (df.delay_k == df.efference_length)].sort_values("delay_k")
    ax.plot(curve.delay_k, curve.old_eval_reward, color=color_for("feedforward"),
            marker="o", markersize=4, linewidth=1.6,
            label="Feedforward, efference-matched (4096 envs)")

    rnn = df[(df.condition == "recurrent") & (df.n_envs == 4096)]
    for _, row in rnn.iterrows():
        short = row.efference_length < row.delay_k
        ax.plot(row.delay_k, row.old_eval_reward, linestyle="none",
                marker=marker_for("recurrent"), markersize=11,
                color=color_for("recurrent"),
                markerfacecolor="none" if short else color_for("recurrent"),
                markeredgewidth=2.0)
        ax.annotate(f"eff {int(row.efference_length)}",
                    (row.delay_k, row.old_eval_reward),
                    textcoords="offset points", xytext=(9, -3), fontsize=8,
                    color=color_for("recurrent"))

    ax.plot([], [], linestyle="none", marker=marker_for("recurrent"), markersize=9,
            color=color_for("recurrent"), label="Recurrent, full efference (4096 envs)")
    ax.plot([], [], linestyle="none", marker=marker_for("recurrent"), markersize=9,
            color=color_for("recurrent"), markerfacecolor="none", markeredgewidth=2.0,
            label="Recurrent, efference truncated to 1")

    ax.set_xlabel("Proprioception delay (control steps)")
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_title("A recurrent decoder with a 1-step efference copy, read against\n"
                 "the feedforward efference-matched curve", fontsize=10)
    ax.set_xlim(-1, 22)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    add_ms_axis(ax, 22)
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "efference_tolerance.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    entries = {p.name: "" for p in (fig_matched(df), fig_efference(df))}
    write_figure_manifest(HERE, entries)
    print("wrote", ", ".join(entries))


if __name__ == "__main__":
    main()
