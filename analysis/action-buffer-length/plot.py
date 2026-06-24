"""Plot the action-buffer-length comparison from data.csv.

Run from the repo root::

    ../.venv/bin/python analysis/action-buffer-length/plot.py

Reads ONLY data.csv (never the WandB API). See analysis/README.md §2. Produces:
  - figures/buffer_length_sweep.png   reward vs delay for full / truncated / no buffer
  - figures/fraction_captured.png     fraction of the full-buffer benefit kept by 5 actions
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import add_ms_axis, apply_style, color_for, marker_for

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
METRIC = "episode_reward/mean"
BUFFER_LEN = 5  # the fixed truncated-buffer length

LABELS = {
    "efference": "Full buffer (length = delay)",
    "efference_trunc": f"Truncated buffer ({BUFFER_LEN} actions)",
    "no_efference": "No buffer",
}
ORDER = ["efference", "efference_trunc", "no_efference"]
FRAC_MAX_DELAY = 50  # beyond this all conditions collapse toward the floor (ratio ill-defined)


def series(df, cond):
    sub = (df[df["condition"] == cond].dropna(subset=[METRIC, "delay_k"])
           .sort_values(METRIC, ascending=False).drop_duplicates("delay_k")
           .sort_values("delay_k"))
    return sub.set_index("delay_k")[METRIC]


def plot_sweep(df):
    s = {c: series(df, c) for c in ORDER}
    max_delay = max(v.index.max() for v in s.values())
    fig, ax = plt.subplots()
    for c in ORDER:
        ax.plot(s[c].index, s[c].values, color=color_for(c), marker=marker_for(c), label=LABELS[c])
    ax.axvline(BUFFER_LEN, ls=":", color="0.5", lw=1.2)
    ax.text(BUFFER_LEN + 1, ax.get_ylim()[1] * 0.05, f"buffer length = {BUFFER_LEN}",
            fontsize=6, color="0.4")
    ax.set_xlim(0, max_delay * 1.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode reward")
    ax.legend(fontsize=7, loc="upper right")
    sns.despine(ax=ax)
    add_ms_axis(ax, max_delay)
    out = FIGURES / "buffer_length_sweep.png"
    fig.savefig(out)
    print(f"Saved {out}")


def plot_fraction(df):
    full, trunc, noeff = series(df, "efference"), series(df, "efference_trunc"), series(df, "no_efference")
    delays = [d for d in trunc.index
              if d in full.index and d in noeff.index and d <= FRAC_MAX_DELAY
              and full[d] > noeff[d]]
    frac = [(trunc[d] - noeff[d]) / (full[d] - noeff[d]) for d in delays]
    fig, ax = plt.subplots()
    ax.plot(delays, [f * 100 for f in frac], color=color_for("efference_trunc"),
            marker=marker_for("efference_trunc"))
    ax.axhline(100, ls="--", color=color_for("efference"), lw=1.0, label="Full buffer")
    ax.axhline(0, ls="--", color=color_for("no_efference"), lw=1.0, label="No buffer")
    ax.set_ylim(0, 115)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel(f"% of full-buffer benefit\nkept by {BUFFER_LEN} actions")
    ax.legend(fontsize=7, loc="lower left")
    sns.despine(ax=ax)
    out = FIGURES / "fraction_captured.png"
    fig.savefig(out)
    print(f"Saved {out}")


def main():
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    plot_sweep(df)
    plot_fraction(df)


if __name__ == "__main__":
    main()
