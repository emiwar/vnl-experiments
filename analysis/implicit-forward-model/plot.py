"""Plot the implicit-forward-model probe from data.csv.

Reads ONLY data.csv. One column of subplots per observation delay; two rows
(decode the current proprioception; decode the delta = current - delayed). In
each subplot, held-out decoding R² is drawn along the actor:

* ``forward model`` — ONE continuous line (same colour) running through the
  predictor layers, the predictor output p̂, then the decoder layers and the
  action output. It is *longer on the left* than the efference line — a visual
  cue that this network does extra upstream computation (the explicit forward
  model) before decoding.
* ``efference``     — the implicit network: decoder layers + output only. Aligned
  to the decoder portion of the x-axis, so the two lines share their decoder
  positions.

Reference lines per subplot: ``delayed input`` (the obs_(t-k)->obs_t baseline,
dashed) and ``current input`` (ceiling, dotted). A line shows an implicit forward
model where it rises above the delayed-input baseline.

Run from the repo root::

    ../.venv/bin/python analysis/implicit-forward-model/plot.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vnl_experiments.wandb_utils.style import apply_style, color_for, label_for

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

# Shared actor x-axis. x=0 is the network INPUT (delayed proprioception +
# efference copy) — the "0th layer", and the principled baseline. The forward
# model then runs its predictor (1..5), and both architectures share the decoder
# (6..10); the efference net jumps straight from the input to its decoder.
INPUT_STAGE = (0, "input\n(delayed+\nefference)", r"^input::delayed_plus_efference$")
PREDICTOR_STAGES = [
    (1, "pred 1", r"3/action/1/predictor/0$"),
    (2, "pred 2", r"3/action/1/predictor/1$"),
    (3, "pred 3", r"3/action/1/predictor/2$"),
    (4, "pred 4", r"3/action/1/predictor/3$"),
    (5, "p̂",  r"3/action/1/predictor/4$"),
]
DECODER_STAGES = [
    (6, "dec 1", r"3/action/1/(decoder/)?0$"),
    (7, "dec 2", r"3/action/1/(decoder/)?1$"),
    (8, "dec 3", r"3/action/1/(decoder/)?2$"),
    (9, "dec 4", r"3/action/1/(decoder/)?3$"),
    (10, "out",  r"3/action/1/(decoder/)?5/action$"),
]
XTICKS = [s[0] for s in [INPUT_STAGE] + PREDICTOR_STAGES + DECODER_STAGES]
XLABELS = [s[1] for s in [INPUT_STAGE] + PREDICTOR_STAGES + DECODER_STAGES]


def _r2(sub, target, rx):
    rx_c = re.compile(rx)
    paths = sub["probe"].str.replace("layer::", "", regex=False)
    hit = sub[(sub.target == target) & paths.str.contains(rx_c)]
    return float(hit["test_r2"].iloc[0]) if len(hit) else None


def _line(sub, target, stages):
    xs, ys = [], []
    for x, _, rx in stages:
        r2 = _r2(sub, target, rx)
        if r2 is not None:
            xs.append(x); ys.append(r2)
    return xs, ys


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    df = df[df.dataset.notna()]

    for dataset in df["dataset"].unique():
        d = df[df.dataset == dataset]
        delays = sorted(d["delay_k"].unique())
        targets = [("proprio", "Decode current proprioception"),
                   ("delta", "Decode delta (current − delayed)")]
        fig, axes = plt.subplots(len(targets), len(delays),
                                 figsize=(3.9 * len(delays), 7.2),
                                 sharex=True, sharey="row", squeeze=False)

        for r, (target, row_title) in enumerate(targets):
            for c, delay in enumerate(delays):
                ax = axes[r][c]
                dd = d[d.delay_k == delay]

                # forward model: input -> predictor -> p̂ -> decoder, ONE line.
                fm = dd[dd.condition == "forward_model"]
                if not fm.empty:
                    xs, ys = _line(fm, target,
                                   [INPUT_STAGE] + PREDICTOR_STAGES + DECODER_STAGES)
                    if xs:
                        ax.plot(xs, ys, color=color_for("forward_model"),
                                marker="o", label=label_for("forward_model"))
                # efference: input -> decoder (no predictor; jumps the gap).
                ef = dd[dd.condition == "efference"]
                if not ef.empty:
                    xs, ys = _line(ef, target, [INPUT_STAGE] + DECODER_STAGES)
                    if xs:
                        ax.plot(xs, ys, color=color_for("efference"),
                                marker="s", label=label_for("efference"))

                # The only reference line is the ceiling (decode from the true
                # current proprioception); the baseline is now the x=0 INPUT
                # datapoint on each line (delayed proprioception + efference copy).
                ref = ef if not ef.empty else fm
                ceil = ref[(ref.target == target)
                           & (ref.probe == "input::current_proprio")]["test_r2"]
                if len(ceil):
                    ax.axhline(ceil.iloc[0], ls=":", color="0.7",
                               label="current input (ceiling)")

                ax.axhline(0, color="k", lw=0.6)
                ax.axvline(5.5, color="0.85", lw=0.8, zorder=0)  # input|... | decoder
                if r == 0:
                    ax.set_title(f"delay = {int(delay)} steps "
                                 f"({int(delay) * 10} ms)")
                if c == 0:
                    ax.set_ylabel(f"{row_title}\nheld-out $R^2$")
            axes[r][0].set_ylim(top=1.05)

        for ax in axes[-1]:
            ax.set_xticks(XTICKS)
            ax.set_xticklabels(XLABELS, fontsize=7.5, rotation=45, ha="right")
            ax.set_xlabel("depth along actor")
        # Single shared legend (de-duplicated).
        handles, labels = axes[0][0].get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        fig.legend(h2, l2, loc="upper center", ncol=4, fontsize=8,
                   bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"Linear decodability of the current state along the actor "
                     f"({dataset})", y=1.06)
        fig.tight_layout()
        out = FIGURES / f"actor_pathway_{dataset}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
