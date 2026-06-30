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

# Shared actor x-axis. Predictor occupies 0..4, decoder 5..9; the efference net
# only fills the decoder portion. (x, label, regex-on-path-without-"layer::").
PREDICTOR_STAGES = [
    (0, "pred 1", r"3/action/1/predictor/0$"),
    (1, "pred 2", r"3/action/1/predictor/1$"),
    (2, "pred 3", r"3/action/1/predictor/2$"),
    (3, "pred 4", r"3/action/1/predictor/3$"),
    (4, "p̂",  r"3/action/1/predictor/4$"),
]
DECODER_STAGES = [
    (5, "dec 1", r"3/action/1/(decoder/)?0$"),
    (6, "dec 2", r"3/action/1/(decoder/)?1$"),
    (7, "dec 3", r"3/action/1/(decoder/)?2$"),
    (8, "dec 4", r"3/action/1/(decoder/)?3$"),
    (9, "out",   r"3/action/1/(decoder/)?5/action$"),
]
XTICKS = [s[0] for s in PREDICTOR_STAGES + DECODER_STAGES]
XLABELS = [s[1] for s in PREDICTOR_STAGES + DECODER_STAGES]


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
                                 figsize=(3.2 * len(delays), 6.4),
                                 sharex=True, sharey="row", squeeze=False)

        for r, (target, row_title) in enumerate(targets):
            for c, delay in enumerate(delays):
                ax = axes[r][c]
                dd = d[d.delay_k == delay]

                # forward model: predictor + decoder as ONE continuous line.
                fm = dd[dd.condition == "forward_model"]
                if not fm.empty:
                    xs, ys = _line(fm, target, PREDICTOR_STAGES + DECODER_STAGES)
                    if xs:
                        ax.plot(xs, ys, color=color_for("forward_model"),
                                marker="o", label=label_for("forward_model"))
                # efference: decoder portion only.
                ef = dd[dd.condition == "efference"]
                if not ef.empty:
                    xs, ys = _line(ef, target, DECODER_STAGES)
                    if xs:
                        ax.plot(xs, ys, color=color_for("efference"),
                                marker="s", label=label_for("efference"))

                # references (prefer efference; else forward_model for that delay)
                ref = ef if not ef.empty else fm
                base = ref[(ref.target == target)
                           & (ref.probe == "input::delayed_proprio")]["test_r2"]
                ceil = ref[(ref.target == target)
                           & (ref.probe == "input::current_proprio")]["test_r2"]
                if len(base):
                    ax.axhline(base.iloc[0], ls="--", color="0.45",
                               label="delayed input (baseline)")
                if len(ceil):
                    ax.axhline(ceil.iloc[0], ls=":", color="0.7",
                               label="current input (ceiling)")

                ax.axhline(0, color="k", lw=0.6)
                if r == 0:
                    ax.set_title(f"delay = {int(delay)} steps "
                                 f"({int(delay) * 10} ms)")
                if c == 0:
                    ax.set_ylabel(f"{row_title}\nheld-out $R^2$")
            axes[r][0].set_ylim(top=1.05)

        for ax in axes[-1]:
            ax.set_xticks(XTICKS)
            ax.set_xticklabels(XLABELS, fontsize=7, rotation=45)
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
