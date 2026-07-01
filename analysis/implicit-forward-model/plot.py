"""Plot the implicit-forward-model probe from data.csv.

Reads ONLY data.csv. Produces per-dataset figures with one column per delay and
two rows (decode the current proprioception; decode the delta = current -
delayed). Held-out decoding R² is drawn along the actor.

The x-axis is a shared "depth along the actor": x=0 is the network INPUT
([delayed proprioception + efference copy] — the fair "layer-0" baseline), then a
forward-model network runs its predictor (1..5) before the decoder (6..10); an
efference network jumps straight from the input to the decoder.

Two figures are produced per dataset:
  * ``actor_pathway_<ds>``     — implicit (efference) vs explicit (forward_model).
  * ``actor_pathway_pg_<ds>``  — explicit (forward_model) vs policy-gradient
                                 (pg_forward_model): same architecture, but the
                                 predictor trained by the policy gradient, not the
                                 self-supervised L2. Tests whether the pg
                                 predictor represents the current state.

Run from the repo root::

    ../.venv/bin/python analysis/implicit-forward-model/plot.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

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
FM_STAGES = [INPUT_STAGE] + PREDICTOR_STAGES + DECODER_STAGES      # has a predictor
EF_STAGES = [INPUT_STAGE] + DECODER_STAGES                        # no predictor
XTICKS = [s[0] for s in FM_STAGES]
XLABELS = [s[1] for s in FM_STAGES]

# Series styling for the two figures (explicit colours; pg not in shared style).
EF_SERIES = [
    dict(cond="forward_model", label="Explicit forward model", color="C2",
         marker="o", stages=FM_STAGES),
    dict(cond="efference", label="With efference copy", color="C1",
         marker="s", stages=EF_STAGES),
]
PG_SERIES = [
    dict(cond="forward_model", label="Explicit FM (L2 loss)", color="C2",
         marker="o", stages=FM_STAGES),
    dict(cond="pg_forward_model", label="Policy-gradient FM (loss=0)", color="C4",
         marker="D", stages=FM_STAGES),
]


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


def _figure(df, dataset, series, out_name, suptitle):
    d = df[df.dataset == dataset]
    if d.empty or not any((d.condition == s["cond"]).any() for s in series):
        return
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
            for s in series:
                sub = dd[dd.condition == s["cond"]]
                if sub.empty:
                    continue
                xs, ys = _line(sub, target, s["stages"])
                if xs:
                    ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                            label=s["label"])
            # ceiling (decode from the true current proprioception)
            ref = next((dd[dd.condition == s["cond"]] for s in series
                        if not dd[dd.condition == s["cond"]].empty), None)
            if ref is not None:
                ceil = ref[(ref.target == target)
                           & (ref.probe == "input::current_proprio")]["test_r2"]
                if len(ceil):
                    ax.axhline(ceil.iloc[0], ls=":", color="0.7",
                               label="current input (ceiling)")
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(5.5, color="0.85", lw=0.8, zorder=0)  # predictor | decoder
            if r == 0:
                ax.set_title(f"delay = {int(delay)} steps ({int(delay) * 10} ms)")
            if c == 0:
                ax.set_ylabel(f"{row_title}\nheld-out $R^2$")
        axes[r][0].set_ylim(top=1.05)

    for ax in axes[-1]:
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XLABELS, fontsize=7.5, rotation=45, ha="right")
        ax.set_xlabel("depth along actor")
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(suptitle, y=1.06)
    fig.tight_layout()
    out = FIGURES / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def main() -> None:
    try:
        from vnl_experiments.wandb_utils.style import apply_style
        apply_style()
    except Exception:
        pass
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "data.csv")
    df = df[df.dataset.notna()]
    for dataset in df["dataset"].unique():
        _figure(df, dataset, EF_SERIES, f"actor_pathway_{dataset}.png",
                f"Linear decodability of the current state along the actor "
                f"({dataset})")
        _figure(df, dataset, PG_SERIES, f"actor_pathway_pg_{dataset}.png",
                f"Explicit vs policy-gradient forward model "
                f"({dataset})")


if __name__ == "__main__":
    main()
