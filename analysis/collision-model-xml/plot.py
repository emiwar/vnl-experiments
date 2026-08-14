"""Figures for the new-XML + reference_root comparison.

Run from the repo root::

    ../.venv/bin/python analysis/collision-model-xml/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/collision-model-xml/plot.py   # for slides

Reads ONLY the committed CSVs (never WandB, never the artifact store) and writes
figures/. See analysis/README.md §3.

Encoding: **colour = network** (the canonical CONDITION_STYLE colours, so these figures
line up with every other question), **line style = arm** — dashed + open marker for the
baseline, solid + filled for the changed configuration.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils import (
    add_ms_axis,
    apply_style,
    color_for,
    marker_for,
    provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"
DATA_EVAL = HERE / "data_eval.csv"

_MANIFEST: dict[str, str] = {}


def stamp(fig, name: str, *inputs) -> None:
    """Add the provenance footer and remember it. Call just before ``savefig``."""
    _MANIFEST[f"{name}.png"] = provenance(fig, HERE, *(inputs or (DATA,)))


NETWORK_OF = {
    "encdec_old_current": "efference", "encdec_old_reference": "efference",
    "encdec_new_current": "efference", "encdec_new_reference": "efference",
    "expfm_old_current": "forward_model", "expfm_old_reference": "forward_model",
    "expfm_new_reference": "forward_model",
    "expfm_old_position": "forward_model", "expfm_new_position": "forward_model",
    "pgfm_old_current": "pg_forward_model", "pgfm_new_reference": "pg_forward_model",
}
NETWORK_LABEL = {"efference": "EncDec + efference", "forward_model": "Explicit FM",
                 "pg_forward_model": "Policy-gradient FM"}
LABEL = {
    "encdec_old_current": "old XML, current_root",
    "encdec_old_reference": "old XML, reference_root",
    "encdec_new_current": "new XML, current_root",
    "encdec_new_reference": "new XML, reference_root",
    "expfm_old_current": "old XML, current_root",
    "expfm_old_reference": "old XML, reference_root",
    "expfm_new_reference": "new XML, reference_root",
    "expfm_old_position": "old XML (position)",
    "expfm_new_position": "new XML (position)",
    "pgfm_old_current": "old XML, current_root",
    "pgfm_new_reference": "new XML, reference_root",
}
BASELINE_STYLE = dict(ls="--", mfc="none")
CHANGED_STYLE = dict(ls="-")

PAIRS = {
    "primary_encdec": ("encdec_old_current", "encdec_new_reference",
                       "EncDec + efference"),
    "primary_expfm": ("expfm_old_current", "expfm_new_reference", "Explicit FM"),
    "primary_pgfm": ("pgfm_old_current", "pgfm_new_reference", "Policy-gradient FM"),
    "xml_at_current": ("encdec_old_current", "encdec_new_current",
                       "XML effect, at current_root"),
    "xml_at_reference": ("encdec_old_reference", "encdec_new_reference",
                         "XML effect, at reference_root"),
    "frame_at_old": ("encdec_old_current", "encdec_old_reference",
                     "Frame effect, on old XML"),
    "frame_at_new": ("encdec_new_current", "encdec_new_reference",
                     "Frame effect, on new XML"),
    "frame_expfm": ("expfm_old_current", "expfm_old_reference",
                    "Frame effect, on old XML (explicit FM)"),
    "position_xml": ("expfm_old_position", "expfm_new_position",
                     "XML effect (position control)"),
}
PRIMARY_PAIRS = ["primary_encdec", "primary_expfm", "primary_pgfm"]
# The 2x2: XML contrasts warm, frame contrasts cool, so main effects read at a glance.
FACTOR_COLOR = {"xml_at_current": "C1", "xml_at_reference": "C3",
                "frame_at_old": "C0", "frame_at_new": "C9",
                "frame_expfm": "C2", "position_xml": "C5"}


def dedup(df: pd.DataFrame, condition: str, metric: str = "reward_mean") -> pd.DataFrame:
    """One point per delay for a condition (best of any duplicate runs at that delay)."""
    sub = df[df["condition"] == condition].dropna(subset=[metric])
    return (sub.sort_values(metric, ascending=False)
               .drop_duplicates(["delay_k"])
               .sort_values("delay_k"))


def paired(df: pd.DataFrame, pair: str, metric: str = "reward_mean") -> pd.DataFrame:
    """Delay-matched baseline/changed table with the absolute and relative difference."""
    base, changed, _ = PAIRS[pair]
    a = dedup(df, base, metric).set_index("delay_k")[metric]
    b = dedup(df, changed, metric).set_index("delay_k")[metric]
    out = pd.DataFrame({"baseline": a, "changed": b}).dropna()
    out["delta"] = out["changed"] - out["baseline"]
    out["pct"] = 100 * (out["changed"] / out["baseline"] - 1)
    return out.reset_index()


def curve_of(curves: pd.DataFrame, condition: str, delay: int) -> pd.DataFrame:
    sub = curves[(curves["condition"] == condition) & (curves["delay_k"] == delay)]
    return sub.sort_values("step")


def draw_arm(ax, df: pd.DataFrame, condition: str, *, baseline: bool, **kw) -> None:
    sub = dedup(df, condition)
    style = dict(color=color_for(NETWORK_OF[condition]),
                 marker=marker_for(NETWORK_OF[condition]),
                 label=LABEL[condition],
                 **(BASELINE_STYLE if baseline else CHANGED_STYLE))
    style.update(kw)
    ax.plot(sub["delay_k"], sub["reward_mean"], **style)


# ---------------------------------------------------------------------------
# Figure 1 (primary): the same contrast in three independent networks
# ---------------------------------------------------------------------------

def fig_primary(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for key in PRIMARY_PAIRS:
        base, changed, label = PAIRS[key]
        draw_arm(ax, df, base, baseline=True,
                 label=f"{label}: old XML, current_root")
        draw_arm(ax, df, changed, baseline=False,
                 label=f"{label}: new XML, reference_root")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode reward (eval on train clips)")
    ax.set_ylim(bottom=0)
    ax.set_title("Torque control: baseline (dashed) vs going-forward config (solid)")
    ax.legend(loc="lower left", fontsize=6.5, ncol=1)
    add_ms_axis(ax, df["delay_k"].max())
    sns.despine(ax=ax)

    ax = axes[1]
    for key in PRIMARY_PAIRS:
        m = paired(df, key)
        net = NETWORK_OF[PAIRS[key][1]]
        ax.plot(m["delay_k"], m["pct"], color=color_for(net), marker=marker_for(net),
                label=PAIRS[key][2])
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.axvspan(28, 62, color="0.85", zorder=0, lw=0)
    ax.annotate("30–60", (45, ax.get_ylim()[1]), ha="center", va="top", fontsize=7,
                color="0.45")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Reward change, new config vs baseline (%)")
    ax.set_title("Same contrast, three networks")
    ax.legend(loc="lower right", fontsize=8)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "primary", DATA)
    fig.savefig(FIGURES / "primary.png")
    print("Saved", FIGURES / "primary.png")


# ---------------------------------------------------------------------------
# Figure 2: the EncDec 2x2 — XML and frame varied independently
# ---------------------------------------------------------------------------

def fig_decomposition(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for cond, baseline, kw in [
        ("encdec_old_current", True, {}),
        ("encdec_old_reference", True, dict(marker="x", alpha=0.75)),
        ("encdec_new_current", False, dict(marker="x", alpha=0.75)),
        ("encdec_new_reference", False, {}),
    ]:
        draw_arm(ax, df, cond, baseline=baseline,
                 color="C0" if "old" in cond else "C3", **kw)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode reward (eval on train clips)")
    ax.set_ylim(bottom=0)
    ax.set_title("All four EncDec cells (blue = old XML, red = new)")
    ax.legend(loc="lower left", fontsize=7)
    add_ms_axis(ax, df["delay_k"].max())
    sns.despine(ax=ax)

    ax = axes[1]
    for key in ("xml_at_current", "xml_at_reference", "frame_at_old", "frame_at_new"):
        m = paired(df, key)
        ax.plot(m["delay_k"], m["pct"], color=FACTOR_COLOR[key], marker="o",
                ls="-" if key.startswith("xml") else "--", label=PAIRS[key][2])
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Reward change vs the other level (%)")
    ax.set_title("Each factor, held at both levels of the other")
    ax.legend(loc="lower left", fontsize=8)
    sns.despine(ax=ax)

    fig.suptitle("The XML carries the effect; the frame is free; they do not interact",
                 fontsize=10)
    fig.tight_layout()
    stamp(fig, "decomposition", DATA)
    fig.savefig(FIGURES / "decomposition.png")
    print("Saved", FIGURES / "decomposition.png")


# ---------------------------------------------------------------------------
# Figure 3: position control
# ---------------------------------------------------------------------------

def fig_position(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    draw_arm(ax, df, "expfm_old_position", baseline=True)
    draw_arm(ax, df, "expfm_new_position", baseline=False)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode reward (eval on train clips)")
    ax.set_ylim(bottom=0)
    ax.set_title("Explicit forward model, position control")
    ax.legend(loc="lower left", fontsize=8)
    add_ms_axis(ax, df["delay_k"].max())
    sns.despine(ax=ax)

    ax = axes[1]
    for key in ("position_xml", "xml_at_current", "xml_at_reference"):
        m = paired(df, key)
        ax.plot(m["delay_k"], m["pct"], color=FACTOR_COLOR[key], marker="o",
                ls="-" if key == "position_xml" else "--",
                label=PAIRS[key][2] + ("" if key == "position_xml" else " (torque)"))
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Reward change, new vs old XML (%)")
    ax.set_title("The XML penalty needs torque control")
    ax.legend(loc="lower left", fontsize=8)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "position", DATA)
    fig.savefig(FIGURES / "position.png")
    print("Saved", FIGURES / "position.png")


# ---------------------------------------------------------------------------
# Figure 4: convergence — lower plateau or slower climb?
# ---------------------------------------------------------------------------

def convergence_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in PRIMARY_PAIRS:
        for cond in PAIRS[key][:2]:
            for delay, group in curves[curves["condition"] == cond].groupby("delay_k"):
                group = group.sort_values("step")
                if len(group) < 12:
                    continue
                tail = group.tail(10)
                final = tail["reward_mean"].mean()
                gained = tail["reward_mean"].iloc[-1] - tail["reward_mean"].iloc[0]
                rows.append({"pair": key, "condition": cond, "delay_k": delay,
                             "final": final, "gained_last_100M": gained,
                             "gained_pct": 100 * gained / final})
    return pd.DataFrame(rows)


def fig_convergence(df: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    base, changed, _ = PAIRS["primary_encdec"]
    show = [30, 40, 50, 60]

    fig, axes = plt.subplots(1, len(show) + 1, figsize=(3.1 * (len(show) + 1), 3.7))
    for ax, delay in zip(axes, show):
        for cond, baseline in [(base, True), (changed, False)]:
            c = curve_of(curves, cond, delay)
            ax.plot(c["step"] / 1e6, c["reward_mean"], color=color_for(NETWORK_OF[cond]),
                    lw=1.2, ls="--" if baseline else "-",
                    label=LABEL[cond] if delay == show[0] else None)
        ax.set_title(f"EncDec, delay {delay}")
        ax.set_xlabel("Env steps (millions)")
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean episode reward")
    axes[0].legend(loc="lower right", fontsize=7)

    # Only the EncDec pair here: all six arms on one axis is unreadable, and the
    # three-network version is in convergence_table.csv.
    ax = axes[-1]
    slopes = convergence_table(curves)
    for cond, baseline in [(base, True), (changed, False)]:
        s = slopes[slopes["condition"] == cond].sort_values("delay_k")
        ax.plot(s["delay_k"], s["gained_pct"], color=color_for(NETWORK_OF[cond]),
                marker=marker_for(NETWORK_OF[cond]), ms=3, lw=1.2,
                label=LABEL[cond], **(BASELINE_STYLE if baseline else CHANGED_STYLE))
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.axvspan(28, 62, color="0.88", zorder=0, lw=0)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Reward still gained in the\nlast 100 M steps (% of final)")
    ax.set_title("Converged at 600 M? (EncDec)")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "convergence", DATA, CURVES)
    fig.savefig(FIGURES / "convergence.png")
    print("Saved", FIGURES / "convergence.png")
    return slopes


# ---------------------------------------------------------------------------
# Figure 5: throughput, GPU- and delay-matched
# ---------------------------------------------------------------------------

def throughput_table(df: pd.DataFrame) -> pd.DataFrame:
    """Delay-matched new/old throughput ratios, A100 only."""
    a100 = df[df["gpu"].astype(str).str.contains("A100")]
    rows = []
    for key in PRIMARY_PAIRS:
        base, changed, label = PAIRS[key]
        for metric in ("train_sps_median", "eval_sps_median"):
            a = dedup(a100, base, metric).set_index("delay_k")[metric]
            b = dedup(a100, changed, metric).set_index("delay_k")[metric]
            m = pd.DataFrame({"old": a, "new": b}).dropna()
            for delay, r in m.iterrows():
                rows.append({"pair": key, "label": label, "metric": metric,
                             "delay_k": delay, "old": r["old"], "new": r["new"],
                             "ratio": r["new"] / r["old"]})
    return pd.DataFrame(rows)


def fig_throughput(df: pd.DataFrame) -> pd.DataFrame:
    table = throughput_table(df)
    train = table[table["metric"] == "train_sps_median"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key in PRIMARY_PAIRS:
        s = train[train["pair"] == key].sort_values("delay_k")
        if s.empty:
            continue
        net = NETWORK_OF[PAIRS[key][1]]
        ax.plot(s["delay_k"], 100 * (s["ratio"] - 1), color=color_for(net),
                marker=marker_for(net), label=f"{PAIRS[key][2]} (n={len(s)})")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Throughput change, new vs old (%)")
    ax.set_title("A100-SXM4-80GB only, delay-matched\n(above zero = the new XML is faster)")
    ax.legend(loc="best", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    stamp(fig, "throughput", DATA)
    fig.savefig(FIGURES / "throughput.png")
    print("Saved", FIGURES / "throughput.png")
    return table


# ---------------------------------------------------------------------------
# Figure 6: offline evaluation, where the batch eval covers both arms of a pair
# ---------------------------------------------------------------------------

def fig_offline_eval(ev: pd.DataFrame) -> None:
    if ev.empty:
        print("No offline eval rows; skipping offline_eval.png")
        return
    covered = set(ev["condition"])
    pairs = [k for k, (a, b, _) in PAIRS.items() if {a, b} <= covered]
    if not pairs:
        print("No pair has offline eval on both arms; skipping offline_eval.png")
        return

    datasets = ["train", "old_eval", "new_eval"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.0 * len(datasets), 3.8),
                             sharex=True)
    for ax, dataset in zip(axes, datasets):
        sub = ev[ev["dataset"] == dataset]
        for key in pairs:
            base, changed, label = PAIRS[key]
            a = sub[sub["condition"] == base].groupby("delay_k")["reward_per_step"].max()
            b = sub[sub["condition"] == changed].groupby("delay_k")[
                "reward_per_step"].max()
            m = pd.DataFrame({"a": a, "b": b}).dropna()
            if m.empty:
                continue
            ax.plot(m.index, 100 * (m["b"] / m["a"] - 1),
                    color=FACTOR_COLOR.get(key, "C7"), marker="o", ls="--", label=label)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_title(dataset)
        ax.set_xlabel("Observation delay (steps)")
        sns.despine(ax=ax)
    axes[0].set_ylabel("Change in reward per step vs baseline (%)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Offline evaluation (batch spec eval3ds-66aaff5b). No primary pair "
                 "appears: none of the new reference_root arms has an offline eval yet.",
                 fontsize=9)
    fig.tight_layout()
    stamp(fig, "offline_eval", DATA_EVAL)
    fig.savefig(FIGURES / "offline_eval.png")
    print("Saved", FIGURES / "offline_eval.png")


# ---------------------------------------------------------------------------

def print_tables(df: pd.DataFrame, slopes: pd.DataFrame,
                 throughput: pd.DataFrame) -> None:
    print("\n=== Reward change vs baseline (%) ===")
    columns = {}
    for key in PAIRS:
        m = paired(df, key)
        columns[key] = m.set_index("delay_k")["pct"].round(1)
    print(pd.DataFrame(columns).to_string())

    print("\n=== Reward still gained in the last 100 M steps (% of final) ===")
    print(slopes.pivot_table(index="delay_k", columns="condition",
                             values="gained_pct").round(1).to_string())

    print("\n=== A100-only, delay-matched throughput ratio (new / old) ===")
    summary = (throughput.groupby(["label", "metric"])["ratio"]
               .agg(["median", "min", "max", "size"]).round(4))
    print(summary.to_string())


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    ev = pd.read_csv(DATA_EVAL)

    fig_primary(df)
    fig_decomposition(df)
    fig_position(df)
    slopes = fig_convergence(df, curves)
    throughput = fig_throughput(df)
    fig_offline_eval(ev)
    print_tables(df, slopes, throughput)

    slopes.to_csv(HERE / "convergence_table.csv", index=False)
    throughput.to_csv(HERE / "throughput_table.csv", index=False)
    write_figure_manifest(HERE, _MANIFEST)


if __name__ == "__main__":
    main()
