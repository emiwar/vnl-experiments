"""Figures for the new-XML + reference_root comparison.

Run from the repo root::

    ../.venv/bin/python analysis/collision-model-xml/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/collision-model-xml/plot.py   # for slides

Reads ONLY the committed CSVs (never WandB, never the artifact store) and writes
figures/. See analysis/README.md §3.

The **offline evaluation** (``data_eval.csv``) carries the conclusions here, on the post-fix
spec ``eval3ds-347333e3``; the training-time reward in ``data.csv`` is a second, independent
measurement rather than a discredited one. ``fig_training_vs_heldout`` shows the two agreeing
to a few percent. Before the 2026-08-18 walker-XML fix they diverged by up to 2.3x and this
module's framing was built on that divergence -- see the retraction in ``report.md``.

Encoding: **colour = network** (the canonical CONDITION_STYLE colours), **line style =
arm** — dashed + open marker for the baseline, solid + filled for the changed config.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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

#: Held-out split, 169 clips, full 502-step rollouts from frame 0.
HELD_OUT = "old_eval"

_MANIFEST: dict[str, str] = {}


def stamp(fig, name: str, *inputs) -> None:
    _MANIFEST[f"{name}.png"] = provenance(fig, HERE, *(inputs or (DATA,)))


NETWORK_OF = {
    "encdec_old_current": "efference", "encdec_old_reference": "efference",
    "encdec_new_current": "efference", "encdec_new_reference": "efference",
    "expfm_old_current": "forward_model", "expfm_old_reference": "forward_model",
    "expfm_new_reference": "forward_model",
    "expfm_old_position": "forward_model", "expfm_new_position": "forward_model",
    "pgfm_old_current": "pg_forward_model", "pgfm_new_reference": "pg_forward_model",
}
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
FACTOR_COLOR = {"xml_at_current": "C1", "xml_at_reference": "C3",
                "frame_at_old": "C0", "frame_at_new": "C9",
                "frame_expfm": "C2", "position_xml": "C5"}


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------

def series(df: pd.DataFrame, condition: str, metric: str,
           dataset: str | None = None) -> pd.Series:
    """One value per delay for a condition, from data.csv or data_eval.csv."""
    sub = df[df["condition"] == condition]
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
    sub = sub.dropna(subset=[metric])
    return (sub.sort_values(metric, ascending=False)
               .drop_duplicates(["delay_k"]).set_index("delay_k")[metric]
               .sort_index())


def paired(df: pd.DataFrame, pair: str, metric: str,
           dataset: str | None = None) -> pd.DataFrame:
    """Delay-matched baseline/changed table with absolute and relative differences."""
    base, changed, _ = PAIRS[pair]
    out = pd.DataFrame({"baseline": series(df, base, metric, dataset),
                        "changed": series(df, changed, metric, dataset)}).dropna()
    out["delta"] = out["changed"] - out["baseline"]
    out["pct"] = 100 * (out["changed"] / out["baseline"] - 1)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Figure 1 (primary): held-out performance, and what drives it
# ---------------------------------------------------------------------------

def fig_primary(ev: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ax = axes[0]
    for key in PRIMARY_PAIRS:
        base, changed, label = PAIRS[key]
        for cond, baseline in [(base, True), (changed, False)]:
            s = series(ev, cond, "survived", HELD_OUT)
            ax.plot(s.index, 100 * s, color=color_for(NETWORK_OF[cond]),
                    marker=marker_for(NETWORK_OF[cond]),
                    label=f"{label}: {'old' if baseline else 'new'}",
                    **(BASELINE_STYLE if baseline else CHANGED_STYLE))
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Clips surviving the full 5 s (%)")
    ax.set_title("Survival")
    ax.legend(loc="upper right", fontsize=6.5)
    add_ms_axis(ax, ev["delay_k"].max())
    sns.despine(ax=ax)

    for ax, metric, title, ylabel in [
        (axes[1], "episode_reward", "Episode reward",
         "Reward change, new vs old (%)"),
        (axes[2], "reward_per_step", "Reward per step (tracking quality)",
         "Change in reward per step (%)"),
    ]:
        for key in PRIMARY_PAIRS:
            m = paired(ev, key, metric, HELD_OUT)
            net = NETWORK_OF[PAIRS[key][1]]
            ax.plot(m["delay_k"], m["pct"], color=color_for(net),
                    marker=marker_for(net), label=PAIRS[key][2])
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        sns.despine(ax=ax)
    # Symmetric, and wide enough for every plotted point. The hard -60..15 window these
    # panels used to carry was sized for the pre-fix numbers and silently clipped the
    # positive tail off the top once the walker-XML fix landed.
    span = max(abs(paired(ev, key, m, HELD_OUT)["pct"]).max()
               for key in PRIMARY_PAIRS
               for m in ("episode_reward", "reward_per_step"))
    for ax in (axes[1], axes[2]):
        ax.set_ylim(-1.1 * span, 1.1 * span)
    axes[1].legend(loc="lower left", fontsize=7)

    fig.suptitle(f"Offline evaluation on held-out clips ({HELD_OUT}, 169 clips, "
                 f"full 502-step rollouts)", fontsize=10)
    fig.tight_layout()
    stamp(fig, "primary", DATA_EVAL)
    fig.savefig(FIGURES / "primary.png")
    print("Saved", FIGURES / "primary.png")


# ---------------------------------------------------------------------------
# Figure 2: why the training-time metric missed it
# ---------------------------------------------------------------------------

def fig_training_vs_heldout(df: pd.DataFrame, ev: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    for key in PRIMARY_PAIRS:
        net = NETWORK_OF[PAIRS[key][1]]
        train = paired(df, key, "reward_mean")
        held = paired(ev, key, "episode_reward", HELD_OUT)
        ax.plot(train["delay_k"], train["pct"], color=color_for(net), ls="--",
                marker=marker_for(net), mfc="none",
                label=f"{PAIRS[key][2]}: training-time")
        ax.plot(held["delay_k"], held["pct"], color=color_for(net), ls="-",
                marker=marker_for(net), label=f"{PAIRS[key][2]}: held-out")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Reward change, new vs old (%)")
    ax.set_title("Same runs, two metrics")
    ax.legend(loc="upper left", fontsize=6, frameon=False)
    sns.despine(ax=ax)

    # The asymmetry is the point, and the walker-XML fix flipped which side carries it:
    # the two lifespan measurements now track each other on the *new* body and diverge on
    # the *old* one from delay ~20 up. Pre-fix this panel read the other way round and was
    # the evidence for "training-time reward is biased in favour of the new body", which is
    # retracted -- the divergence was the offline eval driving the wrong body.
    ax = axes[1]
    base, changed, _ = PAIRS["primary_encdec"]
    for cond, baseline in [(base, True), (changed, False)]:
        train = series(df, cond, "lifespan_mean")
        held = series(ev, cond, "lifespan_steps", HELD_OUT)
        m = pd.DataFrame({"t": train, "h": held}).dropna()
        ax.plot(m.index, m["t"], color="0.45" if baseline else "C3", ls="--",
                marker="o", ms=3, mfc="none",
                label=f"{LABEL[cond]}: training-time")
        ax.plot(m.index, m["h"], color="0.45" if baseline else "C3", ls="-",
                marker="o", ms=3, label=f"{LABEL[cond]}: held-out")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode lifespan (control steps)")
    ax.set_title("EncDec lifespan: the two agree for the new body,\n"
                 "diverge for the old one")
    ax.legend(loc="upper right", fontsize=6.5)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "training_vs_heldout", DATA, DATA_EVAL)
    fig.savefig(FIGURES / "training_vs_heldout.png")
    print("Saved", FIGURES / "training_vs_heldout.png")


# ---------------------------------------------------------------------------
# Figure 3: the EncDec 2x2 on held-out data
# ---------------------------------------------------------------------------

def fig_decomposition(ev: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)

    for ax, metric, ylabel, scale in [
        (axes[0], "episode_reward", "Reward change vs the other level (%)", None),
        (axes[1], "survived", "Change in survival (percentage points)", 100),
    ]:
        for key in ("xml_at_current", "xml_at_reference", "frame_at_old",
                    "frame_at_new"):
            m = paired(ev, key, metric, HELD_OUT)
            y = m["delta"] * scale if scale else m["pct"]
            ax.plot(m["delay_k"], y, color=FACTOR_COLOR[key], marker="o",
                    ls="-" if key.startswith("xml") else "--", label=PAIRS[key][2])
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
    axes[0].legend(loc="lower left", fontsize=8)

    fig.suptitle("EncDec 2x2 on held-out clips: both factors are close to free",
                 fontsize=10)
    fig.tight_layout()
    stamp(fig, "decomposition", DATA_EVAL)
    fig.savefig(FIGURES / "decomposition.png")
    print("Saved", FIGURES / "decomposition.png")


# ---------------------------------------------------------------------------
# Figure 4: position control
# ---------------------------------------------------------------------------

def fig_position(ev: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    for cond, baseline in [("expfm_old_position", True), ("expfm_new_position", False)]:
        s = series(ev, cond, "survived", HELD_OUT)
        ax.plot(s.index, 100 * s, color=color_for("forward_model"),
                marker=marker_for("forward_model"), label=LABEL[cond],
                **(BASELINE_STYLE if baseline else CHANGED_STYLE))
    for cond, baseline in [("expfm_old_current", True), ("expfm_new_reference", False)]:
        s = series(ev, cond, "survived", HELD_OUT)
        ax.plot(s.index, 100 * s, color="0.55", lw=1,
                marker=".", ms=4, label=LABEL[cond] + " [torque]",
                ls="--" if baseline else "-")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Clips surviving the full 5 s (%)")
    ax.set_title("Position control (green) vs torque (grey)")
    ax.legend(loc="upper right", fontsize=6.5)
    sns.despine(ax=ax)

    ax = axes[1]
    for key in ("position_xml", "xml_at_current", "xml_at_reference"):
        m = paired(ev, key, "episode_reward", HELD_OUT)
        ax.plot(m["delay_k"], m["pct"], color=FACTOR_COLOR[key], marker="o",
                ls="-" if key == "position_xml" else "--",
                label=PAIRS[key][2] + ("" if key == "position_xml" else " (torque)"))
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Reward change, new vs old XML (%)")
    ax.set_title("Position control: within a few percent at every delay")
    ax.legend(loc="lower left", fontsize=8)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "position", DATA_EVAL)
    fig.savefig(FIGURES / "position.png")
    print("Saved", FIGURES / "position.png")


# ---------------------------------------------------------------------------
# Figure 5: convergence (training curves — the only thing they are good for here)
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


def fig_convergence(curves: pd.DataFrame) -> pd.DataFrame:
    base, changed, _ = PAIRS["primary_encdec"]
    show = [30, 40, 50, 60]

    fig, axes = plt.subplots(1, len(show) + 1, figsize=(3.1 * (len(show) + 1), 3.7))
    for ax, delay in zip(axes, show):
        for cond, baseline in [(base, True), (changed, False)]:
            c = curves[(curves["condition"] == cond)
                       & (curves["delay_k"] == delay)].sort_values("step")
            ax.plot(c["step"] / 1e6, c["reward_mean"], color=color_for(NETWORK_OF[cond]),
                    lw=1.2, ls="--" if baseline else "-",
                    label=LABEL[cond] if delay == show[0] else None)
        ax.set_title(f"EncDec, delay {delay}")
        ax.set_xlabel("Env steps (millions)")
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Training-time reward")
    axes[0].legend(loc="lower right", fontsize=7)

    ax = axes[-1]
    slopes = convergence_table(curves)
    for cond, baseline in [(base, True), (changed, False)]:
        s = slopes[slopes["condition"] == cond].sort_values("delay_k")
        ax.plot(s["delay_k"], s["gained_pct"], color=color_for(NETWORK_OF[cond]),
                marker=marker_for(NETWORK_OF[cond]), ms=3, lw=1.2, label=LABEL[cond],
                **(BASELINE_STYLE if baseline else CHANGED_STYLE))
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Reward still gained in the\nlast 100 M steps (% of final)")
    ax.set_title("Converged at 600 M? (EncDec)")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    sns.despine(ax=ax)

    fig.suptitle("Training-time curves — from history artifacts, so unaffected by the "
                 "walker-XML bug; read for convergence", fontsize=9)
    fig.tight_layout()
    stamp(fig, "convergence", CURVES)
    fig.savefig(FIGURES / "convergence.png")
    print("Saved", FIGURES / "convergence.png")
    return slopes


# ---------------------------------------------------------------------------
# Figure 6: throughput
# ---------------------------------------------------------------------------

def throughput_table(df: pd.DataFrame) -> pd.DataFrame:
    a100 = df[df["gpu"].astype(str).str.contains("A100")]
    rows = []
    for key in PRIMARY_PAIRS:
        base, changed, label = PAIRS[key]
        for metric in ("train_sps_median", "eval_sps_median"):
            m = pd.DataFrame({"old": series(a100, base, metric),
                              "new": series(a100, changed, metric)}).dropna()
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
    ax.set_title("A100-SXM4-80GB only, delay-matched\n"
                 "(above zero = the new XML is faster)")
    ax.legend(loc="best", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    stamp(fig, "throughput", DATA)
    fig.savefig(FIGURES / "throughput.png")
    print("Saved", FIGURES / "throughput.png")
    return table


# ---------------------------------------------------------------------------

def print_tables(df: pd.DataFrame, ev: pd.DataFrame, slopes: pd.DataFrame,
                 throughput: pd.DataFrame) -> None:
    for metric, label in [("episode_reward", "reward (%)"),
                          ("reward_per_step", "reward per step (%)"),
                          ("survived", "survival (pp)")]:
        print(f"\n=== Held-out ({HELD_OUT}): change in {label} ===")
        cols = {}
        for key in PAIRS:
            m = paired(ev, key, metric, HELD_OUT)
            if m.empty:
                continue
            cols[key] = ((m["delta"] * 100) if metric == "survived"
                         else m["pct"]).round(1)
            cols[key].index = m["delay_k"]
        print(pd.DataFrame(cols).to_string())

    print("\n=== Training-time reward change (%) — for comparison ===")
    cols = {}
    for key in PRIMARY_PAIRS:
        m = paired(df, key, "reward_mean")
        cols[key] = m.set_index("delay_k")["pct"].round(1)
    print(pd.DataFrame(cols).to_string())

    print("\n=== A100-only, delay-matched throughput ratio (new / old) ===")
    print(throughput.groupby(["label", "metric"])["ratio"]
          .agg(["median", "min", "max", "size"]).round(4).to_string())


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    ev = pd.read_csv(DATA_EVAL)

    fig_primary(ev)
    fig_training_vs_heldout(df, ev)
    fig_decomposition(ev)
    fig_position(ev)
    slopes = fig_convergence(curves)
    throughput = fig_throughput(df)
    print_tables(df, ev, slopes, throughput)

    slopes.to_csv(HERE / "convergence_table.csv", index=False)
    throughput.to_csv(HERE / "throughput_table.csv", index=False)
    write_figure_manifest(HERE, _MANIFEST)


if __name__ == "__main__":
    main()
