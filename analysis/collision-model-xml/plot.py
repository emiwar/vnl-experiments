"""Plot the new-vs-old walker-XML comparison from data.csv / curves.csv.

Run from the repo root::

    ../.venv/bin/python analysis/collision-model-xml/plot.py

Reads ONLY the committed CSVs (never the WandB API) and writes figures/.
See analysis/README.md §2.

Encoding used throughout: **colour = network** (the canonical CONDITION_STYLE colours,
so these figures line up with every other question), **line style = XML**
(solid + filled marker = new / almost-full collisions, dashed + open marker = old /
sparse collisions).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

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

# Filled in by `stamp`; written to figures/manifest.json at the end so every png can be
# traced to the exact CSV bytes and commit it came from.
_MANIFEST: dict[str, str] = {}


def stamp(fig, name: str, *inputs) -> None:
    """Add the provenance footer and remember it. Call just before ``savefig``."""
    _MANIFEST[f"{name}.png"] = provenance(fig, HERE, *(inputs or (DATA,)))


NETWORK_LABEL = {
    "efference": "Efference EncDec (torque control)",
    "forward_model": "Explicit forward model (position control)",
    "pg_forward_model": "Policy-gradient FM",
}
XML_LABEL = {"old": "old XML (sparse collisions)", "new": "new XML (almost full)"}
XML_STYLE = {"old": dict(ls="--", mfc="none"), "new": dict(ls="-")}


def dedup(df: pd.DataFrame, by, metric: str) -> pd.DataFrame:
    """Keep the highest-``metric`` row per ``by`` group (duplicate delay-0 runs)."""
    return (
        df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .drop_duplicates(by)
    )


def line(ax, sub, network, xml, metric="reward_mean", label=None, **kw):
    sub = sub.sort_values("delay_k")
    style = dict(
        color=color_for(network), marker=marker_for(network),
        label=label if label is not None else f"{XML_LABEL[xml]}",
        **XML_STYLE[xml],
    )
    style.update(kw)
    ax.plot(sub["delay_k"], sub[metric], **style)


# ---------------------------------------------------------------------------
# Figure 1: reward vs delay, old vs new XML, one facet per matched cell
# ---------------------------------------------------------------------------

def fig_reward(df: pd.DataFrame) -> None:
    cells = ["efference", "forward_model"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, net in zip(axes, cells):
        for xml in ("old", "new"):
            cond = f"{xml}_{net}"
            sub = dedup(df[df["condition"] == cond], ["delay_k"], "reward_mean")
            line(ax, sub, net, xml)
        if net == "efference":
            sub = dedup(df[df["condition"] == "old_efference_refroot"],
                        ["delay_k"], "reward_mean")
            line(ax, sub, net, "old", label="old XML, reference_root frame",
                 ls=":", marker="x", mfc=None, alpha=0.7)
        ax.set_title(NETWORK_LABEL[net])
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", fontsize=8)
        add_ms_axis(ax, df["delay_k"].max())
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean episode reward (eval on train clips)")
    fig.tight_layout()
    stamp(fig, "xml_comparison", DATA)
    fig.savefig(FIGURES / "xml_comparison.png")
    print("Saved", FIGURES / "xml_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: the XML difference (new - old) per cell -- is it condition-dependent?
# ---------------------------------------------------------------------------

def paired(df: pd.DataFrame, net: str, metric: str = "reward_mean",
           old_cond: str | None = None) -> pd.DataFrame:
    old = dedup(df[df["condition"] == (old_cond or f"old_{net}")], ["delay_k"], metric)
    new = dedup(df[df["condition"] == f"new_{net}"], ["delay_k"], metric)
    m = old.merge(new, on="delay_k", suffixes=("_old", "_new"))
    m["delta"] = m[f"{metric}_new"] - m[f"{metric}_old"]
    m["ratio"] = m[f"{metric}_new"] / m[f"{metric}_old"]
    return m.sort_values("delay_k")


def fig_difference(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (col, ylabel) in zip(
        axes, [("delta", "Δ reward (new XML − old XML)"),
               ("ratio", "reward ratio (new / old)")]
    ):
        for net in ("efference", "forward_model"):
            m = paired(df, net)
            ax.plot(m["delay_k"], m[col], color=color_for(net), marker=marker_for(net),
                    label=NETWORK_LABEL[net])
        m = paired(df, "efference", old_cond="old_efference_refroot")
        ax.plot(m["delay_k"], m[col], color=color_for("efference"), marker="x", ls=":",
                alpha=0.7, label="Efference vs reference_root baseline")
        ax.axhline(0 if col == "delta" else 1, color="k", lw=0.8, ls=":")
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower left", fontsize=8)
        sns.despine(ax=ax)
    fig.tight_layout()
    stamp(fig, "xml_difference", DATA)
    fig.savefig(FIGURES / "xml_difference.png")
    print("Saved", FIGURES / "xml_difference.png")


# ---------------------------------------------------------------------------
# Figure 3: the doubly-confounded policy-gradient-FM pair, shown separately
# ---------------------------------------------------------------------------

def fig_confounded(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for cond, net, xml, lab in [
        ("old_pg_forward_model", "pg_forward_model", "old", "PG-FM, old XML, torque"),
        ("new_pg_forward_model", "pg_forward_model", "new", "PG-FM, new XML, position"),
        ("old_forward_model", "forward_model", "old", "Explicit FM, old XML, position"),
        ("old_efference", "efference", "old", "Efference, old XML, torque"),
    ]:
        sub = dedup(df[df["condition"] == cond], ["delay_k"], "reward_mean")
        line(ax, sub, net, xml, label=lab)
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Mean episode reward (eval on train clips)")
    ax.set_ylim(bottom=0)
    ax.set_title("Policy-gradient FM pair: the gap is the actuator mode, not the XML")
    ax.legend(loc="lower left", fontsize=8)
    add_ms_axis(ax, df["delay_k"].max())
    sns.despine(ax=ax)
    fig.tight_layout()
    stamp(fig, "confounded_pg_pair", DATA)
    fig.savefig(FIGURES / "confounded_pg_pair.png")
    print("Saved", FIGURES / "confounded_pg_pair.png")


# ---------------------------------------------------------------------------
# Figure 4: convergence
# ---------------------------------------------------------------------------

def steps_to_target(curve: pd.DataFrame, target: float) -> float:
    """First logged step at which reward reaches ``target`` (NaN if never)."""
    hit = curve[curve["reward_mean"] >= target]
    return float(hit["step"].min()) if len(hit) else np.nan


def reward_at(curve: pd.DataFrame, step: float) -> float:
    """Best reward seen up to ``step`` (running max, so a single noisy eval point
    cannot make a run look worse than it already was)."""
    upto = curve[curve["step"] <= step]
    return float(upto["reward_mean"].max()) if len(upto) else np.nan


def fig_convergence(df: pd.DataFrame, curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    # (a,b) raw learning curves for two representative delays, per cell.
    for ax, (net, delay) in zip(axes[:2], [("efference", 20), ("forward_model", 20)]):
        for xml in ("old", "new"):
            cond = f"{xml}_{net}"
            ids = df[(df["condition"] == cond) & (df["delay_k"] == delay)]["wandb_id"]
            for i, rid in enumerate(ids):
                c = curves[curves["wandb_id"] == rid].sort_values("step")
                ax.plot(c["step"] / 1e6, c["reward_mean"], color=color_for(net),
                        label=XML_LABEL[xml] if i == 0 else None, **XML_STYLE[xml])
        ax.set_title(f"{NETWORK_LABEL[net]}\ndelay {delay}")
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel("Mean episode reward")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower right", fontsize=8)
        sns.despine(ax=ax)

    # (c) steps to reach 80/90% of the pair's common final level, new vs old.
    # (d) reward reached at a fixed half-budget (300M steps), new - old.
    HALF_BUDGET = 300e6
    rows = []
    for net in ("efference", "forward_model"):
        m = paired(df, net)
        for _, r in m.iterrows():
            c_old = curves[curves["wandb_id"] == r["wandb_id_old"]]
            c_new = curves[curves["wandb_id"] == r["wandb_id_new"]]
            common = min(r["reward_mean_old"], r["reward_mean_new"])
            row = dict(network=net, delay_k=r["delay_k"],
                       final_old=r["reward_mean_old"], final_new=r["reward_mean_new"],
                       reward_300M_old=reward_at(c_old, HALF_BUDGET),
                       reward_300M_new=reward_at(c_new, HALF_BUDGET))
            for frac in (0.8, 0.9):
                s_old = steps_to_target(c_old, frac * common)
                s_new = steps_to_target(c_new, frac * common)
                row[f"steps_{int(frac*100)}_old"] = s_old
                row[f"steps_{int(frac*100)}_new"] = s_new
                row[f"ratio_{int(frac*100)}"] = s_new / s_old
            row["delta_300M"] = row["reward_300M_new"] - row["reward_300M_old"]
            rows.append(row)
    conv = pd.DataFrame(rows)

    ax = axes[2]
    for net, sub in conv.groupby("network"):
        sub = sub.sort_values("delay_k")
        ax.plot(sub["delay_k"], sub["ratio_90"], color=color_for(net),
                marker=marker_for(net), label=f"{NETWORK_LABEL[net]} (90%)")
        ax.plot(sub["delay_k"], sub["ratio_80"], color=color_for(net),
                marker=marker_for(net), ls=":", alpha=0.6,
                label=f"{NETWORK_LABEL[net]} (80%)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Steps to reach x% of the pair's\ncommon final reward (new / old)")
    ax.set_title("Convergence speed")
    ax.legend(loc="upper left", fontsize=7)
    sns.despine(ax=ax)

    ax = axes[3]
    for net, sub in conv.groupby("network"):
        sub = sub.sort_values("delay_k")
        ax.plot(sub["delay_k"], sub["delta_300M"], color=color_for(net),
                marker=marker_for(net), label=f"{NETWORK_LABEL[net]}, at 300M steps")
        ax.plot(sub["delay_k"], sub["final_new"] - sub["final_old"], color=color_for(net),
                marker=marker_for(net), ls="--", alpha=0.6,
                label=f"{NETWORK_LABEL[net]}, at 600M steps (final)")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Δ reward (new XML − old XML)")
    ax.set_title("Half-budget vs full-budget gap")
    ax.legend(loc="lower left", fontsize=7)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "convergence", DATA, CURVES)
    fig.savefig(FIGURES / "convergence.png")
    print("Saved", FIGURES / "convergence.png")
    conv.to_csv(FIGURES.parent / "convergence_table.csv", index=False)
    print(conv.to_string(index=False))


# ---------------------------------------------------------------------------
# Figure 5: throughput (same GPU only)
# ---------------------------------------------------------------------------

def fig_throughput(df: pd.DataFrame) -> None:
    a100 = df[df["gpu"].astype(str).str.contains("A100")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in [
        (axes[0], "train_sps_median", "Training throughput (rollout + update)"),
        (axes[1], "eval_sps_median", "Rollout-only throughput (physics + inference)"),
    ]:
        for net in ("efference", "forward_model"):
            for xml in ("old", "new"):
                sub = dedup(a100[a100["condition"] == f"{xml}_{net}"],
                            ["delay_k"], metric)
                line(ax, sub, net, xml, metric=metric,
                     label=f"{NETWORK_LABEL[net].split(' (')[0]}, {XML_LABEL[xml]}")
        sub = dedup(a100[a100["condition"] == "old_efference_refroot"],
                    ["delay_k"], metric)
        line(ax, sub, "efference", "old", metric=metric, ls=":", marker="x", alpha=0.7,
             label="Efference, old XML (reference_root baseline)")
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylabel("Steps per second")
        ax.set_title(title + "\n(A100-SXM4-80GB runs only)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", fontsize=7)
        sns.despine(ax=ax)
    fig.tight_layout()
    stamp(fig, "throughput", DATA)
    fig.savefig(FIGURES / "throughput.png")
    print("Saved", FIGURES / "throughput.png")

    print("\nThroughput ratio new/old at matched delay (A100 only):")
    for metric in ("train_sps_median", "eval_sps_median"):
        for net in ("efference", "forward_model"):
            m = paired(a100, net, metric=metric)
            if len(m):
                print(f"  {metric:18s} {net:14s} "
                      f"delays={list(m['delay_k'])} "
                      f"ratio={[round(x, 3) for x in m['ratio']]} "
                      f"median={np.median(m['ratio']):.3f}")
        m = paired(a100, "efference", metric=metric, old_cond="old_efference_refroot")
        print(f"  {metric:18s} {'eff vs refroot':14s} "
              f"ratio={[round(x, 3) for x in m['ratio']]} "
              f"median={np.median(m['ratio']):.3f}")


# ---------------------------------------------------------------------------

def print_tables(df: pd.DataFrame) -> None:
    print("\nFinal reward, new vs old XML (matched delays):")
    for net in ("efference", "forward_model"):
        m = paired(df, net)
        print(f"\n  {NETWORK_LABEL[net]}")
        print(m[["delay_k", "reward_mean_old", "reward_mean_new", "delta", "ratio"]]
              .round(3).to_string(index=False))
    m = paired(df, "efference", old_cond="old_efference_refroot")
    print("\n  Efference vs the reference_root old-XML baseline")
    print(m[["delay_k", "reward_mean_old", "reward_mean_new", "delta", "ratio"]]
          .round(3).to_string(index=False))


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)

    fig_reward(df)
    fig_difference(df)
    fig_confounded(df)
    fig_convergence(df, curves)
    fig_throughput(df)
    print_tables(df)
    write_figure_manifest(HERE, _MANIFEST)


if __name__ == "__main__":
    main()
