"""Figures for the recurrent-architecture question. Reads only the CSVs in this folder."""

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

#: The one cell every architecture has in common.
REF = dict(delay_k=10, efference_length=10, rollout_length=60)
ARCH_ORDER = ["feedforward", "rnn", "forward_model", "gru", "lstm"]

#: Compact axis labels -- the full `label_for` names collide on a 5-bar axis.
SHORT = {"feedforward": "Feedforward", "rnn": "Vanilla\nRNN",
         "forward_model": "Forward\nmodel", "gru": "GRU", "lstm": "LSTM"}


def cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per experimental cell, averaging the (rare) cross-epoch replicates."""
    key = ["condition", "delay_k", "efference_length", "rollout_length"]
    return (df.groupby(key, dropna=False)
              .agg(reward=("old_eval_reward", "mean"),
                   params=("total_params", "mean"),
                   n=("wandb_id", "size"))
              .reset_index())


def pick(c: pd.DataFrame, condition, **kw):
    sel = c[c.condition == condition]
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    return None if sel.empty else sel.iloc[0]


def fig_architectures(c: pd.DataFrame) -> Path:
    """Which architecture wins at delay 10 -- and how much of it is parameter count."""
    rows = [(a, pick(c, a, **REF)) for a in ARCH_ORDER]
    rows = [(a, r) for a, r in rows if r is not None]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    vals = [float(r.reward) for _, r in rows]
    ax.bar(range(len(rows)), vals, color=[color_for(a) for a, _ in rows], width=0.65)
    ff = pick(c, "feedforward", **REF)
    ax.axhline(float(ff.reward), color="k", linestyle=":", linewidth=1)
    for i, (a, r) in enumerate(rows):
        pct = 100 * (r.reward - ff.reward) / ff.reward
        ax.text(i, r.reward, f"{r.reward:.0f}\n{pct:+.1f}%", ha="center", va="bottom",
                fontsize=8)
        ax.text(i, r.reward * 0.5, f"{r.params / 1e6:.2f} M", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([SHORT[a] for a, _ in rows], fontsize=9)
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title("delay 10, efference 10, rollout 60\n(white text = parameters)",
                 fontsize=10)

    ax = axes[1]
    for a, r in rows:
        ax.plot(r.params / 1e6, r.reward, marker=marker_for(a), markersize=12,
                color=color_for(a), linestyle="none", label=label_for(a))
        ax.annotate(SHORT[a].replace("\n", " "), (r.params / 1e6, r.reward),
                    textcoords="offset points", xytext=(0, 13), ha="center", fontsize=8,
                    color=color_for(a))
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Held-out episode reward")
    ax.set_title("Reward against parameter count:\nthe vanilla RNN breaks the trend",
                 fontsize=10)
    ax.set_ylim(min(vals) * 0.88, max(vals) * 1.10)
    span = max(r.params for _, r in rows) - min(r.params for _, r in rows)
    ax.set_xlim((min(r.params for _, r in rows) - 0.25 * span) / 1e6,
                (max(r.params for _, r in rows) + 0.25 * span) / 1e6)
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "architectures_delay10.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_delay(c: pd.DataFrame, raw: pd.DataFrame) -> Path:
    """Delay tolerance, and whether a one-step action buffer is enough."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    # Extended feedforward reference: the efference-matched sweep at rollout 20.
    ref = c[(c.condition == "feedforward") & (c.rollout_length == 20)
            & (c.delay_k == c.efference_length)].sort_values("delay_k")
    ax.plot(ref.delay_k, ref.reward, color=color_for("feedforward"), linestyle="--",
            linewidth=1.4, alpha=0.75, marker="o", markersize=3,
            label="Feedforward, efference = delay (rollout 20)")

    series = [
        ("feedforward", 60, "eq", "-", "Feedforward, efference = delay (rollout 60)"),
        ("feedforward", 60, "one", ":", "Feedforward, efference = 1 (rollout 60)"),
        ("lstm", 60, "eq", "-", "LSTM, efference = delay (rollout 60)"),
        ("lstm", 60, "one", ":", "LSTM, efference = 1 (rollout 60)"),
    ]
    for cond, roll, mode, ls, lab in series:
        sel = c[(c.condition == cond) & (c.rollout_length == roll)]
        # The `efference = 1` series keeps delay 0, where both architectures still hold a
        # one-step buffer: it is the common anchor from which the two curves diverge.
        sel = (sel[sel.delay_k == sel.efference_length] if mode == "eq"
               else sel[sel.efference_length == 1])
        sel = sel.sort_values("delay_k")
        if sel.empty:
            continue
        ax.plot(sel.delay_k, sel.reward, ls, marker=marker_for(cond), markersize=8,
                color=color_for(cond), linewidth=2.0,
                markerfacecolor=color_for(cond) if mode == "eq" else "none",
                markeredgewidth=1.8, label=lab)

    ax.set_xlabel("Proprioception delay (control steps)")
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_title("A one-step action buffer suffices for the LSTM but not the feedforward\n"
                 "net \u2014 and past delay ~40 the full buffer becomes a liability",
                 fontsize=10.5)
    ax.set_xlim(-2, 63)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    add_ms_axis(ax, 63)
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "delay_tolerance.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def rollout_pairs(c: pd.DataFrame) -> list:
    """Every (condition, delay, efference) cell measured at both rollout 20 and 60."""
    wide = c.pivot_table(index=["condition", "delay_k", "efference_length"],
                         columns="rollout_length", values="reward")
    if 20 not in wide or 60 not in wide:
        return []
    # Rename before reset_index: integer column labels become positional `_3`/`_4`
    # attribute names on the namedtuples otherwise.
    wide = wide.dropna(subset=[20, 60])[[20, 60]].rename(
        columns={20: "r20", 60: "r60"}).reset_index()
    wide["order"] = wide.condition.map(ARCH_ORDER.index)
    wide = wide.sort_values(["order", "delay_k", "efference_length"])
    return list(wide.itertuples(index=False))


def fig_rollout(c: pd.DataFrame) -> Path:
    """Does a longer BPTT horizon help everyone, or only the recurrent net?"""
    pairs = rollout_pairs(c)
    fig, ax = plt.subplots(figsize=(1.05 * len(pairs) + 3.0, 4.4))
    x = np.arange(len(pairs))
    v20 = [float(p.r20) for p in pairs]
    v60 = [float(p.r60) for p in pairs]
    ax.bar(x - 0.19, v20, width=0.36, color="0.72", label="rollout 20 (0.2 s BPTT)")
    ax.bar(x + 0.19, v60, width=0.36, color=[color_for(p.condition) for p in pairs],
           label="rollout 60 (0.6 s BPTT)")
    for i, (a, b) in enumerate(zip(v20, v60)):
        pct = 100 * (b - a) / a
        ax.text(i + 0.19, b, f"{pct:+.1f}%", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color="0.15" if abs(pct) > 5 else "0.45")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{SHORT[p.condition].replace(chr(10), ' ')}\ndelay {p.delay_k:.0f}, "
         f"eff {p.efference_length:.0f}" for p in pairs], fontsize=8)
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_ylim(0, max(v20 + v60) * 1.20)
    ax.set_title("Longer BPTT helps only the recurrent decoder: both LSTM cells gain\n"
                 "~8-10 %, three of four feedforward cells lose, the forward model is flat",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    provenance(fig, HERE, DATA)
    fig.tight_layout()
    out = FIGURES / "rollout_length.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA)
    c = cells(raw)
    figs = (fig_architectures(c), fig_delay(c, raw), fig_rollout(c))
    write_figure_manifest(HERE, {p.name: "" for p in figs})
    print("wrote", ", ".join(p.name for p in figs))


if __name__ == "__main__":
    main()
