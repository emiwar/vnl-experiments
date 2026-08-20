"""Figures for "explicit vs implicit forward model across 600 M / 2 G / 4 G".

Run from the repo root::

    ../.venv/bin/python analysis/explicit-vs-implicit-fm-budgets/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/explicit-vs-implicit-fm-budgets/plot.py

Reads ONLY the committed CSVs (never WandB, never the artifact store). See
analysis/README.md §3.

Encoding, consistent across figures: **colour = arm** where the two arms are compared
(the canonical `CONDITION_STYLE` colours -- green explicit, purple implicit), **colour =
budget** (viridis, dark to light) where budgets are compared, **solid = seed 42, dashed
with open markers = seed 43**. The ±2.9 % run-to-run noise floor from
`xml-ceiling-vs-convergence/` is drawn wherever a difference is plotted.
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
    provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

_MANIFEST: dict[str, str] = {}


def stamp(fig, name: str, *inputs) -> None:
    _MANIFEST[f"{name}.png"] = provenance(fig, HERE, *(inputs or (DATA,)))


NOISE_FLOOR_PCT = 2.9
WINDOW_POINTS = 5  # eval every 10 M steps -> a 50 M-step trailing window

ARM_COLOR = {"explicit": color_for("forward_model"),
             "implicit": color_for("pg_forward_model")}
ARM_LABEL = {"explicit": "Explicit FM (L2 loss, detached)",
             "implicit": "Implicit FM (policy gradient only)"}

#: ``(tier_steps, explicit condition, implicit condition, seed)``. Each entry is a
#: budget at which both arms can be read off the same set of runs. The seed-43 entries
#: all come from the *same four runs per arm*, read at four points along their curves --
#: which is why the 600 M-to-4 G trend within seed 43 is a within-run trend.
TIERS = [
    (600_000_000, "expfm_600m", "pgfm_600m", 42),
    (2_000_000_000, "expfm_2g", "pgfm_2g", 42),
    (600_000_000, "expfm_4g", "pgfm_4g", 43),
    (2_000_000_000, "expfm_4g", "pgfm_4g", 43),
    (2_900_000_000, "expfm_4g", "pgfm_4g", 43),
    (4_000_000_000, "expfm_4g", "pgfm_4g", 43),
]

BUDGETS = (600_000_000, 1_000_000_000, 2_000_000_000, 2_900_000_000, 4_000_000_000)


def tier_column(tier: int, metric: str = "reward") -> str:
    label = (f"{tier // 1_000_000}M" if tier < 1_000_000_000
             else f"{tier / 1e9:g}G".replace(".", "p"))
    return f"{metric}_at_{label}"


def tier_label(tier: int) -> str:
    return f"{tier // 1_000_000} M" if tier < 1_000_000_000 else f"{tier / 1e9:g} G"


def budget_color(tier: int) -> tuple:
    """Dark = short budget, light = long. Position on log(steps) between 600 M and 4 G."""
    lo, hi = np.log(600e6), np.log(4e9)
    frac = (np.log(tier) - lo) / (hi - lo)
    return plt.get_cmap("viridis")(0.08 + 0.80 * frac)


# --------------------------------------------------------------------------------------
# reductions
# --------------------------------------------------------------------------------------


def sweep(df: pd.DataFrame, condition: str, tier: int,
          metric: str = "reward") -> pd.DataFrame:
    """Delay -> mean value over the runs of ``condition`` that reached ``tier``.

    Duplicated delays (the 600 M implicit sweep has two runs at delays 10 and 15; the
    seed-43 explicit arm has two at delay 30, one of which crashed at 1.6 G) are averaged
    over whichever runs got that far, and ``n`` records how many that was.
    """
    column = tier_column(tier, metric)
    sub = df[(df["condition"] == condition)].dropna(subset=[column])
    if sub.empty:
        return pd.DataFrame(columns=["delay_k", "value", "n"])
    out = (sub.groupby("delay_k")[column].agg(["mean", "size"])
              .rename(columns={"mean": "value", "size": "n"}).reset_index())
    return out.sort_values("delay_k")


def advantage(df: pd.DataFrame, explicit: str, implicit: str, tier: int,
              metric: str = "reward") -> pd.DataFrame:
    """Explicit-over-implicit difference, in per cent, at the delays both arms reached."""
    a = sweep(df, explicit, tier, metric).set_index("delay_k")
    b = sweep(df, implicit, tier, metric).set_index("delay_k")
    joined = a.join(b, how="inner", lsuffix="_e", rsuffix="_i")
    if joined.empty:
        return pd.DataFrame(columns=["delay_k", "advantage_pct", "n_e", "n_i"])
    joined["advantage_pct"] = 100 * (joined["value_e"] / joined["value_i"] - 1)
    return joined.reset_index()[["delay_k", "advantage_pct", "n_e", "n_i"]]


def unpaired(df: pd.DataFrame, explicit: str, implicit: str, tier: int) -> pd.DataFrame:
    """Delays where only one arm reached ``tier`` -- examples, never differenced."""
    a = set(sweep(df, explicit, tier)["delay_k"])
    b = set(sweep(df, implicit, tier)["delay_k"])
    rows = [{"delay_k": d, "arm": "explicit"} for d in sorted(a - b)]
    rows += [{"delay_k": d, "arm": "implicit"} for d in sorted(b - a)]
    return pd.DataFrame(rows)


def tier_table(df: pd.DataFrame) -> pd.DataFrame:
    """Every (tier, seed, delay) cell: both arms' reward and the difference."""
    rows = []
    for tier, explicit, implicit, seed in TIERS:
        e = sweep(df, explicit, tier).set_index("delay_k")
        i = sweep(df, implicit, tier).set_index("delay_k")
        for delay in sorted(set(e.index) | set(i.index)):
            ev = e["value"].get(delay, np.nan)
            iv = i["value"].get(delay, np.nan)
            rows.append({
                "tier": tier, "seed": seed, "delay_k": delay,
                "explicit": ev, "implicit": iv,
                "n_explicit": int(e["n"].get(delay, 0)),
                "n_implicit": int(i["n"].get(delay, 0)),
                "advantage_pct": (100 * (ev / iv - 1)
                                  if np.isfinite(ev) and np.isfinite(iv) else np.nan)})
    return pd.DataFrame(rows)


def crossover(table: pd.DataFrame, tier: int, seed: int,
              threshold: float) -> float:
    """Delay at which the explicit advantage first exceeds ``threshold``, interpolated.

    ``NaN`` when the lowest sampled delay is already above the threshold (the seed-43
    tiers start at delay 20, so their crossing is only bounded from above) or when the
    advantage never gets there.
    """
    sub = (table[(table["tier"] == tier) & (table["seed"] == seed)]
           .dropna(subset=["advantage_pct"]).sort_values("delay_k"))
    if sub.empty or sub["advantage_pct"].iat[0] >= threshold:
        return np.nan
    d = sub["delay_k"].to_numpy(dtype=float)
    v = sub["advantage_pct"].to_numpy()
    for i in range(len(v) - 1):
        if v[i] < threshold <= v[i + 1]:
            span = v[i + 1] - v[i]
            frac = 0.0 if span == 0 else (threshold - v[i]) / span
            return float(d[i] + frac * (d[i + 1] - d[i]))
    return np.nan


def curve(curves: pd.DataFrame, wandb_id: str) -> pd.DataFrame:
    sub = curves[curves["wandb_id"] == wandb_id].sort_values("step").copy()
    for column in ("reward_mean", "fm_mse_eval", "action_sigma", "encoder_kl",
                   "joint_l2_error", "lifespan_mean"):
        sub[column] = sub[column].rolling(WINDOW_POINTS, min_periods=1).mean()
    return sub


# --------------------------------------------------------------------------------------
# Figure 1 (headline): delay-vs-reward at every budget, and the contrast
# --------------------------------------------------------------------------------------


def fig_delay_reward(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))

    for ax, arm in ((axes[0], "explicit"), (axes[1], "implicit")):
        for tier, explicit, implicit, seed in TIERS:
            condition = explicit if arm == "explicit" else implicit
            s = sweep(df, condition, tier)
            if s.empty:
                continue
            ax.plot(s["delay_k"], s["value"], marker="o", ms=4,
                    color=budget_color(tier), ls="-" if seed == 42 else "--",
                    mfc=None if seed == 42 else "none",
                    label=f"{tier_label(tier)}, seed {seed}")
        ax.set_title(ARM_LABEL[arm], color=ARM_COLOR[arm])
        ax.set_xlabel("Observation delay (steps)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
        add_ms_axis(ax, 100)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean episode reward (eval on train clips)")
    axes[1].sharey(axes[0])

    ax = axes[2]
    ax.axhspan(-NOISE_FLOOR_PCT, NOISE_FLOOR_PCT, color="0.88", zorder=0, lw=0)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    for tier, explicit, implicit, seed in TIERS:
        m = advantage(df, explicit, implicit, tier)
        if m.empty:
            continue
        ax.plot(m["delay_k"], m["advantage_pct"], marker="o", ms=4,
                color=budget_color(tier), ls="-" if seed == 42 else "--",
                mfc=None if seed == 42 else "none",
                label=f"{tier_label(tier)}, seed {seed}")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([-10, 0, 10, 25, 50, 100, 200],
                  ["−10", "0", "10", "25", "50", "100", "200"])
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Explicit FM advantage over implicit (%)")
    ax.set_title("The contrast, at each budget")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    sns.despine(ax=ax)

    fig.suptitle("Delay-vs-reward at three training budgets. Dashed/open = seed 43 "
                 "(the 4 G runs, read at four points along their own curves).",
                 fontsize=9.5)
    fig.tight_layout()
    stamp(fig, "delay_reward", DATA)
    fig.savefig(FIGURES / "delay_reward.png")
    print("Saved", FIGURES / "delay_reward.png")


# --------------------------------------------------------------------------------------
# Figure 2: training-curve examples
# --------------------------------------------------------------------------------------

#: One example run per (row, delay, arm). Row 1 is the seed-42 pairs that have both a
#: 600 M and a 2 G run of the same configuration; row 2 is the seed-43 long runs.
EXAMPLE_ROWS = [
    ("seed 42", [("expfm_600m", "pgfm_600m"), ("expfm_2g", "pgfm_2g")],
     (0, 10, 20, 50)),
    ("seed 43", [("expfm_4g", "pgfm_4g")], (20, 30, 40, 50)),
]


def fig_training_curves(df: pd.DataFrame, curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.2), sharey=True)
    for row, (seed_label, condition_pairs, delays) in enumerate(EXAMPLE_ROWS):
        for col, delay in enumerate(delays):
            ax = axes[row, col]
            notes = []
            for pair in condition_pairs:
                for condition, arm in zip(pair, ("explicit", "implicit")):
                    sub = df[(df["condition"] == condition) & (df["delay_k"] == delay)]
                    sub = sub.dropna(subset=["max_step"])
                    for _, run in sub.iterrows():
                        c = curve(curves, run["wandb_id"])
                        if c.empty:
                            continue
                        long_run = run["budget"] > 600_000_000
                        ax.plot(c["step"] / 1e6, c["reward_mean"],
                                color=ARM_COLOR[arm], lw=1.7 if long_run else 1.0,
                                alpha=1.0 if long_run else 0.55,
                                label=(ARM_LABEL[arm]
                                       if (row, col, long_run) == (0, 0, True) else None))
                        # Annotate only runs that stopped short of their budget. The
                        # 600 M explicit cohort is all `state = failed` because it died
                        # in the *post-training* eval, which is not a truncation.
                        if run["max_step"] < 0.98 * run["budget"]:
                            notes.append(f"{arm[:3]} {run['state']} "
                                         f"@{run['max_step'] / 1e9:.1f} G")
            title = f"delay {delay}"
            if notes:
                title += "\n" + ", ".join(sorted(set(notes)))
            ax.set_title(title, fontsize=8.5)
            ax.set_ylim(bottom=0)
            if row == 1:
                ax.set_xlabel("Env steps (millions)")
            sns.despine(ax=ax)
        axes[row, 0].set_ylabel(f"{seed_label}\nmean episode reward", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    axes[0, 0].text(0.97, 0.28, "thin = 600 M run\nthick = long run", fontsize=6.5,
                    color="0.45", ha="right", transform=axes[0, 0].transAxes)
    fig.suptitle("Training-curve examples. Row 1: the seed-42 600 M and 2 G runs of the "
                 "same configuration overlaid. Row 2: the seed-43 4 G runs.", fontsize=9.5)
    fig.tight_layout()
    stamp(fig, "training_curves", CURVES)
    fig.savefig(FIGURES / "training_curves.png")
    print("Saved", FIGURES / "training_curves.png")


# --------------------------------------------------------------------------------------
# Figure 3: what the budget buys, and how big the seed effect is
# --------------------------------------------------------------------------------------


def seed_check(df: pd.DataFrame) -> pd.DataFrame:
    """Seed 42 vs seed 43 at matched arm, delay and budget.

    The only place the two seeds overlap is delays 20 and 50 at 600 M and 2 G, because
    that is where the seed-42 tiers and the seed-43 runs' own trajectories cross.
    """
    rows = []
    pairs = [("explicit", {42: ["expfm_600m", "expfm_2g"], 43: ["expfm_4g"]}),
             ("implicit", {42: ["pgfm_600m", "pgfm_2g"], 43: ["pgfm_4g"]})]
    for arm, by_seed in pairs:
        for tier in (600_000_000, 2_000_000_000):
            for delay in (20, 50):
                values = {}
                for seed, conditions in by_seed.items():
                    frames = [sweep(df, c, tier) for c in conditions]
                    frames = [f[f["delay_k"] == delay] for f in frames if not f.empty]
                    frames = [f for f in frames if not f.empty]
                    if frames:
                        values[seed] = float(np.mean([f["value"].iat[0]
                                                      for f in frames]))
                if len(values) == 2:
                    rows.append({"arm": arm, "tier": tier, "delay_k": delay,
                                 "seed42": values[42], "seed43": values[43],
                                 "diff_pct": 100 * (values[43] / values[42] - 1)})
    return pd.DataFrame(rows)


def fig_budget(df: pd.DataFrame, seeds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4))

    # (a) reward vs budget for the delays that have a long run, per arm
    ax = axes[0]
    for arm, condition in (("explicit", "expfm_4g"), ("implicit", "pgfm_4g")):
        for delay in (20, 30, 40, 50):
            xs, ys = [], []
            for tier in BUDGETS:
                s = sweep(df, condition, tier)
                s = s[s["delay_k"] == delay]
                if not s.empty:
                    xs.append(tier / 1e6)
                    ys.append(s["value"].iat[0])
            if xs:
                ax.plot(xs, ys, marker="o", ms=3.5, color=ARM_COLOR[arm],
                        alpha=0.35 + 0.65 * ((delay - 20) / 30), lw=1.4)
                ax.annotate(f"d{delay}", (xs[-1] * 1.04, ys[-1]), fontsize=6.5,
                            va="center", color=ARM_COLOR[arm])
    ax.set_xscale("log")
    ax.set_xticks([b / 1e6 for b in BUDGETS], [tier_label(b) for b in BUDGETS])
    ax.minorticks_off()
    ax.set_xlim(500, 6200)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Training budget")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Seed 43: reward along the budget axis\n(green explicit, purple implicit)",
                 fontsize=9.5)
    sns.despine(ax=ax)

    # (b) the advantage against budget, one line per delay
    ax = axes[1]
    ax.axhspan(-NOISE_FLOOR_PCT, NOISE_FLOOR_PCT, color="0.88", zorder=0, lw=0)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    cmap = plt.get_cmap("plasma")
    for delay in (0, 10, 20, 30, 40, 50):
        for seed, entries in ((42, [t for t in TIERS if t[3] == 42]),
                              (43, [t for t in TIERS if t[3] == 43])):
            xs, ys = [], []
            for tier, explicit, implicit, _ in entries:
                m = advantage(df, explicit, implicit, tier)
                m = m[m["delay_k"] == delay]
                if not m.empty:
                    xs.append(tier / 1e6)
                    ys.append(m["advantage_pct"].iat[0])
            if len(xs) >= 2:
                ax.plot(xs, ys, marker="o", ms=3.5, color=cmap(delay / 60),
                        ls="-" if seed == 42 else "--",
                        mfc=None if seed == 42 else "none",
                        label=f"delay {delay}, seed {seed}")
    ax.set_xscale("log")
    ax.set_xticks([b / 1e6 for b in BUDGETS], [tier_label(b) for b in BUDGETS])
    ax.minorticks_off()
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([0, 10, 25, 50, 100, 200], ["0", "10", "25", "50", "100", "200"])
    ax.set_xlabel("Training budget")
    ax.set_ylabel("Explicit FM advantage (%)")
    ax.set_title("Does the advantage close with budget?", fontsize=9.5)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
    sns.despine(ax=ax)

    # (c) how big is the seed effect we are reading across?
    ax = axes[2]
    labels, values, colors = [], [], []
    for _, row in seeds.iterrows():
        labels.append(f"{row['arm'][:3]} d{int(row['delay_k'])} @{tier_label(row['tier'])}")
        values.append(row["diff_pct"])
        colors.append(ARM_COLOR[row["arm"]])
    ax.barh(range(len(labels)), values, color=colors)
    ax.axvspan(-NOISE_FLOOR_PCT, NOISE_FLOOR_PCT, color="0.88", zorder=0, lw=0)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Seed 43 vs seed 42, matched delay and budget (%)")
    worst = max(abs(v) for v in values) if values else float("nan")
    ax.set_title(f"Seed effect: worst |Δ| = {worst:.1f} %\n"
                 f"(grey = the ±{NOISE_FLOOR_PCT} % within-seed noise floor)",
                 fontsize=9.5)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "budget", DATA)
    fig.savefig(FIGURES / "budget.png")
    print("Saved", FIGURES / "budget.png")


# --------------------------------------------------------------------------------------
# Figure 4: the forward-model prediction
# --------------------------------------------------------------------------------------


def fig_fm_prediction(df: pd.DataFrame, curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    # Only the two extreme tiers, and only from the runs that reach them: 600 M from the
    # seed-42 sweep (the widest delay coverage) and 4 G from the seed-43 long runs.
    for arm, sources in (("explicit", [(600_000_000, "expfm_600m"),
                                       (4_000_000_000, "expfm_4g")]),
                         ("implicit", [(600_000_000, "pgfm_600m"),
                                       (4_000_000_000, "pgfm_4g")])):
        for tier, condition in sources:
            s = sweep(df, condition, tier, metric="fm_mse")
            if s.empty:
                continue
            ax.plot(s["delay_k"], s["value"], marker="o", ms=4, color=ARM_COLOR[arm],
                    ls="-" if tier == 600_000_000 else "--",
                    mfc=None if tier == 600_000_000 else "none",
                    label=f"{arm}, {tier_label(tier)}")
    ax.set_yscale("log")
    ax.set_xlabel("Observation delay (steps)")
    ax.set_ylabel("Forward-prediction MSE (eval)")
    ax.set_title("Prediction error vs delay, at 600 M and 4 G", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax)

    ax = axes[1]
    for arm, condition in (("explicit", "expfm_4g"), ("implicit", "pgfm_4g")):
        for _, run in df[df["condition"] == condition].dropna(
                subset=["max_step"]).iterrows():
            c = curve(curves, run["wandb_id"])
            if c.empty or not np.isfinite(c["fm_mse_eval"]).any():
                continue
            ax.plot(c["step"] / 1e6, c["fm_mse_eval"], color=ARM_COLOR[arm], lw=1.3,
                    alpha=0.35 + 0.65 * ((int(run["delay_k"]) - 20) / 30))
            ax.annotate(f"d{int(run['delay_k'])}",
                        (c["step"].iloc[-1] / 1e6 * 1.02, c["fm_mse_eval"].iloc[-1]),
                        fontsize=6.5, va="center", color=ARM_COLOR[arm])
    ax.set_yscale("log")
    ax.set_xlim(0, 4600)
    ax.set_xlabel("Env steps (millions)")
    ax.set_ylabel("Forward-prediction MSE (eval)")
    ax.set_title("The 4 G runs: the explicit error keeps falling,\n"
                 "the implicit error climbs", fontsize=9.5)
    sns.despine(ax=ax)

    fig.tight_layout()
    stamp(fig, "fm_prediction", DATA, CURVES)
    fig.savefig(FIGURES / "fm_prediction.png")
    print("Saved", FIGURES / "fm_prediction.png")


# --------------------------------------------------------------------------------------


def print_tables(df: pd.DataFrame, table: pd.DataFrame, seeds: pd.DataFrame) -> None:
    print("\n=== Reward and contrast at every (tier, seed, delay) ===")
    print(table.assign(tier=table["tier"] / 1e9).round(2).to_string(index=False))

    print("\n=== Delays where only one arm reached the tier (examples, not contrasts) ===")
    for tier, explicit, implicit, seed in TIERS:
        odd = unpaired(df, explicit, implicit, tier)
        if not odd.empty:
            print(f"  {tier_label(tier)}, seed {seed}: "
                  + ", ".join(f"delay {int(r.delay_k)} ({r.arm} only)"
                              for r in odd.itertuples()))

    print("\n=== Crossover delay: where the explicit advantage clears a threshold ===")
    worst_seed = seeds["diff_pct"].abs().max() if not seeds.empty else np.nan
    for label, threshold in (("within-seed noise (2.9 %)", NOISE_FLOOR_PCT),
                             (f"seed spread ({worst_seed:.1f} %)", worst_seed)):
        print(f"  threshold = {label}")
        for tier, _, _, seed in TIERS:
            value = crossover(table, tier, seed, threshold)
            note = ("already above at the lowest sampled delay"
                    if np.isnan(value) else f"delay ~= {value:.0f}")
            print(f"    {tier_label(tier):>6s}, seed {seed}: {note}")

    print("\n=== Seed 42 vs seed 43, matched arm/delay/budget ===")
    print(seeds.round(2).to_string(index=False))

    print("\n=== Runs, and how far each got ===")
    show = df[["condition", "delay_k", "wandb_id", "state", "seed", "max_step"]].copy()
    show["max_step"] = (show["max_step"] / 1e9).round(2)
    print(show.to_string(index=False))


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)

    table = tier_table(df)
    seeds = seed_check(df)

    fig_delay_reward(df)
    fig_training_curves(df, curves)
    fig_budget(df, seeds)
    fig_fm_prediction(df, curves)
    print_tables(df, table, seeds)

    table.to_csv(HERE / "tier_table.csv", index=False)
    seeds.to_csv(HERE / "seed_table.csv", index=False)
    write_figure_manifest(HERE, _MANIFEST)


if __name__ == "__main__":
    main()
