"""Figures for position-control-open-loop.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

Visual encoding follows [`position-vs-torque-control/`](../position-vs-torque-control/),
the analysis this one follows up: **control mode -> line style and colour** (solid blue =
position, dashed red = torque), so the two folders' figures read the same way. The
decoder-input arms are distinguished by marker and by position on a categorical axis
rather than by colour, because in the Q2 figures colour is already spent on the control
mode and that contrast is the point.

The delay axis is ``symlog`` with ``linthresh=1``: delay 0 has to be on it (it is the
baseline every ratio is taken against) and the sweep runs to 300, so a linear axis would
compress the 0-20 region where the torque arm lives into nothing.

    ../.venv/bin/python analysis/position-control-open-loop/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/position-control-open-loop/plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vnl_experiments.wandb_utils.style import (
    CTRL_DT_MS,
    apply_style,
    provenance,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

MODE_COLOR = {"position": "#2166ac", "torque": "#b2182b"}
MODE_LS = {"position": "-", "torque": "--"}
MODE_LABEL = {"position": "position control", "torque": "torque control"}

DELAY_TICKS = [0, 1, 5, 10, 20, 50, 100, 300]

#: Which decoder inputs each arm has, in the order the Q2 figure lays them out: from the
#: full network down to the two single-stream ablations. The label says what the *policy*
#: can see, which is the thing Q2 is about.
ARMS = [
    ("full_d0", "all inputs\ndelay 0"),
    ("full_d10", "all inputs\ndelay 10"),
    ("noeff_d10", "no efference\ndelay 10"),
    ("noproprio_eff10", "no proprio\neff 10"),
    ("noproprio_eff0", "no proprio\nno efference\n(open loop)"),
    ("nointent_d10", "no intention\ndelay 10"),
]


# --------------------------------------------------------------------------------------
# shaping
# --------------------------------------------------------------------------------------


def sweep(df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    """Mean and spread of ``metric`` per delay for one condition.

    Replicates are averaged rather than best-of: the cohort has 2-4 runs at some delays
    and one at others, and taking the maximum would bias exactly the delays with more
    runs. ``lo``/``hi`` are the min and max of the replicates, drawn as a band.
    """
    sub = df[(df["condition"] == condition)].dropna(subset=[metric])
    return (sub.groupby("delay_k")
            .agg(mean=(metric, "mean"), lo=(metric, "min"), hi=(metric, "max"),
                 n=(metric, "size"))
            .reset_index()
            .sort_values("delay_k"))


def normalised(df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    """``sweep`` divided by the condition's own delay-0 mean.

    The ratio is what makes the new and previous cohorts comparable at all: their absolute
    rewards come from different eval protocols (and, before 2026-08-20, a different
    split), so only the shape of the degradation can be put on one axis.
    """
    points = sweep(df, condition, metric)
    baseline = points.loc[points["delay_k"] == 0, "mean"]
    if baseline.empty:
        raise ValueError(f"{condition} has no delay-0 run to normalise against")
    for column in ("mean", "lo", "hi"):
        points[column] = points[column] / float(baseline.iloc[0])
    return points


def arm_rows(df: pd.DataFrame, mode: str) -> dict[str, pd.DataFrame]:
    """The Q2 arms for one control mode, keyed by the names in ``ARMS``."""
    new = df[(df["setup"] == "new") & (df["control_mode"] == mode)]
    std = new[new["kl_weight"] == 0.001]
    return {
        "full_d0": std[(std["condition"].str.endswith("efference"))
                       & (std["delay_k"] == 0)],
        "full_d10": std[(std["condition"].str.endswith("efference"))
                        & (std["delay_k"] == 10) & (std["efference_length"] == 10)],
        "noproprio_eff10": std[(std["dec_use_proprioception"] == False)  # noqa: E712
                               & (std["efference_length"] == 10)],
        "noproprio_eff0": std[(std["dec_use_proprioception"] == False)  # noqa: E712
                              & (std["efference_length"] == 0)],
        "nointent_d10": std[std["dec_use_intention"] == False],  # noqa: E712
        "noeff_d10": std[std["condition"].str.endswith("no_efference")],
    }


def delay_axis(ax, *, max_delay: float, ms_axis: bool = True):
    """Symlog delay axis with the sweep's own delays as ticks, plus a top ms axis.

    ``style.add_ms_axis`` is not reused here: it builds a linear twin, which cannot line up
    with a symlog bottom axis. Same conversion, same label.
    """
    ax.set_xscale("symlog", linthresh=1)
    ticks = [t for t in DELAY_TICKS if t <= max_delay]
    ax.set_xlim(-0.05, max_delay * 1.25)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Proprioceptive delay (control steps)")
    if not ms_axis:
        return None
    # A sparser tick set than the bottom axis: labelling all eight delays in ms crowds
    # 100/200 and 500/1000 into each other at this figure width.
    ms_ticks = [t for t in (0, 1, 10, 100) if t <= max_delay]
    twin = ax.twiny()
    twin.set_xscale("symlog", linthresh=1)
    twin.set_xlim(ax.get_xlim())
    twin.set_xticks(ms_ticks)
    twin.set_xticklabels([f"{int(t * CTRL_DT_MS)}" for t in ms_ticks])
    twin.set_xlabel("Proprioceptive delay (ms)")
    sns.despine(ax=twin, top=False, right=True, left=True, bottom=True)
    return twin


def _plot_sweep(ax, points, mode, *, label=None, alpha=1.0, lw=1.6, band=True):
    color = MODE_COLOR[mode]
    if band and (points["n"] > 1).any():
        ax.fill_between(points["delay_k"], points["lo"], points["hi"],
                        color=color, alpha=0.18 * alpha, lw=0)
    ax.plot(points["delay_k"], points["mean"], color=color, ls=MODE_LS[mode],
            marker="o" if mode == "position" else "s", ms=4, lw=lw, alpha=alpha,
            label=label if label is not None else MODE_LABEL[mode])


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def fig_q1(df: pd.DataFrame) -> plt.Figure:
    """Q1: does position control's delay tolerance survive the new XML + reference frame?

    The torque arm is drawn from ``torque_efference_aug11`` -- the 2026-08-11 sweep, the
    only complete torque delay sweep on this XML and frame (0 to 100, one commit, one
    launch, one eval source). The later ``torque_efference`` runs are overlaid as open
    markers rather than joined into a second line: they are the same-week comparator for
    the position batch, and at delays 0-20 they agree with the 2026-08-11 sweep to within
    the replicate noise, which is what licenses reading the two eval sources on one axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    ax = axes[0]
    _plot_sweep(ax, sweep(df, "pos_efference", "old_eval_reward"), "position")
    _plot_sweep(ax, sweep(df, "torque_efference_aug11", "old_eval_reward"), "torque",
                label="torque control (2026-08-11 sweep)")
    later = df[df["condition"] == "torque_efference"]
    ax.scatter(later["delay_k"], later["old_eval_reward"], s=30, facecolors="none",
               edgecolors=MODE_COLOR["torque"], lw=1.0, zorder=5,
               label="torque, later batches (individual runs)")
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_title("A. New setup: held-out reward", loc="left", fontsize=9)
    delay_axis(ax, max_delay=300)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    ax = axes[1]
    # Both cohorts on their *train-split* reward, which is the only metric the previous
    # setup has: those runs predate `final_eval/*` and the 2026-08-20 `eval_env = train_env`
    # fix, so their curve is train-split while a new run's curve is held out. Using the new
    # cohort's held-out ratios here would flatter the previous setup by 5-7 pp at long
    # delay, since the train-split ratio degrades less than the held-out one. The protocols
    # still differ (a `final_eval` pass over the whole split vs one inline eval), which is
    # why only the within-cohort ratio is plotted.
    for mode, condition in (("position", "pos_efference"),
                            ("torque", "torque_efference_aug11")):
        _plot_sweep(ax, normalised(df, condition, "train_reward"), mode,
                    label=f"{MODE_LABEL[mode]} (new setup)")
    for mode, condition in (("position", "prev_pos_efference"),
                            ("torque", "prev_torque_efference")):
        points = normalised(df, condition, "window_reward")
        ax.plot(points["delay_k"], points["mean"], color=MODE_COLOR[mode],
                ls=MODE_LS[mode], marker="o" if mode == "position" else "s", ms=3,
                lw=1.0, alpha=0.45, mfc="none",
                label=f"{MODE_LABEL[mode]} (previous setup)")
    ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)
    ax.set_ylabel("Train-split reward / same condition's delay-0 reward")
    ax.set_title("B. Degradation, new vs previous setup (train split both)",
                 loc="left", fontsize=9)
    delay_axis(ax, max_delay=300)
    ax.set_ylim(0.3, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    ax = axes[2]
    _plot_sweep(ax, sweep(df, "pos_efference", "old_eval_hazard"), "position")
    _plot_sweep(ax, sweep(df, "torque_efference_aug11", "old_eval_hazard"), "torque",
                label="torque control (2026-08-11 sweep)")
    ax.set_yscale("log")
    ax.set_ylabel("Held-out failure hazard (failures / s)")
    ax.set_title("C. New setup: failure rate", loc="left", fontsize=9)
    delay_axis(ax, max_delay=300)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    return fig


def fig_q2_open_loop(df: pd.DataFrame) -> plt.Figure:
    """Q2: what does the policy still achieve with each input stream removed?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(ARMS))
    width = 0.38

    for metric, ax, ylabel, logy in (
            ("old_eval_reward", axes[0], "Held-out episode reward (old_eval)", False),
            ("old_eval_hazard", axes[1], "Held-out failure hazard (failures / s)", True)):
        for offset, mode in ((-width / 2, "position"), (width / 2, "torque")):
            rows = arm_rows(df, mode)
            values, errs, xs = [], [], []
            for i, (key, _) in enumerate(ARMS):
                sub = rows[key].dropna(subset=[metric])
                if sub.empty:
                    continue
                xs.append(i + offset)
                values.append(sub[metric].mean())
                errs.append([sub[metric].mean() - sub[metric].min(),
                             sub[metric].max() - sub[metric].mean()])
            errs = np.array(errs).T if errs else None
            ax.bar(xs, values, width, yerr=errs, capsize=2,
                   color=MODE_COLOR[mode], alpha=0.85, label=MODE_LABEL[mode],
                   error_kw={"lw": 0.8})
            for xi, value in zip(xs, values):
                ax.annotate(f"{value:.0f}" if not logy else f"{value:.3g}",
                            (xi, value), textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=6)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in ARMS], fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7)

    axes[0].set_title("A. Held-out reward by decoder input", loc="left", fontsize=9)
    axes[1].set_title("B. Held-out failure rate by decoder input", loc="left", fontsize=9)
    # The missing bar is a missing run, not a zero -- say so rather than leaving a gap.
    open_loop_x = [i for i, (key, _) in enumerate(ARMS) if key == "noproprio_eff0"][0]
    axes[0].annotate("no torque run\nat this arm", (open_loop_x + width / 2, 60),
                     ha="center", va="bottom", fontsize=6, color="0.35", rotation=90)
    fig.tight_layout()
    return fig


def fig_q2b_delay_inert(df: pd.DataFrame) -> plt.Figure:
    """Q2b: with no proprioception, the delay label carries no information.

    Both panels show the same five runs. The point of the pair is that the vertical spread
    at ``delay_k = 5`` in A -- 1284 against 1870, a factor the run names attribute to the
    delay -- lines up perfectly with ``efference_length`` in B, while at fixed efference
    length the delay label does nothing at all (the three ``eff 0`` runs, whose networks
    ``check_delay_inert.py`` proves identical, sit inside one noise band whether they are
    labelled delay 0 or delay 5).
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
    arm = df[(df["setup"] == "new") & (df["control_mode"] == "position")
             & (df["dec_use_proprioception"] == False)]  # noqa: E712
    identical = arm[arm["efference_length"] == 0]["old_eval_reward"]
    reference = df[(df["condition"] == "pos_efference")
                   & (df["delay_k"] == 0)]["old_eval_reward"].mean()

    eff_values = sorted(arm["efference_length"].unique())
    eff_marker = dict(zip(eff_values, ("o", "^", "s", "D")))

    for ax, axis, xlabel, title in (
            (axes[0], "delay_k", "delay_k the run is named after",
             "A. Against the delay label"),
            (axes[1], "efference_length", "efference_length (past actions in the input)",
             "B. Against efference length")):
        ax.axhspan(identical.min(), identical.max(), color="0.85", lw=0, zorder=0)
        for eff in eff_values:
            sub = arm[arm["efference_length"] == eff]
            ax.scatter(sub[axis], sub["old_eval_reward"], s=46,
                       marker=eff_marker[eff], color=MODE_COLOR["position"], zorder=3,
                       label=f"efference {eff}")
        # Join the eff=0 runs: at fixed efference length the delay label is all that
        # differs between them, and the line is flat because the networks are the same one.
        zero = arm[arm["efference_length"] == 0].sort_values(axis)
        ax.plot(zero[axis], zero["old_eval_reward"], color=MODE_COLOR["position"],
                lw=1.0, ls=":", zorder=2)
        for _, row in arm.iterrows():
            ax.annotate(row["wandb_id"], (row[axis], row["old_eval_reward"]),
                        textcoords="offset points", xytext=(6, -2), fontsize=5.5,
                        color="0.35")
        ax.axhline(reference, color="0.4", lw=0.8, ls="--", zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_xlim(-1, 12)
        ax.set_title(title, loc="left", fontsize=9)

    axes[0].set_ylabel("Held-out episode reward (old_eval)")
    axes[0].set_ylim(1150, 2250)
    axes[0].annotate("all inputs, delay 0", (0.03, reference),
                     xycoords=("axes fraction", "data"), fontsize=6.5, color="0.4",
                     va="bottom")
    axes[0].annotate("spread of 3 runs of\none identical network",
                     (0.03, identical.max()), xycoords=("axes fraction", "data"),
                     fontsize=6.5, color="0.45", va="bottom")
    axes[1].legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    return fig


def fig_q2a_generalisation(df: pd.DataFrame) -> plt.Figure:
    """Q2a: is the open-loop performance specific to the training clips?"""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    new = df[df["setup"] == "new"]

    ax = axes[0]
    for mode, group in new.groupby("control_mode"):
        ax.scatter(group["train_reward"], group["old_eval_reward"], s=26,
                   color=MODE_COLOR[mode], label=MODE_LABEL[mode], alpha=0.85)
    lim = [0, 1.05 * new["train_reward"].max()]
    ax.plot(lim, lim, color="0.6", lw=0.8, ls="--", label="no gap")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Reward on the training clips")
    ax.set_ylabel("Reward on held-out clips (old_eval)")
    ax.set_title("A. Train vs held-out, every run", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    # The gap, against how good the run is: an overfitting story would put the weak
    # open-loop arms at a large gap. Position and torque share the axis on purpose.
    ax = axes[1]
    for mode, group in new.groupby("control_mode"):
        ax.scatter(group["old_eval_reward"], 100 * group["heldout_gap_frac"], s=26,
                   color=MODE_COLOR[mode], alpha=0.85, label=MODE_LABEL[mode])
    open_loop = new[(new["dec_use_proprioception"] == False)  # noqa: E712
                    & (new["efference_length"] == 0)]
    ax.scatter(open_loop["old_eval_reward"], 100 * open_loop["heldout_gap_frac"],
               s=90, facecolors="none", edgecolors="k", lw=1.0,
               label="open loop (no proprio, no efference)")
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel("Held-out episode reward (old_eval)")
    ax.set_ylabel("Train - held-out gap (% of train reward)")
    ax.set_title("B. The generalisation gap is small everywhere", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    # Hazard is the length-fair quantity: if the open-loop policy were memorising clips,
    # its failure rate would jump on unseen ones. Three datasets, same weights.
    ax = axes[2]
    keys = ["full_d0", "full_d10", "noproprio_eff10", "noproprio_eff0", "nointent_d10"]
    labels = {"full_d0": "full, delay 0", "full_d10": "full, delay 10",
              "noproprio_eff10": "no proprio,\neff 10", "noproprio_eff0": "open loop",
              "nointent_d10": "no intention"}
    rows = arm_rows(df, "position")
    x = np.arange(len(keys))
    for i, (dataset, hatch, alpha) in enumerate((("train", "", 0.9),
                                                 ("old_eval", "//", 0.65),
                                                 ("new_eval", "..", 0.4))):
        values = [rows[key][f"{dataset}_hazard"].mean() for key in keys]
        ax.bar(x + (i - 1) * 0.27, values, 0.26, color=MODE_COLOR["position"],
               alpha=alpha, hatch=hatch, label=dataset, lw=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[key] for key in keys], fontsize=6.5)
    ax.set_ylabel("Failure hazard (failures / s)")
    ax.set_title("C. Position: hazard on all three datasets", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, title="dataset", title_fontsize=7)

    fig.tight_layout()
    return fig


def fig_q2c_kl(df: pd.DataFrame) -> plt.Figure:
    """Q2c: does a tighter latent bottleneck change the picture?"""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))
    new = df[df["setup"] == "new"]
    sweep_rows = pd.concat([
        new[(new["condition"] == "pos_efference") & (new["delay_k"] == 0)],
        new[new["condition"] == "pos_kl_sweep"],
    ]).sort_values("kl_weight")
    points = (sweep_rows.groupby("kl_weight")
              .agg(reward=("old_eval_reward", "mean"),
                   lo=("old_eval_reward", "min"), hi=("old_eval_reward", "max"),
                   kl=("old_eval_encoder_kl", "mean"))
              .reset_index())

    floor = new[new["condition"] == "pos_nointent"]["old_eval_reward"].mean()
    open_loop = new[(new["condition"] == "pos_noproprio")
                    & (new["efference_length"] == 0)]["old_eval_reward"].mean()

    ax = axes[0]
    ax.fill_between(points["kl_weight"], points["lo"], points["hi"],
                    color=MODE_COLOR["position"], alpha=0.18, lw=0)
    ax.plot(points["kl_weight"], points["reward"], color=MODE_COLOR["position"],
            marker="o", ms=4, label="position, delay 0, full inputs")
    ax.axhline(floor, color="#7b3294", lw=1.0, ls="--",
               label="no intention (encoder removed)")
    ax.axhline(open_loop, color="0.35", lw=1.0, ls=":",
               label="open loop (no proprio, no efference)")
    ax.set_xscale("log")
    ax.set_xlabel("kl_weight")
    ax.set_ylabel("Held-out episode reward (old_eval)")
    ax.set_title("A. A tighter bottleneck only makes it worse", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    ax = axes[1]
    ax.plot(points["kl_weight"], points["kl"], color=MODE_COLOR["position"],
            marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("kl_weight")
    ax.set_ylabel("Measured latent KL (nats / step, old_eval)")
    ax.set_title("B. What the bottleneck actually passes", loc="left", fontsize=9)
    for _, row in points.iterrows():
        ax.annotate(f"{row['kl']:.3g}", (row["kl_weight"], row["kl"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6, color="0.35")
    fig.tight_layout()
    return fig


def fig_curves(curves: pd.DataFrame, df: pd.DataFrame) -> plt.Figure:
    """Convergence check: are the long-delay and ablated runs still improving at 600 M?

    Panel C is on its own y-axis on purpose. The 2026-08-11 sweep predates the 2026-08-20
    `eval_env = train_env` fix, so its `eval/*` curve measures the **training** clips while
    A and B measure held-out ones. Putting all three on one axis would invite exactly the
    cross-fix comparison analysis/README.md §6 forbids; the panel is here for the slope, not
    the level.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    meta = df.set_index("wandb_id")

    ax = axes[0]
    arm = curves[curves["condition"] == "pos_efference"]
    delays = sorted(arm["delay_k"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(delays)))
    for delay, color in zip(delays, colors):
        for wandb_id, run in arm[arm["delay_k"] == delay].groupby("wandb_id"):
            ax.plot(run["step"] / 1e6, run["reward_mean"], color=color, lw=1.0,
                    label=f"delay {delay}" if wandb_id == run["wandb_id"].iat[0] else None)
    ax.set_title("A. Position, full inputs, by delay", loc="left", fontsize=9)
    ax.set_ylabel("Held-out eval episode reward")
    ax.legend(frameon=False, fontsize=6, ncol=2)

    ax = axes[1]
    styles = {"pos_noproprio": "#7b3294", "pos_nointent": "#d95f02",
              "pos_no_efference": "#1b9e77"}
    # One legend entry per (arm, efference length), not per run: three of the
    # no-proprioception runs share an architecture and would otherwise appear three times
    # under the same label.
    seen = set()
    for condition, color in styles.items():
        sub = curves[curves["condition"] == condition]
        for wandb_id, run in sub.groupby("wandb_id"):
            label = (f"{condition.replace('pos_', '')} "
                     f"eff{int(meta.loc[wandb_id, 'efference_length'])}")
            ax.plot(run["step"] / 1e6, run["reward_mean"], color=color, lw=1.0,
                    alpha=0.85, label=None if label in seen else label)
            seen.add(label)
    baseline = curves[(curves["condition"] == "pos_efference")
                      & (curves["delay_k"] == 0)]
    for _, run in baseline.groupby("wandb_id"):
        ax.plot(run["step"] / 1e6, run["reward_mean"], color="0.6", lw=1.0, ls="--")
    ax.set_title("B. Position, decoder-input ablations (grey = delay 0 baseline)",
                 loc="left", fontsize=9)
    ax.sharey(axes[0])
    ax.legend(frameon=False, fontsize=6, ncol=2)

    ax = axes[2]
    arm = curves[curves["condition"] == "torque_efference_aug11"]
    delays = sorted(arm["delay_k"].unique())
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(delays)))
    for delay, color in zip(delays, colors):
        for _, run in arm[arm["delay_k"] == delay].groupby("wandb_id"):
            ax.plot(run["step"] / 1e6, run["reward_mean"], color=color, lw=0.9,
                    label=f"delay {delay}" if delay in (0, 10, 20, 50, 100) else None)
    ax.set_title("C. Torque, 2026-08-11 sweep (train-split curve)", loc="left", fontsize=9)
    ax.set_ylabel("Train-split eval episode reward")
    ax.legend(frameon=False, fontsize=6, ncol=2)

    for ax in axes:
        ax.set_xlabel("Training steps (M)")
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)

    builders = [
        ("q1_delay_tolerance", lambda: fig_q1(df)),
        ("q2_open_loop", lambda: fig_q2_open_loop(df)),
        ("q2b_delay_inert", lambda: fig_q2b_delay_inert(df)),
        ("q2a_generalisation", lambda: fig_q2a_generalisation(df)),
        ("q2c_kl_sweep", lambda: fig_q2c_kl(df)),
        ("training_curves", lambda: fig_curves(curves, df)),
    ]
    manifest = {}
    for name, builder in builders:
        fig = builder()
        manifest[f"{name}.png"] = provenance(fig, HERE, DATA, CURVES)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
