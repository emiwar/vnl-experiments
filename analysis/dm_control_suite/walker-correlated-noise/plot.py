"""Figures for walker-correlated-noise.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/walker-correlated-noise/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/walker-correlated-noise/plot.py

Every reward is the mean of the eval points in the 5e7 steps up to 4.8e8, raw and
unscaled, and a noisy run is evaluated under its own noise. Time to solve is searched only
up to 4.8e8, the budget of every tau > 0 run.

The tau axis is logarithmic, with iid noise (tau = 0) drawn in a separate slot at its left
end, marked "iid".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    apply_style,
    add_ms_axis,
    color_for,
    label_for,
    marker_for,
    provenance,
    reward_label,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

READ_AT = 480e6
YLABEL = reward_label("dmc_eval")
CTRL_MS = 25
#: Where iid noise (tau = 0) sits on the log tau axis.
IID_X = 1.0
#: std -> line alpha; the faint line is the small noise, as in ../walker-motor-noise/.
STD_ALPHA = {0.2: 0.45, 0.5: 1.0}
#: tau values that get a tick label (the others get an unlabelled tick).
TAU_LABELS = {2, 5, 10, 25, 50, 100}
#: Time-to-solve levels: (level, linestyle).
SOLVE_LEVELS = ((900, "-"), (800, "--"))
REWARD_LIM = (0, 1000)
DELAY_XLABEL = "Observation delay (control steps)"

#: data.csv arm -> CONDITION_STYLE key.
STYLE = {"copy": "efference", "no_copy": "no_efference",
         "noise_pre": "motor_noise_pre", "noise_post": "motor_noise_post",
         "noise_no_copy": "motor_noise_no_copy"}
NOISY_ARMS = ("noise_pre", "noise_post", "noise_no_copy")
#: The std the delay > 0 sweeps were run at.
SWEEP_STD = 0.5
#: A delay gets a tau-sweep row once some arm has this many tau > 0 values there.
MIN_TAUS = 3
#: The tau of the delay sweep, drawn against iid (tau = 0).
SWEEP_TAU = 10.0
#: Time-to-solve level for the sweeps at delay (most tau > 0 runs never reach 800).
SWEEP_SOLVE = 500
#: Small multiplicative x offsets so arms at one tau don't sit on each other.
ARM_XSHIFT = {"noise_pre": 1 / 1.06, "noise_post": 1.0, "noise_no_copy": 1.06}
ARM_DSHIFT = {"noise_pre": -0.15, "noise_post": 0.0, "noise_no_copy": 0.15}


def _usable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["reward_n"] > 0]


def _tau_x(tau: pd.Series) -> pd.Series:
    return tau.where(tau > 0, IID_X)


def _tau_axis(ax: plt.Axes, taus) -> None:
    ticks = [IID_X] + sorted(t for t in taus if t > 0)
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    # Every run's tau gets a tick; only well-spaced ones get a label.
    ax.set_xticklabels(["iid"] + [f"{t:g}" if t in TAU_LABELS else ""
                                  for t in ticks[1:]])
    ax.minorticks_off()
    ax.set_xlim(IID_X / 1.35, ticks[-1] * 1.35)
    # The iid slot is not on the log scale: mark the break.
    ax.axvline(np.sqrt(IID_X * ticks[1]), color="0.85", lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel("Noise correlation time τ (control steps)")
    top = ax.secondary_xaxis("top")
    labelled = [t for t in ticks[1:] if t in TAU_LABELS]
    top.set_xticks(labelled)
    top.set_xticklabels([f"{t * CTRL_MS:g}" for t in labelled], fontsize=6.5)
    top.minorticks_off()
    top.set_xlabel("τ (ms)", fontsize=7)


def fig_delay0_calibration(df: pd.DataFrame) -> plt.Figure:
    """Delay 0: what correlated noise costs on its own, in final reward and in speed.

    At delay 0 there is no efference queue, so the site is inert: the iid points pool the
    pre and post launches (and every seed pair that exists). Every tau > 0 point is one
    run, seed pair i42/p1234.
    """
    d0 = _usable(df[df.delay_k == 0])
    clean = d0[d0.arm == "copy"]
    noisy = d0[d0.noise_std > 0].assign(x=lambda d: _tau_x(d.noise_tau))
    color = color_for("motor_noise_post")
    marker = marker_for("motor_noise_post")
    clean_color = color_for("efference")

    fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(10, 3.9))

    # Left: reward.
    ax_r.axhspan(clean.reward.min(), clean.reward.max(), color=clean_color, alpha=0.18,
                 lw=0, label=f"no noise ({len(clean)} runs, range)")
    for std, alpha in STD_ALPHA.items():
        sub = noisy[noisy.noise_std == std]
        mean = sub.groupby("x").reward.mean()
        ax_r.plot(mean.index, mean.values, color=color, alpha=alpha, lw=1.4,
                  marker=marker, ms=4.5, label=f"motor noise std {std:g}")
        iid = sub[sub.noise_tau == 0]
        ax_r.scatter(iid.x, iid.reward, s=9, color=color, alpha=0.5 * alpha,
                     zorder=3, lw=0)
    _tau_axis(ax_r, noisy.noise_tau.unique())
    ax_r.set_ylabel(YLABEL)
    ax_r.set_title("Reward at 4.8e8 steps (zoomed y-axis)", fontsize=9)
    ax_r.legend(frameon=False, fontsize=7, loc="lower left")

    # Right: time to solve, censored at READ_AT.
    censor_y = READ_AT / 1e6 * 1.08
    ax_t.axhspan(*(clean[f"steps_to_{SOLVE_LEVELS[0][0]}"].agg(["min", "max"]) / 1e6),
                 color=clean_color, alpha=0.18, lw=0)
    for std, alpha in STD_ALPHA.items():
        sub = noisy[noisy.noise_std == std]
        for level, ls in SOLVE_LEVELS:
            t = sub[f"steps_to_{level}"] / 1e6
            # Median over the runs in a cell, with not-reached counted as infinitely slow.
            med = (t.fillna(np.inf).groupby(sub.x).median().replace(np.inf, np.nan))
            ax_t.plot(med.index, med.values, color=color, alpha=alpha, lw=1.3, ls=ls,
                      marker=marker, ms=4 if ls == "-" else 3,
                      mfc=color if ls == "-" else "white")
            never = sub[t.isna()]
            ax_t.scatter(never.x, [censor_y] * len(never), s=22, facecolors="none",
                         edgecolors=color, alpha=alpha, marker=marker, zorder=3)
    ax_t.axhline(READ_AT / 1e6, color="0.6", lw=0.6, ls=":")
    ax_t.text(IID_X / 1.3, censor_y, " not reached by 4.8e8", fontsize=6.5,
              color="0.4", va="center")
    ax_t.set_ylim(0, censor_y * 1.05)
    _tau_axis(ax_t, noisy.noise_tau.unique())
    ax_t.set_ylabel("Environment steps (millions)")
    ax_t.set_title("Steps to first reach eval reward 900 (solid) / 800 (dashed)",
                   fontsize=9)
    handles = [plt.Line2D([], [], color=color, alpha=a, lw=1.4, marker=marker, ms=4,
                          label=f"motor noise std {s:g}") for s, a in STD_ALPHA.items()]
    handles += [plt.Line2D([], [], color="0.4", lw=1.2, ls=ls, label=f"reach {lv}")
                for lv, ls in SOLVE_LEVELS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=clean_color, alpha=0.18,
                                 label="no noise, reach 900 (range)"))
    ax_t.legend(handles=handles, frameon=False, fontsize=6.5, loc="upper left",
                bbox_to_anchor=(0, 0.93))

    n_iid = noisy[noisy.noise_tau == 0].groupby("noise_std").size()
    fig.suptitle("WalkerWalk, delay 0 (no efference queue, so the noise site is moot). "
                 f"iid: {n_iid.min()}–{n_iid.max()} runs (dots); "
                 "τ > 0: one run each, seed pair i42/p1234", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _arm_label(arm: str) -> str:
    return label_for(STYLE[arm])


def _censored_steps(ax, sub: pd.DataFrame, x: pd.Series, col: str, color, marker,
                    censor_y: float) -> pd.Series:
    """Scatter steps-to-solve per run, open marker above the horizon if never reached.

    Returns the per-x median with not-reached counted as infinitely slow (NaN once more
    than half the runs are censored).
    """
    t = sub[col] / 1e6
    ax.scatter(x[t.notna()], t[t.notna()], s=14, color=color, marker=marker, zorder=3,
               alpha=0.8, lw=0)
    ax.scatter(x[t.isna()], [censor_y] * int(t.isna().sum()), s=16, zorder=3,
               facecolors="none", edgecolors=color, marker=marker, alpha=0.8)
    return t.fillna(np.inf).groupby(x).median().replace(np.inf, np.nan)


def _horizon(ax, censor_y: float, x_text: float) -> None:
    ax.axhline(READ_AT / 1e6, color="0.6", lw=0.6, ls=":")
    ax.text(x_text, censor_y, " not reached by 4.8e8", fontsize=6, color="0.4",
            va="center")
    ax.set_ylim(0, censor_y * 1.05)


#: The paired contrasts: (arm, minus arm, colour/marker arm, linestyle, open marker).
CONTRASTS = (("noise_pre", "noise_no_copy", "noise_pre", "-", False),
             ("noise_post", "noise_no_copy", "noise_post", "-", False),
             ("noise_pre", "noise_post", "noise_pre", "--", True))


def _short(arm: str) -> str:
    return {"noise_pre": "pre", "noise_post": "post", "noise_no_copy": "no copy"}[arm]


def _paired_diff(df: pd.DataFrame, by: str, arm: str, minus: str,
                 y: str = "reward") -> pd.DataFrame:
    """`arm` minus `minus` at the same `by` value and seed pair.

    Replicates are averaged within a seed pair first; cells without a same-seed partner
    are dropped, not compared across seeds.
    """
    per = df.groupby(["arm", by, "seed_pair"])[y].mean().reset_index()
    base = per[per.arm == minus].set_index([by, "seed_pair"])[y]
    out = per[per.arm == arm].copy()
    out["diff"] = out[y] - out.set_index([by, "seed_pair"]).index.map(base)
    return out.dropna(subset=["diff"])


def _plot_contrasts(ax, cell: pd.DataFrame, by: str, xmap) -> None:
    ax.axhline(0, color="0.5", lw=0.7)
    for arm, minus, style_arm, ls, open_ in CONTRASTS:
        sub = _paired_diff(cell, by, arm, minus)
        if sub.empty:
            continue
        x = xmap(sub[by])
        color, marker = color_for(STYLE[style_arm]), marker_for(STYLE[style_arm])
        ax.scatter(x, sub["diff"], s=14, color=color, marker=marker, alpha=0.5, lw=0,
                   zorder=3)
        mean = sub.groupby(x)["diff"].mean()
        ax.plot(mean.index, mean.values, color=color, marker=marker, ms=4.5, lw=1.4,
                ls=ls, mfc="white" if open_ else color,
                label=f"{_short(arm)} − {_short(minus)}")


def _sweep_delays(df: pd.DataFrame) -> list[int]:
    noisy = df[(df.noise_std == SWEEP_STD) & (df.noise_tau > 0) & (df.delay_k > 0)]
    n = noisy.groupby(["delay_k", "arm"]).noise_tau.nunique()
    return sorted({d for (d, _), k in n.items() if k >= MIN_TAUS})


def fig_tau_sweep(df: pd.DataFrame) -> plt.Figure:
    """Reward, copy advantage and time to solve against tau, one row per swept delay.

    Std 0.5. iid (tau = 0) cells pool every seed pair that exists; tau > 0 cells are
    whatever has been run. The grey line in the reward panel is the same noise at
    delay 0 (post; the site is moot there), so the gap to it is the cost of the delay.
    """
    w = _usable(df)
    delays = _sweep_delays(w)
    d0 = w[(w.delay_k == 0) & (w.noise_std == SWEEP_STD)]
    d0_mean = d0.groupby(_tau_x(d0.noise_tau)).reward.mean()
    fig, axes = plt.subplots(len(delays), 3, figsize=(14, 3.9 * len(delays)),
                             squeeze=False)
    for row, d in zip(axes, delays):
        ax_r, ax_a, ax_t = row
        cell = w[(w.delay_k == d) & (w.noise_std == SWEEP_STD)
                 & w.arm.isin(NOISY_ARMS)]
        clean = w[(w.delay_k == d) & w.arm.isin(["copy", "no_copy"])]
        taus = cell.noise_tau.unique()

        # Reward.
        ax_r.plot(d0_mean.index, d0_mean.values, color="0.55", lw=1.0, ls="--",
                  label="same noise at delay 0")
        for arm in ("copy", "no_copy"):
            r = clean[clean.arm == arm].reward
            if len(r):
                ax_r.axhspan(r.min(), r.max(), color=color_for(STYLE[arm]), alpha=0.15,
                             lw=0, label=f"{label_for(STYLE[arm])}, no noise (range)")
        for arm in NOISY_ARMS:
            sub = cell[cell.arm == arm]
            if sub.empty:
                continue
            x = _tau_x(sub.noise_tau) * ARM_XSHIFT[arm]
            color, marker = color_for(STYLE[arm]), marker_for(STYLE[arm])
            ax_r.scatter(x, sub.reward, s=12, color=color, marker=marker, alpha=0.5,
                         lw=0, zorder=3)
            mean = sub.groupby(x).reward.mean()
            ax_r.plot(mean.index, mean.values, color=color, marker=marker, ms=4.5,
                      lw=1.4, label=_arm_label(arm))
        _tau_axis(ax_r, taus)
        ax_r.set_ylim(*REWARD_LIM)
        ax_r.set_ylabel(YLABEL)
        ax_r.set_title(f"Delay {d} ({d * CTRL_MS} ms): reward at 4.8e8", fontsize=9)
        ax_r.legend(frameon=False, fontsize=6, loc="lower left")

        # Paired contrasts within seed pair.
        _plot_contrasts(ax_a, cell, "noise_tau", _tau_x)
        ax_a.axvline(d, color="0.75", lw=4, alpha=0.4, zorder=0, label="τ = delay")
        _tau_axis(ax_a, taus)
        ax_a.set_ylabel("Reward difference (same seed pair)")
        ax_a.set_title("Paired contrasts between the noisy arms", fontsize=9)
        ax_a.legend(frameon=False, fontsize=6, loc="best")

        # Time to solve.
        censor_y = READ_AT / 1e6 * 1.08
        for arm in NOISY_ARMS:
            sub = cell[cell.arm == arm]
            if sub.empty:
                continue
            x = _tau_x(sub.noise_tau) * ARM_XSHIFT[arm]
            color, marker = color_for(STYLE[arm]), marker_for(STYLE[arm])
            med = _censored_steps(ax_t, sub, x, f"steps_to_{SWEEP_SOLVE}", color,
                                  marker, censor_y)
            ax_t.plot(med.index, med.values, color=color, marker=marker, ms=4,
                      lw=1.3)
        _horizon(ax_t, censor_y, IID_X / 1.3)
        _tau_axis(ax_t, taus)
        ax_t.set_ylabel("Environment steps (millions)")
        ax_t.set_title(f"Steps to first reach eval reward {SWEEP_SOLVE}", fontsize=9)
    fig.suptitle(f"WalkerWalk, motor noise std {SWEEP_STD:g}: correlation time at fixed "
                 "delay. τ > 0 cells are single runs unless dots show more", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_delay_sweep(df: pd.DataFrame) -> plt.Figure:
    """iid against tau = SWEEP_TAU across delay, std 0.5: reward and time to solve.

    Solid = iid, dashed = correlated. The τ = delay crossing is marked.
    """
    w = _usable(df)
    noisy = w[(w.noise_std == SWEEP_STD) & w.noise_tau.isin([0.0, SWEEP_TAU])
              & w.arm.isin(NOISY_ARMS)]
    # d0 has no queue, so its runs are the d0 point of every arm's curve.
    d0 = noisy[noisy.delay_k == 0]
    noisy = pd.concat([noisy[noisy.delay_k > 0]]
                      + [d0.assign(arm=a) for a in NOISY_ARMS])
    clean = w[w.arm.isin(["copy", "no_copy"])]
    fig, (ax_r, ax_a, ax_t) = plt.subplots(1, 3, figsize=(15, 4.0))
    for arm in ("copy", "no_copy"):
        mean = clean[clean.arm == arm].groupby("delay_k").reward.mean()
        ax_r.plot(mean.index, mean.values, color=color_for(STYLE[arm]), lw=0.9,
                  alpha=0.35, label=f"{label_for(STYLE[arm])}, no noise")
    censor_y = READ_AT / 1e6 * 1.08
    for arm in NOISY_ARMS:
        for tau, ls, mfc in ((0.0, "-", None), (SWEEP_TAU, "--", "white")):
            sub = noisy[(noisy.arm == arm) & (noisy.noise_tau == tau)]
            # A line needs more than the shared d0 point.
            if sub.delay_k.nunique() < 3:
                continue
            color, marker = color_for(STYLE[arm]), marker_for(STYLE[arm])
            x = sub.delay_k + ARM_DSHIFT[arm]
            ax_r.scatter(x, sub.reward, s=10, color=color, marker=marker, alpha=0.4,
                         lw=0, zorder=3)
            mean = sub.groupby(x).reward.mean()
            name = "iid" if tau == 0 else f"τ = {tau:g}"
            ax_r.plot(mean.index, mean.values, color=color, ls=ls, marker=marker, ms=4,
                      mfc=mfc or color, lw=1.4, label=f"{_arm_label(arm)}, {name}")
            med = _censored_steps(ax_t, sub, x, f"steps_to_{SWEEP_SOLVE}", color,
                                  marker, censor_y)
            ax_t.plot(med.index, med.values, color=color, ls=ls, marker=marker, ms=4,
                      mfc=mfc or color, lw=1.3)
    # pre - post, paired within seed pair, iid (solid) and tau (dashed). No noisy
    # no-copy run exists at tau = SWEEP_TAU beyond d5, so this is the only contrast.
    ax_a.axhline(0, color="0.5", lw=0.7)
    for tau, ls, mfc in ((0.0, "-", None), (SWEEP_TAU, "--", "white")):
        sub = _paired_diff(noisy[(noisy.noise_tau == tau) & (noisy.delay_k > 0)],
                           "delay_k", "noise_pre", "noise_post")
        color, marker = color_for(STYLE["noise_pre"]), marker_for(STYLE["noise_pre"])
        ax_a.scatter(sub.delay_k, sub["diff"], s=12, color=color, marker=marker,
                     alpha=0.45, lw=0, zorder=3)
        mean = sub.groupby("delay_k")["diff"].mean()
        ax_a.plot(mean.index, mean.values, color=color, ls=ls, marker=marker, ms=4,
                  mfc=mfc or color, lw=1.4,
                  label="pre − post, " + ("iid" if tau == 0 else f"τ = {tau:g}"))
    ax_a.set_ylabel("Reward difference (same seed pair)")
    ax_a.set_title("Seen minus unseen noise (pre − post)", fontsize=9)
    ax_a.legend(frameon=False, fontsize=6.5, loc="lower left")
    for ax in (ax_r, ax_a, ax_t):
        ax.axvline(SWEEP_TAU, color="0.75", lw=4, alpha=0.4, zorder=0)
        ax.set_xlabel(DELAY_XLABEL)
        add_ms_axis(ax, w.delay_k.max(), ctrl_dt_ms=CTRL_MS)
    ax_a.text(SWEEP_TAU, ax_a.get_ylim()[1], " delay = τ", fontsize=6.5, color="0.4",
              va="top")
    ax_r.set_ylim(*REWARD_LIM)
    ax_r.set_ylabel(YLABEL)
    ax_r.set_title("Reward at 4.8e8", fontsize=9)
    ax_r.legend(frameon=False, fontsize=6, loc="upper right")
    _horizon(ax_t, censor_y, 0)
    ax_t.set_ylabel("Environment steps (millions)")
    ax_t.set_title(f"Steps to first reach eval reward {SWEEP_SOLVE}", fontsize=9)
    fig.suptitle(f"WalkerWalk, motor noise std {SWEEP_STD:g}: iid (solid) against "
                 f"τ = {SWEEP_TAU:g} (dashed) across delay; τ = {SWEEP_TAU:g} is one "
                 "run per delay", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_action_std(df: pd.DataFrame) -> plt.Figure:
    """Std of the *executed* rollout actions (after the motor noise), std 0.5.

    The noise adds the same stationary variance at every tau. If the policy can see the
    noise and it persists, it can steer against it, and the executed actions get *less*
    variable than intended action + noise. Post and no-copy cannot do that, so they are
    the baseline for each tau. Pooled over envs, time and the last 5e7 steps.
    """
    w = _usable(df)
    noisy = w[(w.noise_std == SWEEP_STD) & w.arm.isin(NOISY_ARMS)]
    delays = _sweep_delays(w)
    fig, axes = plt.subplots(1, len(delays) + 1, figsize=(4.6 * (len(delays) + 1), 3.8),
                             sharey=True)
    for ax, d in zip(axes, delays):
        cell = noisy[noisy.delay_k == d]
        for arm in NOISY_ARMS:
            sub = cell[cell.arm == arm]
            if sub.empty:
                continue
            x = _tau_x(sub.noise_tau) * ARM_XSHIFT[arm]
            color, marker = color_for(STYLE[arm]), marker_for(STYLE[arm])
            ax.scatter(x, sub.batch_action_std, s=12, color=color, marker=marker,
                       alpha=0.5, lw=0, zorder=3)
            mean = sub.groupby(x).batch_action_std.mean()
            ax.plot(mean.index, mean.values, color=color, marker=marker, ms=4.5, lw=1.4,
                    label=_arm_label(arm))
        _tau_axis(ax, cell.noise_tau.unique())
        ax.set_title(f"Delay {d}", fontsize=9)
    ax = axes[-1]
    for arm in ("noise_pre", "noise_post"):
        for tau, ls, mfc in ((0.0, "-", None), (SWEEP_TAU, "--", "white")):
            sub = noisy[(noisy.arm == arm) & (noisy.noise_tau == tau) & (noisy.delay_k > 0)]
            color, marker = color_for(STYLE[arm]), marker_for(STYLE[arm])
            mean = sub.groupby("delay_k").batch_action_std.mean()
            ax.plot(mean.index, mean.values, color=color, ls=ls, marker=marker, ms=4,
                    mfc=mfc or color, lw=1.4,
                    label=f"{_short(arm)}, " + ("iid" if tau == 0 else f"τ = {tau:g}"))
    ax.set_xlabel(DELAY_XLABEL)
    ax.set_title(f"iid against τ = {SWEEP_TAU:g}, across delay", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("Std of executed actions\n(rollout batch, actuator units)")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.suptitle(f"WalkerWalk, motor noise std {SWEEP_STD:g}: how variable the executed "
                 "action is (lower = more noise cancelled)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _curve_panel(ax, runs: pd.DataFrame, hue: str, cmap, norm, iid_label: bool) -> None:
    labelled = False
    for (_, value), run in runs.groupby(["wandb_id", hue]):
        is_iid = hue == "noise_tau" and value == 0
        label = None
        if is_iid and iid_label and not labelled:
            label, labelled = "iid (each run)", True
        ax.plot(run.step / 1e6, run.reward, lw=1.1 if not is_iid else 1.0,
                color="0.2" if is_iid else cmap(norm(value)),
                ls=":" if is_iid else "-", alpha=0.9, label=label)
    ax.set_ylim(*REWARD_LIM)
    ax.set_xlim(0, READ_AT / 1e6)


def fig_curves(curves: pd.DataFrame) -> plt.Figure:
    """Top: the tau sweep at each swept delay, one panel per arm, coloured by tau.

    Bottom: post noise across delay, iid against tau = SWEEP_TAU, coloured by delay.
    """
    c = curves[curves.noise_std == SWEEP_STD]
    delays = [d for d in _sweep_delays(c.assign(reward_n=1).drop_duplicates("wandb_id"))]
    arms = [a for a in NOISY_ARMS if a in set(c.arm)]
    tau_vals = sorted(t for t in c.noise_tau.unique() if t > 0)
    tau_cmap, tau_norm = plt.cm.viridis, plt.matplotlib.colors.LogNorm(
        min(tau_vals), max(tau_vals))
    d_cmap = plt.cm.plasma
    d_norm = plt.Normalize(0, c.delay_k.max() * 1.15)

    nrows = len(delays) + 1
    fig, axes = plt.subplots(nrows, max(len(arms), 2), figsize=(13, 3.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for row, d in zip(axes, delays):
        for ax, arm in zip(row, arms):
            runs = c[(c.delay_k == d) & (c.arm == arm)]
            _curve_panel(ax, runs, "noise_tau", tau_cmap, tau_norm, iid_label=True)
            ax.set_title(f"delay {d}: {_arm_label(arm)}", fontsize=9)
            if any(ax.get_legend_handles_labels()[1]):
                ax.legend(frameon=False, fontsize=6, loc="lower right")
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=tau_norm, cmap=tau_cmap),
                           ax=row[-1], label="τ (control steps)")
        bar.set_ticks(tau_vals, labels=[f"{t:g}" for t in tau_vals])
        bar.ax.minorticks_off()
    bottom = axes[-1]
    panels = (("noise_post", 0.0), ("noise_post", SWEEP_TAU), ("noise_pre", SWEEP_TAU))
    for ax, (arm, tau) in zip(bottom, panels):
        runs = c[(c.arm == arm) & (c.noise_tau == tau)]
        # One run per delay: the first seed pair, so the panel stays readable.
        first = runs.groupby("delay_k").wandb_id.first()
        _curve_panel(ax, runs[runs.wandb_id.isin(first)], "delay_k", d_cmap, d_norm,
                     iid_label=False)
        ax.set_title(f"{_short(arm)}, "
                     + ("iid" if tau == 0 else f"τ = {tau:g}") + ", across delay",
                     fontsize=9)
    for ax in bottom[len(panels):]:
        ax.set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=d_norm, cmap=d_cmap),
                 ax=bottom[len(panels) - 1], label="delay (control steps)")
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Eval episode reward")
    fig.suptitle(f"WalkerWalk training curves, motor noise std {SWEEP_STD:g} "
                 "(iid dotted black)", fontsize=9)
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    curves = pd.read_csv(CURVES)
    if df["reward_source"].nunique() > 1:
        raise SystemExit("data.csv mixes reward sources")

    manifest = {}
    builders = [
        ("delay0_calibration", lambda: fig_delay0_calibration(df), (DATA,)),
        ("tau_sweep", lambda: fig_tau_sweep(df), (DATA,)),
        ("delay_sweep_tau10", lambda: fig_delay_sweep(df), (DATA,)),
        ("executed_action_std", lambda: fig_action_std(df), (DATA,)),
        ("curves", lambda: fig_curves(curves), (CURVES,)),
    ]
    for name, build, inputs in builders:
        fig = build()
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")
    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
