"""Figures for walker-motor-noise.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/walker-motor-noise/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/walker-motor-noise/plot.py

Every reward is the mean of the eval points in the 5e7 steps up to 4.8e8, raw and
unscaled. A noisy run is evaluated under its own noise. Every reward axis is fixed to
:data:`REWARD_LIM` so panels compare at a glance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    add_ms_axis,
    apply_style,
    color_for,
    label_for,
    marker_for,
    plot_seeds,
    provenance,
    reward_label,
    seed_legend_handles,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CURVES = HERE / "curves.csv"

#: data.csv arm -> CONDITION_STYLE key.
STYLE = {"copy": "efference", "no_copy": "no_efference",
         "noise_pre": "motor_noise_pre", "noise_post": "motor_noise_post",
         "noise_no_copy": "motor_noise_no_copy"}

#: The std of the three-arm comparison (pre / post / no-copy).
ARM_STD = 0.5
NOISY_ARMS = ("noise_pre", "noise_post", "noise_no_copy")
REWARD_LIM = (0, 1000)
READ_AT = 480e6
LONG_AT = 1e9
#: data.csv readout -> (n-points column, horizon label).
HORIZONS = {"reward": ("reward_n", "4.8e8 steps"),
            "reward_1b": ("reward_1b_n", "1e9 steps (runs with a 1e9 budget)")}
#: Time-to-solve levels shown (data.csv has steps_to_{500,800,900}).
SOLVE_PANELS = (900, 500)
MAJOR_ARMS = ("copy", "no_copy", *("noise_pre", "noise_post", "noise_no_copy"))
YLABEL = reward_label("dmc_eval")
DELAY_XLABEL = "Observation delay (control steps)"
CTRL_MS = 25


def _usable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["reward_n"] > 0]


def _arm_label(arm: str, std: float) -> str:
    return f"{label_for(STYLE[arm])}, std {std:g}"


def fig_delay0_scale(df: pd.DataFrame) -> plt.Figure:
    """Reward vs noise std at delay 0: the scale of the noise itself, no delay involved.

    At delay 0 there is no queue, so pre and post are the same network; both are pooled
    and told apart only by marker.
    """
    d0 = _usable(df[df.delay_k == 0])
    clean = d0[d0.arm == "copy"]
    stds = sorted(d0.noise_std.unique())
    mean = d0.groupby("noise_std")["reward"].mean()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(mean.index, mean.values, color="0.4", lw=1.0, zorder=1,
            label="mean over runs")
    for arm in ("copy", "noise_pre", "noise_post"):
        sub = d0[d0.arm == arm]
        if len(sub):
            label = ("no noise" if arm == "copy"
                     else f"noise {arm.removeprefix('noise_')} (= pre at delay 0)")
            ax.scatter(sub.noise_std, sub.reward, s=16, zorder=3, label=label,
                       color=color_for(STYLE[arm]), marker=marker_for(STYLE[arm]))
    for std, n in d0.groupby("noise_std").size().items():
        ax.annotate(f"n={n}", (std, mean[std]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6, color="0.4")
    sigma = clean.policy_sigma.mean()
    ax.axvline(sigma, color="0.6", lw=0.7, ls=":")
    ax.text(sigma, 40, f"policy's own $\\sigma_a$ ≈ {sigma:.2f} ", fontsize=6,
            color="0.4", rotation=90, va="bottom", ha="right")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xticks(stds)
    ax.set_xticklabels([f"{x:g}" for x in stds], fontsize=7)
    ax.minorticks_off()
    ax.set_xlim(-0.01, max(stds) * 1.2)
    ax.set_ylim(*REWARD_LIM)
    ax.set_xlabel("Motor noise std (actuator units, after tanh)")
    ax.set_ylabel(YLABEL)
    ax.set_title("WalkerWalk, delay 0, evaluated under its own noise", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_post_vs_delay(df: pd.DataFrame) -> plt.Figure:
    """Unseen (post) noise at 0.2 and 0.5 against the clean arms: delay amplifies noise."""
    w = _usable(df)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for arm in ("copy", "no_copy"):
        plot_seeds(ax, w[w.arm == arm], x="delay_k", y="reward", seed_col="seed_pair",
                   condition=STYLE[arm], label=label_for(STYLE[arm]) + ", no noise")
    post = w[w.arm == "noise_post"]
    for std, alpha in ((0.2, 0.45), (0.5, 1.0)):
        sub = post[post.noise_std == std]
        plot_seeds(ax, sub, x="delay_k", y="reward", seed_col="seed_pair",
                          condition="motor_noise_post",
                          label=_arm_label("noise_post", std) + " (one seed)")
        ax.lines[-1].set_alpha(alpha)
    ax.set_ylim(*REWARD_LIM)
    ax.set_xlabel(DELAY_XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title("WalkerWalk: unseen motor noise vs delay", fontsize=9)
    add_ms_axis(ax, w["delay_k"].max(), ctrl_dt_ms=CTRL_MS)
    ax.add_artist(ax.legend(frameon=False, fontsize=6.5, loc="lower left"))
    ax.legend(handles=seed_legend_handles(), fontsize=6, loc="upper right",
              frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _copy_advantage(df: pd.DataFrame, y: str = "reward") -> pd.DataFrame:
    """Per seed pair and delay: each copy arm minus the no-copy arm with the same noise."""
    use = df[(df.delay_k > 0) & (df[HORIZONS[y][0]] > 0)]
    cell = (use.groupby(["seed_pair", "delay_k", "arm", "noise_std"])[y].mean()
            .reset_index())
    piv = cell.pivot_table(index=["seed_pair", "delay_k"], columns=["arm", "noise_std"],
                           values=y)
    rows = []
    pairs = [("copy", 0.0, "no_copy", 0.0, "copy"),
             ("noise_pre", ARM_STD, "noise_no_copy", ARM_STD, "noise_pre"),
             ("noise_post", ARM_STD, "noise_no_copy", ARM_STD, "noise_post")]
    for a, sa, b, sb, name in pairs:
        if (a, sa) not in piv or (b, sb) not in piv:
            continue
        diff = (piv[(a, sa)] - piv[(b, sb)]).dropna()
        rows += [{"seed_pair": s, "delay_k": d, "arm": name, "advantage": v}
                 for (s, d), v in diff.items()]
    return pd.DataFrame(rows)


def fig_arms(df: pd.DataFrame, y: str = "reward") -> plt.Figure:
    """The three noisy arms at ARM_STD against the clean arms, and the copy's advantage."""
    n_col, horizon = HORIZONS[y]
    w = df[df[n_col] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for arm in ("copy", "no_copy"):
        plot_seeds(ax, w[w.arm == arm], x="delay_k", y=y, seed_col="seed_pair",
                   condition=STYLE[arm], label=label_for(STYLE[arm]) + ", no noise")
        ax.lines[-1].set_alpha(0.45)
    for arm in NOISY_ARMS:
        sub = w[(w.arm == arm) & (w.noise_std == ARM_STD)]
        plot_seeds(ax, sub, x="delay_k", y=y, seed_col="seed_pair",
                   condition=STYLE[arm], label=_arm_label(arm, ARM_STD))
    ax.set_ylim(*REWARD_LIM)
    ax.set_xlabel(DELAY_XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(f"WalkerWalk, motor noise {ARM_STD:g}, read at {horizon}", fontsize=9)
    add_ms_axis(ax, w["delay_k"].max(), ctrl_dt_ms=CTRL_MS)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    ax = axes[1]
    adv = _copy_advantage(df, y)
    labels = {"copy": "Copy − no copy, no noise",
              "noise_pre": f"Copy (pre) − no copy, std {ARM_STD:g}",
              "noise_post": f"Copy (post) − no copy, std {ARM_STD:g}"}
    for arm in ("copy", "noise_pre", "noise_post"):
        plot_seeds(ax, adv[adv.arm == arm], x="delay_k", y="advantage",
                   seed_col="seed_pair", condition=STYLE[arm], label=labels[arm])
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xlabel(DELAY_XLABEL)
    ax.set_ylabel("Reward difference, same seed pair")
    ax.set_title("What the copy is worth (paired within seed)", fontsize=9)
    add_ms_axis(ax, w["delay_k"].max(), ctrl_dt_ms=CTRL_MS)
    ax.add_artist(ax.legend(frameon=False, fontsize=6.5, loc="lower left"))
    ax.legend(handles=seed_legend_handles(), fontsize=6, loc="upper left",
              frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_policy_width(df: pd.DataFrame) -> plt.Figure:
    """Was the noise big? Against the policy's own exploration width."""
    t = df[df["loglik_n"] > 0]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for arm in ("copy", "no_copy"):
        plot_seeds(ax, t[t.arm == arm], x="delay_k", y="policy_sigma",
                   seed_col="seed_pair", condition=STYLE[arm])
    for arm in NOISY_ARMS:
        plot_seeds(ax, t[(t.arm == arm) & (t.noise_std == ARM_STD)], x="delay_k",
                   y="policy_sigma", seed_col="seed_pair", condition=STYLE[arm],
                   label=_arm_label(arm, ARM_STD))
    for std in (0.2, 0.5):
        ax.axhline(std, color="0.6", lw=0.7, ls=":")
        ax.text(t.delay_k.max(), std, f" {std:g}", fontsize=6, va="center", color="0.4")
    ax.set_yscale("log")
    ax.set_title("WalkerWalk: dotted = motor noise std", fontsize=9)
    ax.set_xlabel(DELAY_XLABEL)
    add_ms_axis(ax, t["delay_k"].max(), ctrl_dt_ms=CTRL_MS)
    ax.set_ylabel("Policy exploration width $\\sigma_a$\n(actuator units, "
                  "from log-likelihood)")
    ax.set_ylim(0.09, 1.0)
    ax.legend(frameon=False, fontsize=6, loc="upper right", ncols=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_curves(curves: pd.DataFrame) -> plt.Figure:
    """Training curves of the three noisy arms at ARM_STD, clean arms faint behind them.

    Out to 1e9 steps; the dashed line is the 4.8e8 readout. Runs with a 4.8e8 budget stop
    there.
    """
    delays = (3, 5, 7, 10, 12, 15)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey=True, sharex=True)
    for ax, d in zip(axes.flat, delays):
        cell = curves[curves.delay_k == d]
        for arm in ("copy", "no_copy", *NOISY_ARMS):
            noisy = arm.startswith("noise")
            sub = cell[(cell.arm == arm) & (cell.noise_std == (ARM_STD if noisy else 0))]
            for i, (_, run) in enumerate(sub.groupby("wandb_id")):
                ax.plot(run.step / 1e6, run.reward, color=color_for(STYLE[arm]),
                        lw=1.3 if noisy else 0.7, alpha=0.95 if noisy else 0.3,
                        label=(_arm_label(arm, ARM_STD) if noisy
                               else label_for(STYLE[arm])) if i == 0 else None)
        ax.axvline(READ_AT / 1e6, color="0.5", lw=0.7, ls="--")
        ax.set_title(f"delay {d} ({d * CTRL_MS} ms)", fontsize=9)
        ax.set_ylim(*REWARD_LIM)
        ax.set_xlim(0, LONG_AT / 1e6)
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Eval episode reward")
    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axes[0, 0].legend(handles.values(), handles.keys(), frameon=False, fontsize=6,
                      loc="lower right")
    fig.suptitle("WalkerWalk training curves, motor noise 0.5 (clean runs faint; "
                 "dashed = 4.8e8 readout)", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_time_to_solve(df: pd.DataFrame) -> plt.Figure:
    """Steps until the eval reward first reaches a level, for runs trained to 1e9.

    Only runs whose history reaches 1e9 are used, so every run has the same horizon. A
    run that never reaches the level by 1e9 is drawn as an open marker above the axis
    break; the cell median counts it as infinitely slow, and is omitted once more than
    half the runs are censored.
    """
    clean = df.arm.isin(["copy", "no_copy"])
    noisy = df.arm.isin(NOISY_ARMS) & (df.noise_std == ARM_STD)
    long = df[(df.last_eval_step >= 0.99 * LONG_AT) & (clean | noisy)]
    censor_y = LONG_AT / 1e6 * 1.08
    offsets = dict(zip(MAJOR_ARMS, (-0.3, -0.15, 0.0, 0.15, 0.3)))
    fig, axes = plt.subplots(1, len(SOLVE_PANELS), figsize=(11, 4.0), sharey=True)
    for ax, level in zip(axes, SOLVE_PANELS):
        col = f"steps_to_{level}"
        for arm in MAJOR_ARMS:
            sub = long[long.arm == arm].copy()
            if sub.empty:
                continue
            t = sub[col].where(sub[col] <= LONG_AT) / 1e6
            x = sub.delay_k + offsets[arm]
            color, marker = color_for(STYLE[arm]), marker_for(STYLE[arm])
            ax.scatter(x[t.notna()], t[t.notna()], s=14, color=color, marker=marker,
                       zorder=3, alpha=0.8)
            ax.scatter(x[t.isna()], [censor_y] * t.isna().sum(), s=14, zorder=3,
                       facecolors="none", edgecolors=color, marker=marker, alpha=0.8)
            med = (t.fillna(float("inf")).groupby(sub.delay_k).median()
                   .replace(float("inf"), float("nan")))
            label = (label_for(STYLE[arm]) + ", no noise" if arm in ("copy", "no_copy")
                     else _arm_label(arm, ARM_STD))
            ax.plot(med.index + offsets[arm], med.values, color=color, lw=1.3,
                    marker=marker, ms=4, label=label)
        ax.axhline(LONG_AT / 1e6, color="0.6", lw=0.6, ls=":")
        ax.text(ax.get_xlim()[0], censor_y, " not reached by 1e9", fontsize=6,
                color="0.4", va="center")
        ax.set_ylim(0, censor_y * 1.04)
        ax.set_xlabel(DELAY_XLABEL)
        ax.set_title(f"Steps to first reach eval reward {level}", fontsize=9)
        add_ms_axis(ax, long["delay_k"].max(), ctrl_dt_ms=CTRL_MS)
    axes[0].set_ylabel("Environment steps (millions)")
    axes[-1].legend(frameon=False, fontsize=6, loc="upper left",
                    bbox_to_anchor=(0, 0.92))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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
        ("delay0_noise_scale", lambda: fig_delay0_scale(df), (DATA,)),
        ("post_noise_vs_delay", lambda: fig_post_vs_delay(df), (DATA,)),
        ("three_arms", lambda: fig_arms(df), (DATA,)),
        ("noise_vs_policy_width", lambda: fig_policy_width(df), (DATA,)),
        ("three_arms_1b", lambda: fig_arms(df, "reward_1b"), (DATA,)),
        ("time_to_solve", lambda: fig_time_to_solve(df), (DATA,)),
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
