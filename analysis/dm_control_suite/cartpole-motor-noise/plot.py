"""Figures for cartpole-motor-noise.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/dm_control_suite/cartpole-motor-noise/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/dm_control_suite/cartpole-motor-noise/plot.py

Every reward is the mean of the eval points in the 5e7 steps up to 4.8e8, raw and
unscaled. A noisy run is evaluated under its own noise. Every reward axis is fixed to
:data:`REWARD_LIM` so panels compare at a glance, and with the WalkerWalk sibling
(``../walker-motor-noise/``).

Unlike WalkerWalk there is no noisy no-copy arm, so the copy's *advantage* under noise
cannot be computed; the pre-vs-post contrast exists at delay 5 only.
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
         "noise_pre": "motor_noise_pre", "noise_post": "motor_noise_post"}

#: The only delay with both noise sites.
SITE_DELAY = 5
REWARD_LIM = (0, 1000)
YLABEL = reward_label("dmc_eval")
STD_XLABEL = "Motor noise std (actuator units, after tanh)"
CTRL_MS = 10
#: data.csv column -> panel title. The README window mean, then the crash-robust median
#: (see extract.py ROBUST_WINDOW): every reward figure shows both side by side.
READOUTS = {"reward": "mean, last 5e7 steps (README §6)",
            "reward_median": "median, last 2.4e8 steps (crash-robust)"}


def _usable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["reward_n"] > 0]


def _std_axis(ax, stds) -> None:
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xticks(stds)
    ax.set_xticklabels([f"{x:g}" for x in stds], fontsize=6, rotation=90)
    ax.minorticks_off()
    ax.set_xlim(-0.005, max(stds) * 1.2)
    ax.set_xlabel(STD_XLABEL)


def _dose_panel(ax, cell: pd.DataFrame, arms, y: str) -> None:
    """Every run as a dot, mean per std as a line; std 0 is the clean copy cell."""
    clean = cell[cell.arm == "copy"]
    ax.scatter([0] * len(clean), clean[y], s=16, zorder=3, label="no noise",
               color=color_for("efference"), marker=marker_for("efference"))
    for arm in arms:
        sub = cell[cell.arm == arm]
        mean = pd.concat([clean, sub]).groupby("noise_std")[y].mean()
        ax.plot(mean.index, mean.values, color=color_for(STYLE[arm]), lw=1.2, zorder=2)
        ax.scatter(sub.noise_std, sub[y], s=16, zorder=3, label=label_for(STYLE[arm]),
                   color=color_for(STYLE[arm]), marker=marker_for(STYLE[arm]))
    _std_axis(ax, sorted(cell.noise_std.unique()))
    ax.set_ylim(*REWARD_LIM)


def fig_delay0_scale(df: pd.DataFrame) -> plt.Figure:
    """Reward vs noise std at delay 0: the scale of the noise itself, no delay involved."""
    d0 = _usable(df[df.delay_k == 0])
    sigma = d0.loc[d0.arm == "copy", "policy_sigma"].mean()
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(10.5, 3.8), sharey=True)
    for ax, (y, title) in zip(axes, READOUTS.items()):
        _dose_panel(ax, d0, ("noise_post",), y)
        for std, n in d0.groupby("noise_std").size().items():
            ax.annotate(f"n={n}", (std, d0[d0.noise_std == std][y].max()),
                        textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=6, color="0.4")
        ax.axvline(sigma, color="0.6", lw=0.7, ls=":")
        ax.set_title(f"Delay 0, {title}", fontsize=9)
    axes[0].text(sigma, 260, f"policy's own $\\sigma_a$ ≈ {sigma:.3f} ", fontsize=6,
                 color="0.4", rotation=90, va="bottom", ha="right")
    axes[0].set_ylabel(YLABEL)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("CartpoleSwingup, evaluated under its own noise", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_sites(df: pd.DataFrame) -> plt.Figure:
    """Pre vs post at SITE_DELAY across stds (both readouts), and the paired difference."""
    cell = _usable(df[df.delay_k == SITE_DELAY])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    nc = cell[cell.arm == "no_copy"]
    for ax, (y, title) in zip(axes, READOUTS.items()):
        _dose_panel(ax, cell, ("noise_pre", "noise_post"), y)
        ax.scatter([0] * len(nc), nc[y], s=16, zorder=3, label="no copy, no noise",
                   color=color_for("no_efference"), marker=marker_for("no_efference"))
        ax.set_title(f"Delay {SITE_DELAY}, {title}", fontsize=9)
    axes[0].set_ylabel(YLABEL)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")

    ax = axes[2]
    noisy = cell[cell.arm.isin(["noise_pre", "noise_post"])]
    for (y, title), marker, ls in zip(READOUTS.items(), ("o", "s"), (":", "-")):
        piv = noisy.pivot_table(index=["noise_std", "seed_pair"], columns="arm",
                                values=y, aggfunc="mean").dropna()
        diff = (piv.noise_pre - piv.noise_post).rename("diff").reset_index()
        for i, (seed, g) in enumerate(diff.groupby("seed_pair")):
            ax.plot(g.noise_std, g["diff"], marker=marker, ms=4, lw=0.9, ls=ls,
                    color=f"C{7 + i}", label=f"{seed}, {title.split(',')[0]}")
    ax.axhline(0, color="0.5", lw=0.7)
    _std_axis(ax, sorted(diff.noise_std.unique()))
    ax.set_xlim(0.007, diff.noise_std.max() * 1.2)
    ax.set_ylim(-500, 500)
    ax.set_ylabel("Pre − post reward, same seed pair")
    ax.set_title("Paired difference (> 0: seeing the noise helps)", fontsize=9)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_post_vs_delay(df: pd.DataFrame) -> plt.Figure:
    """Unseen (post) noise at 0.2, 0.5 and 1.0 against the clean arms, over delay."""
    w = _usable(df)
    post = w[w.arm == "noise_post"]
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(10.5, 3.8), sharey=True)
    for ax, (y, title) in zip(axes, READOUTS.items()):
        for arm in ("copy", "no_copy"):
            plot_seeds(ax, w[w.arm == arm], x="delay_k", y=y, seed_col="seed_pair",
                       condition=STYLE[arm], label=label_for(STYLE[arm]) + ", no noise")
        for std, alpha, ls in ((0.2, 0.4, "-"), (0.5, 1.0, "-"), (1.0, 1.0, "--")):
            plot_seeds(ax, post[post.noise_std == std], x="delay_k", y=y,
                       seed_col="seed_pair", condition="motor_noise_post",
                       label=f"{label_for('motor_noise_post')}, std {std:g}")
            ax.lines[-1].set_alpha(alpha)
            ax.lines[-1].set_linestyle(ls)
        ax.set_ylim(*REWARD_LIM)
        ax.set_xlabel("Observation delay (control steps)")
        ax.set_title(title, fontsize=9)
        add_ms_axis(ax, w["delay_k"].max(), ctrl_dt_ms=CTRL_MS)
    axes[0].set_ylabel(YLABEL)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    axes[1].legend(handles=seed_legend_handles(), fontsize=6, loc="lower right",
                   frameon=False)
    fig.suptitle("CartpoleSwingup: unseen motor noise vs delay", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_policy_width(df: pd.DataFrame) -> plt.Figure:
    """Was the noise big? The policy's own width against the motor noise std."""
    t = df[(df["loglik_n"] > 0) & df.delay_k.isin([0, SITE_DELAY])]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for (delay, arm), g in t[t.arm != "no_copy"].groupby(["delay_k", "arm"]):
        if arm == "copy":
            continue
        clean = t[(t.delay_k == delay) & (t.arm == "copy")]
        mean = pd.concat([clean, g]).groupby("noise_std")["policy_sigma"].mean()
        ls = "-" if delay == SITE_DELAY else "--"
        ax.plot(mean.index, mean.values, color=color_for(STYLE[arm]), ls=ls, lw=1.2,
                marker=marker_for(STYLE[arm]), ms=4,
                label=f"{label_for(STYLE[arm])}, delay {delay}")
    stds = sorted(t.noise_std.unique())
    ax.plot([0.01, max(stds)], [0.01, max(stds)], color="0.6", lw=0.7, ls=":",
            label="policy width = noise std")
    _std_axis(ax, stds)
    ax.set_yscale("log")
    ax.set_ylabel("Policy exploration width $\\sigma_a$\n(actuator units, "
                  "from log-likelihood)")
    ax.set_title("CartpoleSwingup: the policy barely explores", fontsize=9)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_curves(curves: pd.DataFrame) -> plt.Figure:
    """Training curves where the readout is least trustworthy: the d5 collapses, d20."""
    panels = [(SITE_DELAY, 0.05, ("noise_pre", "noise_post")),
              (SITE_DELAY, 0.2, ("noise_pre", "noise_post")),
              (20, 0.2, ("noise_post",)),
              (20, 0.5, ("noise_post",))]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 3.7), sharey=True)
    handles = {}
    for ax, (d, std, arms) in zip(axes, panels):
        cell = curves[curves.delay_k == d]
        for arm in ("copy", *arms):
            noisy = arm != "copy"
            sub = cell[(cell.arm == arm) & (cell.noise_std == (std if noisy else 0))]
            for _, run in sub.groupby("wandb_id"):
                (h,) = ax.plot(run.step / 1e6, run.reward, color=color_for(STYLE[arm]),
                               lw=1.3 if noisy else 0.7, alpha=0.9 if noisy else 0.35)
                handles.setdefault(label_for(STYLE[arm]), h)
        ax.set_title(f"delay {d} ({d * CTRL_MS} ms), noise std {std:g}", fontsize=9)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylim(*REWARD_LIM)
    axes[0].set_ylabel(YLABEL)
    axes[0].legend(handles.values(), handles.keys(), frameon=False, fontsize=6,
                   loc="lower right")
    fig.suptitle("CartpoleSwingup training curves (clean copy runs faint)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


#: (arm, std, line style) series in the time-to-solve figure.
SOLVE_SERIES = (("copy", 0.0, "-"), ("no_copy", 0.0, "-"),
                ("noise_post", 0.2, ":"), ("noise_post", 1.0, "--"))


def fig_time_to_solve(df: pd.DataFrame) -> plt.Figure:
    """Steps until the eval reward first reaches a level; every run has a 4.8e8 horizon.

    A run that never reaches the level is an open marker above the axis break; the cell
    median counts it as infinitely slow and is omitted once more than half are censored.
    """
    use = _usable(df)
    horizon = 480.0
    censor_y = horizon * 1.08
    offsets = dict(zip(SOLVE_SERIES, (-0.45, -0.15, 0.15, 0.45)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, level in zip(axes, (800, 600)):
        col = f"steps_to_{level}"
        for key in SOLVE_SERIES:
            arm, std, ls = key
            sub = use[(use.arm == arm) & (use.noise_std == std)]
            t = sub[col] / 1e6
            x = sub.delay_k + offsets[key]
            color, marker = color_for(STYLE[arm]), marker_for(STYLE[arm])
            ax.scatter(x[t.notna()], t[t.notna()], s=14, color=color, marker=marker,
                       zorder=3, alpha=0.8)
            ax.scatter(x[t.isna()], [censor_y] * t.isna().sum(), s=14, zorder=3,
                       facecolors="none", edgecolors=color, marker=marker, alpha=0.8)
            med = (t.fillna(float("inf")).groupby(sub.delay_k).median()
                   .replace(float("inf"), float("nan")))
            label = (label_for(STYLE[arm]) + ", no noise" if std == 0
                     else f"{label_for(STYLE[arm])}, std {std:g}")
            ax.plot(med.index + offsets[key], med.values, color=color, lw=1.3, ls=ls,
                    marker=marker, ms=4, label=label)
        ax.axhline(horizon, color="0.6", lw=0.6, ls=":")
        ax.text(ax.get_xlim()[0], censor_y, " not reached by 4.8e8", fontsize=6,
                color="0.4", va="center")
        ax.set_ylim(0, censor_y * 1.04)
        ax.set_xlabel("Observation delay (control steps)")
        ax.set_title(f"Steps to first reach eval reward {level}", fontsize=9)
        add_ms_axis(ax, use["delay_k"].max(), ctrl_dt_ms=CTRL_MS)
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
        ("sites_at_delay5", lambda: fig_sites(df), (DATA,)),
        ("post_noise_vs_delay", lambda: fig_post_vs_delay(df), (DATA,)),
        ("noise_vs_policy_width", lambda: fig_policy_width(df), (DATA,)),
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
