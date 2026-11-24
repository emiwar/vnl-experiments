"""Figures for the behaviour, attribution and efference-loop strands.

    ../../../../.venv/bin/python plot.py

Reads only ``gait.csv``, ``gait_cycles.npz``, ``gait_distances.csv``,
``attribution.csv``, ``attribution_inputs.csv`` and ``loop.csv`` (all written by the
extract-layer scripts here); ``probe.py`` and ``weights.py`` draw their own figures.
Every dot is one run; a run's colour is the factor the panel is about.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import apply_style, provenance

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FACTORS = ["efference", "forward_model", "sensor", "privileged_critic", "servo"]
LABEL = {"efference": "efference copy", "forward_model": "forward model",
         "sensor": "block sensor", "privileged_critic": "privileged critic",
         "servo": "servo actuators"}
ON, OFF = "#1b7f79", "#c2571a"   # factor on / off
JOINT_LABEL = {"r_hip": "right hip", "r_knee": "right knee", "r_ankle": "right ankle",
               "pitch": "torso pitch", "r_foot_h": "right foot height",
               "l_foot_h": "left foot height"}
_manifest: dict[str, str] = {}


def save(fig, name, *inputs):
    _manifest[name] = provenance(fig, HERE, *inputs)
    fig.savefig(FIG / name)
    plt.close(fig)


def write_manifest():
    path = FIG / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(_manifest)
    path.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")


def jitter(n, w=0.12, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n)


def strip(ax, df, col, by, y, order=(False, True), labels=None, color_by=None):
    """One dot per run at x = level of ``by``; mean as a bar."""
    for k, lv in enumerate(order):
        d = df[df[by] == lv]
        c = [ON if v else OFF for v in d[color_by]] if color_by else (ON if lv else OFF)
        ax.scatter(k + jitter(len(d), seed=k), d[y], s=14, c=c, alpha=0.8, lw=0)
        ax.hlines(d[y].mean(), k - 0.28, k + 0.28, color="k", lw=1.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels or [f"no {LABEL[by]}", LABEL[by]], rotation=0)
    ax.set_xlim(-0.6, len(order) - 0.4)


# --------------------------------------------------------------------------------------

def fig_gait_signatures():
    z = np.load(HERE / "gait_cycles.npz")
    g = pd.read_csv(HERE / "gait.csv")
    meta = g[g.course == "flat"].set_index("wandb_id").loc[z["wandb_id"]]
    ch = list(z["channels"])
    show = ["r_hip", "r_knee", "r_ankle", "r_foot_h", "l_foot_h", "pitch"]
    phase = np.linspace(0, 100, z["sig"].shape[2], endpoint=False)
    fig, axs = plt.subplots(2, len(show), figsize=(13, 5.2), sharex=True)
    for row, fac in enumerate(["servo", "sensor"]):
        for col, name in enumerate(show):
            ax = axs[row, col]
            c = ch.index(name)
            for lv in (False, True):
                m = (meta[fac] == lv).values
                color = ON if lv else OFF
                ax.plot(phase, z["sig"][m, c].T, color=color, lw=0.6, alpha=0.25)
                ax.plot(phase, z["sig"][m, c].mean(0), color=color, lw=2.4,
                        label=(LABEL[fac] if lv else f"no {LABEL[fac]}") + f" (n={m.sum()})")
            if row == 0:
                ax.set_title(JOINT_LABEL[name])
            if row == 1:
                ax.set_xlabel("% of right-foot stride")
            if col == 0:
                ax.set_ylabel("rad" if name not in ("r_foot_h", "l_foot_h") else "m")
        axs[row, 0].legend(loc="upper left", fontsize=7)
    # joint limits on the hip/ankle panels
    for row in range(2):
        for col, (lo, hi) in [(0, (-0.349, 1.745)), (2, (-0.785, 0.785)), (1, (-2.618, 0))]:
            for v in (lo, hi):
                axs[row, col].axhline(v, color="0.5", lw=0.7, ls=":")
    fig.suptitle("Stride-averaged kinematics on the flat course, one thin line per run "
                 "(dotted: joint limits)", fontsize=10)
    fig.tight_layout()
    save(fig, "gait_signatures.png", HERE / "gait.csv")


def fig_gait_features():
    g = pd.read_csv(HERE / "gait.csv")
    flat, terr = g[g.course == "flat"], g[g.course == "h0.3_irr1"]
    panels = [
        (flat, "servo", "hip_rom", "hip range of motion (rad)\nflat course", None),
        (flat, "servo", "hip_freq", "dominant hip frequency (Hz)\nflat course", None),
        (flat, "sensor", "swing_peak", "mean swing-foot peak height (m)\nflat course", None),
        (terr, "sensor", "swing_peak_vs_block",
         "swing peak vs height of the block ahead\n(slope, m/m; irregular terrain)", None),
        (flat, "efference", "act_hf", "action power above 4 Hz (fraction)\nflat course",
         "servo"),
        (flat, "forward_model", "lr_phase_R",
         "left/right alternation regularity (R)\nflat course", None),
        (flat, "forward_model", "pitch_std", "torso pitch std (rad)\nflat course", None),
        (flat, "sensor", "flight", "flight fraction (both feet off ground)\nflat course",
         "forward_model"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, (d, by, y, title, cb) in zip(axs.flat, panels):
        strip(ax, d, None, by, y, color_by=cb)
        ax.set_title(title, fontsize=9)
        if cb:
            ax.text(0.98, 0.98, f"teal: with {LABEL[cb]}, orange: without",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="0.3")
    axs.flat[3].axhline(0, color="0.5", lw=0.7, ls=":")
    fig.suptitle("Gait features that separate conditions (one dot per run, bar = mean)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "gait_features.png", HERE / "gait.csv")


def fig_gait_effects():
    g = pd.read_csv(HERE / "gait.csv")
    feats = {"speed": "speed", "stuck_frac": "time stuck", "cadence": "cadence",
             "stride_len": "stride length", "duty": "duty factor", "flight": "flight",
             "swing_peak": "swing peak height", "lr_phase_R": "L/R regularity",
             "lr_asym": "L/R joint asymmetry", "pitch_std": "torso pitch std",
             "height_mean": "torso height", "knee_flex": "knee flexion",
             "hip_rom": "hip range of motion", "hip_freq": "hip frequency",
             "act_abs": "|action|", "act_jerk": "action jerk", "act_hf": "action >4 Hz"}
    cols = []
    for course, title in [("flat", "flat"), ("h0.3_irr1", "irregular terrain")]:
        d = g[g.course == course]
        X = np.c_[np.ones(len(d)), d[FACTORS].astype(float).values]
        t = {}
        for f in feats:
            y = d[f].values
            y = (y - y.mean()) / y.std()
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            r = y - X @ b
            se = np.sqrt(np.diag(r @ r / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
            t[feats[f]] = b[1:] / se[1:]
            if course != "flat" and f in ("duty", "flight", "stride_len"):
                # a foot resting on a block edge with its origin past the edge reads as
                # airborne, so stance/flight are only measured on the flat course
                t[feats[f]] = np.full(len(FACTORS), np.nan)
        cols.append((title, pd.DataFrame(t, index=[LABEL[f] for f in FACTORS]).T))
    fig, axs = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    for ax, (title, tt) in zip(axs, cols):
        v = np.clip(tt.values, -10, 10)
        im = ax.imshow(v, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
        for (i, j), val in np.ndenumerate(tt.values):
            if abs(val) >= 3:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                        color="w" if abs(val) > 7 else "k")
        ax.set_xticks(range(len(FACTORS)))
        ax.set_xticklabels(tt.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(tt)))
        ax.set_yticklabels(tt.index)
        ax.set_title(title)
    fig.colorbar(im, ax=axs, shrink=0.6, label="t of main effect (clipped at ±10)")
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle("Main effect of each factor on each gait feature (OLS, 62 runs, seeds "
                 "not modelled; |t| ≥ 3 printed)", fontsize=9)
    save(fig, "gait_effects.png", HERE / "gait.csv")


def fig_gait_distance():
    d = pd.read_csv(HERE / "gait_distances.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    rel = ["replicate", "same condition", "same seed", "neither"]
    rel_lab = ["replicate\n(same cond. + seeds)", "same condition\n(other seed)",
               "same seed\n(other cond.)", "neither"]
    for k, r in enumerate(rel):
        v = d[d.relation == r].dist
        axs[0].scatter(k + jitter(len(v), 0.25), v, s=6 if len(v) > 50 else 16,
                       alpha=0.25 if len(v) > 50 else 0.9, color="0.3", lw=0)
        axs[0].hlines(v.median(), k - 0.3, k + 0.3, color="k", lw=1.8)
    axs[0].set_xticks(range(4))
    axs[0].set_xticklabels(rel_lab, fontsize=8)
    axs[0].set_ylabel("RMS distance between stride-\naveraged kinematics (rad / m)")
    one = d[d.n_diff == 1]
    order = ["privileged_critic", "forward_model", "efference", "servo", "sensor"]
    for k, f in enumerate(order):
        v = one[one["diff"] == f].dist
        axs[1].scatter(k + jitter(len(v), 0.25), v, s=8, alpha=0.5, color="0.3", lw=0)
        axs[1].hlines(v.median(), k - 0.3, k + 0.3, color="k", lw=1.8)
    axs[1].axhline(d[d.relation == "same condition"].dist.median(), color=ON, ls="--",
                   lw=1, label="median, same condition / other seed")
    axs[1].set_xticks(range(len(order)))
    axs[1].set_xticklabels([LABEL[f] for f in order], rotation=25, ha="right", fontsize=8)
    axs[1].set_title("pairs differing in exactly one factor", fontsize=9)
    axs[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Whole-stride gait similarity on the flat course (bar = median)", fontsize=10)
    fig.tight_layout()
    save(fig, "gait_distance.png", HERE / "gait_distances.csv")


def fig_attribution():
    a = pd.read_csv(HERE / "attribution.csv")
    a = a[a.scope == "policy"].assign(group=lambda x: x.group.replace({"sensor": "blk"}))
    p = a.pivot_table(index=["wandb_id", "condition", "reward_480M", *FACTORS],
                      columns="group",
                      values="share").fillna(0).reset_index()
    lags = [f"eff_lag{k}" for k in range(1, 6)]
    for c in lags + ["blk"]:
        if c not in p:
            p[c] = 0.0
    p["efference copy"] = p[lags].sum(axis=1)
    p["arch"] = (p.forward_model.map({True: "FM", False: "MLP"})
                 + p.efference.map({True: "+eff", False: ""})
                 + p.servo.map({True: " servo", False: " torque"}))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0),
                            gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    groups = ["orient", "vel", "blk", "efference copy"]
    glab = ["orientation", "velocity", "block sensor", "efference copy"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b3"]
    order = p.groupby("condition").reward_480M.mean().sort_values().index
    m = p.groupby("condition")[groups].mean().loc[order]
    bottom = np.zeros(len(m))
    for gname, lab, col in zip(groups, glab, colors):
        axs[0].bar(range(len(m)), m[gname], bottom=bottom, color=col, label=lab, width=0.8)
        bottom += m[gname].values
    axs[0].set_xticks(range(len(m)))
    axs[0].set_xticklabels(m.index, rotation=70, ha="right", fontsize=6.5)
    axs[0].set_ylabel("share of action sensitivity")
    axs[0].set_title("What the action is sensitive to (mean over runs per condition;\n"
                     "conditions ordered by reward_480M)", fontsize=9)
    axs[0].legend(fontsize=7, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.42))
    # servo vs torque: velocity and sensor shares
    for ax, col, title in [(axs[1], "vel", "velocity share"),
                           (axs[2], "blk", "block-sensor share (sensor runs)")]:
        d = p if col == "vel" else p[p.sensor.astype(bool)]
        strip(ax, d, None, "servo", col, color_by="efference")
        ax.set_title(title + "\nteal: with efference copy, orange: without", fontsize=9)
    fig.tight_layout()
    save(fig, "attribution.png", HERE / "attribution.csv")

    # efference lag profile
    fig, ax = plt.subplots(figsize=(5, 3.4))
    e = p[p.efference]
    sty = {"MLP+eff torque": (OFF, "-"), "MLP+eff servo": (OFF, "--"),
           "FM+eff torque": (ON, "-"), "FM+eff servo": (ON, "--")}
    for arch, d in e.groupby("arch"):
        prof = d[lags].div(d["efference copy"], axis=0)
        col, ls = sty[arch]
        ax.plot(range(1, 6), prof.T.values, color=col, ls=ls, lw=0.6, alpha=0.35)
        ax.plot(range(1, 6), prof.mean(), color=col, ls=ls, lw=2.2,
                label=f"{arch} (n={len(d)})")
    ax.axhline(0.2, color="0.5", lw=0.7, ls=":")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("efference lag (control steps of 25 ms; 5 = the delayed obs' own step)")
    ax.set_ylabel("share of efference sensitivity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "attribution_lags.png", HERE / "attribution.csv")


def fig_loop():
    lp = pd.read_csv(HERE / "loop.csv")
    g = pd.read_csv(HERE / "gait.csv")
    g = g[g.course == "all"]
    lp["arch"] = (lp.forward_model.map({True: "FM+eff", False: "MLP+eff"})
                  + lp.servo.map({True: "\nservo", False: "\ntorque"}))
    order = ["MLP+eff\ntorque", "FM+eff\ntorque", "MLP+eff\nservo", "FM+eff\nservo"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for k, a in enumerate(order):
        d = lp[lp.arch == a]
        c = [ON if s else OFF for s in d.sensor]
        axs[0].scatter(k + jitter(len(d)), d.rho_median, c=c, s=16, lw=0)
        axs[0].hlines(d.rho_median.mean(), k - .28, k + .28, color="k")
        axs[1].scatter(k + jitter(len(d)), d.frac_rho_ge1, c=c, s=16, lw=0)
        axs[1].hlines(d.frac_rho_ge1.mean(), k - .28, k + .28, color="k")
    for ax, t in [(axs[0], "median spectral radius of the\nlinearised efference loop"),
                  (axs[1], "fraction of visited states\nwhere the loop is unstable (ρ ≥ 1)")]:
        ax.set_xticks(range(4))
        ax.set_xticklabels(order, fontsize=8)
        ax.set_title(t, fontsize=9)
    axs[0].axhline(1, color="0.5", ls=":", lw=0.8)
    axs[0].text(0.02, 0.02, "teal: with block sensor, orange: blind",
                transform=axs[0].transAxes,
                fontsize=7, color="0.3")
    m = lp.merge(g[["wandb_id", "act_hf"]], on="wandb_id")
    for s, mk in [(False, "o"), (True, "^")]:
        d = m[m.servo == s]
        axs[2].scatter(d.rho_median, d.act_hf, marker=mk, s=18,
                       c=[ON if f else OFF for f in d.forward_model],
                       label=("servo" if s else "torque"), lw=0)
    nef = g[~g.efference]
    for s, mk in [(False, "o"), (True, "^")]:
        v = nef[nef.servo == s].act_hf
        axs[2].hlines(v.mean(), 0.6, 1.1, color="0.5", ls="--" if s else ":", lw=1)
    axs[2].set_xlabel("median loop spectral radius")
    axs[2].set_ylabel("action power above 4 Hz (fraction)")
    axs[2].set_title("loop gain vs high-frequency action power\n"
                     "teal: forward model, orange: MLP; grey: no-efference means\n"
                  "(dotted torque, dashed servo)", fontsize=8)
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "efference_loop.png", HERE / "loop.csv", HERE / "gait.csv")


def main():
    apply_style()
    FIG.mkdir(exist_ok=True)
    fig_gait_signatures()
    fig_gait_features()
    fig_gait_effects()
    fig_gait_distance()
    fig_attribution()
    fig_loop()
    write_manifest()


if __name__ == "__main__":
    main()
