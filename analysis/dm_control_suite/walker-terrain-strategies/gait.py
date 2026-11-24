"""Gait and terrain-negotiation features of every run, from the activations_dmc recordings.

    OMP_NUM_THREADS=8 ../../../../.venv/bin/python gait.py

Writes ``gait.csv``: one row per (run, course set), features averaged over that set's
three 1000-step episodes, plus a ``course == "all"`` row averaging the three sets.
The first ``SKIP`` steps of each episode (the stand-up from the random initial pose)
are dropped from every feature.

Contact is the foot body's origin within ``CONTACT_H`` of the ground under it (the
foot capsule's radius is 0.05 m; the recorded heights are bimodal with the stance mode
at 0.03-0.07 m). A "swing" is a maximal run of non-contact frames of at least
``MIN_SWING`` steps; a touchdown is the frame a swing ends.

Features (all per episode, then averaged):

* ``speed`` m/s (mean torso x velocity), ``stuck_frac`` (fraction of 1 s windows with
  < 0.25 m of progress), ``fallen_frac`` (torso upright term < 0.5).
* ``cadence`` touchdowns/s per foot, ``stride_len`` = speed / cadence (m per stride
  of one foot), ``duty`` stance fraction, ``double_support``, ``flight`` fractions.
  **Read duty/flight on the flat course only**: on blocks, a foot resting on a block's
  edge with its origin past the edge is measured against the ground below and reads as
  airborne.
* ``lr_phase``: circular mean of where a left touchdown falls in the right foot's
  touchdown cycle (0.5 = alternating gait), and ``lr_phase_R`` its concentration.
* ``swing_peak``: mean peak foot height above the ground under it, per swing; and on the
  ``h0.3_irr1`` set ``swing_peak_vs_block`` -- the slope (m/m) of swing peak against
  the top height of the next block whose near edge is ahead of *both* feet at swing
  start, over swings whose foot is within 1 m of that edge. Neither foot has touched
  that block, so a blind policy cannot know its height: this measures *anticipatory*
  scaling of step height, and the blind runs are its null. (Undefined on stairs, where
  every block is 0.3 m.) ``swing_peak_near_block`` / ``_far_block``: mean swing peak
  for those swings / all others. ``block_strike`` (foot >2 cm below the top of the block
  it is over) turned out to be always 0 and is not used.
* posture: mean/std of torso pitch, torso height (rootz), each joint angle; ``knee_flex``.
* action: ``act_abs`` mean |a|, ``act_sat`` fraction |a| > 0.98, ``act_jerk`` mean |a_t -
  a_{t-1}|, ``act_hf`` fraction of action power above 4 Hz. Their meaning differs
  between torque and servo runs (torque vs target angle), so compare them within
  actuator type only.
* reward terms: mean per-step ``standing``, ``upright``, ``move``; ``return``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

SKIP = 40
DT = 0.025
CONTACT_H = 0.08
MIN_SWING = 3
WINDOW = 40  # 1 s


def _runs_of(mask: np.ndarray):
    """(start, stop) of each maximal True run in a 1-D bool array."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fx_swing(d, i, a, foot):
    return d["foot_pos"][SKIP + a, i, foot, 0]


def _circmean(phases):
    if len(phases) == 0:
        return np.nan, np.nan
    z = np.exp(2j * np.pi * np.asarray(phases)).mean()
    return float((np.angle(z) / (2 * np.pi)) % 1.0), float(np.abs(z))


def episode_features(d: dict, i: int, terrain: bool) -> dict:
    s = slice(SKIP, None)
    qpos, qvel = d["qpos"][s, i], d["qvel"][s, i]
    act = d["actions"][s, i]
    fz = d["foot_pos"][s, i, :, 1]
    gh = d["foot_ground_height"][s, i]
    h = fz - gh                                     # [T, 2] (right, left)
    blk = d["block_sensor"][s, i]
    rt = d["reward_terms"][s, i]
    T = qpos.shape[0]
    f = {}

    x = d["qpos"][:, i, 1]
    f["speed"] = float(qvel[:, 1].mean())
    prog = x[SKIP + WINDOW::WINDOW] - x[SKIP:-WINDOW:WINDOW][: len(x[SKIP + WINDOW::WINDOW])]
    f["stuck_frac"] = float((prog < 0.25).mean())
    f["fallen_frac"] = float((rt[:, 1] < 0.5).mean())
    f["return"] = float(d["reward"][:, i].sum())
    for j, name in enumerate(["standing", "upright", "stand", "move"]):
        f[f"r_{name}"] = float(rt[:, j].mean())

    contact = h < CONTACT_H
    f["duty"] = float(contact.mean())
    f["double_support"] = float(contact.all(axis=1).mean())
    f["flight"] = float((~contact).all(axis=1).mean())
    touchdowns, peaks, peak_block, starts = [], [], [], []
    for foot in range(2):
        swings = [(a, b) for a, b in _runs_of(~contact[:, foot]) if b - a >= MIN_SWING
                  and a > 0 and b < T]
        touchdowns.append(np.array([b for _, b in swings]))
        for a, b in swings:
            peaks.append(h[a:b, foot].max())
            if terrain:
                # The next block whose near edge is ahead of *both* feet at swing start:
                # neither foot has touched it, so only the sensor can tell its height.
                dist0, top0, dist1, top1 = blk[a]
                tx = d["qpos"][SKIP + a, i, 1]
                feet_x = d["foot_pos"][SKIP + a, i, :, 0]
                edge0, edge1 = tx + dist0, tx + dist1
                if edge0 > feet_x.max():
                    edge, top = edge0, top0
                elif edge1 > feet_x.max():
                    edge, top = edge1, top1
                else:
                    edge, top = np.inf, np.nan
                peak_block.append((h[a:b, foot].max(), top, edge - fx_swing(d, i, a, foot)))
    f["cadence"] = float(np.mean([len(t) for t in touchdowns]) / (T * DT))
    f["stride_len"] = f["speed"] / f["cadence"] if f["cadence"] > 0 else np.nan
    f["swing_peak"] = float(np.mean(peaks)) if peaks else np.nan
    f["swing_peak_p90"] = float(np.percentile(peaks, 90)) if peaks else np.nan
    if terrain and peak_block:
        pb = np.array(peak_block)
        near = (pb[:, 2] < 1.0) & np.isfinite(pb[:, 1])
        if near.sum() > 10 and pb[near, 1].std() > 0.02:
            f["swing_peak_vs_block"] = float(np.polyfit(pb[near, 1], pb[near, 0], 1)[0])
        f["swing_peak_near_block"] = float(pb[near, 0].mean()) if near.any() else np.nan
        f["swing_peak_far_block"] = float(pb[~near, 0].mean()) if (~near).any() else np.nan
    # left touchdowns within the right foot's cycle
    tr, tl = touchdowns
    ph = []
    for t in tl:
        k = np.searchsorted(tr, t)
        if 0 < k < len(tr):
            ph.append((t - tr[k - 1]) / (tr[k] - tr[k - 1]))
    f["lr_phase"], f["lr_phase_R"] = _circmean(ph)
    f["block_strike"] = float((h < -0.02).mean())

    # rooty is unwrapped: a somersault while standing up adds 2*pi (several runs do one)
    pitch = wrap(qpos[:, 2])
    f["pitch_mean"] = float(pitch.mean())
    f["pitch_std"] = float(pitch.std())
    f["height_mean"] = float(qpos[:, 0].mean())
    for j, name in enumerate(C.JOINTS):
        f[f"q_{name}_mean"] = float(qpos[:, 3 + j].mean())
        f[f"q_{name}_std"] = float(qpos[:, 3 + j].std())
    f["knee_flex"] = float(-(qpos[:, 4].mean() + qpos[:, 7].mean()) / 2)
    f["hip_rom"] = float((np.ptp(qpos[:, 3]) + np.ptp(qpos[:, 6])) / 2)
    f["lr_asym"] = float(np.abs(qpos[:, 3:6].mean(0) - qpos[:, 6:9].mean(0)).mean())

    f["act_abs"] = float(np.abs(act).mean())
    f["act_sat"] = float((np.abs(act) > 0.98).mean())
    f["act_jerk"] = float(np.abs(np.diff(act, axis=0)).mean())
    spec = np.abs(np.fft.rfft(act - act.mean(0), axis=0)) ** 2
    freqs = np.fft.rfftfreq(T, DT)
    f["act_hf"] = float(spec[freqs > 4].sum() / spec.sum())
    # dominant gait frequency from the hip angles
    hs = np.abs(np.fft.rfft(qpos[:, [3, 6]] - qpos[:, [3, 6]].mean(0), axis=0)) ** 2
    band = (freqs > 0.3) & (freqs < 5)
    f["hip_freq"] = float(freqs[band][hs[band].sum(1).argmax()])
    return f


def main():
    runs = C.load_runs()
    rows = []
    for _, r in runs.iterrows():
        per_course = []
        for course in C.COURSE_SETS:
            rec = C.load_rec(r.wandb_id, course, acts=False)
            feats = [episode_features(rec.data, i, terrain=course != "flat")
                     for i in range(rec.data["reward"].shape[1])]
            mean = pd.DataFrame(feats).mean(numeric_only=True).to_dict()
            # lr_phase is circular: average as a circular mean
            ph = np.array([f["lr_phase"] for f in feats])
            mean["lr_phase"] = _circmean(ph[np.isfinite(ph)])[0]
            per_course.append(mean)
            rows.append({"wandb_id": r.wandb_id, "course": course, **mean})
        allm = pd.DataFrame(per_course).mean(numeric_only=True).to_dict()
        rows.append({"wandb_id": r.wandb_id, "course": "all", **allm})
    g = pd.DataFrame(rows)
    meta = runs[["wandb_id", "condition", "arch", *C.FACTORS, "seed", "seed_pair",
                 "reward_480M"]]
    g = meta.merge(g, on="wandb_id")
    g.to_csv(Path(__file__).parent / "gait.csv", index=False, float_format="%.5g")
    print(g[g.course == "all"].groupby("condition")[
        ["return", "speed", "cadence", "stride_len", "duty", "flight", "swing_peak",
         "lr_phase", "stuck_frac", "act_sat", "act_jerk"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
