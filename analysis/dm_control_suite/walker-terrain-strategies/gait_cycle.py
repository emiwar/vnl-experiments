"""Stride-averaged kinematics ("gait signature") per run on the flat course, and distances.

    OMP_NUM_THREADS=8 ../../../../.venv/bin/python gait_cycle.py

A stride is right-foot touchdown to the next right-foot touchdown (contact as in
``gait.py``); strides shorter than 0.5x or longer than 2x the run's median are dropped
(stumbles, stand-ups). Each stride is resampled to ``N_PHASE`` points and averaged over
every stride of the three flat episodes. Channels: the six joint angles, torso pitch,
and the two feet's heights -- 9 x ``N_PHASE``.

Writes ``gait_cycles.npz`` (``sig`` [run, channel, phase], ``sd`` the across-stride std,
``wandb_id``) and ``gait_distances.csv``: every pair of runs, the RMS distance between
their signatures (radians / metres, unnormalised, so a joint's own excursion sets its
weight) and the pair's relation (replicate = same condition and seed pair; same seed;
same condition; neither) and which factors differ.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from gait import CONTACT_H, MIN_SWING, SKIP, _runs_of, wrap  # noqa: E402

N_PHASE = 50
CHANNELS = C.JOINTS + ["pitch", "r_foot_h", "l_foot_h"]


def strides(d: dict, i: int) -> list[np.ndarray]:
    qpos = d["qpos"][SKIP:, i]
    h = d["foot_pos"][SKIP:, i, :, 1] - d["foot_ground_height"][SKIP:, i]
    sig = np.c_[qpos[:, 3:9], wrap(qpos[:, 2]), h]                       # [T, 9]
    contact = h[:, 0] < CONTACT_H
    td = np.array([b for a, b in _runs_of(~contact) if b - a >= MIN_SWING
                   and a > 0 and b < len(contact)])
    if len(td) < 3:
        return []
    dur = np.diff(td)
    med = np.median(dur)
    out = []
    grid = np.linspace(0, 1, N_PHASE, endpoint=False)
    for a, b in zip(td[:-1], td[1:]):
        if 0.5 * med <= b - a <= 2 * med:
            t = np.arange(a, b + 1)
            ph = (t - a) / (b - a)
            out.append(np.stack([np.interp(grid, ph, sig[t, c]) for c in range(sig.shape[1])]))
    return out


def main():
    runs = C.load_runs()
    sigs, sds, n = [], [], []
    for _, r in runs.iterrows():
        rec = C.load_rec(r.wandb_id, "flat", acts=False)
        st = [s for i in range(rec.data["reward"].shape[1]) for s in strides(rec.data, i)]
        st = np.stack(st)
        sigs.append(st.mean(0))
        sds.append(st.std(0))
        n.append(len(st))
    sig, sd = np.stack(sigs), np.stack(sds)
    here = Path(__file__).parent
    np.savez_compressed(here / "gait_cycles.npz", sig=sig, sd=sd,
                        wandb_id=np.array(runs.wandb_id.tolist(), dtype="U16"), channels=np.array(CHANNELS, dtype="U16"),
                        n_strides=np.array(n))
    rows = []
    for a, b in itertools.combinations(range(len(runs)), 2):
        ra, rb = runs.iloc[a], runs.iloc[b]
        diff = [f for f in C.FACTORS if ra[f] != rb[f]]
        if ra.condition == rb.condition and ra.seed_pair == rb.seed_pair:
            rel = "replicate"
        elif ra.seed == rb.seed:
            rel = "same seed"
        elif ra.condition == rb.condition:
            rel = "same condition"
        else:
            rel = "neither"
        rows.append(dict(a=ra.wandb_id, b=rb.wandb_id, relation=rel,
                         n_diff=len(diff), diff="+".join(diff) or "-",
                         dist=float(np.sqrt(((sig[a] - sig[b]) ** 2).mean())),
                         dist_joints=float(np.sqrt(((sig[a, :6] - sig[b, :6]) ** 2).mean()))))
    df = pd.DataFrame(rows)
    df.to_csv(here / "gait_distances.csv", index=False, float_format="%.5g")
    print(df.groupby("relation").dist.describe().round(3))
    one = df[df.n_diff == 1].groupby("diff").dist.describe().round(3)
    print("pairs differing in exactly one factor:\n", one)
    print("strides per run:", np.min(n), np.median(n), np.max(n))


if __name__ == "__main__":
    main()
