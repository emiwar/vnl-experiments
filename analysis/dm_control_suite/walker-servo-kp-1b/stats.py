"""Exact rank tests behind the report's time-to-solve claims. Reads only data.csv.

    ../.venv/bin/python analysis/dm_control_suite/walker-servo-kp-1b/stats.py

For each delay, ``servo_kp = 64`` against torque on ``steps_to_900``, efference-copy runs
only, both budgets pooled (nothing in training depends on ``total_steps``; see
extract.py). A run that never reached 900 is ranked *after* every run that did, which is
conservative for a 4.8e8 run censored early only if its group is the slower one -- such
runs are listed so that can be checked. Within one delay every run has a distinct seed
pair or launch, so runs are the unit; nothing here pools across delays, which would
re-use seeds (memory: seed shares init across delays).

The test is the exact two-sided permutation p of the difference in mean rank, i.e.
Mann-Whitney U with ties, enumerated rather than approximated (n <= 8 per arm).
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
T = "steps_to_900"


def exact_rank_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    pooled = np.concatenate([a, b])
    ranks = rankdata(pooled)
    n = len(a)
    obs = ranks[:n].mean() - ranks[n:].mean()
    diffs = []
    for idx in combinations(range(len(pooled)), n):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        diffs.append(ranks[m].mean() - ranks[~m].mean())
    diffs = np.array(diffs)
    return obs, float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))


def fmt(s: pd.Series) -> str:
    return ", ".join("never" if pd.isna(v) else f"{v / 1e6:.0f}" for v in sorted(
        s, key=lambda v: np.inf if pd.isna(v) else v))


def main() -> None:
    df = pd.read_csv(HERE / "data.csv")
    out = [__doc__.split("    ../")[0].rstrip(), "", "=" * 78]
    for eff in ("copy", "none"):
        out.append(f"\nefference = {eff}: servo_kp 64 vs torque, steps_to_900 (M)")
        for delay in sorted(df.delay.unique()):
            sub = df[(df.delay == delay) & (df.efference == eff)]
            s = sub[sub.servo_kp == 64][T]
            t = sub[sub.servo_kp == 0][T]
            if len(s) < 1 or len(t) < 1:
                continue
            line = (f"  delay {delay:2d}  kp64 n={len(s)} [{fmt(s)}]  "
                    f"torque n={len(t)} [{fmt(t)}]")
            if len(s) >= 2 and len(t) >= 2:
                obs, p = exact_rank_test(s.fillna(np.inf).to_numpy(),
                                         t.fillna(np.inf).to_numpy())
                line += f"\n            mean-rank diff (kp64 - torque) {obs:+.2f}, exact p = {p:.3f}"
            out.append(line)
    # Delay 0: the stiff end of the new sweep against torque.
    d0 = df[(df.delay == 0)]
    stiff = d0[d0.servo_kp >= 128][T]
    torque = d0[d0.servo_kp == 0][T]
    obs, p = exact_rank_test(stiff.fillna(np.inf).to_numpy(), torque.fillna(np.inf).to_numpy())
    out += ["", f"delay 0: servo_kp >= 128 (n={len(stiff)}, one seed) [{fmt(stiff)}] vs "
            f"torque (n={len(torque)}) [{fmt(torque)}]",
            f"  mean-rank diff {obs:+.2f}, exact p = {p:.3f}"]
    mid = d0[d0.servo_kp.between(4, 64)].sort_values("servo_kp")
    out += ["", "delay 0, servo_kp ladder (seed in brackets):"]
    out += [f"  kp {r.servo_kp:6g}  {r.budget / 1e6:5.0f}M  {r.seed_pair:9s}  "
            f"{'never' if pd.isna(r[T]) else f'{r[T] / 1e6:.1f}'}"
            for _, r in d0[d0.servo_kp > 0].sort_values(["servo_kp", "budget"]).iterrows()]
    # Within the single-seed sweep (i45/p45, 1e9, efference copy): does time-to-solve fall
    # with stiffness at every delay, or only past the settling-time crossover? Spearman on
    # ranks, censored runs ranked last; exact p by enumerating all permutations is too many
    # for n = 12, so 20 000 random permutations. One seed: this describes *this seed's*
    # ladder, not the population.
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    out += ["", "within the i45/p45 sweep at 1e9: Spearman(servo_kp, steps_to_900), "
            "censored = last; servo_kp >= 2 only (below that every run fails)"]
    for delay in (0, 5, 7):
        sw = df[(df.seed_pair == "i45/p45") & (df.budget == 1_000_000_000)
                & (df.efference == "copy") & (df.condition == "servo")
                & (df.delay == delay) & (df.servo_kp >= 2)].sort_values("servo_kp")
        if len(sw) < 3:
            continue
        y = sw[T].fillna(np.inf).to_numpy()
        x = sw.servo_kp.to_numpy()
        rho = spearmanr(x, rankdata(y)).statistic
        null = np.array([spearmanr(x, rng.permutation(rankdata(y))).statistic
                         for _ in range(20_000)])
        pval = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
        out.append(f"  delay {delay}: n={len(sw)}  rho = {rho:+.2f}  perm p = {pval:.3f}   "
                   + "  ".join(f"{k:g}:{'never' if not np.isfinite(v) else f'{v / 1e6:.0f}'}"
                               for k, v in zip(x, y)))
    text = "\n".join(out) + "\n"
    (HERE / "stats.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
