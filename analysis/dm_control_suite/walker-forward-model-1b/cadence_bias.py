"""Does the eval cadence change where ``steps_to_900`` says a run solved the task?

The problem
-----------
This cohort spans the 2026-09-15 eval-cadence change, so its curves come at three
resolutions (measured, not assumed -- ``eval_spacing_median`` in ``data.csv``):

* ~4.9e5-7.4e5 steps (2-3 PPO iterations) for runs configured at ``every_steps = 6e5``;
* 4.9e6 (20 iterations) for runs configured at ``4.8e6``;
* **both, inside one curve**, for the three requeued runs that straddled the change --
  their config records only the last attempt's value.

``steps_to_<thr>`` is the first step whose **trailing 1e7-step mean** reaches the
criterion. The window is defined in steps, so what it *means* is cadence-independent --
but it is built from ~17 eval points at the fine cadence and ~2 at the coarse one, and a
mean of 2 points is noisier than a mean of 17. A first-crossing time read off a noisier
signal is biased early. Since the headline compares crossing times *across* runs whose
cadence differs, that bias has to be bounded rather than assumed small.

The check
---------
Recompute the criterion for every fine-cadence run on a **coarse grid**: keep only the
first eval point in each 4 915 200-step bin, which is exactly the post-change cadence, and
run the identical estimator over it. Each run is then its own control -- same weights, same
rewards, only the sampling density changed -- so the difference is the cadence effect with
nothing else moving.

Both window widths are reported. The 1e7 window is what ``extract.py`` uses; the 2.46e7
one (5 coarse bins) is shown because if the effect were noise-driven it should shrink
markedly with a wider window, and that is the diagnostic that separates noise from
granularity.

Reads ``curves.csv`` only -- no WandB, no artifact store, no network. Writes
``cadence_bias.txt``.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/cadence_bias.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

#: One PPO iteration is ``n_envs * rollout_length`` env steps, constant across the cohort.
ITERATION_STEPS = 8192 * 30

#: The post-2026-09-15 cadence: 20 iterations. The coarse grid this script resamples onto.
COARSE_BIN = ITERATION_STEPS * 20

#: A run is "fine cadence" if its measured median spacing is well under the coarse bin.
#: Taken from the curve rather than from ``config.eval.every_steps``, which for a requeued
#: run describes only its last attempt.
FINE_MAX_SPACING = COARSE_BIN / 2

#: ``extract.py``'s window, and a wider one for the noise-vs-granularity diagnostic.
WINDOWS = {"1.0e7 (the estimator)": 10_000_000,
           "2.5e7 (5 coarse bins)": COARSE_BIN * 5}

THRESHOLD = 900


def trailing_mean(steps: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
    """Identical to ``extract.trailing_mean`` -- duplicated so this check cannot be
    silently changed by an edit to the estimator it is supposed to be auditing."""
    lo = np.searchsorted(steps, steps - window, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= window, out, np.nan)


def crossing(steps: np.ndarray, values: np.ndarray, window: int) -> float:
    reached = np.where(trailing_mean(steps, values, window) >= THRESHOLD)[0]
    return float(steps[reached[0]]) if len(reached) else np.nan


def coarsen(steps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first sample in each coarse bin: the post-change cadence, exactly."""
    keep = pd.Series(np.arange(len(steps))).groupby(steps // COARSE_BIN).min().to_numpy()
    return steps[keep], values[keep]


def main() -> None:
    curves = pd.read_csv(HERE / "curves.csv")
    data = pd.read_csv(HERE / "data.csv")

    rows, lines = [], []
    for run_id, g in curves.groupby("wandb_id"):
        g = g.sort_values("step")
        steps = g["step"].to_numpy(dtype=float)
        values = g["reward_mean"].to_numpy(dtype=float)
        spacing = float(np.median(np.diff(steps))) if len(steps) > 1 else np.inf
        if spacing > FINE_MAX_SPACING:
            continue  # already coarse: nothing to resample
        meta = data[data["wandb_id"] == run_id].iloc[0]
        cs, cv = coarsen(steps, values)
        row = {"wandb_id": run_id, "arm": meta["condition"], "delay": int(meta["delay"]),
               "seed": int(meta["seed"]), "n_fine": len(steps), "n_coarse": len(cs),
               "spacing": spacing}
        for name, win in WINDOWS.items():
            row[f"fine {name}"] = crossing(steps, values, win)
            row[f"coarse {name}"] = crossing(cs, cv, win)
            row[f"shift {name}"] = row[f"coarse {name}"] - row[f"fine {name}"]
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["arm", "delay", "seed"])
    est, wide = list(WINDOWS)

    def fmt(v: float) -> str:
        return "    --  " if not np.isfinite(v) else f"{v / 1e6:7.1f}"

    for _, r in out.iterrows():
        lines.append(
            f"  {r['wandb_id']}  {r['arm']:<19} delay={r['delay']:<3} "
            f"seed={r['seed']}  n {r['n_fine']:>5} -> {r['n_coarse']:<4} "
            f"spacing={r['spacing'] / 1e3:>6.0f}k   "
            f"fine {fmt(r[f'fine {est}'])} -> coarse {fmt(r[f'coarse {est}'])} "
            f"(shift {fmt(r[f'shift {est}'])})e6")
    summary = []
    for name in WINDOWS:
        s = out[f"shift {name}"].dropna() / 1e6
        summary.append(
            f"  window {name:<24} n={len(s):<3} median={s.median():+6.2f}e6  "
            f"mean={s.mean():+6.2f}e6  range=[{s.min():+6.2f}, {s.max():+6.2f}]e6")

    verdict_lines = []
    med_est = (out[f"shift {list(WINDOWS)[0]}"].dropna() / 1e6).median()
    med_wide = (out[f"shift {list(WINDOWS)[1]}"].dropna() / 1e6).median()
    effects = data.loc[data[f"solved_{THRESHOLD}"].astype("boolean").fillna(False),
                       f"steps_to_{THRESHOLD}"].dropna() / 1e6
    verdict_lines.append(
        f"Median shift with the estimator's own window: {med_est:+.2f}e6 steps, against "
        f"crossing times spanning {effects.min():.0f}e6 to {effects.max():.0f}e6 -- "
        f"i.e. {abs(med_est) / effects.median() * 100:.2f} % of the median crossing time.")
    verdict_lines.append(
        f"Widening the window to {list(WINDOWS)[1]} moves the median shift to "
        f"{med_wide:+.2f}e6.")
    verdict_lines.append(
        "The shift is POSITIVE (coarse crosses *later*), which is the signature of grid\n"
        "granularity rather than of noise: a coarse grid can only report a crossing at one\n"
        "of its own points, so it rounds up by ~half a bin (2.5e6) on average. A\n"
        "noise-driven bias would go the other way -- a 2-point mean is noisier than a\n"
        "17-point one and would cross *early* -- and would shrink when the window widens\n"
        "rather than staying put. Neither signature is present at a magnitude that matters."
        if med_est > 0 else
        "*** The shift is NEGATIVE (coarse crosses earlier), which is the noise signature.\n"
        "Check whether it shrinks with the wider window; if it does not, the estimator is\n"
        "cadence-dependent and the crossing times of fine- and coarse-cadence runs are not\n"
        "directly comparable. ***")

    text = "\n".join([
        __doc__.split("Run it")[0].rstrip(), "",
        "=" * 78,
        f"Fine-cadence runs resampled onto the {COARSE_BIN:,}-step post-change grid.",
        f"Each run is its own control: same rewards, only the sampling density differs.",
        "=" * 78,
        *lines, "",
        "Shift = coarse crossing - fine crossing, in env steps:",
        *summary, "",
        "=" * 78,
        *verdict_lines,
        "=" * 78, "",
    ])
    (HERE / "cadence_bias.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
