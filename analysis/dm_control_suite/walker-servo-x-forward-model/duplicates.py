"""What did de-duplication throw away, and how much do identical runs differ?

The problem
-----------
Four pairs of runs in this cohort's candidate pool agree on every experimental parameter
*and* on **both** seeds -- the network-initialisation seed and the PPO seed that keys env
resets, rollouts and eval (track README). They are not two seeds. They are one experiment
run twice, differing only in MuJoCo/XLA nondeterminism, and counting both weights that
seed double in every cell mean and every spread. ``extract._first_of_duplicates`` keeps
the earlier-launched run of each pair; this script is what keeps the discarded one from
being lost.

Two jobs
--------
1. **Record what was dropped**, with the readouts recomputed the same way ``extract.py``
   computes them, so the decision is auditable from the committed text rather than from a
   re-run.
2. **Measure end-to-end training nondeterminism at fixed seed**, which nothing else in
   this project does. README §6 measures the *eval*'s nondeterminism (~1 % on this laptop,
   exactly reproducible on the cluster) and the final training-curve point's (a few
   percent). Neither says what happens when a whole 1e9-step *training* run is repeated
   with the same seeds: every gradient step after the first divergence in the physics is
   computed on different data, so the two runs are different trajectories through weight
   space from very early on. These four pairs are the only direct measurement of that
   available here, and the number matters -- it is the floor below which no arm difference
   in ``report.md`` means anything, however many seeds are averaged.

What to expect, and what it means if the pairs disagree
-------------------------------------------------------
Reward at the end of training should agree closely; a pair that does not is telling you
the cell is in a regime where the outcome is not determined by the seed at all. The
*time*-to-criterion may agree much less well, because ``steps_to_900`` is defined inside
the gap of a bimodal learning curve (report.md caveat 8) -- so a pair that takes the same
mode should agree, and a pair that splits modes will differ by hundreds of millions of
steps. Both facts are worth having explicitly.

Reads the committed index and the history artifact store -- no network. Its output is the
committed artifact, in the same role as ``attrition.txt`` and ``divergence_check.txt``.

    ../.venv/bin/python analysis/dm_control_suite/walker-servo-x-forward-model/duplicates.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import index

HERE = Path(__file__).resolve().parent

import extract as E  # noqa: E402  (after the path-relative docstring, deliberately)


def _readouts(store: Store, wandb_id: str) -> dict:
    """The same numbers ``extract.build_row`` would produce, for any run in the pool."""
    series = E.reward_series(E.history_of(store, wandb_id, E.HISTORY_SPEC_ID))
    if series is None or series.empty:
        return {}
    steps = series["step"].to_numpy()
    window = series[series["step"] > steps.max() - E.FINAL_WINDOW_STEPS]
    out = {"reward_final": float(window["value"].mean())}
    out.update(E.time_to_criterion(series))
    return out


def main() -> None:
    df = index.load(project=E.PROJECT)
    store = Store()

    # The candidate pool *before* de-duplication: `_cohort` applies the dedupe itself, so
    # it is rebuilt here without that last step.
    pool_mask = (
        df["env"].eq(E.TASK)
        & df["net_params.min_std"].eq(E.MIN_STD)
        & df["config.ppo.total_steps"].eq(E.BUDGET)
        & df["net_params.delay_k"].isin(E.DELAYS)
        & ~df["git_commit"].str.startswith(E.PRE_EVAL_FIX_COMMIT, na=False)
        & (df["state"].eq("finished") | df["summary._step"].ge(E.BUDGET))
        & df["net_params.network_class"].isin(["DelayedMLP", "FlatForwardModel"])
        & df["env_params.servo_kp"].fillna(0.0).isin([0.0, E.SERVO_KP])
    )
    pool = df[pool_mask].copy()
    for col in ("env_params.servo_kp", "env_params.servo_damping_ratio"):
        pool[col] = pool[col].fillna(0.0)
    pool["env_params.servo_center"] = pool["env_params.servo_center"].fillna("absent")
    pool["arm"] = (
        pool["net_params.network_class"].map({"DelayedMLP": "mlp",
                                              "FlatForwardModel": "fm"})
        + "_" + pool["env_params.servo_kp"].map(
            lambda k: "servo" if k == E.SERVO_KP else "torque"))
    kept = set(pool.sort_values(["created_at", "wandb_id"])
                   .groupby(E.DEDUPE_KEY, dropna=False)["wandb_id"].first())

    groups = [g for _, g in pool.groupby(E.DEDUPE_KEY, dropna=False) if len(g) > 1]
    lines = [f"candidate pool (before de-duplication): {len(pool)} runs",
             f"duplicate groups                      : {len(groups)}",
             f"runs discarded                        : "
             f"{sum(len(g) - 1 for g in groups)}", ""]

    d_reward, d_steps, same_mode = [], [], []
    for g in sorted(groups, key=lambda x: (x["arm"].iloc[0],
                                           x["net_params.delay_k"].iloc[0])):
        g = g.sort_values(["created_at", "wandb_id"])
        arm = g["arm"].iloc[0]
        delay = int(g["net_params.delay_k"].iloc[0])
        seed = f"i{int(g['seed'].iloc[0])}/p{int(g['config.seed'].iloc[0])}"
        lines += ["=" * 78, f"{arm}  delay {delay}  seed {seed}", "=" * 78,
                  f"  {'id':<10} {'kept':<5} {'launched':<11} {'gpu':<38} "
                  f"{'reward':>8} {'to900':>9} {'dwell':>8}"]
        rows = []
        for _, r in g.iterrows():
            o = _readouts(store, r["wandb_id"])
            rows.append(o)
            to900 = o.get(f"steps_to_{E.HEADLINE_THRESHOLD}")
            lines.append(
                f"  {r['wandb_id']:<10} {'yes' if r['wandb_id'] in kept else 'NO':<5} "
                f"{str(r['created_at'])[:10]:<11} "
                f"{str(r['gpu']).replace('NVIDIA ', ''):<38} "
                f"{o.get('reward_final', float('nan')):>8.1f} "
                + (f"{to900 / 1e6:>8.1f}e6" if to900 else f"{'censored':>9}")
                + f" {o.get('band_dwell', 0) / 1e6:>7.0f}e6")
        if len(rows) == 2 and all("reward_final" in r for r in rows):
            dr = abs(rows[0]["reward_final"] - rows[1]["reward_final"])
            d_reward.append(dr)
            lines.append(f"  -> |delta reward| = {dr:.1f}")
            a, b = (r.get(f"steps_to_{E.HEADLINE_THRESHOLD}") for r in rows)
            if a and b:
                d_steps.append(abs(a - b) / 1e6)
                stalled = [r["band_dwell"] > E.STALL_DWELL_STEPS for r in rows]
                same_mode.append(stalled[0] == stalled[1])
                lines.append(
                    f"  -> |delta steps_to_{E.HEADLINE_THRESHOLD}| = "
                    f"{abs(a - b) / 1e6:.1f}e6   both runs took the "
                    f"{'SAME' if stalled[0] == stalled[1] else 'OPPOSITE'} mode "
                    f"(report.md caveat 8)")
        lines.append("")

    if d_reward:
        lines += [
            "=" * 78,
            "End-to-end training nondeterminism at fixed seed (both seeds identical)",
            "=" * 78,
            f"  |delta reward_final| over {len(d_reward)} pairs: "
            f"median {np.median(d_reward):.1f}, max {max(d_reward):.1f} "
            f"(on rewards of ~9.7e2, i.e. {max(d_reward) / 970 * 100:.2f} % at worst)",
        ]
        if d_steps:
            lines.append(
                f"  |delta steps_to_{E.HEADLINE_THRESHOLD}| over {len(d_steps)} pairs: "
                f"median {np.median(d_steps):.1f}e6, max {max(d_steps):.1f}e6")
            lines.append(f"  pairs taking the same learning mode: "
                         f"{sum(same_mode)}/{len(same_mode)}")
        lines += [
            "",
            "  This is the floor. No arm difference in report.md that is smaller than the",
            "  reward figure above means anything, however many seeds are averaged -- and",
            "  the steps figure is the floor for the *time* readout, which is looser",
            "  because it is defined inside a bimodal gap. Both are measured here on the",
            "  same task, budget, code and hardware pool as every number in the report,",
            "  which is what makes them usable as a floor rather than an analogy.",
            "=" * 78,
        ]
    else:
        lines += ["=" * 78, "No duplicate groups: nothing was discarded.", "=" * 78]

    text = "\n".join([__doc__.split("Reads the committed index")[0].rstrip(), "",
                      "=" * 78, "", *lines, ""])
    (HERE / "duplicates.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
