"""Factor grid at 0.3 m / delay 5: one row per run, five true/false factors, two readouts.

    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/extract_factors.py
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/extract_factors.py --sync

A side table for hand-made plots, not part of ``extract.py``'s frozen selection (that
one fixes ``efference_length == delay``, ``servo_kp == 0``, ``DelayedMLP`` and the
privileged critic, i.e. it holds every factor here at its baseline). Locked:
``WalkerWalkTerrain``, ``terrain.height == 0.3``, ``delay_k == 5``,
``terrain.irregularity == 1``, no motor noise, no distillation, ``min_std == 0.001``.

Factors (parametric settings collapsed to true/false; the raw value is kept beside each):

* ``efference``: ``efference_length > 0`` (raw: ``efference_length``, 0 or 5). In
  ``DelayedMLP`` the action queue goes to the actor; in ``FlatForwardModel`` it goes to
  the predictor.
* ``forward_model``: ``network_class == "FlatForwardModel"`` (else ``DelayedMLP``).
* ``blind``: ``not terrain.observed``.
* ``privileged_critic``: ``FlatForwardModel`` has no such key; its critic always gets
  the undelayed observation (``make_forward_model_actor_critic``), so it is ``True``.
* ``servo``: ``servo_kp > 0`` (raw: ``servo_kp``).

Readouts, the same definitions as ``data.csv`` (functions imported from ``extract.py``):

* ``reward_480M``: mean eval reward over (430 M, 480 M]; NaN if the run has not got there.
* ``steps_to_{900,800,700,600}``: first eval step at or above that level, up to 480 M;
  NaN if not reached within ``min(steps_trained, 480 M)``. Read ``steps_trained`` before
  treating NaN as "never". (No run in this grid reaches 900 by 480 M, hence the lower
  levels.)

Every matching run is listed, including crashed or still-running ones (``complete``
False), and runs trained past 480 M (``budget``), so nothing is filtered silently. The
selection is read live from the run index; ``--sync`` refreshes it first. Writes
``factors.csv``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import index

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import (  # noqa: E402  (this folder's; same history spec and readouts)
    INVARIANTS, MIN_STD, PRE_EVAL_FIX_COMMIT, PROJECT, READ_AT, _completed, _first_at,
    _noise, _series, _window)

HERE = Path(__file__).resolve().parent
OUT = HERE / "factors.csv"
HEIGHT = 0.3
DELAY = 5
NETWORKS = ("DelayedMLP", "FlatForwardModel")
#: 900 was asked for; no run here reaches it by 480 M, so lower levels ride along.
LEVELS = (900, 800, 700, 600)
FACTORS = ["efference", "forward_model", "blind", "privileged_critic", "servo"]


def select(df: pd.DataFrame) -> pd.DataFrame:
    m = (df["env"].eq("WalkerWalkTerrain")
         & df["env_params.terrain.height"].eq(HEIGHT)
         & df["net_params.delay_k"].eq(DELAY)
         & df["env_params.terrain.irregularity"].eq(1.0)
         & df["net_params.network_class"].isin(NETWORKS)
         & df["net_params.min_std"].eq(MIN_STD)
         & _noise(df).eq(0.0)
         & df["distill.teacher"].isna()
         & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False))
    return df[m].copy()


def check_invariants(runs: pd.DataFrame) -> None:
    """Every non-factor setting must be single-valued across the grid."""
    varying = {c: sorted(map(str, runs[c].dropna().unique())) for c in INVARIANTS
               if c in runs.columns and runs[c].dropna().astype(str).nunique() > 1}
    if varying:
        raise SystemExit(f"non-factor settings vary across the grid: {varying}")


def build(runs: pd.DataFrame, store: Store) -> pd.DataFrame:
    rows = []
    for _, r in runs.iterrows():
        try:
            s = _series(store, r["wandb_id"])
        except KeyError:
            raise SystemExit(f"{r['wandb_id']}: no history artifact; run `python -m "
                             "vnl_experiments.artifacts ensure --kind history ...`")
        fm = r["net_params.network_class"] == "FlatForwardModel"
        eff = int(r["net_params.efference_length"])
        kp = float(r["env_params.servo_kp"] if pd.notna(r["env_params.servo_kp"]) else 0.0)
        priv = True if fm else bool(r["net_params.privileged_critic"])
        reward, n = _window(s, READ_AT)
        rows.append({
            "wandb_id": r["wandb_id"],
            "efference": eff > 0,
            "forward_model": fm,
            "blind": not bool(r["env_params.terrain.observed"]),
            "privileged_critic": priv,
            "servo": kp > 0,
            "seed_pair": f"i{int(r['seed'])}/p{int(r['config.seed'])}",
            "reward_480M": reward,
            **{f"steps_to_{lv}": _first_at(s, lv) for lv in LEVELS},
            "efference_length": eff,
            "servo_kp": kp,
            "network_class": r["net_params.network_class"],
            "state": r["state"],
            "complete": bool(_completed(r.to_frame().T).iloc[0]),
            "steps_trained": float(s.step.max()) if s is not None else np.nan,
            "budget": int(r["config.ppo.total_steps"]),
            "reward_n": n,
            "git_commit": r["git_commit"],
            "created_at": r["created_at"],
        })
    return (pd.DataFrame(rows)
            .sort_values([*FACTORS, "seed_pair", "wandb_id"],
                         ascending=[False, True, False, False, True, True, True])
            .reset_index(drop=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sync", action="store_true", help="refresh the WandB run index first")
    args = p.parse_args()
    if args.sync:
        index.sync(PROJECT)
    runs = select(index.load(PROJECT))
    check_invariants(runs)
    out = build(runs, Store())
    out.to_csv(OUT, index=False)

    n_cells = out[out.complete].groupby(FACTORS).ngroups
    print(out[[*FACTORS, "seed_pair", "reward_480M", *[f"steps_to_{lv}" for lv in LEVELS[1:]], "state",
               "steps_trained", "wandb_id"]].to_string())
    print(f"\nWrote {OUT} ({len(out)} runs, {out.complete.sum()} complete, "
          f"{n_cells} of {2 ** len(FACTORS)} cells with a complete run)")


if __name__ == "__main__":
    main()
