"""Did any servo run in this cohort diverge?

``envs/servo_control.md`` §3.6: a stiffer servo raises the divergence rate, and on the
training env ``NaNGuardWrapper`` turns a divergence into a silent zero-reward termination,
so a diverging stiffness can read as "bad at the task". The new sweep goes to
``servo_kp = 512`` -- 8x stiffer than anything ``../walker-servo-x-forward-model/``
cleared, with the ankle servo settling in 6.2 ms against a 2.5 ms ``sim_dt``.

Signal (as in that folder's ``divergence_check.py``): WalkerWalk never terminates on its
own, so ``rollout_batch/done_rate > rollout_batch/truncation_rate`` at any logged
iteration means a divergence. Queried from WandB directly because those keys are not in
the pinned ``history`` spec. Torque runs are not re-checked: every torque run here at
1e9 is already in the sibling's check, and torque has no stiffness to destabilise.

Run it (needs network + WandB auth; ~1 s per run)
-------------------------------------------------
    ../.venv/bin/python analysis/dm_control_suite/walker-servo-kp-1b/divergence_check.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-delays"
KEYS = ["rollout_batch/done_rate", "rollout_batch/truncation_rate"]
SAMPLES = 2000
TOL = 1.0 / (8192 * 30) / 10   # a tenth of one terminated step in a rollout batch


def main() -> None:
    import wandb

    runs = pd.read_csv(HERE / "data.csv")
    runs = runs[runs.condition == "servo"]
    api = wandb.Api(timeout=120)
    lines, bad = [], []
    for _, r in runs.sort_values(["servo_kp", "delay", "budget", "seed_pair"]).iterrows():
        frame = api.run(f"{PROJECT}/{r['wandb_id']}").history(
            keys=KEYS, samples=SAMPLES, pandas=True)
        if frame.empty:
            lines.append(f"  {r['wandb_id']}  kp={r['servo_kp']:<6g} delay={r['delay']:<3} "
                         f"*** keys not logged ***")
            bad.append((r["wandb_id"], "no data"))
            continue
        excess = frame[KEYS[0]] - frame[KEYS[1]]
        n_over = int((excess > TOL).sum())
        if n_over:
            bad.append((r["wandb_id"], n_over))
        lines.append(
            f"  {r['wandb_id']}  kp={r['servo_kp']:<6g} delay={r['delay']:<3} "
            f"{r['budget'] / 1e6:5.0f}M {r['seed_pair']:<9} iters={len(frame):<5} "
            f"max(done-trunc)={float(excess.max()):+.3e}  "
            f"{'OK' if not n_over else f'*** {n_over} ITERATIONS WITH TERMINATIONS ***'}")
    verdict = ("INERT: no servo run ever terminated an episode, up to servo_kp = 512. The\n"
               "stiff end of the sweep did not destabilise the physics, so no stiffness's\n"
               "reward is depressed by silent zero-reward terminations." if not bad else
               "*** NOT INERT: " + ", ".join(f"{i} ({n})" for i, n in bad))
    text = "\n".join([__doc__.split("Run it")[0].rstrip(), "", "=" * 78, *lines, "",
                      "=" * 78, verdict, "=" * 78, ""])
    (HERE / "divergence_check.txt").write_text(text)
    print(text)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
