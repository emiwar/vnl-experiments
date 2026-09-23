"""Did any run in this 2x2 ever diverge -- and does the torque half's commit spread matter?

Two problems, one signal.

**The code spread on the torque side.** ``comparability.txt``'s DESIGN AXES section shows
``git_commit`` (three values), ``repos.nnx_ppo.commit`` (two) and
``repos.vnl_experiments.dirty`` (True on 10 runs) all varying -- and *only* on the torque
side. Both servo arms are a single clean commit. A dirty flag voids the hash outright
(README §6), so the hashes cannot be the comparability argument for the torque half.
Reading the diffs across those commits leaves exactly one change that could alter a
trajectory, and ``../walker-forward-model-1b/nan_guard_inert.py`` already enumerated them
for most of these same runs:

* ``f9c8960`` adds ``train.video.render_kwargs.camera``. Render path only.
* nnx-ppo ``1725d20 -> 5314279`` adds non-finite *counters* plus a ``check_diagnostics``
  that raises. ``optimizer.update`` is called identically.
* ``b487eed`` adds requeue / ``resume_reset``; ``7bedb5c`` lowers the eval and video
  cadence. Neither touches the training step.
* ``9e216ae`` adds ``NaNGuardWrapper`` to the *training* env stack. **This one can change
  a trajectory** -- but only on a step where MJX has already diverged.

**The servo side's own risk.** ``envs/servo_control.md`` §3.6 is explicit that a stiff
servo raises the divergence rate, and that the two sides of the run fail differently: on
the training env ``NaNGuardWrapper`` turns a divergence into a silent zero-reward
termination, so a diverging arm can read as "bad at the task"; on the eval env there is no
guard at all. ``servo_kp = 64`` is the stiff end of the sweep, so "check the divergence
rate per arm" is part of that document's stated protocol and not an optional extra. This
is the first analysis where it can be checked on a servo cohort at full budget.

The signal
----------
``NaNGuardWrapper`` turns a diverged step into ``done=1`` with ``truncated`` left False --
a *termination*. dm_control_suite's WalkerWalk never terminates on its own, so under
``EpisodeWrapper`` every episode end in a healthy run is a **truncation**. Hence

    rollout_batch/done_rate > rollout_batch/truncation_rate  <=>  something terminated

and the only thing that can terminate is a divergence. The same test works on the
pre-guard runs, because the dm_control tasks already set
``done = isnan(qpos).any() | isnan(qvel).any()`` themselves -- the guard changed what the
*reward* does on that step, not whether ``done`` fires.

So if ``done_rate == truncation_rate`` at every logged iteration of every run: no run
diverged, the guard never fired, every code stack in the cohort is functionally identical
**on these runs** (which survives ``dirty = True`` in a way a hash does not), *and* the
stiff servo did not destabilise the physics. One check, both jobs.

It also covers the twelve requeued runs for free: a resume redraws episode phases but
never sets ``done``, so a resume that had corrupted the population would show here too.

These two keys are not in the pinned ``history`` spec, and adding them would change
``HISTORY_SPEC_ID`` and unpin the curves this folder shares with both siblings. So this
script queries WandB directly and commits its verdict as ``divergence_check.txt``.

Run it (needs network + WandB auth; ~1 s per run)
-------------------------------------------------
    ../.venv/bin/python analysis/dm_control_suite/walker-servo-x-forward-model/divergence_check.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-delays"

#: Sampled history, never `scan_history` (README §6). 2000 samples covers the ~1667
#: logged eval iterations of a 1e9-step run.
KEYS = ["rollout_batch/done_rate", "rollout_batch/truncation_rate"]
SAMPLES = 2000

#: An episode end is logged as a rate over 8192 envs x 30 rollout steps, so the smallest
#: non-zero difference a single terminated step can produce is 1/245760 = 4.07e-6.
#: Compare against a tenth of that: anything larger is a real termination, anything
#: smaller is float noise in WandB's own sampling.
TOL = 1.0 / (8192 * 30) / 10


def main() -> None:
    import wandb

    # data.csv rather than runs.csv: the frozen selection carries only ids, and this
    # report is only legible with the arm / delay / seed / commit beside each one.
    runs = pd.read_csv(HERE / "data.csv")
    api = wandb.Api(timeout=120)
    lines, bad = [], []

    for _, r in runs.sort_values(["condition", "delay", "seed_pair"]).iterrows():
        run = api.run(f"{PROJECT}/{r['wandb_id']}")
        frame = run.history(keys=KEYS, samples=SAMPLES, pandas=True)
        excess = (frame["rollout_batch/done_rate"]
                  - frame["rollout_batch/truncation_rate"])
        worst = float(excess.max())
        n_over = int((excess > TOL).sum())
        if n_over:
            bad.append((r["wandb_id"], r["condition"], n_over))
        lines.append(
            f"  {r['wandb_id']}  {r['condition']:<10} delay={int(r['delay']):<3} "
            f"{r['seed_pair']:<11} {str(r['git_commit'])[:7]:<8} "
            f"dirty={str(r['vnl_experiments_dirty']):<5} "
            f"iters={len(frame):<5} max(done-trunc)={worst:+.3e}  "
            f"{'OK' if not n_over else f'*** {n_over} ITERATIONS WITH TERMINATIONS ***'}")

    by_arm = runs.groupby("condition").size()
    verdict = (
        "INERT: no run in this 2x2 ever terminated an episode.\n"
        "  (a) MJX never diverged, so NaNGuardWrapper never fired, so the varying\n"
        "      git_commit / nnx_ppo.commit / dirty flags on the torque side do not\n"
        "      threaten any contrast in this folder.\n"
        "  (b) servo_kp = 64 did not destabilise the physics at either network arm, so\n"
        "      servo_control.md §3.6's divergence caveat is discharged for this cohort\n"
        "      and no arm's reward is depressed by silent zero-reward terminations."
        if not bad else
        "*** NOT INERT: " + ", ".join(f"{i} ({c}, {n} iters)" for i, c, n in bad)
        + "\n  A divergence occurred. Two consequences, and both must be carried into\n"
          "  report.md: the pre-/post-guard runs took different paths on those steps, so\n"
          "  the torque half's commit split is a real confound; and a terminated episode\n"
          "  scores zero, so the affected arm's reward is depressed for a reason that is\n"
          "  not the manipulation. Check whether the affected runs are concentrated in\n"
          "  the servo arms (servo_control.md §3.6) before reading any arm difference.")

    text = "\n".join([
        __doc__.split("Run it")[0].rstrip(),
        "",
        "=" * 78,
        "Per run: max over logged iterations of (done_rate - truncation_rate).",
        f"Tolerance {TOL:.3e} = a tenth of one terminated step in a "
        f"{8192 * 30}-transition rollout batch.",
        f"Cohort: {len(runs)} runs -- "
        + ", ".join(f"{k} {v}" for k, v in by_arm.items()),
        "=" * 78,
        *lines,
        "",
        "=" * 78,
        verdict,
        "=" * 78,
        "",
    ])
    (HERE / "divergence_check.txt").write_text(text)
    print(text)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
