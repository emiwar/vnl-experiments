"""Did the one functional code difference inside this cohort ever change the physics?

The problem
-----------
``comparability.txt``'s DESIGN AXES section shows ``git_commit``,
``repos.nnx_ppo.commit`` and ``repos.vnl_experiments.dirty`` all varying, and the
variation is **not balanced across the arms**: the MLP arm spans four vnl-experiments
commits and the forward-model arm two, and at some delays commit and seed coincide, so
configuration alone cannot separate them. ``repos.vnl_experiments.dirty`` is ``True`` on
most runs, which per README §6 voids the commit hash outright -- so the hashes cannot be
the argument even where they agree.

Reading the diffs across the cohort's commits leaves exactly one change that could alter a
trajectory:

* ``f9c8960`` adds ``train.video.render_kwargs.camera`` to six env configs. Render path
  only -- it cannot reach the physics or the loss.
* nnx-ppo ``1725d20 -> 5314279`` adds non-finite *counters* plus a ``check_diagnostics``
  that raises. ``optimizer.update`` is called identically; the update math is untouched.
  (A run that completed under ``5314279`` therefore also *proves* it saw no non-finite
  reward, observation, action or gradient -- the check would have killed it.)
* ``540e356`` adds ``servo_kp``; ``b487eed`` adds requeue/``resume_reset``; ``7bedb5c``
  lowers the eval and video cadence. None of these touches the training step for a run
  with ``servo_kp = 0`` (``apply_servo`` returns before writing any field), and
  ``extract.py`` admits only such runs.
* ``9e216ae`` adds ``NaNGuardWrapper`` to the training env stack. **This one can change a
  trajectory** -- but only on a step where MJX has already diverged, which is what this
  script checks for.

The check
---------
``NaNGuardWrapper`` turns a diverged step into ``done=1`` with ``truncated`` left False --
a *termination*. dm_control_suite's WalkerWalk never terminates on its own, so under
``EpisodeWrapper`` every episode end in a healthy run is a **truncation**. Hence

    rollout_batch/done_rate > rollout_batch/truncation_rate  <=>  something terminated

and the only thing that can terminate is a divergence. The same signal works on the
pre-guard side, because the dm_control tasks already set
``done = isnan(qpos).any() | isnan(qvel).any()`` themselves -- the guard changed what the
*reward* does on that step, not whether ``done`` fires.

So: if ``done_rate == truncation_rate`` at every logged iteration of every run, no run in
this cohort ever diverged, the guard never fired, and every code stack in the cohort is
functionally identical **on these runs** -- which is a stronger statement than a clean
commit hash, and one that survives ``repos.vnl_experiments.dirty = True`` (README §6: a
dirty flag voids the hash, so the hash cannot be the argument).

It also covers the requeued runs for free: a resume redraws episode phases but never sets
``done``, so a resume that had somehow corrupted the population would show up here too.

These two keys are not in the pinned ``history`` spec, and adding them would change
``HISTORY_SPEC_ID`` and unpin the 4.8e8 curves this folder shares with
``../explicit-forward-model/``. So this script queries WandB directly and commits its
verdict as ``nan_guard_inert.txt``. Same role as
``rodent/efference-copy-vs-proprioception/code_identity.py``: a one-off comparability
check whose *output* is the committed artifact.

Run it (needs network + WandB auth; ~1 s per run)
-------------------------------------------------
    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/nan_guard_inert.py
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
#: non-zero difference a single terminated step can produce is 1/245760 = 4.07e-6. Compare
#: against a tenth of that: anything larger is a real termination, anything smaller is
#: float noise in WandB's own sampling.
TOL = 1.0 / (8192 * 30) / 10


def main() -> None:
    import wandb

    # data.csv rather than runs.csv: the frozen selection carries only ids, and this
    # report is only legible with the delay / seed / commit of each run beside it.
    runs = pd.read_csv(HERE / "data.csv")
    api = wandb.Api(timeout=120)
    lines, bad = [], []

    for _, r in runs.sort_values(["budget", "condition", "delay", "seed"]).iterrows():
        run = api.run(f"{PROJECT}/{r['wandb_id']}")
        frame = run.history(keys=KEYS, samples=SAMPLES, pandas=True)
        excess = (frame["rollout_batch/done_rate"]
                  - frame["rollout_batch/truncation_rate"])
        worst = float(excess.max())
        n_over = int((excess > TOL).sum())
        ok = n_over == 0
        if not ok:
            bad.append(r["wandb_id"])
        lines.append(
            f"  {r['wandb_id']}  {r['condition']:<19} delay={int(r['delay']):<3} "
            f"seed={int(r['seed'])} budget={int(r['budget']) // 10 ** 6:>4}M  "
            f"{str(r['git_commit'])[:7]:<8} "
            f"iters={len(frame):<5} max(done-trunc)={worst:+.3e}  "
            f"{'OK' if ok else f'*** {n_over} ITERATIONS WITH TERMINATIONS ***'}")

    verdict = ("INERT: no run in this cohort ever terminated an episode, so MJX never "
               "diverged, so NaNGuardWrapper never fired. Every code stack in this\n"
               "cohort is functionally identical on these runs, and the varying "
               "git_commit / nnx_ppo.commit / dirty flags in comparability.txt do not\n"
               "threaten the arm contrast."
               if not bad else
               "*** NOT INERT: " + ", ".join(bad) + " terminated at least one episode, "
               "so a divergence occurred and the pre-/post-guard runs took different\n"
               "paths on those steps. The commit split in comparability.txt is a real "
               "confound and must be treated as one in report.md. ***")

    text = "\n".join([
        __doc__.split("Run it")[0].rstrip(),
        "",
        "=" * 78,
        f"Per run: max over logged iterations of (done_rate - truncation_rate).",
        f"Tolerance {TOL:.3e} = a tenth of one terminated step in a "
        f"{8192 * 30}-transition rollout batch.",
        "=" * 78,
        *lines,
        "",
        "=" * 78,
        verdict,
        "=" * 78,
        "",
    ])
    (HERE / "nan_guard_inert.txt").write_text(text)
    print(text)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
