"""Machinery for training runs that survive being preempted and requeued.

The cluster's ``gpu_requeue`` partition runs on idle dedicated nodes: it is
nearly free, but a job is killed and requeued whenever the node's owner wants it
back, and Slurm then re-runs the *same* sbatch script from the top. A run that is
to survive that needs four things, none of which is specific to an architecture:

1. **An identity that outlives the process.** :func:`run_token` derives it from
   the Slurm job id, which is preserved across a requeue (``SLURM_RESTART_COUNT``
   is what increments). Attempt N therefore lands in the same run directory as
   attempt 1 with no bookkeeping and no coordination.

2. **A record of what the run is.** :class:`RunState` (``run_state.json`` in the
   run directory) holds the WandB id, the attempt history and whether training
   finished. It is written atomically: losing it loses the run's WandB identity,
   which is not recoverable from anything else on disk.

3. **A way to notice the axe.** :class:`PreemptionWatcher` catches the signals
   Slurm sends and exposes itself as ``train_ppo``'s ``stop_fn``, so the training
   loop saves a checkpoint at the next iteration boundary and returns instead of
   dying with up to a full checkpoint interval of progress unsaved.

4. **One WandB run, not N.** :func:`init_wandb_resumable` reopens the stored run
   id, so every attempt appends to a single curve. This is seamless because a
   graceful save records the exact step it stopped on, which is also the run's
   last logged step; only a hard kill leaves logged steps ahead of the newest
   checkpoint, and then a rewind (if the account has that WandB feature) rolls
   them back -- see :func:`rewind_needed`.

Nothing here imports jax or an environment, so it stays cheap to import and easy
to test.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import socket
import time
from pathlib import Path
from typing import Any, Callable, Optional

#: Exit code meaning "training was interrupted, state is saved, please run me
#: again". The sbatch wrapper requeues on exactly this code and lets every other
#: non-zero code fail the job, so a genuine crash cannot loop forever.
EXIT_PREEMPTED = 42

#: Signals that mean "you are about to be killed". Slurm sends SIGTERM when it
#: preempts a job (after the partition's GraceTime, SIGKILL follows), and SIGUSR1
#: when asked to with ``--signal=B:USR1@<seconds>`` before the time limit.
PREEMPTION_SIGNALS = (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)

RUN_STATE_FILE = "run_state.json"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def run_token() -> str:
    """A token identifying this job, stable across requeues of the same job.

    Slurm keeps the job id when it requeues a job, and keeps both
    ``SLURM_ARRAY_JOB_ID`` and ``SLURM_ARRAY_TASK_ID`` when it requeues an array
    task, so either is a usable key. A *resubmitted* job gets a new id and so
    starts a new run, which is the intended behaviour: pass ``--run-name`` when
    you want a new submission to continue an existing run.

    Off-cluster there is nothing to be stable against, so a timestamp is used and
    each invocation is its own run.
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"job{array_job}_{array_task}"
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if job:
        return f"job{job}"
    return time.strftime("local-%Y%m%d-%H%M%S")


def restart_count() -> int:
    """How many times Slurm has requeued this job (0 on the first attempt)."""
    try:
        return int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Persistent run state
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunState:
    """What attempt N needs to know about attempts 1..N-1.

    Lives at ``{run_dir}/run_state.json`` next to ``config.json`` and the
    ``step_*`` checkpoints, so the whole run is one self-describing directory.
    """

    wandb_id: str
    exp_name: str
    #: One entry per attempt: restart count, job id, host, the step it resumed
    #: from and how it ended. Provenance for "why does this curve have a seam".
    attempts: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    #: Set once training reached total_steps *and* the final eval finished. A
    #: stray requeue after that must not start training again.
    done: bool = False

    @classmethod
    def path(cls, run_dir) -> Path:
        return Path(run_dir) / RUN_STATE_FILE

    @classmethod
    def load(cls, run_dir) -> Optional["RunState"]:
        """Read the state, or None when this is the first attempt.

        A truncated file is treated as absent rather than fatal: an attempt that
        died mid-write has nothing worth resuming into anyway, and refusing to
        start would strand the run.
        """
        path = cls.path(run_dir)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, run_dir) -> None:
        """Write atomically -- a half-written state file loses the WandB id."""
        path = self.path(run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        os.replace(tmp, path)

    def record_attempt(self, run_dir, *, resumed_from_step: Optional[int],
                       outcome: str = "started", **extra) -> dict[str, Any]:
        """Append an attempt and persist. Returns the appended entry."""
        entry = {
            "attempt": len(self.attempts) + 1,
            "restart_count": restart_count(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "host": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "resumed_from_step": resumed_from_step,
            "outcome": outcome,
            **extra,
        }
        self.attempts.append(entry)
        self.save(run_dir)
        return entry

    def finish_attempt(self, run_dir, outcome: str, **extra) -> None:
        """Update the last attempt's outcome in place."""
        if self.attempts:
            self.attempts[-1].update(outcome=outcome,
                                     ended_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     **extra)
        self.save(run_dir)


# ---------------------------------------------------------------------------
# Preemption
# ---------------------------------------------------------------------------

class PreemptionWatcher:
    """Turns a preemption signal into a clean ``train_ppo`` early stop.

    Usable directly as ``stop_fn``: the handler only sets a flag, and the
    training loop reads it once per iteration. Python runs signal handlers
    between bytecodes, so with ~1.5 s iterations the save starts within a couple
    of seconds of the signal -- comfortable inside a short grace period, given a
    light checkpoint takes about a second to write.

    Signalling a second time restores the original handlers, so a repeated Ctrl-C
    or SIGTERM is not swallowed while the current iteration finishes.
    """

    def __init__(self, signals=PREEMPTION_SIGNALS,
                 on_signal: Optional[Callable[[int], None]] = None):
        self.triggered = False
        self.signum: Optional[int] = None
        self._on_signal = on_signal
        self._previous: dict[int, Any] = {}
        for sig in signals:
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (OSError, ValueError):
                # Not the main thread, or the signal does not exist here.
                pass

    def _handle(self, signum, _frame) -> None:
        if self.triggered:
            # Signalled twice: whoever is doing it means it. Hand the signals
            # back to their original handlers so the next one takes effect
            # immediately rather than waiting for the iteration to finish.
            self.restore()
            return
        self.triggered = True
        self.signum = signum
        if self._on_signal is not None:
            self._on_signal(signum)

    def __call__(self, steps: int) -> bool:
        return self.triggered

    @property
    def signal_name(self) -> str:
        return signal.Signals(self.signum).name if self.signum else ""

    def restore(self) -> None:
        """Put the original handlers back (so a later Ctrl-C behaves normally)."""
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass


# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------

def last_logged_step(state: RunState) -> Optional[int]:
    """Last step a *previous* attempt is known to have reached, or None.

    None means no previous attempt recorded an ending -- it was killed outright
    (SIGKILL, node failure) rather than saving on its way out.
    """
    for a in reversed(state.attempts):
        for key in ("stopped_at_step", "final_step"):
            if a.get(key) is not None:
                return int(a[key])
    return None


def rewind_needed(state: RunState, resume_from_step: Optional[int]) -> bool:
    """Whether WandB's history runs past the checkpoint we are resuming from.

    A *graceful* preemption checkpoints at the exact step it stopped on, so the
    resume point already is the run's last logged step and the history continues
    seamlessly -- no rewind, nothing dropped. Only a hard kill leaves the last
    periodic checkpoint behind the last log, and only then is there history after
    the restored weights that ought to be rolled back.
    """
    if resume_from_step is None:
        return False
    last = last_logged_step(state)
    return last is None or last > resume_from_step


def init_wandb_resumable(
    state: RunState,
    *,
    project: str,
    config: dict,
    tags,
    notes: str,
    first_attempt: bool,
    resume_from_step: Optional[int] = None,
    rewind: bool = False,
    verbose: bool = True,
):
    """Open (or reopen) this run's single WandB run.

    First attempt: an ordinary ``wandb.init`` under the id stored in
    ``state.wandb_id``.

    Later attempts: the same id with ``resume="allow"``, so every attempt appends
    to one run and one curve. This works because every script logs
    ``wandb.log(metrics, step=env steps)`` -- WandB's ``_step`` *is* the env-step
    count the checkpoint directory is named after, and a resumed attempt
    continues from there. ``"allow"`` rather than ``"must"`` because a later
    attempt is not proof the run exists: an attempt can die between writing
    ``run_state.json`` and reaching ``wandb.init``, and a run whose checkpoints
    were deleted restarts from scratch under the id it already has.

    ``rewind`` additionally passes ``resume_from="{id}?_step={step}"``, which
    rolls the history back to the checkpoint being restored. That is only worth
    doing when the run logged past the checkpoint -- see :func:`rewind_needed` --
    because WandB drops any ``log`` whose step is not greater than the last one
    it saw, so those replayed steps would otherwise be missing from the curve.

    Rewind is a private-preview WandB feature (as of wandb 0.25 the server
    answers ``400 Rewind is in private preview`` unless it has been enabled for
    the account), so a rejection is expected and falls back to a plain resume: an
    unattended requeue loop must not die over a cosmetic gap.
    """
    import wandb

    common = dict(project=project, id=state.wandb_id, name=state.exp_name,
                  config=config, tags=tags, notes=notes)
    if first_attempt:
        return wandb.init(**common)

    if rewind and resume_from_step is not None:
        try:
            run = wandb.init(
                resume_from=f"{state.wandb_id}?_step={int(resume_from_step)}",
                **common,
            )
            if verbose:
                print(f"  wandb: rewound {state.wandb_id} to step {resume_from_step}")
            return run
        except Exception as exc:  # noqa: BLE001 -- see docstring
            print(f"  wandb: could not rewind to step {resume_from_step} ({exc}); "
                  f"resuming without it. Metrics logged after step "
                  f"{resume_from_step} by the attempt that was killed stay in the "
                  f"history, and the steps replayed since then will not be "
                  f"re-logged.")

    if verbose:
        where = ("from scratch" if resume_from_step is None
                 else f"at step {resume_from_step}")
        print(f"  wandb: resuming run {state.wandb_id} {where}")
    return wandb.init(resume="allow", **common)


def wandb_requeue_config(state: RunState, run_dir, resume_from_step,
                         attempt: int) -> dict:
    """Requeue provenance to merge into the WandB config on every attempt.

    ``attempt`` is passed in rather than derived from ``state.attempts``, whose
    length depends on whether :meth:`RunState.record_attempt` has run yet.
    """
    return {
        "requeue": {
            "attempt": attempt,
            "restart_count": restart_count(),
            "run_token": run_token(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "checkpoint_dir": str(run_dir),
            "resumed_from_step": resume_from_step,
            "resumed_steps": [a.get("resumed_from_step") for a in state.attempts
                              if a.get("resumed_from_step") is not None],
        }
    }


# ---------------------------------------------------------------------------
# JAX compilation cache
# ---------------------------------------------------------------------------

def maybe_enable_jax_cache(path: Optional[str], *, min_compile_secs: float = 1.0) -> None:
    """Enable JAX's persistent compilation cache at ``path`` (no-op if None).

    Every attempt otherwise re-compiles ``ppo_step``, the eval rollout and the
    render scan from scratch, which is minutes of GPU time per preemption and is
    paid again by every run in a sweep that shares the same shapes.

    Must be called before the first jit'd call. Set via ``jax.config`` rather
    than an env var so the flag holds regardless of import order.
    """
    if not path:
        return
    import jax

    os.makedirs(path, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_secs)
    print(f"JAX persistent compilation cache: {path}")
