"""Preemption-safe PPO training for the delays architectures.

Same runs as ``train_rodent.py``, but able to survive being killed and requeued
arbitrarily often, which is what the cluster's ``gpu_requeue`` partition does to
you in exchange for being nearly free. Use ``train_rodent.py`` on the dedicated
partitions; use this one where jobs get preempted.

Every attempt produces the *same* WandB run and the *same* checkpoint directory,
so a 600M-step run that was preempted four times still shows up as one run with
one continuous curve -- seamlessly, because the save triggered by the preemption
signal records the exact step it stopped on, which is also the last step WandB
saw, so the resumed attempt's first log continues straight on from it. The three
moving parts:

* **Identity.** The run directory is ``checkpoints/{stem}-{job token}``, where the
  token comes from the Slurm job id -- preserved across requeue. No timestamp: it
  could not be stable across attempts. ``run_state.json`` in that directory
  carries the WandB id and the attempt history.

* **Light checkpoints.** ``include_env_state=False``: weights, optimizer,
  ``rng_key`` and ``steps_taken``, ~43 MB instead of ~750 MB, about a second to
  write, which is what makes saving inside a short preemption grace period
  realistic. The offline eval path only ever reads ``networks/``, so nothing
  downstream notices.

* **Phase-spread resume.** A light checkpoint has no env states, so they are
  redrawn -- but resetting 4096 envs at once would *synchronise* the population:
  with the run's ``start_frame_range`` of [0, 44) every episode would start within
  88 of its ~480 env steps of every other, and agents good enough to reach the
  end of a clip would then keep restarting together. Instead each env is reset
  with an explicit ``start_frame`` drawn over the whole valid range, which makes
  the remaining-episode-time distribution uniform -- the steady state of a
  surviving population, and closer to it than the default reset is.

Run as::

    python -m vnl_experiments.delays.train_rodent_requeue --delay 5

with the same flags ``train_rodent.py`` accepts (see ``slurm_rodent_requeue.sh``
for how it is launched and requeued). Re-running the identical command outside
Slurm resumes only if you pass the same ``--run-name``, since there is no job id
to key on.

Exit codes: 0 = training finished (or was already finished), 42 = interrupted and
saved, please requeue, anything else = a real failure that should not be retried.
"""

import os


os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import dataclasses
import gc
import json
import signal
import sys
from pathlib import Path

import jax
import wandb
from flax import nnx

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.algorithms.checkpointing import (
    latest_checkpoint,
    load_checkpoint,
    make_checkpoint_fn,
)

from vnl_experiments import requeue
from vnl_experiments.delays import evaluation, train_rodent
from vnl_experiments.delays.network_builders import ARCHITECTURES
from vnl_experiments.envs.absolute_imitation import AbsoluteImitation

#: Default cap on attempts per run. Preemption is normal and cheap, but a run
#: that has been requeued this many times is more likely stuck than unlucky.
DEFAULT_MAX_ATTEMPTS = 30


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--network", default=train_rodent.DEFAULT_NETWORK,
                   choices=sorted(ARCHITECTURES),
                   help=f"Architecture to train (default: {train_rodent.DEFAULT_NETWORK}).")
    p.add_argument("--net-config", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a net-config key of the chosen architecture. "
                        "Repeatable. Lists accept JSON or comma-separated values.")
    p.add_argument("--list-networks", action="store_true",
                   help="Print the registered architectures and their defaults, then exit.")
    train_rodent.add_common_args(p)
    p.add_argument("--run-name", default=None,
                   help="Directory name under checkpoints/ for this run. Defaults "
                        "to {arch}_delay..._eff...-{slurm job id}. Pass it to make "
                        "a resubmitted (not requeued) job continue an existing run.")
    p.add_argument("--checkpoint-root", default="checkpoints",
                   help="Where run directories live (default: checkpoints).")
    p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                   help=f"Refuse to start a further attempt beyond this many "
                        f"(default: {DEFAULT_MAX_ATTEMPTS}).")
    p.add_argument("--jax-cache-dir", default=os.environ.get("JAX_COMPILATION_CACHE_DIR"),
                   help="Enable JAX's persistent compilation cache here, so each "
                        "attempt does not re-compile from scratch. Defaults to "
                        "$JAX_COMPILATION_CACHE_DIR; unset means no cache.")
    p.add_argument("--full-checkpoints", action="store_true",
                   help="Pickle the env states too (~750 MB per checkpoint, seconds "
                        "to write). Only useful if you would rather risk an "
                        "unfinished save than redraw the env states on resume.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def spread_env_states(train_env, nets, n_envs: int, key):
    """Fresh env and carry states, with episode phases spread over the clip.

    ``start_frame`` is passed explicitly, so the env's own
    ``config.start_frame_range`` (only the first 44 of 250 mocap frames for these
    runs) is bypassed for *this* reset and used as normal for every reset during
    the rollout afterwards. Drawing it over the whole valid range makes the time
    each env has left in its episode uniform, so the population does not march in
    lockstep after a resume.

    ``_last_valid_frame`` is the env's own definition of the last frame an episode
    may start at (``clip_length - (reference_length-1)*reference_stride - 2``).
    Re-deriving the formula here instead would be one silent drift away from
    spreading over the wrong range, so we ask the env; if vnl-playground renames
    it, this fails loudly at startup rather than quietly mis-resetting.
    """
    reset_key, frame_key = jax.random.split(key)
    last_valid_frame = int(train_env._last_valid_frame())
    frames = jax.random.randint(frame_key, (n_envs,), 0, last_valid_frame + 1)
    env_states = nnx.vmap(
        lambda k, f: train_env.reset(k, start_frame=f)
    )(jax.random.split(reset_key, n_envs), frames)
    return env_states, nets.initialize_state(n_envs), last_valid_frame


def warn_on_config_drift(stored, rebuilt) -> None:
    """Compare the checkpoint's TrainConfig with the one this attempt rebuilt.

    The config is rebuilt from the CLI rather than restored, so that a resumed run
    is described by the code in front of you. That is only safe while the code has
    not changed under the run: editing ``make_train_config`` between attempts would
    otherwise silently change the algorithm mid-run.
    """
    if stored is None:
        return
    try:
        old, new = dataclasses.asdict(stored), dataclasses.asdict(rebuilt)
    except TypeError:
        return
    changed = [k for k in set(old) | set(new) if old.get(k) != new.get(k)]
    if changed:
        print("  WARNING: this attempt's TrainConfig differs from the one the "
              f"checkpoint was written with, in: {sorted(changed)}")
        print(f"    checkpoint: { {k: old.get(k) for k in sorted(changed)} }")
        print(f"    this run:   { {k: new.get(k) for k in sorted(changed)} }")


def restore(step_dir: str, setup, *, n_envs: int):
    """Rebuild a resumable TrainingState from a checkpoint.

    The template is built with ``n_envs=1``: the optimizer state does not depend
    on the number of envs, and a full-width template would allocate env states
    only to throw them away. Weights and optimizer state are restored in place
    into ``setup.nets``.
    """
    ppo_cfg = setup.config.ppo
    template = ppo.new_training_state(
        setup.train_env, setup.nets, n_envs=1, seed=setup.seed,
        learning_rate=ppo_cfg.learning_rate,
        gradient_clipping=ppo_cfg.gradient_clipping,
        weight_decay=ppo_cfg.weight_decay,
    )
    ckpt = load_checkpoint(step_dir, template.networks, template.optimizer)
    state = ckpt["training_state"]
    step = int(ckpt["step"])
    warn_on_config_drift(ckpt["config"], setup.config)

    if state.env_states is None or state.network_states is None:
        # A light checkpoint: redraw what it did not store. Note this is decided
        # by what is *in* the checkpoint, not by --full-checkpoints, so switching
        # that flag mid-run resumes fine either way.
        key = jax.random.fold_in(jax.random.key(setup.seed), step)
        env_states, network_states, last_frame = spread_env_states(
            setup.train_env, setup.nets, n_envs, key
        )
        print(f"  env states redrawn with start_frame ~ U[0, {last_frame}] "
              f"over {n_envs} envs")
        state = dataclasses.replace(
            state, env_states=env_states, network_states=network_states
        )
    else:
        print("  env states restored exactly from the checkpoint")

    return state, step


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace, **build_kwargs) -> int:
    """One attempt. Returns the process exit code.

    ``build_kwargs`` are forwarded to :func:`train_rodent.build_run`, so a thin
    wrapper can supply the same ``net_config_overrides`` / ``extra_wandb_config``
    / ``name_token`` its non-preemptible counterpart does (see
    ``train_rodent_forward_model.py``). Launched directly, a forward-model run
    sets those knobs through ``--net-config`` instead, which records them under
    ``net_params.*`` but not as the historical top-level WandB columns.
    """
    requeue.maybe_enable_jax_cache(args.jax_cache_dir)

    setup = train_rodent.build_run(args, **build_kwargs)

    token = requeue.run_token()
    suffix = f"-{args.exp_name_suffix}" if args.exp_name_suffix else ""
    exp_name = args.run_name or f"{setup.name_stem}-{token}{suffix}"
    run_dir = Path(args.checkpoint_root) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    state = requeue.RunState.load(run_dir)
    attempt = 1 if state is None else len(state.attempts) + 1
    print(f"=== {exp_name} | attempt {attempt} "
          f"(slurm restart count {requeue.restart_count()}) ===")
    print(f"    run dir: {run_dir}")

    if state is not None and state.done:
        print("Run already finished (run_state.json says done). Nothing to do.")
        return 0
    if state is not None and attempt > args.max_attempts:
        print(f"ERROR: {len(state.attempts)} attempts already made, "
              f"--max-attempts is {args.max_attempts}. Refusing to continue; "
              f"raise the cap if this run is merely unlucky.")
        return 1

    # config.json is written once, by the first attempt: it records what the run
    # *started* with, which is what the offline eval path reconstructs the env
    # from. Later attempts must not rewrite it.
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(setup.config_json(), indent=2, default=str))

    if state is None:
        state = requeue.RunState(wandb_id=wandb.util.generate_id(), exp_name=exp_name)
        state.save(run_dir)

    # Resume from the newest complete checkpoint, if there is one.
    step_dir = latest_checkpoint(str(run_dir))
    initial_state = None
    resumed_from_step = None
    total_steps = args.total_steps or setup.config.ppo.total_steps

    if step_dir is not None:
        print(f"  restoring {step_dir}")
        initial_state, resumed_from_step = restore(
            step_dir, setup, n_envs=setup.config.ppo.n_envs
        )
        print(f"  resumed at step {resumed_from_step} of {total_steps}")
    else:
        print("  no checkpoint found; starting from scratch")

    state.record_attempt(run_dir, resumed_from_step=resumed_from_step)

    notes = getattr(args, "notes", train_rodent.DEFAULT_NOTES)
    if resumed_from_step is not None:
        notes = (f"{notes}\nRequeue-safe run: attempt {attempt}, resumed from "
                 f"step {resumed_from_step} ({step_dir}).")
    requeue.init_wandb_resumable(
        state,
        project=args.wandb_project,
        config={**setup.wandb_config,
                **requeue.wandb_requeue_config(state, run_dir, resumed_from_step,
                                               attempt)},
        tags=(*setup.tags, "requeue"),
        notes=notes,
        # Not "resumed_from_step is not None": a run whose checkpoints were
        # deleted starts over, but its WandB run already exists and must be
        # reopened rather than initialised fresh under the same id.
        first_attempt=attempt == 1,
        resume_from_step=resumed_from_step,
        # Rewind only when the previous attempt logged past this checkpoint,
        # which means it was killed outright rather than saving on its way out.
        rewind=requeue.rewind_needed(state, resumed_from_step),
    )

    # Training already complete but the final eval never ran (preempted during
    # it, most likely). Skip straight to the eval.
    if resumed_from_step is not None and resumed_from_step >= total_steps:
        print(f"  step {resumed_from_step} >= total_steps {total_steps}; "
              f"skipping training and going to the final eval")
        del initial_state
        jax.clear_caches()
        gc.collect()
        return finish(args, setup, state, run_dir, exp_name, resumed_from_step)

    watcher = requeue.PreemptionWatcher(
        on_signal=lambda s: print(
            f"\n>>> caught {signal.Signals(s).name}: saving a checkpoint at the "
            f"next iteration boundary and exiting for requeue", flush=True)
    )

    result = ppo.train_ppo(
        setup.train_env,
        setup.nets,
        setup.config,
        log_fn=wandb.log,
        video_fn=wandb_video_fn(fps=50),
        checkpoint_fn=make_checkpoint_fn(
            str(run_dir), setup.config,
            include_env_state=args.full_checkpoints,
        ),
        eval_env=setup.eval_env,
        initial_state=initial_state,
        stop_fn=watcher,
        # A resumed attempt must not repeat the eval, video and checkpoint of the
        # step it restored from -- that is work the previous attempt already did
        # and logged, and on a frequently preempted run it would be most of what
        # the job does.
        initial_eval=initial_state is None,
        **({} if args.total_steps is None else {"total_steps": args.total_steps}),
    )
    watcher.restore()

    print(f"Attempt {attempt} ran {result.total_iterations} iterations, "
          f"now at step {result.total_steps} of {total_steps}")

    # "Is there work left?" rather than "did the watcher fire?". These differ in
    # exactly one case: the signal arriving during the iteration that crosses
    # total_steps. Training is then complete, and asking for a requeue would burn
    # a whole job -- rebuilding the env only to discover there is nothing to do --
    # before reaching the final eval. So ask about the work, not the signal.
    if result.total_steps < total_steps:
        state.finish_attempt(run_dir, "preempted",
                             stopped_at_step=int(result.total_steps),
                             signal=watcher.signal_name)
        # exit_code=1 leaves the WandB run marked as not-finished. That is the
        # honest state -- analyses select on state == "finished" (wandb_utils/
        # index.py), and a preempted attempt must not make an incomplete run look
        # complete to them. The last attempt finishes it normally.
        wandb.finish(exit_code=1)
        print(f"Saved at step {result.total_steps}; exiting {requeue.EXIT_PREEMPTED} "
              f"for requeue.")
        return requeue.EXIT_PREEMPTED

    if result.eval_history:
        print("Final eval reward: "
              f"{result.eval_history[-1].get('eval/episode_reward/mean', 'N/A')}")

    final_step = int(result.total_steps)
    # Release the training state (n_envs env states + optimizer moments) before
    # the eval allocates. `setup.nets` is the same object as
    # result.training_state.networks, so the trained weights survive.
    del result, initial_state
    jax.clear_caches()
    gc.collect()

    return finish(args, setup, state, run_dir, exp_name, final_step)


def finish(args, setup, state, run_dir, exp_name: str, step: int) -> int:
    """End-of-training eval, mark the run done, close WandB."""
    if args.final_eval:
        evaluation.run_final_eval(
            setup.nets, AbsoluteImitation, setup.env_config,
            ckpt_dir=run_dir,
            wandb_id=state.wandb_id, wandb_name=exp_name, step=step,
            net_params=setup.net_params,
            train_env=setup.train_env, train_clips=setup.train_clips,
            test_clips=setup.test_clips,
            seed=setup.seed, limit_clips=args.eval_limit_clips,
            summary_fn=wandb.run.summary.update,
        )

    state.done = True
    state.finish_attempt(run_dir, "finished", final_step=step)
    wandb.finish()
    print(f"Run complete at step {step}.")
    return 0


def main() -> None:
    # Slurm redirects stdout to a file, which makes it block-buffered, and a
    # preempted attempt's progress would then be whatever happened to be flushed.
    # Line buffering costs nothing here and keeps the log readable while the job
    # is still running.
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if args.list_networks:
        train_rodent.print_networks()
        return
    sys.exit(run(args))


if __name__ == "__main__":
    main()
