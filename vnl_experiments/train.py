"""The training entry point. Config is Hydra; see ``conf/train.yaml``.

    python -m vnl_experiments.train delay=5
    python -m vnl_experiments.train delay=5 net=recurrent net.rnn_cell=gru
    python -m vnl_experiments.train env.ctrl_dt=0.02 env.reward_terms.joints.weight=0.5
    python -m vnl_experiments.train train.ppo.n_envs=64 train.ppo.clip_range=0.3
    python -m vnl_experiments.train train=smoke final_eval=false

Every field of the env config, the network config and the PPO config is reachable this
way -- not just the handful that used to have a flag. Keys are validated against the real
schema, so a typo lists what is available instead of being silently ignored.

Two run modes share everything except how a run is named and what happens when it is
interrupted:

* the default, for the dedicated partitions: one timestamped run, full checkpoints;
* ``requeue.enabled=true``, for ``gpu_requeue``: the run directory is keyed on the Slurm
  job id so it is stable across preemptions, checkpoints omit the env states (94 % of the
  bytes) so a save fits in the grace period, and an interrupted attempt exits 42 asking
  the batch script to requeue it. See ``slurm_rodent_requeue.sh``.
"""

import os

# Before any JAX/MuJoCo import: the render path needs a headless GL backend, and the
# choice has to be made before the libraries look at it.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import dataclasses
import gc
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import jax
import wandb
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.algorithms.checkpointing import (
    latest_checkpoint,
    load_checkpoint,
    make_checkpoint_fn,
)
from nnx_ppo.algorithms.config import TrainConfig
from vnl_experiments import requeue as requeue_lib
from vnl_experiments.config import (
    OverrideError,
    build_env_config,
    build_net_config,
    build_train_config,
    validate_train_config,
)
from vnl_experiments.delays import evaluation
from vnl_experiments.delays.network_builders import (
    ARCHITECTURES,
    build_network,
    get_architecture,
)
from vnl_experiments.conf_schema import register as register_schemas
from vnl_experiments.envs import registry as env_registry
from vnl_experiments.provenance import repo_versions
from vnl_playground.tasks.reference_clips import ReferenceClips

# Must happen before @hydra.main composes: the group files refer to these nodes.
register_schemas()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunSetup:
    """Everything needed to start -- or resume -- one run.

    Built once by :func:`build_run` so the plain and preemption-safe paths construct the
    envs, the network, the ``config.json`` payload and the WandB config from the same
    code. The only thing left to the caller is the run's *name*, which the two disagree
    about: a plain run timestamps it, a requeued run needs one stable across attempts.
    """

    arch: Any
    env_config: Any
    net_params: dict
    nets: Any
    train_env: Any
    eval_env: Any
    train_clips: Any
    test_clips: Any
    config: TrainConfig
    ablations: tuple[str, ...]
    #: Run name up to (but excluding) the trailing timestamp or job token.
    name_stem: str
    wandb_config: dict
    tags: tuple[str, ...]
    seed: int
    env_spec: Any

    def config_json(self) -> dict:
        """The ``config.json`` payload the offline eval path reconstructs from.

        Shape is fixed by everything that reads it back -- the eval scripts, the artifact
        producers, `network_builders.load_network`. `net_params` in particular must stay
        a flat dict of JSON scalars.
        """
        return {
            "env_params": self.env_config.to_dict(),
            "net_params": self.net_params,
        }


def _task_overrides() -> list[str]:
    """The override strings this invocation was launched with, for the record.

    The resolved config is the authoritative account of what a run used, but it cannot
    say which values were *deliberately* changed. Keeping the raw overrides makes that
    legible, and the command line reproducible, without diffing against the defaults.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        return list(HydraConfig.get().overrides.task)
    except Exception:  # noqa: BLE001 - not launched through @hydra.main
        return []


def build_run(cfg: DictConfig) -> RunSetup:
    """Build the envs, network, configs and WandB metadata for one run."""
    arch = get_architecture(cfg.net_spec.architecture)
    if arch is None:
        raise OverrideError(
            f"Unknown net_spec.architecture {cfg.net_spec.architecture!r}. "
            f"Available: {sorted(ARCHITECTURES)}"
        )

    spec = env_registry.get(cfg.env_spec.task)
    if spec.obs_layout != arch.obs_layout:
        raise OverrideError(
            f"{cfg.env_spec.task} has a {spec.obs_layout!r} observation but "
            f"{arch.name} expects {arch.obs_layout!r}. Pick a network whose layout "
            f"matches: {sorted(a.name for a in ARCHITECTURES.values() if a.obs_layout == spec.obs_layout)}"
        )

    efference_length = cfg.efference if cfg.efference is not None else cfg.delay
    seed = cfg.seed

    env_config = build_env_config(spec.default_config, cfg.env)
    net_config = build_net_config(arch.defaults, cfg.net)
    config = validate_train_config(build_train_config(cfg.train))

    if spec.uses_clips:
        clips = ReferenceClips(env_config.reference_data_path,
                               env_config.clip_length,
                               env_config.keep_clips_idx)
        train_clips, test_clips = clips.split()
        train_env = spec.build(env_config, clips=train_clips)
        # In-training eval runs on the held-out clips. Until 2026-08-20 this line was
        # followed by `eval_env = train_env`, so every delays run's WandB `eval/*` series
        # actually measured train-split performance -- see the trap note in
        # analysis/README.md. Do not "simplify" this back.
        eval_env = spec.build(env_config, clips=test_clips)
    else:
        # A self-contained task has no held-out data to hold out, so train and eval see
        # the same env. They are still separate instances, for the same reason as above.
        train_clips = test_clips = None
        train_env = spec.build(env_config)
        eval_env = spec.build(env_config)

    # One source of truth: config.json (for offline reconstruction by the eval scripts),
    # the network built here, and the end-of-training eval's metadata all read this dict.
    net_params = {
        **net_config.to_dict(),
        "delay_k": cfg.delay,
        "efference_length": efference_length,
        "network_class": arch.name,
    }

    # The registry builder -- the same call the offline eval path makes, so the trained
    # architecture and its reconstruction are the same code.
    nets = build_network(net_params, train_env, nnx.Rngs(seed))
    if nets is None:
        raise OverrideError(f"build_network returned None for {arch.name!r}")

    # Decoder-input ablations get their own name token and tag. Without this a
    # no-intention run at eff == delay is indistinguishable from a standard-arch
    # efference baseline to the analyses that select on delay_k / efference_length /
    # hidden sizes alone. Empty when both flags are at their default, so existing run
    # names and tag sets are unchanged.
    ablations = tuple(
        token for key, token in (("dec_use_intention", "nointent"),
                                 ("dec_use_proprioception", "noproprio"))
        if not net_params.get(key, True)
    )

    overrides = _task_overrides()
    env_overridden = [o for o in overrides if o.startswith("env.")]

    return RunSetup(
        arch=arch,
        env_config=env_config,
        net_params=net_params,
        nets=nets,
        train_env=train_env,
        eval_env=eval_env,
        train_clips=train_clips,
        test_clips=test_clips,
        config=config,
        ablations=ablations,
        env_spec=spec,
        name_stem=(
            f"{arch.run_label(net_params)}_delay{cfg.delay}_eff{efference_length}"
            f"{''.join(f'_{t}' for t in ablations)}"
        ),
        wandb_config={
            # This payload's shape is load-bearing: the run index flattens it to dotted
            # columns (`env_params.walker_xml_path`, `config.ppo.total_steps`, ...) and
            # the committed analyses select on them. Adding keys is safe; renaming or
            # nesting existing ones silently breaks cohorts built from older runs.
            "env": cfg.env_spec.task,
            "delay_k": cfg.delay,
            "efference_length": efference_length,
            # Logged so the run index can group by architecture without having to parse
            # the run name.
            "network_class": arch.name,
            "seed": seed,
            # Git state of all three repos. WandB's own `git_commit` covers only
            # vnl-experiments; nnx-ppo (the algorithm) and vnl-playground (the task) were
            # previously unrecorded, and `dirty` flags a working copy that has drifted
            # from its commit -- which has happened on the cluster before.
            "repos": repo_versions(),
            "config": dataclasses.asdict(config),
            "net_params": net_params,
            "env_params": env_config.to_dict(),
            **({"overrides": overrides} if overrides else {}),
        },
        # `env-override` marks a run whose env differs from the study's standard config.
        # The comparability protocol (analysis/README.md) reads `env_params`, but a tag is
        # what makes such a run obvious in a run list before anyone thinks to check.
        tags=(*arch.tags, "warp", "TrainEvalSplit", arch.name,
              f"delay{cfg.delay}", f"eff{efference_length}",
              *ablations,
              *(("env-override",) if env_overridden else ()),
              *tuple(cfg.wandb.tags)),
        seed=seed,
    )


def write_config_json(run_dir: Path, setup: RunSetup) -> None:
    """Write ``config.json`` if absent. Never overwrite: it records what the run started
    with, and the offline eval path rebuilds the env from it."""
    path = run_dir / "config.json"
    if not path.exists():
        path.write_text(json.dumps(setup.config_json(), indent=2, default=str))


def write_resolved_config(run_dir: Path, cfg: DictConfig) -> None:
    """Snapshot the composed config beside the checkpoints.

    ``config.json`` is the machine-readable contract for reloading; this is the
    human-readable account of how the run was configured, overrides included. Hydra's own
    output directory is disabled (see conf/train.yaml) because the run directory is ours
    and has to stay stable across requeue attempts.
    """
    (run_dir / "hydra_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))


def run_final_eval(cfg: DictConfig, setup: RunSetup, run_dir: Path, exp_name: str,
                   wandb_id: str, step: int) -> None:
    """End-of-training evaluation on the held-out clips.

    Only meaningful for a clip-driven task: it reports train-split versus test-split
    performance, and a self-contained env has no such split. The in-training eval covers
    those, so skipping here loses nothing.
    """
    if not setup.env_spec.uses_clips:
        print("Final eval skipped: this task has no held-out clip split.")
        return
    evaluation.run_final_eval(
        setup.nets, setup.env_spec.cls, setup.env_config,
        ckpt_dir=run_dir,
        wandb_id=wandb_id, wandb_name=exp_name, step=step,
        net_params=setup.net_params,
        train_env=setup.train_env, train_clips=setup.train_clips,
        test_clips=setup.test_clips,
        seed=setup.seed, limit_clips=cfg.eval_limit_clips,
        summary_fn=wandb.run.summary.update,
    )


# ---------------------------------------------------------------------------
# Plain training
# ---------------------------------------------------------------------------

def run_plain(cfg: DictConfig, setup: RunSetup) -> int:
    """One uninterrupted run on a dedicated partition."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{cfg.run.suffix}" if cfg.run.suffix else ""
    exp_name = cfg.run.name or f"{setup.name_stem}-{timestamp}{suffix}"

    run_dir = Path(cfg.checkpoint_root) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_json(run_dir, setup)
    write_resolved_config(run_dir, cfg)

    wandb.init(project=cfg.wandb.project, config=setup.wandb_config,
               name=exp_name, tags=list(setup.tags), notes=cfg.wandb.notes)

    result = ppo.train_ppo(
        setup.train_env, setup.nets, setup.config,
        log_fn=wandb.log,
        video_fn=wandb_video_fn(fps=50),
        checkpoint_fn=make_checkpoint_fn(str(run_dir), setup.config),
        eval_env=setup.eval_env,
    )

    print(f"Training complete: {result.total_steps} steps, "
          f"{result.total_iterations} iterations")
    if result.eval_history:
        print("Final eval reward: "
              f"{result.eval_history[-1].get('eval/episode_reward/mean', 'N/A')}")

    total_steps = int(result.total_steps)
    if cfg.final_eval:
        # Release the training state (n_envs env states + optimizer moments) before the
        # eval allocates. `setup.nets` is the same object as
        # result.training_state.networks, so the trained weights survive.
        del result
        jax.clear_caches()
        gc.collect()
        run_final_eval(cfg, setup, run_dir, exp_name, wandb.run.id, total_steps)

    wandb.finish()
    return 0


# ---------------------------------------------------------------------------
# Preemption-safe training
# ---------------------------------------------------------------------------

def spread_env_states(train_env, nets, n_envs: int, key):
    """Fresh env and carry states, with episode phases spread over the clip.

    ``start_frame`` is passed explicitly, so the env's own ``config.start_frame_range``
    (only the first 44 of 250 mocap frames for these runs) is bypassed for *this* reset
    and used as normal for every reset during the rollout afterwards. Drawing it over the
    whole valid range makes the time each env has left in its episode uniform, so the
    population does not march in lockstep after a resume.

    ``_last_valid_frame`` is the env's own definition of the last frame an episode may
    start at. Re-deriving the formula here would be one silent drift away from spreading
    over the wrong range, so we ask the env; if vnl-playground renames it, this fails
    loudly at startup rather than quietly mis-resetting.
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

    The config is rebuilt from the command line rather than restored, so that a resumed
    run is described by the config in front of you. That is only safe while the config has
    not changed under the run: editing conf/train/rodent.yaml between attempts would
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


def warn_on_env_drift(run_dir: Path, setup: RunSetup) -> None:
    """Compare this attempt's env config with the one the run started with.

    The env config is rebuilt on every attempt, and since it is overridable a changed
    override -- or a changed group file -- would silently move the task under a run that
    is still reported as one continuous curve. ``config.json`` is written once, by the
    first attempt, so it is the record of what the run set out to be; anything the offline
    eval path later reconstructs comes from it, not from this attempt's overrides.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return
    try:
        stored = json.loads(config_path.read_text()).get("env_params", {})
    except (json.JSONDecodeError, OSError):
        return
    # Round-trip through JSON so paths and tuples compare like they were stored.
    rebuilt = json.loads(json.dumps(setup.env_config.to_dict(), default=str))
    changed = [k for k in set(stored) | set(rebuilt) if stored.get(k) != rebuilt.get(k)]
    if changed:
        print("  WARNING: this attempt's env config differs from the one in "
              f"config.json, in: {sorted(changed)}")
        for k in sorted(changed):
            print(f"    {k}: config.json={stored.get(k)!r} this run={rebuilt.get(k)!r}")
        print("    config.json is NOT rewritten, so the offline eval path will keep "
              "rebuilding the env the run started with.")


def restore(step_dir: str, setup: RunSetup, *, n_envs: int):
    """Rebuild a resumable TrainingState from a checkpoint.

    The template is built with ``n_envs=1``: the optimizer state does not depend on the
    number of envs, and a full-width template would allocate env states only to throw
    them away. Weights and optimizer state are restored in place into ``setup.nets``.
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
        # A light checkpoint: redraw what it did not store. Note this is decided by what
        # is *in* the checkpoint, not by the config, so flipping full_checkpoints
        # mid-run resumes fine either way.
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


def run_requeue(cfg: DictConfig, setup: RunSetup) -> int:
    """One attempt of a preemption-safe run. Returns the process exit code."""
    requeue_lib.maybe_enable_jax_cache(cfg.requeue.jax_cache_dir)

    token = requeue_lib.run_token()
    suffix = f"-{cfg.run.suffix}" if cfg.run.suffix else ""
    exp_name = cfg.run.name or f"{setup.name_stem}-{token}{suffix}"
    run_dir = Path(cfg.checkpoint_root) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    state = requeue_lib.RunState.load(run_dir)
    attempt = 1 if state is None else len(state.attempts) + 1
    print(f"=== {exp_name} | attempt {attempt} "
          f"(slurm restart count {requeue_lib.restart_count()}) ===")
    print(f"    run dir: {run_dir}")

    if state is not None and state.done:
        print("Run already finished (run_state.json says done). Nothing to do.")
        return 0
    if state is not None and attempt > cfg.requeue.max_attempts:
        print(f"ERROR: {len(state.attempts)} attempts already made, "
              f"requeue.max_attempts is {cfg.requeue.max_attempts}. Refusing to "
              f"continue; raise the cap if this run is merely unlucky.")
        return 1

    write_config_json(run_dir, setup)
    write_resolved_config(run_dir, cfg)

    if state is None:
        state = requeue_lib.RunState(wandb_id=wandb.util.generate_id(),
                                     exp_name=exp_name)
        state.save(run_dir)

    # Resume from the newest complete checkpoint, if there is one.
    step_dir = latest_checkpoint(str(run_dir))
    initial_state = None
    resumed_from_step = None
    total_steps = setup.config.ppo.total_steps

    if step_dir is not None:
        print(f"  restoring {step_dir}")
        warn_on_env_drift(run_dir, setup)
        initial_state, resumed_from_step = restore(
            step_dir, setup, n_envs=setup.config.ppo.n_envs
        )
        print(f"  resumed at step {resumed_from_step} of {total_steps}")
    else:
        print("  no checkpoint found; starting from scratch")

    state.record_attempt(run_dir, resumed_from_step=resumed_from_step)

    notes = cfg.wandb.notes
    if resumed_from_step is not None:
        notes = (f"{notes}\nRequeue-safe run: attempt {attempt}, resumed from "
                 f"step {resumed_from_step} ({step_dir}).")
    requeue_lib.init_wandb_resumable(
        state,
        project=cfg.wandb.project,
        config={**setup.wandb_config,
                **requeue_lib.wandb_requeue_config(state, run_dir, resumed_from_step,
                                                   attempt)},
        tags=(*setup.tags, "requeue"),
        notes=notes,
        # Not "resumed_from_step is not None": a run whose checkpoints were deleted starts
        # over, but its WandB run already exists and must be reopened rather than
        # initialised fresh under the same id.
        first_attempt=attempt == 1,
        resume_from_step=resumed_from_step,
        # Rewind only when the previous attempt logged past this checkpoint, which means
        # it was killed outright rather than saving on its way out.
        rewind=requeue_lib.rewind_needed(state, resumed_from_step),
    )

    # Training already complete but the final eval never ran (preempted during it, most
    # likely). Skip straight to the eval.
    if resumed_from_step is not None and resumed_from_step >= total_steps:
        print(f"  step {resumed_from_step} >= total_steps {total_steps}; skipping "
              f"training and going to the final eval")
        del initial_state
        jax.clear_caches()
        gc.collect()
        return _finish_requeue(cfg, setup, state, run_dir, exp_name, resumed_from_step)

    watcher = requeue_lib.PreemptionWatcher(
        on_signal=lambda s: print(
            f"\n>>> caught {signal.Signals(s).name}: saving a checkpoint at the next "
            f"iteration boundary and exiting for requeue", flush=True)
    )

    result = ppo.train_ppo(
        setup.train_env, setup.nets, setup.config,
        log_fn=wandb.log,
        video_fn=wandb_video_fn(fps=50),
        checkpoint_fn=make_checkpoint_fn(
            str(run_dir), setup.config,
            include_env_state=cfg.requeue.full_checkpoints,
        ),
        eval_env=setup.eval_env,
        initial_state=initial_state,
        stop_fn=watcher,
        # A resumed attempt must not repeat the eval, video and checkpoint of the step it
        # restored from -- that is work the previous attempt already did and logged, and
        # on a frequently preempted run it would be most of what the job does.
        initial_eval=initial_state is None,
    )
    watcher.restore()

    print(f"Attempt {attempt} ran {result.total_iterations} iterations, "
          f"now at step {result.total_steps} of {total_steps}")

    # "Is there work left?" rather than "did the watcher fire?". These differ in exactly
    # one case: the signal arriving during the iteration that crosses total_steps.
    # Training is then complete, and asking for a requeue would burn a whole job --
    # rebuilding the env only to discover there is nothing to do -- before reaching the
    # final eval. So ask about the work, not the signal.
    if result.total_steps < total_steps:
        state.finish_attempt(run_dir, "preempted",
                             stopped_at_step=int(result.total_steps),
                             signal=watcher.signal_name)
        # exit_code=1 leaves the WandB run marked as not-finished. That is the honest
        # state -- analyses select on state == "finished" (wandb_utils/index.py), and a
        # preempted attempt must not make an incomplete run look complete to them. The
        # last attempt finishes it normally.
        wandb.finish(exit_code=1)
        print(f"Saved at step {result.total_steps}; exiting "
              f"{requeue_lib.EXIT_PREEMPTED} for requeue.")
        return requeue_lib.EXIT_PREEMPTED

    if result.eval_history:
        print("Final eval reward: "
              f"{result.eval_history[-1].get('eval/episode_reward/mean', 'N/A')}")

    final_step = int(result.total_steps)
    del result, initial_state
    jax.clear_caches()
    gc.collect()

    return _finish_requeue(cfg, setup, state, run_dir, exp_name, final_step)


def _finish_requeue(cfg, setup, state, run_dir: Path, exp_name: str, step: int) -> int:
    """End-of-training eval, mark the run done, close WandB."""
    if cfg.final_eval:
        run_final_eval(cfg, setup, run_dir, exp_name, state.wandb_id, step)
    state.done = True
    state.finish_attempt(run_dir, "finished", final_step=step)
    wandb.finish()
    print(f"Run complete at step {step}.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    # Slurm redirects stdout to a file, which makes it block-buffered, and a preempted
    # attempt's progress would then be whatever happened to be flushed. Line buffering
    # costs nothing and keeps the log readable while the job is still running.
    sys.stdout.reconfigure(line_buffering=True)

    try:
        setup = build_run(cfg)
    except (OverrideError, KeyError) as e:
        sys.exit(f"error: {e}")

    code = run_requeue(cfg, setup) if cfg.requeue.enabled else run_plain(cfg, setup)
    sys.exit(code)


if __name__ == "__main__":
    main()
