"""Which env a config group names, how to default it, and how to construct it.

``env_spec.task`` in a Hydra group is a *string*, for the same reason
``net_params.network_class`` is: it is written into ``config.json`` and has to keep
resolving years later, when the module layout has moved on. A ``_target_``-style import
path would tie every stored checkpoint to a source location.

The table is also what makes the config system extensible past the rodent. A new task is
one entry here plus one YAML group; the training entry point needs to know nothing about
it. The two families currently registered differ in more than their config:

* the **imitation** tasks are driven by reference clips, so they are constructed with a
  train/test clip split and expose a structured (``dict``) observation;
* the **dm_control_suite** tasks are self-contained, expose a single flat observation
  vector, and -- importantly -- never terminate on their own, so they need an
  ``EpisodeWrapper`` to produce episodes at all.

They also differ in how a preempted run resumes: a light checkpoint omits the env states,
so they have to be redrawn, and "spread the episode phases out" means something different
for a clip than for a fixed-length episode. That is ``resume_reset`` below; it is the last
thing in the training path that was rodent-only.

``obs_layout`` is checked against the chosen network's, so a rodent architecture asked to
train on a dm_control task fails at startup with a sentence rather than a shape error
inside the first forward pass.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from vnl_playground.tasks.rodent.imitation import (
    Imitation,
    default_config as imitation_default_config,
)

from vnl_experiments.envs.absolute_imitation import (
    AbsoluteImitation,
    default_config as absolute_default_config,
)


class EnvSpec(NamedTuple):
    """How to default and construct one task."""

    #: ``() -> ConfigDict``. The authoritative schema; YAML groups hold deltas onto it.
    default_config: Callable[[], Any]
    #: ``(config, clips=None, for_eval=False) -> env``. Applies whatever wrappers the
    #: family needs. ``for_eval`` matters because some training wrappers must *not* be
    #: on the measurement env -- see :func:`_dmc_builder`.
    build: Callable[..., Any]
    #: ``"dict"`` or ``"flat"``; must match the architecture's.
    obs_layout: str = "dict"
    #: Whether the task is clip-driven, and so needs a train/test split and a `clips=`
    #: argument. False for a self-contained env such as a dm_control task.
    uses_clips: bool = True
    #: The env class, for the clip-based final eval, which builds its own instances.
    #: None for a task that has no held-out split to evaluate on.
    cls: Any = None
    #: ``(env, env_config, n_envs, key) -> (env_states, note)``. How to draw a fresh
    #: population of env states with their episode phases spread out, for a resume from a
    #: light checkpoint (one that omitted the env states). ``note`` is a one-line
    #: description of what was drawn, for the resume log.
    #:
    #: Deliberately has **no default**: a family that does not say how to do this would
    #: resume with every env at the same point in its episode, and a population marching
    #: in lockstep looks like a training artefact rather than a bug. ``train.py``'s
    #: ``validate_resumable`` rejects a spec without one at startup, so the failure lands
    #: on attempt 1 rather than on the first preemption hours later.
    resume_reset: Callable[..., Any] | None = None


def _imitation_builder(cls):
    def build(config, *, clips=None, for_eval=False):
        # The imitation tasks need no wrappers, so train and eval differ only in which
        # clips they are handed -- which the caller decides.
        return cls(config, clips=clips)

    return build


def _imitation_resume_reset(env, env_config, n_envs: int, key):
    """Fresh imitation env states, with episode phases spread over the clip.

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
    import jax
    from flax import nnx

    reset_key, frame_key = jax.random.split(key)
    last_valid_frame = int(env._last_valid_frame())
    frames = jax.random.randint(frame_key, (n_envs,), 0, last_valid_frame + 1)
    env_states = nnx.vmap(
        lambda k, f: env.reset(k, start_frame=f)
    )(jax.random.split(reset_key, n_envs), frames)
    return env_states, f"start_frame ~ U[0, {last_valid_frame}] over {n_envs} envs"


def _dmc_resume_reset(env, env_config, n_envs: int, key):
    """Fresh dm_control env states, with episode phases spread over the whole episode.

    ``EpisodeWrapper.reset`` already spreads the phase, but only over
    ``[0, max_len // 2)`` -- so a freshly reset population has *nothing* in the second
    half of its episode, where a population that has been running a while keeps about two
    thirds of its mass (occupancy is proportional to ``min(phase + 1, max_len // 2)``).
    Resuming that way would leave the whole population unable to truncate for ~500 steps
    and then truncate in a burst.

    Overwriting ``step_counter`` over the full range is the direct analogue of what
    :func:`_imitation_resume_reset` does with ``start_frame``: bypass the narrow default
    spread for this one reset, and let every reset during the rollout afterwards use it as
    normal. ``truncated`` needs no fixing up -- the draw excludes ``max_len`` itself, so no
    env starts already truncated.
    """
    import jax
    from flax import nnx

    reset_key, phase_key = jax.random.split(key)
    env_states = nnx.vmap(env.reset)(jax.random.split(reset_key, n_envs))
    max_len = int(env_config.get("episode_length", 1000))
    counter = env_states.info["step_counter"]
    phases = jax.random.randint(phase_key, (n_envs,), 0, max_len).astype(counter.dtype)
    env_states = env_states.replace(
        info={**env_states.info, "step_counter": phases}
    )
    return env_states, f"step_counter ~ U[0, {max_len}) over {n_envs} envs"


def _dmc_builder(task: str):
    """Construct a dm_control_suite task with the wrappers it needs to be trainable.

    Three wrappers, all load-bearing:

    ``EpisodeWrapper``
        dm_control_suite envs from the registry never set ``done``: left unwrapped they
        run as one infinite episode, so nothing is ever reset and every episode statistic
        is meaningless. This supplies the truncation. The vnl-playground tasks
        self-truncate and must *not* be wrapped this way.

    ``RewardScalingWrapper``
        brax's ``reward_scaling`` for every dm_control task is 10.0, and the pre-Hydra
        `train_delays.py` applied it. It is not cosmetic -- it scales the advantages, so
        dropping it changes the effective learning rate and makes runs incomparable to the
        existing `nnx-ppo-delays` cohort. Set ``env.reward_scale=1.0`` to disable.

    ``NaNGuardWrapper``
        MJX diverges every few hundred million steps, and whether the resulting reward is
        finite is luck (see `vnl_experiments.envs.nan_guard`). A non-finite one is fatal:
        it turns every gradient NaN in a single update. This makes the diverged step a
        plain zero-reward termination, which is what the survivable case already looks
        like. It wraps *innermost* so ``EpisodeWrapper`` folds its ``done`` in with
        ``truncated`` still False -- a termination, matching the task's own NaN handling.

    The first two read their parameter from the env config rather than closing over a
    constant, so the value is overridable per run *and* recorded in ``config.json``'s
    ``env_params``.

    **All three are training-only.** ``for_eval=True`` returns the bare env. The first
    two corrupt the measurement: reward scaling reports reward in 10x units, and
    ``EpisodeWrapper.reset`` seeds ``step_counter`` to a random value in
    ``[0, max_len/2)`` -- a deliberate phase-spread so training envs do not truncate in
    lockstep, but on an eval env it means an episode can begin 499 steps in and the
    reported lifespan lands near 750 instead of 1000, with large variance. The NaN guard
    is left off eval for a different reason: a divergence there costs one NaN eval point
    rather than the whole run, and the eval numbers are the quantity the `nnx-ppo-delays`
    cohort is compared on, so it is not worth changing them for a ~1-in-1e9-steps event.
    Revisit if the divergence rate ever rises.
    """
    def build(config, *, clips=None, for_eval=False):
        import mujoco_playground
        from nnx_ppo.wrappers import episode_wrapper, reward_scaling_wrapper

        from vnl_experiments.envs import actuator_mode
        from vnl_experiments.envs.nan_guard import NaNGuardWrapper

        env = mujoco_playground.registry.load(task, config=config)
        # Control mode is a property of the *plant*, so it is applied before the for_eval
        # branch: train and eval must simulate the same physics. At servo_kp=0 this returns
        # the env untouched. See envs/servo_control.md.
        env = actuator_mode.apply_servo(
            env,
            kp=float(config.get("servo_kp", 0.0)),
            damping_ratio=float(config.get("servo_damping_ratio", 1.0)),
            center=str(config.get("servo_center", "qpos0")),
            unlimited_half_range=float(config.get("servo_unlimited_half_range", 0.0)),
        )
        if for_eval:
            # None of the three belongs on the measurement env. RewardScalingWrapper
            # reports reward in 10x units and EpisodeWrapper's randomised start counter
            # truncates the episode early; the NaN guard would not distort much, but it
            # would still move the numbers this project's cohort is compared on, for an
            # event rare enough that a NaN eval point is the cheaper failure. The eval
            # rollout bounds itself with `eval.max_episode_length`, so the raw env is
            # both correct and what the pre-Hydra script used.
            return env
        env = NaNGuardWrapper(env)
        env = episode_wrapper.EpisodeWrapper(env, int(config.get("episode_length", 1000)))
        scale = float(config.get("reward_scale", 1.0))
        if scale != 1.0:
            env = reward_scaling_wrapper.RewardScalingWrapper(env, scale)
        return env

    return build


def _brax_params(task: str):
    """brax's reference PPO config for a dm_control task, or None if unavailable.

    Only the two wrapper parameters are taken from here; every optimisation
    hyperparameter is pinned explicitly in ``conf/train/dmc.yaml`` instead, so that one
    ``train=dmc`` means the same thing across envs.
    """
    try:
        import mujoco_playground.config.dm_control_suite_params as params

        return params.brax_ppo_config(task)
    except Exception:  # noqa: BLE001 - a task brax has no entry for still has defaults
        return None


def _dmc_default_config(task: str):
    def default_config():
        import mujoco_playground

        cfg = mujoco_playground.registry.get_default_config(task)
        # These two belong to the *wrappers*, not the task, so the task config has no
        # field for them -- but they have to be overridable per run and recorded in
        # env_params like everything else, so they are added here. Both defaults come from
        # brax_ppo_config, which is what the pre-Hydra script derived them from.
        params = _brax_params(task)
        if "episode_length" not in cfg:
            cfg.episode_length = int(getattr(params, "episode_length", 1000) or 1000)
        if "reward_scale" not in cfg:
            cfg.reward_scale = float(getattr(params, "reward_scaling", 1.0) or 1.0)
        # Joint control mode. These belong to the *model*, not to the task, so the playground
        # config has no field for them -- but they have to be overridable per run and
        # recorded in env_params, for the same reason the two wrapper fields above are.
        # servo_kp is a dimensionless normalised stiffness and 0.0 leaves the model exactly
        # as shipped; the whole argument is in envs/servo_control.md.
        cfg.servo_kp = 0.0
        cfg.servo_damping_ratio = 1.0
        cfg.servo_center = "qpos0"
        cfg.servo_unlimited_half_range = 0.0
        return cfg

    return default_config


def _dmc_spec(task: str) -> EnvSpec:
    return EnvSpec(
        default_config=_dmc_default_config(task),
        build=_dmc_builder(task),
        obs_layout="flat",
        uses_clips=False,
        resume_reset=_dmc_resume_reset,
    )


ENVS: dict[str, EnvSpec] = {
    "Imitation": EnvSpec(imitation_default_config, _imitation_builder(Imitation),
                        cls=Imitation, resume_reset=_imitation_resume_reset),
    "AbsoluteImitation": EnvSpec(absolute_default_config,
                                 _imitation_builder(AbsoluteImitation),
                                 cls=AbsoluteImitation,
                                 resume_reset=_imitation_resume_reset),
    # dm_control_suite tasks, by their mujoco_playground registry name. Add more as
    # they are needed -- the entry is the only code a new one requires. The set below is
    # every env the `nnx-ppo-delays` project already has runs for, plus CheetahRun and
    # WalkerStand; `mujoco_playground.registry.dm_control_suite.ALL_ENVS` lists the rest.
    "CartpoleBalance": _dmc_spec("CartpoleBalance"),
    "CartpoleSwingup": _dmc_spec("CartpoleSwingup"),
    "BallInCup": _dmc_spec("BallInCup"),
    "CheetahRun": _dmc_spec("CheetahRun"),
    "HumanoidStand": _dmc_spec("HumanoidStand"),
    "HumanoidWalk": _dmc_spec("HumanoidWalk"),
    "WalkerStand": _dmc_spec("WalkerStand"),
    "WalkerWalk": _dmc_spec("WalkerWalk"),
    "WalkerRun": _dmc_spec("WalkerRun"),
    # Added for the joint-stiffness (equilibrium-point) sweep: small, cheap bodies where a
    # peripheral servo is a plausible model of muscle mechanics. Reacher is the two-joint
    # arm the lambda model was posed about, Hopper is unstable and contact-rich, Finger
    # manipulates an unactuated external object. envs/servo_control.md section 3 lists what
    # is idiosyncratic about each -- HopperStand's action-dependent reward in particular.
    "ReacherEasy": _dmc_spec("ReacherEasy"),
    "ReacherHard": _dmc_spec("ReacherHard"),
    "HopperStand": _dmc_spec("HopperStand"),
    "HopperHop": _dmc_spec("HopperHop"),
    "FingerTurnEasy": _dmc_spec("FingerTurnEasy"),
    "FingerTurnHard": _dmc_spec("FingerTurnHard"),
}


def get(task: str) -> EnvSpec:
    """Resolve a task name, or raise naming what is available."""
    try:
        return ENVS[task]
    except KeyError:
        raise KeyError(f"Unknown env task {task!r}. Available: {sorted(ENVS)}") from None
