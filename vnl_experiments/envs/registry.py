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
    #: ``(config, clips=None) -> env``. Applies whatever wrappers the family needs.
    build: Callable[..., Any]
    #: ``"dict"`` or ``"flat"``; must match the architecture's.
    obs_layout: str = "dict"
    #: Whether the task is clip-driven, and so needs a train/test split and a `clips=`
    #: argument. False for a self-contained env such as a dm_control task.
    uses_clips: bool = True
    #: The env class, for the clip-based final eval, which builds its own instances.
    #: None for a task that has no held-out split to evaluate on.
    cls: Any = None


def _imitation_builder(cls):
    def build(config, *, clips=None):
        return cls(config, clips=clips)

    return build


def _dmc_builder(task: str):
    """Construct a dm_control_suite task with the wrappers it needs to be trainable.

    Two wrappers, both load-bearing:

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

    Both read their parameter from the env config rather than closing over a constant, so
    the value is overridable per run *and* recorded in ``config.json``'s ``env_params``.
    """
    def build(config, *, clips=None):
        import mujoco_playground
        from nnx_ppo.wrappers import episode_wrapper, reward_scaling_wrapper

        env = mujoco_playground.registry.load(task, config=config)
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
        return cfg

    return default_config


def _dmc_spec(task: str) -> EnvSpec:
    return EnvSpec(
        default_config=_dmc_default_config(task),
        build=_dmc_builder(task),
        obs_layout="flat",
        uses_clips=False,
    )


ENVS: dict[str, EnvSpec] = {
    "Imitation": EnvSpec(imitation_default_config, _imitation_builder(Imitation),
                        cls=Imitation),
    "AbsoluteImitation": EnvSpec(absolute_default_config,
                                 _imitation_builder(AbsoluteImitation),
                                 cls=AbsoluteImitation),
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
}


def get(task: str) -> EnvSpec:
    """Resolve a task name, or raise naming what is available."""
    try:
        return ENVS[task]
    except KeyError:
        raise KeyError(f"Unknown env task {task!r}. Available: {sorted(ENVS)}") from None
