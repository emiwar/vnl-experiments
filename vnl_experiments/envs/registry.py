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
    """Construct a dm_control_suite task, wrapped so that it produces episodes.

    dm_control_suite envs from the registry never set ``done``: left unwrapped they run
    as one infinite episode, so nothing is ever reset and every episode statistic is
    meaningless. ``EpisodeWrapper`` supplies the truncation. The vnl-playground tasks
    self-truncate and must *not* be wrapped this way.
    """
    def build(config, *, clips=None):
        import mujoco_playground
        from nnx_ppo.wrappers import episode_wrapper

        env = mujoco_playground.registry.load(task, config=config)
        return episode_wrapper.EpisodeWrapper(env, config.get("episode_length", 1000))

    return build


def _dmc_default_config(task: str):
    def default_config():
        import mujoco_playground

        cfg = mujoco_playground.registry.get_default_config(task)
        # Episode length is the wrapper's, not the task's, so it has nowhere else to
        # live -- and it has to be overridable per run like everything else.
        if "episode_length" not in cfg:
            cfg.episode_length = 1000
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
    # they are needed -- the entry is the only code a new one requires.
    "WalkerWalk": _dmc_spec("WalkerWalk"),
    "WalkerRun": _dmc_spec("WalkerRun"),
    "CheetahRun": _dmc_spec("CheetahRun"),
    "CartpoleBalance": _dmc_spec("CartpoleBalance"),
}


def get(task: str) -> EnvSpec:
    """Resolve a task name, or raise naming what is available."""
    try:
        return ENVS[task]
    except KeyError:
        raise KeyError(f"Unknown env task {task!r}. Available: {sorted(ENVS)}") from None
