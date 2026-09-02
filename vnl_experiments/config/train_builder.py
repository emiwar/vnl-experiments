"""Turning a Hydra config group into a real ``nnx_ppo`` :class:`TrainConfig`.

The schema is *derived* from nnx-ppo's own dataclasses rather than restated here, so a
field added upstream becomes overridable with no change on this side and the two cannot
drift. ``OmegaConf.structured(TrainConfig)`` accepts them as-is.

Two fields need help, and both are worth understanding before touching this:

``logging_percentiles``
    Declared ``Optional[tuple[int, ...]]``. OmegaConf has no tuple type, so it comes back
    as a *list*. That is not cosmetic: the value is passed into
    ``nnx.jit(..., static_argnums=...)`` (``nnx_ppo/algorithms/ppo.py:115``, used at L147
    and L224), static arguments must be hashable, and a list is not. It is re-tupled here.

``logging_level``
    An ``enum.Flag``. OmegaConf round-trips a *named* member fine, but the values actually
    used are combinations (``LOSSES | THROUGHPUT | ENV_METRICS``) and a combination has no
    name to write in YAML. So the schema retypes it to accept a list of member names,
    which :func:`resolve_logging_level` reduces with ``|``.

The subclasses below exist only for those two fields; everything else is inherited.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import reduce
from operator import or_
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from nnx_ppo.algorithms.config import (
    EvalConfig,
    PPOConfig,
    TrainConfig,
    VideoConfig,
)
from nnx_ppo.algorithms.types import LoggingLevel


@dataclass
class PPOSchema(PPOConfig):
    """:class:`PPOConfig` with the two OmegaConf-hostile fields widened."""

    logging_level: Any = "LOSSES"
    logging_percentiles: Optional[list[int]] = None


@dataclass
class EvalSchema(EvalConfig):
    """:class:`EvalConfig`, same two fields widened."""

    logging_level: Any = "NONE"
    logging_percentiles: Optional[list[int]] = field(
        default_factory=lambda: [0, 25, 50, 75, 100]
    )


@dataclass
class TrainSchema(TrainConfig):
    """:class:`TrainConfig` using the widened sub-schemas."""

    ppo: PPOSchema = field(default_factory=PPOSchema)
    eval: EvalSchema = field(default_factory=EvalSchema)


def resolve_logging_level(value: Any) -> LoggingLevel:
    """A ``LoggingLevel`` from a member name, a list of names, or an existing flag.

    A list is reduced with ``|``, which is how a combination gets expressed in YAML:
    ``logging_level: [LOSSES, THROUGHPUT, ENV_METRICS]``.
    """
    if isinstance(value, LoggingLevel):
        return value
    if value is None:
        return LoggingLevel.NONE
    names = [value] if isinstance(value, str) else list(value)
    try:
        return reduce(or_, (LoggingLevel[str(n).strip()] for n in names), LoggingLevel.NONE)
    except KeyError as e:
        known = [m.name for m in LoggingLevel]
        raise ValueError(f"Unknown LoggingLevel {e.args[0]!r}. Known: {known}") from e


def _rebuild(cls, widened):
    """A real ``cls`` from its widened counterpart, with jit-safe field types."""
    kwargs = {f.name: getattr(widened, f.name) for f in dataclasses.fields(cls)}
    kwargs["logging_level"] = resolve_logging_level(kwargs["logging_level"])
    percentiles = kwargs.get("logging_percentiles")
    kwargs["logging_percentiles"] = tuple(percentiles) if percentiles is not None else None
    return cls(**kwargs)


def build_train_config(cfg: DictConfig | dict) -> TrainConfig:
    """Compose ``cfg`` over the nnx-ppo defaults and return a genuine ``TrainConfig``.

    The result is the real dataclass, not a ``DictConfig``: it is handed to ``train_ppo``,
    pickled into checkpoint metadata, and ``dataclasses.asdict``-ed into the WandB config,
    all of which expect the actual type.
    """
    merged = OmegaConf.merge(OmegaConf.structured(TrainSchema), cfg)
    widened = OmegaConf.to_object(merged)
    return TrainConfig(
        ppo=_rebuild(PPOConfig, widened.ppo),
        eval=_rebuild(EvalConfig, widened.eval),
        video=VideoConfig(
            **{f.name: getattr(widened.video, f.name) for f in dataclasses.fields(VideoConfig)}
        ),
        seed=widened.seed,
        checkpoint_every_steps=widened.checkpoint_every_steps,
    )


def validate_train_config(config: TrainConfig) -> TrainConfig:
    """Reject combinations the PPO update cannot honour. Returns ``config``."""
    if config.ppo.n_envs % config.ppo.n_minibatches:
        raise ValueError(
            f"n_envs ({config.ppo.n_envs}) must be divisible by n_minibatches "
            f"({config.ppo.n_minibatches})."
        )
    return config
