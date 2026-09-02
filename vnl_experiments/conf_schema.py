"""Registering the env and net schemas with Hydra, generated from the real defaults.

Hydra composes in *struct* mode: an override naming a key the composed config does not
already have is refused. That is the behaviour we want -- it is how `net.latnet_size=64`
gets caught -- but the keys live in ml_collections configs that Hydra cannot see, so
without help every legitimate override would need a ``+`` prefix, and only the ones a
group file happened to mention would not. Two different syntaxes for the same operation,
decided by an implementation detail, is worse than either.

So the full key set is registered into Hydra's ConfigStore, generated at import time from
the same ``default_config()`` and ``Architecture.defaults()`` the run actually uses. There
is no second copy to drift: change a default in vnl-playground and the schema changes with
it. What the group YAML files hold stays what it should be -- the handful of values this
project deliberately differs on.

Side benefits: ``--cfg job`` prints the complete configuration surface, and
``--help`` / shell completion know every key.

The conversion to plain YAML types is lossy in one direction only. ``epath.Path`` becomes
a string and tuples become lists, because OmegaConf cannot hold either. Both are converted
back when the values are applied to the real ConfigDict, which is type-strict and rebuilds
each field in its own type -- see :mod:`vnl_experiments.config.overrides`.
"""

from __future__ import annotations

from typing import Any

from hydra.core.config_store import ConfigStore

from vnl_experiments.delays.network_builders import ARCHITECTURES
from vnl_experiments.envs.registry import ENVS

#: ConfigStore groups the YAML files refer to as `/env_defaults@env` and `/net_defaults@net`.
ENV_GROUP = "env_defaults"
NET_GROUP = "net_defaults"


def to_plain(value: Any) -> Any:
    """A config value as YAML-representable types.

    ``epath`` paths and tuples are the two things OmegaConf refuses; everything else in
    these configs is already a scalar, list or dict.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    if hasattr(value, "items"):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return str(value)


def register() -> None:
    """Register one schema node per env task and per network architecture.

    Idempotent: ConfigStore overwrites by name, so importing this twice is harmless.
    """
    cs = ConfigStore.instance()
    for name, spec in ENVS.items():
        cs.store(group=ENV_GROUP, name=name, node=to_plain(spec.default_config()))
    for name, arch in ARCHITECTURES.items():
        cs.store(group=NET_GROUP, name=name, node=to_plain(arch.defaults()))
