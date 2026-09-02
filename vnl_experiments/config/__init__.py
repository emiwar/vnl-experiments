"""The boundary between Hydra/OmegaConf and the config objects the libraries expect.

Hydra composes YAML into a ``DictConfig``. Neither library takes one: ``vnl-playground``
wants an ``ml_collections.ConfigDict`` and ``nnx-ppo`` wants its own dataclasses. This
package is the (small, tested) conversion, and it is the only place that knows about both
sides -- so neither library needs to change to support Hydra.

* :mod:`~vnl_experiments.config.overrides` -- validated, type-preserving assignment onto
  a ``ConfigDict``.
* :mod:`~vnl_experiments.config.env_builder` -- ``default_config()`` + YAML deltas ->
  ``ConfigDict``, for both env and net configs.
* :mod:`~vnl_experiments.config.train_builder` -- YAML -> a real ``TrainConfig``.
"""

from vnl_experiments.config.env_builder import build_env_config, build_net_config
from vnl_experiments.config.overrides import (
    OverrideError,
    apply_overrides,
    apply_strings,
    apply_tree,
    coerce,
    flatten,
)
from vnl_experiments.config.train_builder import (
    build_train_config,
    resolve_logging_level,
    validate_train_config,
)

__all__ = [
    "OverrideError",
    "apply_overrides",
    "apply_strings",
    "apply_tree",
    "build_env_config",
    "build_net_config",
    "build_train_config",
    "coerce",
    "flatten",
    "resolve_logging_level",
    "validate_train_config",
]
