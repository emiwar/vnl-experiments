"""Turning a Hydra config group into an ``ml_collections`` env or net config.

The YAML groups under ``conf/env/`` and ``conf/net/`` hold **deltas only**. The
authoritative defaults stay where they already live -- ``default_config()`` in
vnl-playground (or in ``vnl_experiments.envs``) for the env, and the architecture
registry's ``defaults()`` for the network. Restating them in YAML would create a second
source of truth that silently drifts from the first.

So the composition is: take the real default config, apply the group's deltas through
:mod:`vnl_experiments.config.overrides` (which validates every key against the schema and
preserves each field's type), and return the ``ConfigDict`` the env constructor wants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from omegaconf import DictConfig, OmegaConf

from vnl_experiments.config.overrides import apply_tree, flatten

#: Config fields naming an asset file. A YAML group gives the *basename*; the directory
#: comes from whatever the default config points at on this machine.
ASSET_FIELDS = ("walker_xml_path", "arena_xml_path", "reference_data_path")


def _as_tree(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _localise_assets(deltas: dict[str, Any], default_cfg) -> dict[str, Any]:
    """Resolve bare asset basenames against the default config's own directory.

    The XML and reference files live inside the installed ``vnl-playground``, so their
    absolute path differs between this laptop and the cluster and must not be written
    into a committed YAML. A group therefore names the *file*
    (``walker_xml_path: rodent_no_tail_collisions.xml``) and the directory is taken from
    the default. A value that already has a directory component is left alone, so an
    out-of-tree asset can still be pointed at explicitly.

    This mirrors what ``envs.config_io`` does when *reloading* a checkpoint: keep the
    file, localise the directory. Getting that wrong is what silently swaps the body.
    """
    out = dict(deltas)
    for field in ASSET_FIELDS:
        value = out.get(field)
        if value is None or field not in default_cfg:
            continue
        if Path(str(value)).parent != Path("."):
            continue
        out[field] = str(Path(str(default_cfg[field])).parent / str(value))
    return out


def build_env_config(
    default_fn: Callable[[], Any],
    deltas: DictConfig | Mapping[str, Any] | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
):
    """The env ``ConfigDict`` for a run: ``default_fn()`` with ``deltas`` applied.

    ``extra`` is applied after ``deltas`` and is for values computed at runtime rather
    than written in YAML. Both are validated against the config's own schema, so an
    unknown key raises instead of being silently ignored.
    """
    cfg = default_fn()
    apply_tree(cfg, _localise_assets(flatten(_as_tree(deltas)), cfg))
    if extra:
        apply_tree(cfg, _localise_assets(flatten(dict(extra)), cfg))
    return cfg


def build_net_config(
    defaults_fn: Callable[[], Any],
    deltas: DictConfig | Mapping[str, Any] | None = None,
):
    """The net ``ConfigDict`` for a run: the architecture's ``defaults()`` plus ``deltas``.

    Net configs are flat, but this goes through the same validated path as the env config
    so that an unknown ``net.*`` key fails loudly and with the available keys listed.
    """
    cfg = defaults_fn()
    apply_tree(cfg, _as_tree(deltas))
    return cfg
