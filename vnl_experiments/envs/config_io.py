"""Helpers for rebuilding an env config from a checkpoint's logged ``env_params``.

A checkpoint records the *cluster* paths of the assets it trained on, and those paths do
not exist on a laptop, so every offline rebuild (batch eval, activation recording, video)
has to repair them. The naive repair -- take the local default -- silently swapped the
**body**: a run trained on ``rodent_no_tail_collisions.xml`` was re-simulated on
``rodent.xml``, because that is what ``default_config()`` points at. Symptom, measured on
the new-XML forward-model cohort: offline ``old_eval`` reward fell 2 % / 13 % / 27 % / 42 %
below the inline eval at delays 0 / 10 / 20 / 50, with survival 0.23 vs 0.67 at delay 50.
The inline end-of-training eval was unaffected because it is handed the live ``env_config``
rather than a reconstruction.

:func:`resolve_local_xml_paths` repairs the *directory* while preserving the *file*, which
is what the callers actually wanted. Reference-data paths keep the old behaviour: the mocap
file is not run-specific beyond ``clip_set``/``clip_length``, which are honoured separately.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Config fields that name a run-specific asset and must survive the repair.
XML_FIELDS: tuple[str, ...] = ("walker_xml_path", "arena_xml_path")


def _assign(cfg, field: str, value: Any, template: Any) -> None:
    """Set ``cfg.field`` to ``value``, keeping the type the config already holds.

    These configs are ``ml_collections`` ConfigDicts, which are type-strict, and the XML
    fields hold ``epath`` paths rather than ``str``.
    """
    try:
        cfg[field] = type(template)(str(value))
    except (TypeError, ValueError):
        with cfg.ignore_type():
            cfg[field] = str(value)


def _choose(field: str, env_params: Mapping[str, Any], default, *, warn: bool) -> Path:
    """The local file to use for one asset field: the run's own, else the default."""
    default_path = Path(str(default[field]))
    stored = env_params.get(field)
    if not stored:
        return default_path

    candidate = default_path.parent / Path(str(stored)).name
    if candidate.name == default_path.name:
        return default_path
    if candidate.exists():
        return candidate
    if warn:
        warnings.warn(
            f"{field}: the run trained on {Path(str(stored)).name!r}, which is not present "
            f"in {default_path.parent}; falling back to the local default "
            f"{default_path.name!r}. Results will describe a DIFFERENT asset than the run "
            f"was trained with.",
            stacklevel=3,
        )
    return default_path


def local_xml_names(
    env_params: Mapping[str, Any],
    default,
    *,
    fields: Sequence[str] = XML_FIELDS,
    warn: bool = False,
) -> dict[str, str]:
    """``{field: basename}`` that :func:`resolve_local_xml_paths` would select.

    Pure -- it needs no config to mutate. Producers call this to stamp the sidecar with the
    body an artifact was made on, and the audit compares that stamp against the run's
    ``env_params``. Sharing ``_choose`` with the resolver is what keeps the stamp honest:
    the two cannot drift apart into "what we recorded" vs "what we simulated".
    """
    return {field: _choose(field, env_params, default, warn=warn).name
            for field in fields if field in default}


def resolve_local_xml_paths(
    cfg,
    env_params: Mapping[str, Any],
    default,
    *,
    fields: Sequence[str] = XML_FIELDS,
) -> dict[str, str]:
    """Point ``cfg``'s asset fields at *local copies of the files the run trained on*.

    For each field, keep the basename recorded in ``env_params`` and look for it beside the
    corresponding default. Fall back to the default -- the old behaviour -- only when that
    basename is not available locally, and warn when doing so, because falling back means
    simulating a different body than the one that was trained.

    Returns ``{field: basename}`` of what was actually used.
    """
    used: dict[str, str] = {}
    for field in fields:
        if field not in cfg:
            continue
        chosen = _choose(field, env_params, default, warn=True)
        _assign(cfg, field, chosen, cfg[field])
        used[field] = chosen.name
    return used
