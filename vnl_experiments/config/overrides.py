"""Applying overrides onto an ``ml_collections`` config, safely.

The env and net configs are ``ConfigDict``s owned by ``vnl-playground`` and by the network
registry. They are the authoritative schema -- this module never invents a key, it only
sets ones that already exist, and it keeps the type the field already holds.

Two properties are load-bearing and are the reason this is not a two-line ``dict.update``:

* **An unknown key is an error, not a no-op.** A typo that quietly did nothing would
  produce a run whose recorded config says one thing and whose behaviour is another --
  precisely the failure that recording ``env_params`` exists to prevent. Every component
  of a dotted path is checked, and the error names what *is* available at that level.
* **The existing value's type is the specification.** ``ConfigDict`` is type-strict, and
  some fields hold types a string cannot be parsed into directly -- the XML and reference
  paths are ``epath`` objects rather than ``str``, so they are rebuilt via
  ``type(current)(value)``.

Overrides arrive from two places and both land here: Hydra config groups (a nested tree,
flattened to dotted keys) and any remaining ``KEY=VALUE`` command-line strings. Strings go
through :func:`coerce` first; values that arrive already typed from YAML do not need it.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


class OverrideError(ValueError):
    """An override names a key that does not exist, or a value that does not fit."""


def coerce(default: Any, raw: str) -> Any:
    """Coerce a string override to the type of the value it is replacing.

    The type of the value already in the config is the specification: it is what keeps
    ``latent_size=32`` an int and ``ctrl_dt=0.02`` a float, and it is what ml_collections
    requires, since a ConfigDict refuses a value of a different type than the field holds.
    """
    if isinstance(default, bool):
        # Before the int branch: bool is a subclass of int, and int("true") is not what
        # anyone means.
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(default, (list, tuple)):
        text = raw.strip()
        if text.startswith("["):
            items = json.loads(text)
        elif text == "":
            items = []
        else:
            items = text.split(",")
        # Element type from the existing elements, so int lists (hidden sizes,
        # start_frame_range) stay ints and float ones (healthy_z_range) stay floats.
        # Empty default: assume ints, which is what the net configs use.
        element = type(default[0]) if len(default) else int
        return [element(x) for x in items]
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if default is None:
        # An unset optional (e.g. latent_ar1_weight): infer from the literal.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}``.

    Leaves are anything that is not a mapping, so a list value stays one value rather
    than becoming a set of indexed keys.
    """
    out: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{path}."))
        else:
            out[path] = value
    return out


def _descend(cfg, parts: list[str], path: str):
    """Walk to the container holding the final component, checking each step."""
    node = cfg
    for i, part in enumerate(parts[:-1]):
        if part not in node:
            where = ".".join(parts[:i]) or "the config"
            raise OverrideError(
                f"Unknown key {part!r} under {where}. Available: {sorted(node.keys())}"
            )
        node = node[part]
        if not hasattr(node, "keys"):
            raise OverrideError(
                f"{path}: {'.'.join(parts[:i + 1])} is a value, not a group, so it has "
                f"no {parts[i + 1]!r} inside it."
            )
    return node


def set_one(cfg, path: str, value: Any) -> None:
    """Set one dotted ``path`` on ``cfg``, keeping the field's existing type.

    ``value`` may be a string (coerced against the current value) or an already-typed
    value from YAML (used as-is, with a type rebuild if ml_collections refuses it).
    """
    parts = path.split(".")
    node = _descend(cfg, parts, path)
    leaf = parts[-1]
    if leaf not in node:
        where = ".".join(parts[:-1]) or "the config"
        raise OverrideError(
            f"Unknown key {leaf!r} under {where}. Available: {sorted(node.keys())}"
        )

    current = node[leaf]
    if isinstance(value, str) and not isinstance(current, str):
        try:
            value = coerce(current, value)
        except (ValueError, json.JSONDecodeError) as e:
            raise OverrideError(f"Bad value for {path}: {value!r} ({e})") from e

    try:
        node[leaf] = value
    except TypeError:
        # ConfigDict is type-strict and some fields hold a type the incoming value is not
        # -- the XML/reference paths are `epath` objects, not str. Rebuild the field's own
        # type around the value.
        try:
            node[leaf] = type(current)(value)
        except (TypeError, ValueError) as e:
            raise OverrideError(
                f"Bad value for {path}: {value!r} does not fit a "
                f"{type(current).__name__} field ({e})"
            ) from e


def apply_overrides(cfg, overrides: Mapping[str, Any]):
    """Apply a ``{dotted_key: value}`` mapping onto ``cfg``, in place. Returns ``cfg``."""
    for path, value in overrides.items():
        set_one(cfg, path, value)
    return cfg


def apply_tree(cfg, tree: Mapping[str, Any]):
    """Apply a nested mapping (e.g. a Hydra config group) onto ``cfg``, in place."""
    return apply_overrides(cfg, flatten(tree))


def apply_strings(cfg, overrides: Iterable[str]):
    """Apply ``KEY=VALUE`` strings onto ``cfg``, in place. Returns ``cfg``.

    Nested keys are dotted::

        ctrl_dt=0.02
        reward_terms.joints.weight=0.5
        termination_criteria.pose_error.max_l2_error=6.0
        start_frame_range=0,120
    """
    for item in overrides:
        if "=" not in item:
            raise OverrideError(f"expected KEY=VALUE, got {item!r}")
        path, _, raw = item.partition("=")
        set_one(cfg, path.strip(), raw.strip())
    return cfg
