"""Distilling a frozen, privileged teacher into a delayed student.

The training entry point switches to ``nnx_ppo.algorithms.distillation.train_distillation``
when ``distill.teacher`` names a run directory (and the train group is a distillation one,
e.g. ``train=dmc_distill``). This module holds the two pieces specific to that path:

* :func:`load_teacher` -- rebuild and restore the teacher from its run directory, through
  the same ``network_builders.load_network`` the offline eval uses;
* :func:`sampler_target` -- the ``target_fn`` that maps the teacher's rollout_extras onto
  the student's tree.

Why a ``target_fn`` is needed: nnx-ppo's distillation feeds the stored target back to the
student as its own ``rollout_extras`` during the loss replay, and containers route extras
by position. An undelayed teacher and a delayed student do not have the same skeleton --
the student has an extra ``Delay`` layer, and may be recurrent or a forward model -- so the
teacher's tree cannot be used as it is. :func:`sampler_target` keeps the student's own
extras and replaces only the sampler leaf (the raw action, which for the teacher in eval
mode is its mean) with the teacher's. See nnx-ppo's ``docs/reference/distillation.rst``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax

from vnl_experiments.config import OverrideError
from vnl_experiments.delays.network_builders import (
    _parse_net_params,
    get_architecture,
    load_network,
)


@dataclasses.dataclass
class Teacher:
    """A restored teacher plus what the run record needs to know about it."""

    network: Any
    run_dir: Path
    step: int
    net_params: dict
    env_params: dict

    def info(self) -> dict:
        """The ``distill`` block of the WandB config and ``config.json``."""
        return {
            "teacher": str(self.run_dir),
            "teacher_name": self.run_dir.name,
            "teacher_step": self.step,
            "teacher_network_class": self.net_params.get("network_class"),
            "teacher_delay_k": self.net_params.get("delay_k"),
            "teacher_efference_length": self.net_params.get("efference_length"),
        }


def resolve_teacher_dir(teacher: str, checkpoint_root: str) -> Path:
    """The teacher's run directory: as given, else under ``checkpoint_root``."""
    for candidate in (Path(teacher), Path(checkpoint_root) / teacher):
        if (candidate / "config.json").is_file():
            return candidate
    raise OverrideError(
        f"distill.teacher={teacher!r}: no config.json found there or under "
        f"{checkpoint_root!r}. The teacher must be a run trained through "
        f"`python -m vnl_experiments.train` (pre-Hydra dm_control runs wrote no "
        f"config.json and cannot be rebuilt)."
    )


def load_teacher(teacher: str, checkpoint_root: str, *, env, env_config,
                 obs_layout: str, seed: int) -> Teacher:
    """Rebuild the teacher on ``env`` and restore its latest checkpoint.

    Refuses a teacher whose observation layout differs from the env's, since it could not
    even be built on it. Warns -- does not refuse -- when the teacher is delayed or was
    trained on a differently configured env: both are legitimate experiments, but neither
    is the "privileged undelayed teacher" this path is for, so they should be deliberate.
    """
    run_dir = resolve_teacher_dir(teacher, checkpoint_root)
    stored = json.loads((run_dir / "config.json").read_text())
    net_params = _parse_net_params(stored.get("net_params", {}))
    env_params = stored.get("env_params", {})

    arch = get_architecture(net_params.get("network_class", ""))
    if arch is None:
        raise OverrideError(
            f"distill.teacher: unknown network_class "
            f"{net_params.get('network_class')!r} in {run_dir / 'config.json'}.")
    if arch.obs_layout != obs_layout:
        raise OverrideError(
            f"distill.teacher is a {arch.name} ({arch.obs_layout!r} observation) but this "
            f"env has a {obs_layout!r} observation.")

    if int(net_params.get("delay_k", 0) or 0) != 0:
        print(f"  WARNING: the teacher {run_dir.name} was trained with "
              f"delay_k={net_params['delay_k']}; distillation is meant to start from an "
              f"undelayed (privileged) teacher.")

    rebuilt = json.loads(json.dumps(env_config.to_dict(), default=str))
    changed = sorted(k for k in set(env_params) | set(rebuilt)
                     if env_params.get(k) != rebuilt.get(k))
    if changed:
        print(f"  WARNING: the teacher's env config differs from this run's in: {changed}")
        for k in changed:
            print(f"    {k}: teacher={env_params.get(k)!r} this run={rebuilt.get(k)!r}")

    loaded = load_network(run_dir, net_params, env, seed)
    if loaded is None:
        raise OverrideError(f"distill.teacher: could not restore a checkpoint from {run_dir}.")
    network, step = loaded
    print(f"  teacher: {arch.name} from {run_dir} at step {step}")
    return Teacher(network=network, run_dir=run_dir, step=step,
                   net_params=net_params, env_params=env_params)


# ---------------------------------------------------------------------------
# target_fn
# ---------------------------------------------------------------------------

def _is_action_path(path) -> bool:
    """Whether a pytree path runs through the ``"action"`` port of a ``PPOAdapter``."""
    return any(isinstance(k, jax.tree_util.DictKey) and k.key == "action" for k in path)


def _action_leaf(extras, who: str):
    """Index (in flatten order) and value of the single sampler leaf in ``extras``.

    Under a ``PPOAdapter``'s ``action`` port every module but the sampler emits None
    (``Delay``, ``EfferenceCopy`` passes through, the MLP layers, the RNN cells, the
    forward model's predictor), so the one array leaf there *is* the sampler's raw action.
    The Normalizer's emission sits outside the adapter and is never under ``action``.
    Anything else -- several samplers, or none -- is refused rather than guessed at.
    """
    leaves = jax.tree_util.tree_flatten_with_path(extras)[0]
    hits = [(i, leaf) for i, (path, leaf) in enumerate(leaves) if _is_action_path(path)]
    if len(hits) != 1:
        raise ValueError(
            f"sampler_target: expected exactly one sampler leaf under an 'action' key in "
            f"the {who}'s rollout_extras, found {len(hits)}. Multi-head networks need a "
            f"per-head mapping.")
    return hits[0]


def sampler_target(teacher_extras, student_extras):
    """The student's own rollout_extras, with its sampler leaf replaced by the teacher's.

    The ``target_fn`` passed to ``train_distillation``. Module-level so it is hashable and
    stable across calls (it is a static argument of the jitted step). Every check here
    runs at trace time, so it costs nothing per step.
    """
    _, t_leaf = _action_leaf(teacher_extras, "teacher")
    s_index, s_leaf = _action_leaf(student_extras, "student")
    if t_leaf.shape != s_leaf.shape:
        raise ValueError(
            f"sampler_target: teacher action {t_leaf.shape} vs student action "
            f"{s_leaf.shape}; the two networks do not act in the same space.")
    leaves, treedef = jax.tree_util.tree_flatten(student_extras)
    leaves[s_index] = t_leaf.astype(s_leaf.dtype)
    return jax.tree_util.tree_unflatten(treedef, leaves)
