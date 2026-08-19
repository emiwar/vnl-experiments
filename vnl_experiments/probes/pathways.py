"""Where each recorded activation sits along a pathway of the network.

Layer keys in an activation recording are *positional* pytree paths
(``3/action/1/predictor/4``), so turning them into a readable depth axis is a manual
mapping that has to be re-derived whenever the container tree changes. Keeping it here
rather than in a plotting script means ``extract.py`` can resolve it **once into
``data.csv``**, and the figures then need no regexes at all -- the stage labels become part
of the committed data.

Verified against the ``RodentForwardModel`` recordings of 2026-08 (30 leaf datasets at
``delay_k == 0``, 31 otherwise): every pattern below matches exactly one leaf, and the
tables together cover every probe, so an unclassified probe means the tree changed.

The actor axis is a shared "depth along the actor": stage 0 is the network *input*
(delayed proprioception + efference copy -- the fair layer-0 baseline), a forward-model
network then runs its predictor (1-5, ending in the prediction p-hat) before the decoder
(6-11). An architecture with no predictor jumps from 0 straight to 6.
"""

from __future__ import annotations

import re
from typing import Iterable

#: Bump when the probe set or the stage mapping changes, so a stale ``data.csv`` shows up
#: as a diff on every row instead of a silent shift. Recorded in ``data.csv``.
PROBE_SET_VERSION = 1

#: (stage_index, label, pattern). Patterns are anchored on the full ``probe`` string.
ACTOR_STAGES: list[tuple[int, str, str]] = [
    (0, "input\n(delayed+eff)", r"^input::delayed_plus_efference$"),
    (1, "pred 1", r"^layer::3/action/1/predictor/0$"),
    (2, "pred 2", r"^layer::3/action/1/predictor/1$"),
    (3, "pred 3", r"^layer::3/action/1/predictor/2$"),
    (4, "pred 4", r"^layer::3/action/1/predictor/3$"),
    (5, "p̂ (277)", r"^layer::3/action/1/predictor/4$"),
    (6, "dec 1", r"^layer::3/action/1/(decoder/)?0$"),
    (7, "dec 2", r"^layer::3/action/1/(decoder/)?1$"),
    (8, "dec 3", r"^layer::3/action/1/(decoder/)?2$"),
    (9, "dec 4", r"^layer::3/action/1/(decoder/)?3$"),
    (10, "head (76)", r"^layer::3/action/1/(decoder/)?4$"),
    (11, "action (38)", r"^layer::3/action/1/(decoder/)?5/action$"),
]

#: The task/goal encoder. Proprioception does **not** pass through it -- it reaches the
#: actor via ``3/action/0/proprioception`` -- so decoding current proprioception here is a
#: leakage control: anything above baseline means the current pose is inferable from the
#: *reference target*, not that the encoder was handed it.
ENCODER_STAGES: list[tuple[int, str, str]] = [
    (0, "task_obs (640)", r"^layer::3/action/0/task_obs/0$"),
    (1, "enc 1", r"^layer::3/action/0/task_obs/1$"),
    (2, "enc 2", r"^layer::3/action/0/task_obs/2$"),
    (3, "enc 3", r"^layer::3/action/0/task_obs/3$"),
    (4, "enc 4", r"^layer::3/action/0/task_obs/4$"),
    (5, "head (64)", r"^layer::3/action/0/task_obs/5$"),
    (6, "latent z (32)", r"^layer::3/action/0/task_obs/6$"),
]

#: The critic sees *current* proprioception (privileged), so it is expected to decode the
#: target well; it is a positive control, not a result.
CRITIC_STAGES: list[tuple[int, str, str]] = [
    (0, "critic in (917)", r"^layer::3/value/0$"),
    (1, "critic 1", r"^layer::3/value/1$"),
    (2, "critic 2", r"^layer::3/value/2$"),
    (3, "value (1)", r"^layer::3/value/3$"),
]

#: Probes that are references rather than points on a pathway.
REFERENCE_PROBES: dict[str, str] = {
    "input::current_proprio":
        "ceiling: decode the target from the target. R^2 ~ 1 or the pipeline is broken.",
    "input::delayed_proprio":
        "floor: how far autocorrelation alone gets you (obs_(t-k) -> obs_t).",
    "layer::3/action/1/delay":
        "the network's own delay buffer; must match the k-shifted floor to ~3 dp.",
    "layer::3/action/0/proprioception":
        "current proprioception entering the actor, post-Normalizer: decodes ~1 by "
        "construction.",
}

#: Normalizer chain, *before* the in-network Delay, so these carry the un-delayed
#: proprioception and decode at R^2 ~ 1. Known leak, kept as a check rather than dropped.
PREPROCESSING = r"^layer::(0/state/|1/state/|2/)"

#: Recorded alongside the action but not an activation of interest.
DIAGNOSTIC = r"^layer::3/action/1/(decoder/)?5/log_likelihood$"

_TABLES: list[tuple[str, list[tuple[int, str, str]]]] = [
    ("actor", ACTOR_STAGES),
    ("encoder", ENCODER_STAGES),
    ("critic", CRITIC_STAGES),
]
_COMPILED = [(pathway, index, label, re.compile(pattern))
             for pathway, table in _TABLES for index, label, pattern in table]
_PREPROCESSING = re.compile(PREPROCESSING)
_DIAGNOSTIC = re.compile(DIAGNOSTIC)


def resolve_stage(probe: str) -> tuple[str, int | None, str]:
    """``probe`` -> ``(pathway, stage_index, stage_label)``.

    Off-pathway probes get ``("reference" | "preprocessing" | "diagnostic", None, "")``,
    and anything unrecognised ``("other", None, "")`` -- which callers should treat as an
    error rather than a filter, since a silently unclassified leaf vanishes from figures.
    """
    if probe in REFERENCE_PROBES:
        return "reference", None, ""
    for pathway, index, label, pattern in _COMPILED:
        if pattern.match(probe):
            return pathway, index, label
    if _DIAGNOSTIC.match(probe):
        return "diagnostic", None, ""
    if _PREPROCESSING.match(probe):
        return "preprocessing", None, ""
    return "other", None, ""


def stage_axis(pathway: str) -> list[tuple[int, str]]:
    """``[(stage_index, label)]`` for one pathway, for building a plot axis."""
    table = {"actor": ACTOR_STAGES, "encoder": ENCODER_STAGES,
             "critic": CRITIC_STAGES}[pathway]
    return [(index, label) for index, label, _ in table]


def unclassified(probes: Iterable[str]) -> list[str]:
    """Probes no table claims. Should be empty; assert on it in ``extract.py``."""
    return [p for p in probes if resolve_stage(p)[0] == "other"]
