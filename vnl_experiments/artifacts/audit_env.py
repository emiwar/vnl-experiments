"""Whether a stored artifact was built on the body its run actually trained on.

Until 2026-08-18 every offline env rebuild replaced the run's ``walker_xml_path`` with the
local default (``rodent.xml``), so a run trained on ``rodent_no_tail_collisions.xml`` was
re-simulated on the wrong body -- see :mod:`vnl_experiments.envs.config_io`. 395 artifacts
were affected and one analysis was retracted.

That incident is closed (the reporting and repair-planning tooling was retired on
2026-09-01; the narrative is in ``analysis/README.md``). What survives is the *predicate*,
because it is the safety check on an operation that is still available: ``artifacts
adopt`` hardlinks an artifact from an older producer version onto the current
``spec_id``, and may only do so for artifacts whose bytes the version bump provably
cannot change. :func:`classify` is what decides that, per artifact, from evidence rather
than from the caller's assertion.

The producers stamp ``resolved.walker_xml_path``, which makes the question decidable:

* stamp present -> the artifact says which body it used; compare it to the run's config.
* stamp absent  -> the artifact predates the fix, so the body it used *was* the local
  default. That inference is the one assumption in this module, and it is why such rows
  are labelled ``pre-fix`` rather than pretending to have been measured.

Four verdicts per artifact:

``broken``     the body used differs from the one trained; the artifact must be re-produced,
               and :func:`~vnl_experiments.artifacts.cli.cmd_adopt` refuses it outright.
``adoptable``  pre-fix, but the run trained on the default body, so the fix cannot change
               the output: the file can be hardlinked to the new ``spec_id``.
``repaired``   mismatched, but a current-version replacement already exists.
``ok``         already carries a stamp that agrees with the run's config.
``unknown``    the run is not in the index, so there is nothing to compare against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from vnl_experiments.artifacts.store import MANIFEST_PATH

#: Kinds that rebuild an env from a checkpoint, and so could have used the wrong body.
ENV_KINDS = ("eval", "activations", "video")

#: Config fields the audit compares. ``config_io.XML_FIELDS``, restated so a change there
#: is a visible change here too.
ASSET_FIELDS = ("walker_xml_path", "arena_xml_path")


@dataclass(frozen=True)
class Row:
    """One stored artifact, classified."""

    kind: str
    spec_id: str
    wandb_id: str
    wandb_name: str
    env_class: str
    variant: str
    verdict: str          # broken | adoptable | ok | unknown
    trained: dict[str, str]
    used: dict[str, str]
    stamped: bool
    bytes: int
    reason: str = ""
    spec: tuple[tuple[str, Any], ...] = ()   # hashable, so Row stays frozen-friendly


def _hashable(value: Any) -> Any:
    """Lists in a spec (``datasets``) would make Row unhashable; tuples do not."""
    return tuple(value) if isinstance(value, list) else value


def _default_names(env_class: str) -> dict[str, str]:
    """Asset basenames ``default_config()`` points at, i.e. what pre-fix rebuilds used."""
    from vnl_experiments.delays.evaluation import resolve_env_class

    _, default_config_fn = resolve_env_class(env_class or "AbsoluteImitation", {}, "AbsoluteImitation")
    default = default_config_fn()
    return {field: Path(str(default[field])).name
            for field in ASSET_FIELDS if field in default}


def variant_label(kind: str, spec: Mapping[str, Any]) -> str:
    """Short name for one spec family, from the producer's own ``prefix``.

    Delegating rather than reimplementing matters: a hand-rolled label that dropped the
    dataset count would put ``eval1ds`` and ``eval3ds`` artifacts in one bucket, and the
    ``adopt --from-spec`` that follows would then name the wrong spec_id for half of them.
    """
    from vnl_experiments.artifacts import get_producer

    try:
        producer = get_producer(kind)
        return producer.prefix(producer.spec(**{k: (list(v) if isinstance(v, tuple) else v)
                                                for k, v in dict(spec).items()}))
    except Exception:  # noqa: BLE001 - legacy specs have no reconstructible prefix
        return "legacy"


def _index_frame():
    from vnl_experiments.wandb_utils import index

    df = index.load()
    keep = ["wandb_id", "wandb_name"] + [f"env_params.{f}" for f in ASSET_FIELDS]
    keep += ["env_params.body_target_frame"]
    return df[[c for c in keep if c in df.columns]].set_index("wandb_id")


def classify(*, manifest: Iterable[Mapping[str, Any]] | None = None) -> list[Row]:
    """Classify every stored artifact of an env-rebuilding kind."""
    from vnl_experiments.artifacts.store import Store

    idx = _index_frame()
    store = Store()
    if manifest is None:
        manifest = [json.loads(line) for line in
                    MANIFEST_PATH.read_text().splitlines() if line.strip()]

    rows: list[Row] = []
    for rec in manifest:
        kind = rec["kind"]
        if kind not in ENV_KINDS:
            continue
        wid, sid = rec["wandb_id"], rec["spec_id"]
        spec = rec.get("spec") or {}
        resolved = rec.get("resolved") or {}
        env_class = str(resolved.get("env_class") or "AbsoluteImitation")

        if wid not in idx.index:
            rows.append(Row(kind, sid, wid, "?", env_class, variant_label(kind, spec),
                            "unknown", {}, {}, bool(resolved.get("walker_xml_path")),
                            rec.get("bytes") or 0,
                            "run is not in the local index; sync or it was deleted"))
            continue

        run = idx.loc[wid]
        trained = {f: Path(str(run.get(f"env_params.{f}"))).name
                   for f in ASSET_FIELDS
                   if run.get(f"env_params.{f}") not in (None, "", "nan")}
        defaults = _default_names(env_class)
        stamped = any(resolved.get(f) for f in ASSET_FIELDS)
        used = ({f: str(resolved[f]) for f in ASSET_FIELDS if resolved.get(f)}
                if stamped else dict(defaults))

        # A field the run never logged was at the default during training too, so only
        # compare the fields the run actually recorded.
        mismatched = [f for f in trained if f in used and used[f] != trained[f]]

        # A pre-fix file stays on disk as the historical record even once its replacement
        # exists, so "does this still need work?" is about the *replacement*, not this file.
        # Without this the todo lists could never reach zero.
        new_sid = current_spec_id(kind, spec)
        superseded = bool(new_sid) and new_sid != sid and store.have(kind, wid, new_sid)

        if mismatched and not superseded:
            verdict, reason = "broken", ", ".join(
                f"{f}: used {used[f]} but trained {trained[f]}" for f in mismatched)
        elif mismatched:
            verdict, reason = "repaired", f"superseded by {new_sid}"
        elif stamped:
            verdict, reason = "ok", ""
        elif sid.startswith("legacy-") or spec.get("legacy"):
            verdict, reason = "ok", "legacy spec: unaffected body, no v2 counterpart"
        elif superseded:
            verdict, reason = "repaired", f"already adopted as {new_sid}"
        else:
            verdict, reason = "adoptable", "pre-fix, but trained on the default body"

        rows.append(Row(kind, sid, wid, str(run.get("wandb_name", "?")), env_class,
                        variant_label(kind, spec), verdict, trained, used, stamped,
                        rec.get("bytes") or 0, reason,
                        tuple(sorted((k, _hashable(v)) for k, v in spec.items()))))
    return rows


def current_spec_id(kind: str, spec: Mapping[str, Any] | Iterable[Any]) -> str | None:
    """The ``spec_id`` today's producer would give this spec, or None if it cannot.

    Legacy specs (``{"legacy": True}``) are not reconstructible and return None -- they
    have no v2 counterpart by design, since their true spec was never recorded.
    """
    from vnl_experiments.artifacts import get_producer

    fields = dict(spec)
    fields = {k: list(v) if isinstance(v, tuple) else v for k, v in fields.items()}
    try:
        producer = get_producer(kind)
        return producer.spec_id(producer.spec(**fields))
    except Exception:  # noqa: BLE001
        return None
