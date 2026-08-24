"""The artifact store: derived per-run data, with a record of how it was made.

Anything expensive that is computed *from* a run -- an offline evaluation, a training
history, recorded unit activations, a rollout video -- is an **artifact**. Artifacts are
keyed by ``(kind, wandb_id, spec_id)`` and live under a single root::

    $VNL_ARTIFACTS/                       (default: <repo>/artifacts/, gitignored)
      eval/<wandb_id>/<spec_id>.json
      eval/<wandb_id>/<spec_id>.meta.json     <- how it was made
      history/<wandb_id>/<spec_id>.parquet
      activations/<wandb_id>/<spec_id>.h5
      video/<wandb_id>/<spec_id>.mp4

Every artifact carries a **sidecar** ``.meta.json`` holding its full spec plus the
producer (module, version, git commit, host, GPU) and a checksum. The sidecar -- not any
central file -- is the source of truth, which is what makes the store work unchanged on
the laptop and on the cluster: ``rsync`` a ``<kind>/<wandb_id>/`` directory across and
the provenance travels with the bytes.

``analysis/_artifacts/manifest.jsonl`` is a **committed index** of the local store, one
line per artifact, sorted by ``(kind, wandb_id, spec_id)``. It is derived -- rebuild it
at any time with ``python -m vnl_experiments.artifacts reindex`` -- but committing it
means an analysis folder's requirements can be checked, and the history of what was
computed can be read, without the (gitignored, possibly multi-gigabyte) store present.

The ``spec_id``
---------------
``spec_id`` is a readable prefix plus an 8-character hash of the normalised spec *and*
the producer version, e.g. ``eval3ds-s600064000-4f1a2b91``. Two consequences worth
stating explicitly:

* Changing a producer's ``VERSION`` changes every ``spec_id`` it emits, so a fixed eval
  code version is baked into the identity of its outputs. Records from different eval
  code versions can never be silently mixed into one figure.
* The eval spec pins the **resolved** ``checkpoint_step``, never ``"last"``. Since
  ``0b4de48`` training no longer writes an extra checkpoint when it finishes, so an
  inline end-of-training ``eval.json`` (which measured the in-memory network) and a later
  ``eval_runs.py`` pass (which restores the newest checkpoint on disk) may not describe
  the same weights. They are different specs and are never interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from vnl_experiments.provenance import repo_versions

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "analysis" / "_artifacts" / "manifest.jsonl"
DEFAULT_ROOT = REPO_ROOT / "artifacts"

KINDS = ("history", "eval", "activations", "video")

#: Files larger than this are recorded by size only. Checksumming the 22 GB of
#: activations on every reindex is not worth the minutes it costs.
HASH_SIZE_LIMIT = 256 * 1024 * 1024


def store_root() -> Path:
    """The artifact store root: ``$VNL_ARTIFACTS`` or ``<repo>/artifacts``."""
    return Path(os.environ.get("VNL_ARTIFACTS", DEFAULT_ROOT)).expanduser()


# --------------------------------------------------------------------------------------
# spec identity
# --------------------------------------------------------------------------------------


def normalise_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical form of a spec: sorted keys, lists (not tuples/sets), no ``None`` noise.

    ``None`` values are dropped so that adding an explicitly-defaulted key to a producer
    does not invalidate every artifact it made before.
    """
    out: dict[str, Any] = {}
    for key in sorted(spec):
        value = spec[key]
        if value is None:
            continue
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        elif isinstance(value, tuple):
            value = list(value)
        out[key] = value
    return out


def spec_id(prefix: str, spec: Mapping[str, Any], producer_version: int) -> str:
    """``<prefix>-<8 hex chars>``, stable across machines and Python versions."""
    payload = json.dumps({"spec": normalise_spec(spec), "v": producer_version},
                         sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{prefix}-{digest}"


def file_sha256(path: Path, *, limit: int = HASH_SIZE_LIMIT) -> str | None:
    """SHA-256 of ``path``, or ``None`` if it is larger than ``limit``."""
    if path.stat().st_size > limit:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return (out.stdout.strip().splitlines() or [None])[0]
    except Exception:  # noqa: BLE001
        return None


def _cuda_driver() -> str | None:
    """NVIDIA driver version, which nothing else records.

    WandB's per-run metadata has ``cudaVersion`` (the toolkit) and the GPU model, but
    not the driver -- and the driver is exactly what a cluster OS update changes.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return (out.stdout.strip().splitlines() or [None])[0]
    except Exception:  # noqa: BLE001
        return None


#: Packages whose version can change a *number* an artifact reports: the physics, the
#: linear algebra, and the checkpoint reader. Recorded per artifact because artifacts are
#: produced long after the run, often on a different software stack -- a run's WandB
#: ``requirements.txt`` says nothing about the environment that later evaluated it.
STACK_PACKAGES = ("jax", "jaxlib", "mujoco", "mujoco-mjx", "warp-lang", "flax",
                  "orbax-checkpoint", "numpy")


def _stack_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in STACK_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def producer_stamp(module: str, version: int, **extra: Any) -> dict[str, Any]:
    """The ``producer`` block written into every sidecar."""
    stamp = {
        "module": module,
        "version": version,
        "git_commit": _git_commit(),
        "host": socket.gethostname(),
        "gpu": _gpu_name(),
        "cuda_driver": _cuda_driver(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": _stack_versions() or None,
        # git state of all three editable repos, not just vnl-experiments: nnx-ppo is
        # the algorithm and vnl-playground is the task, and both can change a number.
        "repos": repo_versions() or None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    stamp.update(extra)
    return {k: v for k, v in stamp.items() if v is not None}


# --------------------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One manifest / sidecar record."""

    kind: str
    wandb_id: str
    spec_id: str
    spec: dict[str, Any]
    producer: dict[str, Any]
    created_at: str
    path: str          # relative to the store root
    bytes: int
    sha256: str | None
    #: Facts discovered while producing, e.g. the ``checkpoint_step`` actually restored
    #: or the number of clips evaluated. Deliberately **not** part of ``spec_id``: they
    #: cannot be known before the artifact exists, so hashing them would make
    #: "do I already have this?" unanswerable. Coverage checks assert on them instead.
    resolved: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "wandb_id": self.wandb_id, "spec_id": self.spec_id,
            "spec": self.spec, "producer": self.producer, "resolved": self.resolved,
            "created_at": self.created_at,
            "path": self.path, "bytes": self.bytes, "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Entry":
        return cls(
            kind=obj["kind"], wandb_id=obj["wandb_id"], spec_id=obj["spec_id"],
            spec=obj.get("spec", {}), producer=obj.get("producer", {}),
            resolved=obj.get("resolved", {}),
            created_at=obj.get("created_at", ""), path=obj["path"],
            bytes=obj.get("bytes", 0), sha256=obj.get("sha256"),
        )


class Store:
    """Filesystem artifact store rooted at :func:`store_root` (override with ``root``)."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else store_root()

    # -- paths ---------------------------------------------------------------------

    def dir_for(self, kind: str, wandb_id: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"unknown artifact kind {kind!r}; expected one of {KINDS}")
        return self.root / kind / wandb_id

    def path_for(self, kind: str, wandb_id: str, sid: str, ext: str) -> Path:
        return self.dir_for(kind, wandb_id) / f"{sid}{ext}"

    def meta_path(self, kind: str, wandb_id: str, sid: str) -> Path:
        return self.dir_for(kind, wandb_id) / f"{sid}.meta.json"

    # -- read ----------------------------------------------------------------------

    def lookup(self, kind: str, wandb_id: str, sid: str) -> Entry | None:
        """The sidecar for one artifact, or ``None`` if it (or its data file) is absent."""
        meta = self.meta_path(kind, wandb_id, sid)
        if not meta.exists():
            return None
        entry = Entry.from_json(json.loads(meta.read_text()))
        return entry if (self.root / entry.path).exists() else None

    def have(self, kind: str, wandb_id: str, sid: str) -> bool:
        return self.lookup(kind, wandb_id, sid) is not None

    def missing(self, kind: str, sid: str, wandb_ids: Iterable[str]) -> list[str]:
        """Which of ``wandb_ids`` lack this artifact locally."""
        return [wid for wid in wandb_ids if not self.have(kind, wid, sid)]

    def other_specs(self, kind: str, wandb_ids: Iterable[str],
                    exclude: str) -> dict[str, int]:
        """Other ``spec_id``s of this kind held for these runs, and how many have each.

        Surfaced when a requirement is unmet so the difference between "no data" and
        "data made by a different eval version" is visible rather than mystifying.
        """
        counts: dict[str, int] = {}
        for wid in wandb_ids:
            directory = self.dir_for(kind, wid)
            if not directory.exists():
                continue
            for meta in directory.glob("*.meta.json"):
                sid = meta.name[: -len(".meta.json")]
                if sid != exclude:
                    counts[sid] = counts.get(sid, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def scan(self) -> list[Entry]:
        """Every artifact present in the store, discovered from its sidecars."""
        entries = []
        for meta in sorted(self.root.glob("*/*/*.meta.json")):
            try:
                entry = Entry.from_json(json.loads(meta.read_text()))
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"  ! unreadable sidecar {meta}: {exc}")
                continue
            entries.append(entry)
        return sorted(entries, key=lambda e: (e.kind, e.wandb_id, e.spec_id))

    # -- write ---------------------------------------------------------------------

    def record(self, kind: str, wandb_id: str, sid: str, data_path: Path, *,
               spec: Mapping[str, Any], producer: Mapping[str, Any],
               resolved: Mapping[str, Any] | None = None) -> Entry:
        """Write the sidecar for an artifact file that has just been produced."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        entry = Entry(
            kind=kind, wandb_id=wandb_id, spec_id=sid,
            spec=normalise_spec(spec), producer=dict(producer),
            resolved=dict(resolved or {}),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            path=str(data_path.relative_to(self.root)),
            bytes=data_path.stat().st_size,
            sha256=file_sha256(data_path),
        )
        meta = self.meta_path(kind, wandb_id, sid)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(entry.to_json(), indent=2, sort_keys=True) + "\n")
        return entry

    # -- manifest ------------------------------------------------------------------

    def reindex(self, manifest_path: Path = MANIFEST_PATH) -> list[Entry]:
        """Rewrite the committed manifest from the sidecars actually present."""
        entries = self.scan()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as fh:
            for entry in entries:
                json.dump(entry.to_json(), fh, sort_keys=True, separators=(",", ":"))
                fh.write("\n")
        return entries


def read_manifest(manifest_path: Path = MANIFEST_PATH) -> list[Entry]:
    """The committed manifest (works with no store present)."""
    if not manifest_path.exists():
        return []
    entries = []
    with manifest_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(Entry.from_json(json.loads(line)))
    return entries


def manifest_df(manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """The manifest as a frame: one row per artifact, ``spec``/``producer`` flattened."""
    rows = []
    for entry in read_manifest(manifest_path):
        row = {"kind": entry.kind, "wandb_id": entry.wandb_id, "spec_id": entry.spec_id,
               "created_at": entry.created_at, "bytes": entry.bytes, "path": entry.path}
        row.update({f"spec.{k}": v for k, v in entry.spec.items()})
        row.update({f"resolved.{k}": v for k, v in entry.resolved.items()})
        row.update({f"producer.{k}": v for k, v in entry.producer.items()})
        rows.append(row)
    return pd.DataFrame(rows)
