"""Local, incremental mirror of a WandB project's run metadata.

Why this exists
---------------
Hitting the WandB API is slow: listing a 466-run project takes ~70 s, and pulling
``config`` / ``summary`` / ``metadata`` costs a further ~1 s per run serially. Every
``extract.py`` used to pay that cost from scratch, for the same runs, in 17 different
folders. This module pays it **once** and keeps the answer on disk:

    analysis/_runs/<project-slug>.jsonl     committed, one JSON object per run,
                                            sorted by wandb_id

Because the file is sorted by id and has one line per run, ``git diff`` after a sync
shows exactly which runs entered the project and how their summaries moved — the index
doubles as a version-controlled log of the experiment history.

Usage
-----
Refresh the mirror (incremental; only new/changed runs are re-fetched)::

    python -m vnl_experiments.wandb_utils.index sync
    python -m vnl_experiments.wandb_utils.index sync --full      # ignore the cache
    python -m vnl_experiments.wandb_utils.index info

Read it (no network, no WandB import needed)::

    from vnl_experiments.wandb_utils import index

    df = index.load()
    runs = index.select(df, tags="TrainEvalSplit", state="finished",
                        **{"env_params.walker_xml_path": NEW_XML})

Frame layout
------------
:func:`load` returns one row per run with **dotted column names**:

===============================================  ==========================================
``wandb_id``, ``wandb_name``, ``state``, ...     run-level metadata (see :data:`META_FIELDS`)
``gpu``, ``host``, ``slurm_job_id``              from ``run.metadata`` -- the GPU model is a
                                                 first-class column because A100-vs-H200 is a
                                                 throughput confound that has already bitten
                                                 one analysis
``env_params.body_target_frame``                 flattened ``run.config``
``config.ppo.n_envs``                            (the PPO block is nested under a config key
                                                 literally named ``config``)
``summary.eval/episode_reward/mean``             flattened ``run.summary``, scalars only
===============================================  ==========================================

List-valued config entries (``enc_hidden_sizes``) are stored as their JSON string so they
stay hashable and can be grouped / compared for equality.

This module never fetches history. Training curves are a *Tier 1* artifact -- see
``vnl_experiments.artifacts`` -- because they are per-run bulk data, not metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "analysis" / "_runs"

DEFAULT_PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: Run-level fields stored outside ``config`` / ``summary``. These become bare columns.
META_FIELDS = (
    "wandb_id",
    "wandb_name",
    "wandb_entity",
    "wandb_project",
    "state",
    "created_at",
    "heartbeat_at",
    "runtime_s",
    "tags",
    "notes",
    "git_commit",
    "gpu",
    "gpu_count",
    "host",
    "slurm_job_id",
    "program",
    "args",
    # Software stack. WandB records these per run in wandb-metadata.json but nothing
    # used to surface them, so a driver/OS/CUDA change was an invisible confound -- the
    # same failure mode as reading a training script instead of `env_params`. Add them to
    # INVARIANTS in any analysis whose runs straddle a cluster upgrade.
    "os",
    "cuda_version",
    "python",
)

_SCALAR = (str, int, float, bool, type(None))


# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------


def project_slug(project: str) -> str:
    """``entity/project`` (or bare ``project``) -> the filename stem used on disk."""
    return project.split("/")[-1]


def index_path(project: str = DEFAULT_PROJECT) -> Path:
    return INDEX_DIR / f"{project_slug(project)}.jsonl"


# --------------------------------------------------------------------------------------
# record building
# --------------------------------------------------------------------------------------


def _runtime_seconds(created_at: str | None, heartbeat_at: str | None) -> float | None:
    if not created_at or not heartbeat_at:
        return None
    fmt = "%Y-%m-%dT%H:%M:%S%z"
    try:
        t0 = datetime.strptime(created_at.replace("Z", "+0000"), fmt)
        t1 = datetime.strptime(heartbeat_at.replace("Z", "+0000"), fmt)
    except ValueError:
        return None
    return (t1 - t0).total_seconds()


def _clean_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Keep scalar summary entries; drop WandB's nested bookkeeping and media objects."""
    out: dict[str, Any] = {}
    for key, value in summary.items():
        if key.startswith("_") and key not in ("_step", "_runtime", "_timestamp"):
            continue
        if isinstance(value, _SCALAR):
            out[key] = value
    return out


def build_record(run) -> dict[str, Any]:
    """Flatten one WandB ``Run`` into the JSON object stored on a line of the index.

    ``run.metadata`` downloads the run's ``wandb-metadata.json`` (this is the expensive
    part, ~0.8 s), which is where the GPU model, host and SLURM job id live.
    """
    meta = {}
    try:
        meta = run.metadata or {}
    except Exception:  # noqa: BLE001 - a missing metadata file must not abort a sync
        meta = {}

    created_at = getattr(run, "createdAt", None)
    heartbeat_at = getattr(run, "heartbeatAt", None)
    slurm = meta.get("slurm") or {}

    return {
        "wandb_id": run.id,
        "wandb_name": run.name,
        "wandb_entity": run.entity,
        "wandb_project": run.project,
        "state": run.state,
        "created_at": created_at,
        "heartbeat_at": heartbeat_at,
        "runtime_s": _runtime_seconds(created_at, heartbeat_at),
        "tags": sorted(run.tags or []),
        "notes": getattr(run, "notes", None),
        # `run.commit` is served with the run listing; `run.metadata["git"]["commit"]`
        # says the same thing but costs a file download, so prefer the cheap one.
        "git_commit": getattr(run, "commit", None) or (meta.get("git") or {}).get("commit"),
        "gpu": meta.get("gpu"),
        "gpu_count": meta.get("gpu_count"),
        "host": meta.get("host"),
        "slurm_job_id": slurm.get("job_id") or slurm.get("jobid"),
        "program": meta.get("program"),
        "args": meta.get("args"),
        # Kernel + glibc, CUDA toolkit, interpreter. Backfilled only by `sync --full`:
        # `_is_fresh` keeps cached records, so pre-existing rows lack these until then.
        "os": meta.get("os"),
        "cuda_version": meta.get("cudaVersion"),
        "python": meta.get("python"),
        "config": run.config or {},
        "summary": _clean_summary(run.summary),
        "_synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _is_fresh(cached: Mapping[str, Any], run) -> bool:
    """True if the cached record still describes ``run``.

    A finished run's ``heartbeatAt`` stops advancing, so (state, heartbeat) is a
    sufficient staleness key. A run that is still training is always re-fetched.
    """
    if cached.get("state") != run.state:
        return False
    if run.state not in ("finished", "failed", "crashed", "killed"):
        return False
    return cached.get("heartbeat_at") == getattr(run, "heartbeatAt", None)


# --------------------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------------------


def read_index(project: str = DEFAULT_PROJECT) -> dict[str, dict[str, Any]]:
    """Read the committed index as ``{wandb_id: record}`` (empty if it does not exist)."""
    path = index_path(project)
    if not path.exists():
        return {}
    records = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["wandb_id"]] = rec
    return records


def write_index(records: Mapping[str, Mapping[str, Any]],
                project: str = DEFAULT_PROJECT) -> Path:
    """Write ``{wandb_id: record}`` sorted by id, one compact JSON object per line."""
    path = index_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for wid in sorted(records):
            json.dump(records[wid], fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
    return path


def sync(project: str = DEFAULT_PROJECT, *, workers: int = 8, full: bool = False,
         verbose: bool = True) -> dict[str, int]:
    """Refresh the on-disk index for ``project``. Returns counts of what changed.

    Only runs that are new or whose ``(state, heartbeatAt)`` moved are re-fetched, so a
    routine sync after a few new runs takes seconds. ``full=True`` re-fetches everything.

    Fetching is threaded (``workers``) because it is entirely I/O-bound; 8 threads take
    the per-run cost from ~1 s to ~0.14 s.
    """
    import wandb  # imported lazily: reading the index must not require wandb

    cached = {} if full else read_index(project)

    t0 = time.perf_counter()
    api = wandb.Api(timeout=120)
    runs = list(api.runs(project, per_page=500, order="-created_at"))
    if verbose:
        print(f"listed {len(runs)} runs in {time.perf_counter() - t0:.0f} s")

    stale = [r for r in runs if not _is_fresh(cached.get(r.id, {}), r)]
    if verbose:
        print(f"{len(stale)} to fetch, {len(runs) - len(stale)} reused from the index")

    t0 = time.perf_counter()
    fetched: list[dict[str, Any]] = []
    if stale:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, rec in enumerate(pool.map(build_record, stale), start=1):
                fetched.append(rec)
                if verbose and i % 50 == 0:
                    print(f"  {i}/{len(stale)} ({time.perf_counter() - t0:.0f} s)")

    added = sum(1 for rec in fetched if rec["wandb_id"] not in cached)
    records = dict(cached)
    records.update({rec["wandb_id"]: rec for rec in fetched})

    # A run deleted from WandB should leave the index too, or selectors would keep
    # picking up a run nobody can look at any more.
    live = {r.id for r in runs}
    removed = [wid for wid in records if wid not in live]
    for wid in removed:
        del records[wid]

    path = write_index(records, project)

    counts = {"total": len(records), "fetched": len(fetched), "added": added,
              "updated": len(fetched) - added, "removed": len(removed)}
    if verbose:
        print(f"wrote {path.relative_to(REPO_ROOT)}: " +
              ", ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


# --------------------------------------------------------------------------------------
# load / select
# --------------------------------------------------------------------------------------


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), sub, out)
    elif isinstance(value, (list, tuple)):
        # Kept as a JSON string so the column stays hashable: analyses group and compare
        # architectures by equality (`enc_hidden_sizes == "[512, 512, 512, 512]"`).
        out[prefix] = json.dumps(list(value))
    else:
        out[prefix] = value


def flatten_record(rec: Mapping[str, Any]) -> dict[str, Any]:
    """One index line -> one flat row with dotted column names."""
    row: dict[str, Any] = {}
    for field in META_FIELDS:
        value = rec.get(field)
        if field == "tags":
            row[field] = ",".join(value or [])
        elif isinstance(value, (list, tuple)):
            row[field] = json.dumps(list(value))
        else:
            row[field] = value
    _flatten("", rec.get("config") or {}, row)
    for key, value in (rec.get("summary") or {}).items():
        row[f"summary.{key}"] = value
    row["_synced_at"] = rec.get("_synced_at")
    return row


def load(project: str = DEFAULT_PROJECT) -> pd.DataFrame:
    """Load the index as a flat DataFrame. No network access.

    Reading and flattening 466 runs takes ~0.08 s, so there is deliberately no cache
    layer here -- a stale cache would cost more in confusion than it saves in time.
    """
    path = index_path(project)
    if not path.exists():
        raise FileNotFoundError(
            f"No run index at {path}. Create it with:\n"
            f"    python -m vnl_experiments.wandb_utils.index sync --project {project}"
        )

    df = pd.DataFrame([flatten_record(rec) for rec in read_index(project).values()])
    return df.sort_values("wandb_id", ignore_index=True)


def has_tags(df: pd.DataFrame, tags: str | Iterable[str]) -> pd.Series:
    """Boolean mask: rows carrying *all* of ``tags``."""
    wanted = [tags] if isinstance(tags, str) else list(tags)
    mask = pd.Series(True, index=df.index)
    for tag in wanted:
        mask &= df["tags"].fillna("").str.split(",").apply(lambda ts: tag in ts)
    return mask


def select(df: pd.DataFrame, *, tags: str | Iterable[str] | None = None,
           **filters: Any) -> pd.DataFrame:
    """Filter the index frame.

    ``tags`` requires *all* of the given tags. Every other keyword is a column name
    (dotted, so pass ``env_params.walker_xml_path`` via ``**{...}``) matched against a
    scalar or, if a list/tuple/set is given, against membership::

        select(df, tags=["TrainEvalSplit"], state="finished",
               **{"env_params.torque_actuators": True, "net_params.delay": [0, 5, 10]})

    Unknown columns raise, rather than silently returning nothing -- a mistyped config
    key that quietly yields an empty cohort is the failure mode this guards against.
    """
    out = df
    if tags is not None:
        out = out[has_tags(out, tags)]
    for column, wanted in filters.items():
        if column not in out.columns:
            raise KeyError(
                f"{column!r} is not a column of the index. Close matches: "
                f"{[c for c in df.columns if column.split('.')[-1] in c][:8]}"
            )
        if isinstance(wanted, (list, tuple, set)):
            out = out[out[column].isin(list(wanted))]
        else:
            out = out[out[column] == wanted]
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _cmd_info(project: str) -> None:
    df = load(project)
    path = index_path(project)
    print(f"{path.relative_to(REPO_ROOT)}: {len(df)} runs, "
          f"{path.stat().st_size / 1e6:.1f} MB, {len(df.columns)} columns")
    print(f"  synced:  {df['_synced_at'].max()}")
    print(f"  created: {df['created_at'].min()} .. {df['created_at'].max()}")
    print("\nby state:")
    print(df["state"].value_counts().to_string())
    if "gpu" in df:
        print("\nby gpu:")
        print(df["gpu"].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sync", "info"])
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full", action="store_true",
                        help="re-fetch every run instead of only new/changed ones")
    args = parser.parse_args()

    if args.command == "sync":
        sync(args.project, workers=args.workers, full=args.full)
    else:
        _cmd_info(args.project)


if __name__ == "__main__":
    main()
