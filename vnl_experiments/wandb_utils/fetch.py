"""Fetch runs from WandB and flatten them into tidy records.

This is the *only* place WandB-fetching logic should live. Analysis ``extract.py``
scripts call these helpers, build a :class:`pandas.DataFrame`, and write it to a
committed ``data.csv``. Plotting scripts then read that CSV and never touch the API
(see ``analysis/README.md``).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import wandb

# Default metadata columns that every record carries, so any analysis can audit
# provenance and comparability without re-fetching.
PROVENANCE_KEYS = (
    "wandb_id",
    "wandb_name",
    "wandb_project",
    "state",
    "git_commit",
    "tags",
    "notes",
)


def fetch_runs(
    project: str,
    *,
    finished_only: bool = True,
    tags: Sequence[str] | None = None,
    per_page: int = 300,
    timeout: int = 60,
):
    """Return WandB runs for ``project`` (``entity/project`` form).

    Args:
        finished_only: keep only runs whose ``state == "finished"``.
        tags: if given, keep only runs that carry *all* of these tags.
        per_page / timeout: forwarded to the WandB API.
    """
    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(project, per_page=per_page, order="-created_at"))
    if finished_only:
        runs = [r for r in runs if r.state == "finished"]
    if tags:
        wanted = set(tags)
        runs = [r for r in runs if wanted.issubset(set(r.tags))]
    return runs


def _git_commit(run) -> str | None:
    meta = getattr(run, "metadata", None)
    if not meta:
        return None
    return (meta.get("git", {}) or {}).get("commit")


def run_record(
    run,
    *,
    config_keys: Iterable[str] = (),
    net_param_keys: Iterable[str] = (),
    ppo_keys: Iterable[str] = (),
    metrics: Iterable[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten a single WandB run into a flat dict.

    Always includes the :data:`PROVENANCE_KEYS` (notably ``wandb_id`` and
    ``git_commit``). Pulls top-level ``config`` entries (``config_keys``), nested
    ``config['net_params']`` (``net_param_keys``) and ``config['config']['ppo']``
    (``ppo_keys``), and ``summary`` metrics (``metrics``). The actual number of
    trained steps is exposed as ``actual_step`` (``summary['_step']``).
    """
    cfg = run.config or {}
    net = cfg.get("net_params", {}) or {}
    inner = cfg.get("config", {}) if isinstance(cfg.get("config"), dict) else {}
    ppo = inner.get("ppo", {}) or {}
    summary = run.summary

    rec: dict[str, Any] = {
        "wandb_id": run.id,
        "wandb_name": run.name,
        "wandb_project": run.project,
        "state": run.state,
        "git_commit": _git_commit(run),
        "tags": ",".join(run.tags),
        "notes": getattr(run, "notes", None),
    }
    for k in config_keys:
        rec[k] = cfg.get(k)
    for k in net_param_keys:
        rec[k] = net.get(k)
    for k in ppo_keys:
        rec[k] = ppo.get(k)
    for k in metrics:
        rec[k] = summary.get(k)
    rec["actual_step"] = summary.get("_step")
    if extra:
        rec.update(extra)
    return rec


def records_to_df(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame from records, with provenance columns ordered first."""
    df = pd.DataFrame(list(records))
    front = [c for c in PROVENANCE_KEYS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]
