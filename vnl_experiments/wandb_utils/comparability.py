"""Programmatic comparability checks for a set of runs.

The pipeline policy (``analysis/README.md``) requires that before any two
conditions are plotted together, we verify the runs are actually comparable:
trained for the same number of steps, same hyperparameters, same env, and ideally
the same git commit. These helpers produce a human-readable report that
``extract.py`` prints *and* writes to ``comparability.txt`` next to ``data.csv``.

This is the programmatic half. The manual half — independently eyeballing configs,
tags and notes, and running ``git diff`` when commits differ — is done by the
analyst (see the policy).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

# Columns that should normally be constant across every run in a comparison.
DEFAULT_INVARIANTS = (
    "git_commit",
    "env",
    "latent_size",
    "kl_weight",
    "body_target_frame",
    "enc_hidden_sizes",
    "dec_hidden_sizes",
    "critic_hidden_sizes",
    "n_envs",
    "total_steps",
    "actual_step",
)


def _hashable(v):
    """Make list/dict config values hashable so they can be deduplicated."""
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


def _unique(series: pd.Series):
    return sorted({_hashable(v) for v in series.dropna()}, key=lambda x: str(x))


def comparability_report(
    df: pd.DataFrame,
    *,
    invariant_cols: Sequence[str] = DEFAULT_INVARIANTS,
    group_col: str | None = None,
) -> str:
    """Return a printable comparability report for ``df``.

    For each column in ``invariant_cols`` present in ``df``, list the unique values
    and flag ``*** VARIES ***`` when there is more than one. If ``group_col`` is
    given, the report is produced per group as well as overall, so you can confirm
    invariants hold within each condition.
    """
    lines: list[str] = []

    def section(label: str, sub: pd.DataFrame) -> None:
        lines.append("=" * 70)
        lines.append(f"{label}  (n={len(sub)})")
        lines.append("=" * 70)
        for col in invariant_cols:
            if col not in sub.columns:
                continue
            vals = _unique(sub[col])
            flag = "  *** VARIES ***" if len(vals) > 1 else ""
            lines.append(f"  {col:34s}: {vals}{flag}")

    if group_col and group_col in df.columns:
        for name, sub in df.groupby(group_col):
            section(f"Group {group_col}={name!r}", sub)
            lines.append("")
    section("All runs", df)
    return "\n".join(lines)


def git_commit_summary(df: pd.DataFrame) -> dict[str, int]:
    """Count runs per distinct ``git_commit``.

    When this returns more than one commit, the policy requires the analyst to
    confirm (via ``git diff``) that the differences are additive / do not touch
    shared training, env, network or reward code, and to record that in the report.
    """
    if "git_commit" not in df.columns:
        return {}
    return df["git_commit"].value_counts(dropna=False).to_dict()
