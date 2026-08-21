"""Helpers for the ``extract.py`` half of an analysis folder.

These implement the two things every ``extract.py`` needs and that are easy to get
subtly wrong: the **freeze/refresh gate** (a figure must rebuild from exactly the runs it
was made from unless you explicitly ask for more) and the **coverage check** (an analysis
must say out loud which runs are missing the data it claims to use).

The gate
--------
``runs.csv`` -- committed, one row per included run -- *is* the analysis's dataset
definition. ``CONDITIONS`` in ``extract.py`` is the *query* that produced it. Both are
version-controlled, and they are consulted in different modes:

======================  ==================================================================
``extract.py``          **frozen**: read ``runs.csv``, rebuild ``data.csv`` from exactly
                        those runs. Deterministic, no selector evaluation.
``extract.py --refresh``re-run ``CONDITIONS`` against the index, print the added/removed
                        runs, rewrite ``runs.csv`` and ``data.csv``.
``extract.py --check``  frozen rebuild compared against the committed ``data.csv``;
                        exits non-zero on drift.
======================  ==================================================================

So "redo this plot with exactly the same data" is the default, and "now include the new
runs" is one flag and a visible diff.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from vnl_experiments.wandb_utils import index

SELECTION_COLUMNS = ("condition", "wandb_id", "wandb_name", "created_at", "state")

#: Net-config flags that ablate one of the enc-dec decoder's input streams
#: (see ``vnl_experiments.delays.network_builders.build_delay_network``). Absent or
#: True means the stream is present, which is why every run predating the flags reads
#: as un-ablated.
DECODER_INPUT_FLAGS = ("dec_use_intention", "dec_use_proprioception")


# --------------------------------------------------------------------------------------
# cohort guards
# --------------------------------------------------------------------------------------


def full_decoder_inputs(net_params: Mapping[str, Any] | None) -> bool:
    """False if this run ablated one of the decoder's input streams.

    A decoder-input ablation keeps the standard hidden sizes and can keep
    ``efference_length == delay_k``, so it satisfies every "standard architecture,
    efference-matched" test the analyses apply and would silently join their baseline
    cohort. Any analysis whose baseline means "the decoder sees all three streams"
    has to say so explicitly -- this is that test, in dict form for the extractors
    that classify a live ``run.config``.
    """
    net = net_params or {}
    return all(net.get(key, True) for key in DECODER_INPUT_FLAGS)


def full_decoder_inputs_mask(df: pd.DataFrame) -> pd.Series:
    """:func:`full_decoder_inputs` as a row mask over an index frame.

    Tolerates the columns being missing entirely (no ablation run synced yet), which
    ``index.select`` deliberately does not -- it raises on unknown columns. Tests for
    the ablated value rather than the present one so that a missing cell -- every run
    older than the flags -- reads as un-ablated without an ``fillna`` downcast.
    """
    mask = pd.Series(True, index=df.index)
    for key in DECODER_INPUT_FLAGS:
        column = f"net_params.{key}"
        if column in df.columns:
            mask &= ~df[column].isin([False, 0, "False", "false"])
    return mask


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(description: str = "", argv: Sequence[str] | None = None):
    """The standard ``extract.py`` argument set."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true",
                        help="re-run the condition selectors and pick up new runs")
    parser.add_argument("--sync", action="store_true",
                        help="refresh the WandB run index first (implies --refresh)")
    parser.add_argument("--check", action="store_true",
                        help="rebuild frozen and diff against the committed CSVs; "
                             "exit non-zero if they differ")
    parser.add_argument("--project", default=index.DEFAULT_PROJECT)
    args = parser.parse_args(argv)
    if args.sync:
        args.refresh = True
    return args


# --------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------


def select_conditions(df: pd.DataFrame, conditions: Mapping[str, Any]) -> pd.DataFrame:
    """Apply one selector per condition; return the tagged union.

    A selector is either a mapping of :func:`index.select` filters (the common case) or,
    when a cell needs logic that column equality cannot express, a callable taking the
    index frame and returning a boolean mask or a sub-frame.

    A run matching two conditions is an error, not a warning: it means the conditions
    are not the mutually exclusive cells the analysis is about to treat them as.
    """
    frames = []
    for name, selector in conditions.items():
        if callable(selector):
            result = selector(df)
            picked = (df[result] if isinstance(result, pd.Series) else result).copy()
        else:
            filters = dict(selector)
            tags = filters.pop("tags", None)
            picked = index.select(df, tags=tags, **filters).copy()
        picked.insert(0, "condition", name)
        frames.append(picked)

    if not frames:
        raise ValueError("no conditions defined")
    out = pd.concat(frames, ignore_index=True)

    dupes = out[out.duplicated("wandb_id", keep=False)]
    if not dupes.empty:
        overlap = (dupes.groupby("wandb_id")["condition"].apply(list).to_dict())
        raise ValueError(
            "these runs match more than one condition, so the conditions overlap:\n  " +
            "\n  ".join(f"{wid}: {conds}" for wid, conds in overlap.items()))
    return out.sort_values(["condition", "wandb_id"], ignore_index=True)


def selection_path(here: Path) -> Path:
    return Path(here) / "runs.csv"


def read_selection(here: Path) -> pd.DataFrame:
    path = selection_path(here)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet. Create it by running this script once with "
            f"--refresh (add --sync to update the run index first).")
    return pd.read_csv(path, dtype={"wandb_id": str})


def write_selection(runs: pd.DataFrame, here: Path) -> Path:
    path = selection_path(here)
    columns = [c for c in SELECTION_COLUMNS if c in runs.columns]
    out = runs[columns].sort_values(["condition", "wandb_id"], ignore_index=True)
    out.to_csv(path, index=False)
    return path


def describe_selection_change(old: pd.DataFrame | None,
                              new: pd.DataFrame) -> str:
    """Human-readable added/removed summary for a ``--refresh``."""
    if old is None or old.empty:
        counts = new["condition"].value_counts().sort_index()
        return ("new selection: " +
                ", ".join(f"{k}={v}" for k, v in counts.items()) +
                f" ({len(new)} runs)")

    old_ids, new_ids = set(old["wandb_id"]), set(new["wandb_id"])
    added, removed = sorted(new_ids - old_ids), sorted(old_ids - new_ids)
    by_cond = new.set_index("wandb_id")["condition"].to_dict()
    old_by_cond = old.set_index("wandb_id")["condition"].to_dict()

    lines = [f"selection: {len(old_ids)} -> {len(new_ids)} runs "
             f"(+{len(added)}, -{len(removed)})"]
    for wid in added:
        lines.append(f"  + {wid}  {by_cond.get(wid, '?')}")
    for wid in removed:
        lines.append(f"  - {wid}  {old_by_cond.get(wid, '?')}")
    moved = [wid for wid in new_ids & old_ids if by_cond[wid] != old_by_cond[wid]]
    for wid in sorted(moved):
        lines.append(f"  ~ {wid}  {old_by_cond[wid]} -> {by_cond[wid]}")
    return "\n".join(lines)


def resolve_selection(here: Path, conditions: Mapping[str, Mapping[str, Any]], *,
                      refresh: bool, sync: bool = False,
                      project: str = index.DEFAULT_PROJECT) -> pd.DataFrame:
    """The freeze/refresh gate. Returns index rows for the selected runs, ``condition``
    column included.

    Frozen (the default) never evaluates ``conditions`` -- it reads ``runs.csv`` and
    looks those ids up in the index -- so adding runs to WandB cannot change a committed
    figure behind your back.
    """
    if sync:
        index.sync(project)
    df = index.load(project)

    if refresh:
        new = select_conditions(df, conditions)
        try:
            old = read_selection(here)
        except FileNotFoundError:
            old = None
        print(describe_selection_change(old, new))
        write_selection(new, here)
        return new

    selection = read_selection(here)
    indexed = df.set_index("wandb_id")
    unknown = [w for w in selection["wandb_id"] if w not in indexed.index]
    if unknown:
        raise KeyError(
            f"{len(unknown)} run(s) in runs.csv are not in the index "
            f"({unknown[:5]}...). Run `python -m vnl_experiments.wandb_utils.index sync`, "
            f"or if they were deleted from WandB, re-run with --refresh.")
    rows = indexed.loc[selection["wandb_id"]].reset_index()
    rows.insert(0, "condition", selection["condition"].to_numpy())
    return rows.sort_values(["condition", "wandb_id"], ignore_index=True)


# --------------------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------------------


def _requirement_spec_id(requirement: str) -> tuple[str, str | None]:
    """``"eval:legacy-batch"`` -> ``("eval", "legacy-batch")``; ``"eval"`` -> default."""
    kind, _, sid = requirement.partition(":")
    if kind == "index":
        return kind, None
    if sid:
        return kind, sid
    from vnl_experiments.artifacts import get_producer

    producer = get_producer(kind)
    return kind, producer.spec_id(producer.spec())


def coverage_table(runs: pd.DataFrame, requires: Iterable[str]) -> pd.DataFrame:
    """Per condition, how many runs have each required artifact."""
    from vnl_experiments.artifacts import Store

    store = Store()
    rows = []
    for requirement in requires:
        kind, sid = _requirement_spec_id(requirement)
        for condition, group in runs.groupby("condition"):
            if kind == "index":
                have = len(group)
            else:
                have = sum(store.have(kind, wid, sid) for wid in group["wandb_id"])
            rows.append({"requirement": requirement, "spec_id": sid or "-",
                         "condition": condition, "have": have, "n": len(group)})
    return pd.DataFrame(rows)


def write_coverage(runs: pd.DataFrame, requires: Sequence[str], here: Path,
                   *, strict: bool = False) -> pd.DataFrame:
    """Write ``coverage.txt`` and shout about gaps.

    Runs are never silently dropped for a missing artifact -- an analysis that quietly
    reports on the subset that happened to have data is how a cohort with zero offline
    evals got written up as if it had them.
    """
    table = coverage_table(runs, requires)
    lines = [f"Coverage for {len(runs)} runs in {Path(here).name}", ""]
    gaps = []
    for requirement, group in table.groupby("requirement", sort=False):
        total_have, total_n = int(group["have"].sum()), int(group["n"].sum())
        sid = group["spec_id"].iloc[0]
        lines.append(f"{requirement}  (spec_id={sid}): {total_have}/{total_n}")
        for _, row in group.iterrows():
            flag = "" if row["have"] == row["n"] else "   *** GAP ***"
            lines.append(f"    {row['condition']:<28s} {row['have']:>4d}/{row['n']:<4d}"
                         f"{flag}")
        if total_have < total_n:
            gaps.append((requirement, total_n - total_have))
        lines.append("")

    if gaps:
        from vnl_experiments.artifacts import Store

        store = Store()
        lines.append("Missing artifacts. Produce or fetch them with:")
        for requirement, n in gaps:
            kind, sid = _requirement_spec_id(requirement)
            lines.append(f"  # {n} runs missing {requirement}")
            # Distinguish "nothing there" from "made by a different producer version":
            # the second reads as a total gap unless the other specs are named.
            others = store.other_specs(kind, runs["wandb_id"], sid or "")
            if others:
                lines.append("  # other specs held for these runs: " +
                             ", ".join(f"{k} ({v})" for k, v in others.items()))
            lines.append(f"  python -m vnl_experiments.artifacts plan  --kind {kind} "
                         f"--runs {Path(here).name}/runs.csv")
            lines.append(f"  python -m vnl_experiments.artifacts pull  --kind {kind} "
                         f"--runs {Path(here).name}/runs.csv")
        lines.append("")

    text = "\n".join(lines)
    (Path(here) / "coverage.txt").write_text(text)
    print(text)
    if gaps and strict:
        raise SystemExit("coverage gaps (strict mode); see coverage.txt")
    return table


# --------------------------------------------------------------------------------------
# metrics + output
# --------------------------------------------------------------------------------------


def first_present(row: Mapping[str, Any], *keys: str) -> Any:
    """First of ``keys`` that is present and non-null in ``row``.

    For coalescing metrics whose logged name changed mid-project, e.g.
    ``first_present(row, "summary.eval/episode_reward/mean",
    "summary.episode_reward/mean")``.
    """
    for key in keys:
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    return None


def write_csv(frame: pd.DataFrame, path: Path, *, check: bool = False) -> bool:
    """Write ``frame`` to ``path``, or in ``check`` mode diff it against what is there.

    Returns True if the file is (or would be) unchanged.
    """
    path = Path(path)
    text = frame.to_csv(index=False)
    if not check:
        path.write_text(text)
        print(f"wrote {path.name}: {len(frame)} rows x {len(frame.columns)} cols")
        return True

    if not path.exists():
        print(f"CHECK: {path.name} does not exist")
        return False
    committed = path.read_text()
    if committed == text:
        print(f"CHECK: {path.name} unchanged ({len(frame)} rows)")
        return True
    diff = list(difflib.unified_diff(committed.splitlines(), text.splitlines(),
                                     fromfile=f"committed/{path.name}",
                                     tofile=f"rebuilt/{path.name}", lineterm="", n=1))
    print(f"CHECK: {path.name} DIFFERS ({len(diff)} diff lines); first 40:")
    print("\n".join(diff[:40]))
    return False
