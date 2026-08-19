"""Which stored artifacts were built on a different body than the run trained on.

Until 2026-08-18 every offline env rebuild replaced the run's ``walker_xml_path`` with the
local default (``rodent.xml``), so a run trained on ``rodent_no_tail_collisions.xml`` was
re-simulated on the wrong body -- see :mod:`vnl_experiments.envs.config_io`. The producers
now stamp ``resolved.walker_xml_path``, which makes the question decidable:

* stamp present -> the artifact says which body it used; compare it to the run's config.
* stamp absent  -> the artifact predates the fix, so the body it used *was* the local
  default. That inference is the one assumption in this module, and it is why the tables
  below label those rows ``pre-fix`` rather than pretending to have measured them.

Three outcomes per artifact:

``broken``     the body used differs from the one trained; the artifact must be re-produced.
``adoptable``  pre-fix, but the run trained on the default body, so the fix cannot change
               the output: the file can be hardlinked to the new ``spec_id`` instead of
               being recomputed (``artifacts adopt``).
``ok``         already carries a stamp that agrees with the run's config.

Run it::

    python -m vnl_experiments.artifacts audit-env
    python -m vnl_experiments.artifacts audit-env --out-dir artifact_repair/<slug>
    python -m vnl_experiments.artifacts audit-env --by-analysis
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from vnl_experiments.artifacts.store import MANIFEST_PATH, REPO_ROOT

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


def summarise(rows: Iterable[Row]) -> str:
    """The per-kind and per-spec tables, as text (also written to ``summary.txt``)."""
    rows = list(rows)
    out = ["Artifact env audit -- which artifacts used a different body than was trained",
           "=" * 78, ""]

    per_kind: dict[str, dict[str, list[Row]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        per_kind[row.kind][row.verdict].append(row)

    out.append(f"{'kind':<13}{'total':>7}{'BROKEN':>8}{'adoptable':>11}{'repaired':>10}"
               f"{'ok':>5}{'unknown':>9}   broken GB")
    for kind in ENV_KINDS:
        buckets = per_kind.get(kind, {})
        total = sum(len(v) for v in buckets.values())
        broken = buckets.get("broken", [])
        out.append(f"{kind:<13}{total:>7}{len(broken):>8}"
                   f"{len(buckets.get('adoptable', [])):>11}"
                   f"{len(buckets.get('repaired', [])):>10}"
                   f"{len(buckets.get('ok', [])):>5}"
                   f"{len(buckets.get('unknown', [])):>9}"
                   f"{sum(r.bytes for r in broken) / 1e9:>12.2f}")
    out += ["",
            "broken    = must be re-produced (todo_* lists)",
            "adoptable = pre-fix, but the fix cannot change the bytes: hardlink it "
            "(adopt_* lists)",
            "repaired  = a current-version replacement already exists; the pre-fix file is "
            "kept as the record",
            "ok        = carries a body stamp that matches the run, or a legacy spec with "
            "no counterpart",
            ""]

    broken = [r for r in rows if r.verdict == "broken"]
    if broken:
        out += ["Broken, by kind x spec (these need re-producing):", ""]
        groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
        for row in broken:
            groups[(row.kind, row.spec_id, row.variant)].append(row)
        for (kind, sid, variant), group in sorted(groups.items()):
            new_sid = current_spec_id(kind, group[0].spec)
            note = f"  ->  {new_sid}" if new_sid else "  ->  (no v2 counterpart)"
            out.append(f"  {kind:<12} {variant:<14} {len(group):>4} runs "
                       f"{sum(r.bytes for r in group) / 1e9:>7.2f} GB  {sid:<22}{note}")
        out.append("")
        example = broken[0]
        out += ["Example:", f"  {example.kind} {example.wandb_id} ({example.wandb_name})",
                f"  {example.reason}", ""]

    unknown = [r for r in rows if r.verdict == "unknown"]
    if unknown:
        out.append(f"{len(unknown)} artifacts could not be classified "
                   f"(run absent from the index): "
                   f"{', '.join(sorted({r.wandb_id for r in unknown})[:8])}")
        out.append("")
    return "\n".join(out)


def write_lists(rows: Iterable[Row], out_dir: Path) -> list[Path]:
    """Write ``todo_*`` / ``adopt_*`` run lists in the ``plan --out`` format.

    Format is ``wandb_id \\t wandb_name \\t env_class`` -- what ``slurm_eval.sh`` consumes.
    """
    rows = list(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Group by spec_id, not by label: one file per spec family means the `--set` (to
    # re-produce) and the `--from-spec` (to adopt) in each header are unambiguous.
    groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        if row.verdict in ("broken", "adoptable"):
            action = "todo" if row.verdict == "broken" else "adopt"
            groups[(action, row.kind, row.spec_id)].append(row)

    for (action, kind, spec_id), group in sorted(groups.items()):
        variant = group[0].variant
        new_sid = current_spec_id(kind, group[0].spec)
        path = out_dir / f"{action}_{variant}.txt"
        spec = dict(group[0].spec)
        noise = spec.get("action_noise")

        header = [f"# {kind} / {variant} -- {len(group)} runs, "
                  f"{sum(r.bytes for r in group) / 1e9:.2f} GB",
                  f"# from spec_id : {spec_id}",
                  f"# to   spec_id : {new_sid or '(none: legacy spec)'}"]
        if action == "todo":
            extra = "" if noise is None else f" --set action_noise={noise:g}"
            header.append(f"#   sbatch slurm_eval.sh <this file> {kind}{extra}")
        else:
            header.append(f"#   python -m vnl_experiments.artifacts adopt --kind {kind} "
                          f"--runs <this file> --from-spec {spec_id}")
        header.append("# generated by `python -m vnl_experiments.artifacts audit-env`; "
                      "regenerate rather than edit")

        lines = header + [f"{r.wandb_id}\t{r.wandb_name}\t{r.env_class}"
                          for r in sorted(group, key=lambda r: r.wandb_id)]
        path.write_text("\n".join(lines) + "\n")
        written.append(path)
    return written


def _pinned_strings(path: Path) -> str:
    """Every string literal in ``path`` that a spec id could actually be *pinned* in.

    Searching the raw source instead makes a folder that merely *mentions* a retired spec id
    -- in a comment, or in the docstring recording why it was retired -- look like it still
    reads that data, so a repointed folder could never come back clean. Docstrings and
    bare-string statements are dropped for the same reason; a real pin is a string in an
    expression (an assignment right-hand side, a list element, an argument).

    A file that will not parse falls back to its raw text: over-reporting is the safe
    direction for an audit.
    """
    import ast

    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return path.read_text()

    commentary = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    return "\n".join(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in commentary)


def by_analysis(rows: Iterable[Row]) -> str:
    """Which committed analyses consumed a broken artifact.

    Pinned ``REQUIRES`` keep resolving the old ``spec_id`` after a producer version bump, so
    ``extract.py --check`` will *not* flag these folders -- the damage is real but quiet and
    has to be located deliberately.

    Two conditions must both hold, and testing only one gives the wrong answer:

    1. the folder *names* a broken ``spec_id`` somewhere in its scripts -- a run overlap alone
       proves nothing, since a folder may use only WandB summaries or ``history`` for the very
       same runs (``explicit-vs-implicit-fm-2g`` is exactly that case);
    2. some of its runs are among those whose artifact under that ``spec_id`` is broken.

    Folders with neither ``runs.csv`` nor a ``wandb_id`` column are reported as unknown rather
    than silently passed.
    """
    import pandas as pd

    broken_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.verdict == "broken":
            broken_ids[(row.kind, row.spec_id)].add(row.wandb_id)

    out = ["Analyses consuming a broken artifact", "=" * 78, ""]
    for folder in sorted((REPO_ROOT / "analysis").iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue

        scripts = "\n".join(_pinned_strings(p) for p in folder.glob("*.py"))
        pinned = [(kind, sid) for (kind, sid) in broken_ids if sid in scripts]

        ids: set[str] = set()
        source = None
        for candidate in ["runs.csv", *sorted(p.name for p in folder.glob("data*.csv"))]:
            path = folder / candidate
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path)
            except Exception:  # noqa: BLE001
                continue
            if "wandb_id" in frame.columns:
                ids = set(frame["wandb_id"].dropna().astype(str))
                source = candidate
                break

        if not pinned:
            out.append(f"  {folder.name:<38} clean     "
                       f"pins no affected spec_id")
            continue
        if not ids:
            out.append(f"  {folder.name:<38} ?         pins "
                       f"{len(pinned)} affected spec_id(s) but has no runs.csv/wandb_id "
                       f"column - check by hand")
            continue

        hits = {sid: len(ids & runs) for (_, sid), runs in
                ((key, runs) for key, runs in broken_ids.items() if key in pinned)
                if ids & runs}
        if not hits:
            out.append(f"  {folder.name:<38} clean     "
                       f"pins an affected spec_id, but none of its {len(ids)} runs are broken")
            continue
        detail = ", ".join(f"{sid} {n}/{len(ids)}" for sid, n in sorted(hits.items()))
        out.append(f"  {folder.name:<38} AFFECTED  {detail}  (from {source})")

    out += ["",
            "Pinned spec ids keep these folders rebuilding their old numbers, so "
            "`--check` stays",
            "green and a refresh is a deliberate per-folder decision, not an automatic one."]
    return "\n".join(out)


def main(args) -> int:
    rows = classify()
    report = summarise(rows)
    print(report)

    if getattr(args, "by_analysis", False):
        print()
        print(by_analysis(rows))
        return 0

    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None
    if out_dir is not None:
        (out_dir).mkdir(parents=True, exist_ok=True)
        written = write_lists(rows, out_dir)
        (out_dir / "summary.txt").write_text(report + "\n")
        print(f"wrote {len(written) + 1} files to {out_dir}:")
        for path in [*written, out_dir / "summary.txt"]:
            n = sum(1 for line in path.read_text().splitlines()
                    if line.strip() and not line.startswith("#"))
            print(f"  {path.name:<34} {n:>4} lines")
    else:
        print("(pass --out-dir to write the todo_*/adopt_* run lists)")
    return 0
