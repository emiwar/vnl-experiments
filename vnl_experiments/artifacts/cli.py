"""Command line for the artifact store.

    python -m vnl_experiments.artifacts <command> [options]

    plan    --kind eval --runs analysis/<q>/runs.csv [--out todo.txt]
    ensure  --kind history --runs analysis/<q>/runs.csv
    pull    --kind eval --runs analysis/<q>/runs.csv
    verify  [--kind eval]
    reindex
    ls      [--kind eval]
    import-legacy [--dry-run]

``--runs`` accepts an analysis ``runs.csv`` (uses its ``wandb_id`` column), a plain text
file of ids one per line, or a comma-separated list of ids.

The intended cluster loop is::

    python -m vnl_experiments.artifacts plan --kind eval --runs analysis/q/runs.csv --out todo.txt
    scp todo.txt cluster:...            &&  sbatch slurm_eval.sh todo.txt
    python -m vnl_experiments.artifacts pull --kind eval --runs analysis/q/runs.csv

so that checkpoints stay on the cluster and only the (small) results come down.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from vnl_experiments.artifacts.producers import PRODUCERS, get_producer
from vnl_experiments.artifacts.store import (
    KINDS,
    MANIFEST_PATH,
    REPO_ROOT,
    Store,
    file_sha256,
    store_root,
)

#: Where checkpoints are looked for, in order. Same list as ``eval_runs.py`` uses.
CHECKPOINT_DIRS = (REPO_ROOT / "downloaded_checkpoints", REPO_ROOT / "checkpoints")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def read_run_ids(spec: str) -> list[str]:
    """Resolve ``--runs`` to a list of wandb ids, preserving order and deduping."""
    path = Path(spec)
    if path.exists():
        if path.suffix == ".csv":
            ids = pd.read_csv(path)["wandb_id"].astype(str).tolist()
        else:
            ids = [line.split()[0] for line in path.read_text().splitlines()
                   if line.strip() and not line.startswith("#")]
    elif "/" in spec or path.suffix in (".csv", ".txt"):
        # A value that looks like a path but does not resolve is a mistake, not a
        # one-element id list. Falling through to the comma-split below turns it into a
        # single bogus wandb id, and since `pull` passes the resulting file list to rsync
        # with --ignore-missing-args, the remote's "no such file" is swallowed: rsync
        # exits 0, reports "total size is 0", and `pull` claims success while having
        # transferred nothing. Fail loudly instead.
        raise SystemExit(
            f"--runs {spec!r}: no such file (cwd {Path.cwd()}).\n"
            f"Relative paths resolve against the current directory -- run from the "
            f"repo root ({REPO_ROOT}) or pass an absolute path. To pass wandb ids "
            f"directly, use a comma-separated list with no '/'."
        )
    else:
        ids = [s.strip() for s in spec.split(",") if s.strip()]
    seen, out = set(), []
    for wid in ids:
        if wid not in seen:
            seen.add(wid)
            out.append(wid)
    return out


def find_checkpoint(wandb_name: str) -> Path | None:
    for base in CHECKPOINT_DIRS:
        candidate = base / wandb_name
        if (candidate / "config.json").exists():
            return candidate
    return None


def run_context(wandb_ids: list[str]) -> dict[str, dict]:
    """Per-run context (name, env class, checkpoint dir) looked up in the run index."""
    from vnl_experiments.wandb_utils import index

    try:
        df = index.load().set_index("wandb_id")
    except FileNotFoundError:
        df = pd.DataFrame()
    ctx = {}
    for wid in wandb_ids:
        row = df.loc[wid] if wid in df.index else None
        name = str(row["wandb_name"]) if row is not None else wid
        ctx[wid] = {
            "wandb_name": name,
            "env_class": (str(row["env"]) if row is not None and "env" in row
                          and pd.notna(row.get("env")) else "AbsoluteImitation"),
            "checkpoint_dir": find_checkpoint(name),
        }
    return ctx


def _spec_from_args(kind: str, overrides: list[str] | None) -> dict:
    """Build a spec from the producer defaults plus ``--set key=jsonvalue`` overrides."""
    producer = get_producer(kind)
    parsed = {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return producer.spec(**parsed)


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------


def cmd_plan(args) -> int:
    store = Store()
    producer = get_producer(args.kind)
    spec = _spec_from_args(args.kind, args.set)
    sid = producer.spec_id(spec)
    wandb_ids = read_run_ids(args.runs)
    missing = store.missing(args.kind, sid, wandb_ids)

    print(f"kind={args.kind}  spec_id={sid}")
    print(f"spec: {json.dumps(spec, sort_keys=True)}")
    print(f"{len(wandb_ids) - len(missing)}/{len(wandb_ids)} present in {store.root}")
    if not missing:
        print("nothing to do")
        return 0

    ctx = run_context(missing)
    with_ckpt = [w for w in missing if ctx[w]["checkpoint_dir"]]
    without = [w for w in missing if not ctx[w]["checkpoint_dir"]]

    others = store.other_specs(args.kind, wandb_ids, sid)
    if others:
        print("other specs held for these runs: " +
              ", ".join(f"{k} ({v})" for k, v in others.items()))

    def show(label: str, ids: list[str]) -> None:
        if not ids:
            return
        print(f"\n{label} ({len(ids)}):")
        for wid in ids:
            print(f"  {wid}\t{ctx[wid]['wandb_name']}\t{ctx[wid]['env_class']}")

    print(f"{len(missing)} missing")
    if producer.NEEDS_CHECKPOINT:
        show("producible here (checkpoint present)", with_ckpt)
        show("need the cluster (no local checkpoint)", without)
        if without:
            print("\n  -> send the run list to the cluster and pull the results:")
            print(f"     sbatch slurm_eval.sh <run-list> {args.kind}")
            print(f"     python -m vnl_experiments.artifacts pull --kind {args.kind} "
                  f"--runs {args.runs}")
    else:
        show("to produce", missing)

    lines = [f"{w}\t{ctx[w]['wandb_name']}\t{ctx[w]['env_class']}" for w in missing]
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out} ({len(lines)} runs)")
    return 0


def cmd_ensure(args) -> int:
    store = Store()
    producer = get_producer(args.kind)
    spec = _spec_from_args(args.kind, args.set)
    sid = producer.spec_id(spec)
    wandb_ids = read_run_ids(args.runs)
    todo = wandb_ids if args.override else store.missing(args.kind, sid, wandb_ids)
    print(f"kind={args.kind} spec_id={sid}: {len(todo)}/{len(wandb_ids)} to produce")

    ctx = run_context(todo)
    made = failed = 0

    def one(wid: str) -> tuple[str, object]:
        try:
            return wid, producer.ensure(store, wid, spec, ctx=ctx[wid],
                                        override=args.override)
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop the batch
            return wid, exc

    # GPU-bound producers stay serial; pure-I/O ones (history) parallelise well.
    workers = args.workers if producer.PARALLEL_SAFE else 1
    results = (ThreadPoolExecutor(max_workers=workers).map(one, todo) if workers > 1
               else map(one, todo))
    for i, (wid, outcome) in enumerate(results, start=1):
        if isinstance(outcome, Exception):
            failed += 1
            print(f"  [{i}/{len(todo)}] {wid}: FAILED "
                  f"{type(outcome).__name__}: {outcome}")
        else:
            made += 1
            print(f"  [{i}/{len(todo)}] {wid}: {outcome.path} "
                  f"({outcome.bytes / 1e3:.0f} kB) {outcome.resolved}")
    store.reindex()
    print(f"produced {made}, failed {failed}; manifest reindexed")
    return 1 if failed else 0


def cmd_pull(args) -> int:
    remote = args.remote or os.environ.get("VNL_CLUSTER_ARTIFACTS")
    if not remote:
        print("Set --remote or $VNL_CLUSTER_ARTIFACTS, e.g.\n"
              "  export VNL_CLUSTER_ARTIFACTS="
              "cluster:/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/artifacts",
              file=sys.stderr)
        return 2

    store = Store()
    wandb_ids = read_run_ids(args.runs)
    kinds = [args.kind] if args.kind else list(KINDS)
    # Ask rsync for whole <kind>/<wandb_id>/ directories: the sidecars travel with the
    # bytes, so provenance survives the copy and `reindex` can rebuild from them.
    listing = [f"{kind}/{wid}/" for kind in kinds for wid in wandb_ids]
    list_file = Path(args.out or "/tmp/vnl_pull_list.txt")
    list_file.write_text("\n".join(listing) + "\n")

    store.root.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-rlpt", "--info=stats1,progress2", "--ignore-missing-args",
           f"--files-from={list_file}", f"{remote.rstrip('/')}/", str(store.root)]
    if args.dry_run:
        cmd.insert(1, "--dry-run")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode not in (0, 23, 24):  # 23/24: some sources vanished, fine here
        return result.returncode
    if not args.dry_run:
        entries = store.reindex()
        print(f"manifest reindexed: {len(entries)} artifacts")
    return 0


def cmd_verify(args) -> int:
    store = Store()
    entries = store.scan()
    if args.kind:
        entries = [e for e in entries if e.kind == args.kind]

    missing_files, bad_hash, unhashed = [], [], 0
    for entry in entries:
        path = store.root / entry.path
        if not path.exists():
            missing_files.append(entry)
            continue
        if entry.sha256 is None:
            unhashed += 1
            continue
        if file_sha256(path) != entry.sha256:
            bad_hash.append(entry)

    # Some producers leave companion files beside the recorded one (a video's .h5 and
    # .stats.json), all sharing the spec_id stem. Those belong to the artifact; only a
    # file matching no known spec_id in its directory is genuinely orphaned.
    described = {(store.root / e.path).resolve() for e in entries}
    stems: dict[Path, set[str]] = {}
    for entry in entries:
        stems.setdefault((store.root / entry.path).parent.resolve(),
                         set()).add(entry.spec_id)
    orphans = [
        p for p in store.root.glob("*/*/*")
        if p.is_file() and not p.name.endswith(".meta.json")
        and p.resolve() not in described
        and not any(p.name.startswith(s) for s in stems.get(p.parent.resolve(), ()))
    ]

    total_bytes = sum(e.bytes for e in entries)
    print(f"store: {store.root}")
    print(f"  {len(entries)} artifacts, {total_bytes / 1e9:.1f} GB")
    for kind in sorted({e.kind for e in entries}):
        n = sum(1 for e in entries if e.kind == kind)
        print(f"    {kind:12s} {n:5d}")
    print(f"  {unhashed} too large to checksum (recorded by size only)")
    print(f"  {len(missing_files)} sidecars with no data file")
    print(f"  {len(bad_hash)} checksum mismatches")
    print(f"  {len(orphans)} orphan files with no sidecar")
    for entry in missing_files[:10]:
        print(f"    missing: {entry.path}")
    for entry in bad_hash[:10]:
        print(f"    corrupt: {entry.path}")
    for path in orphans[:10]:
        print(f"    orphan:  {path.relative_to(store.root)}")
    return 1 if (missing_files or bad_hash) else 0


def cmd_reindex(args) -> int:
    entries = Store().reindex()
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}: {len(entries)} artifacts")
    return 0


def cmd_ls(args) -> int:
    from vnl_experiments.artifacts.store import manifest_df

    df = manifest_df()
    if df.empty:
        print("manifest is empty; run `reindex` (or `pull`) first")
        return 0
    if args.kind:
        df = df[df["kind"] == args.kind]
    summary = (df.groupby(["kind", "spec_id"])
                 .agg(runs=("wandb_id", "nunique"), gb=("bytes", lambda s: s.sum() / 1e9))
                 .sort_values("runs", ascending=False))
    print(summary.to_string(float_format=lambda v: f"{v:.2f}"))
    return 0


def cmd_import_legacy(args) -> int:
    """Adopt the pre-store directories into the store, without copying the bytes.

    Files are **hardlinked**, so the 22 GB of activations is not duplicated and the old
    directories can be deleted once `verify` is clean. Legacy artifacts get fixed,
    unhashed ``spec_id``s (``legacy-batch`` etc.): their true spec was never recorded, and
    inventing a hash would imply a precision that does not exist.
    """
    from vnl_experiments.wandb_utils import index

    store = Store()
    sources = [
        ("eval", REPO_ROOT / "eval_results" / "eval_results", "*.json", "legacy-batch"),
        ("eval", REPO_ROOT / "eval_results" / "old_eval_results", "*.json",
         "legacy-batch-v0"),
        ("activations", REPO_ROOT / "eval_results" / "activations", "*.h5", None),
    ]
    try:
        name_to_id = {rec["wandb_name"]: rec["wandb_id"]
                      for rec in index.read_index().values()}
    except Exception:  # noqa: BLE001
        name_to_id = {}

    imported = skipped = unresolved = 0
    for kind, src_dir, pattern, fixed_sid in sources:
        if not src_dir.exists():
            print(f"(skipping absent {src_dir})")
            continue
        files = sorted(src_dir.glob(pattern))
        print(f"{src_dir.relative_to(REPO_ROOT)}: {len(files)} files")
        for path in files:
            if kind == "eval":
                wandb_id, sid = path.stem, fixed_sid
                resolved = {}
                try:
                    record = json.loads(path.read_text())
                    resolved = {"checkpoint_step": record.get("step"),
                                "env_class": record.get("env_class"),
                                "datasets": sorted(record.get("datasets", {}))}
                except json.JSONDecodeError:
                    pass
            else:
                # activations are named <wandb_name>__<dataset>.h5
                stem, _, dataset = path.stem.rpartition("__")
                wandb_id = name_to_id.get(stem)
                if wandb_id is None:
                    unresolved += 1
                    print(f"  ? no run in the index named {stem!r}; skipped")
                    continue
                sid, resolved = f"legacy-{dataset}", {"dataset": dataset}

            dest = store.path_for(kind, wandb_id, sid, path.suffix)
            if store.have(kind, wandb_id, sid):
                skipped += 1
                continue
            if args.dry_run:
                print(f"  would link {path.name} -> {dest.relative_to(store.root)}")
                imported += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    os.link(path, dest)
                except OSError:       # different filesystem: fall back to a copy
                    dest.write_bytes(path.read_bytes())
            store.record(kind, wandb_id, sid, dest, spec={"legacy": True},
                         producer={"legacy": True, "source": str(path.relative_to(REPO_ROOT))},
                         resolved=resolved)
            imported += 1

    print(f"\nimported {imported}, already present {skipped}, unresolved {unresolved}")
    if not args.dry_run:
        entries = store.reindex()
        print(f"manifest reindexed: {len(entries)} artifacts")
    return 0


def cmd_audit_env(args) -> int:
    """Report artifacts built on a different body than the run trained on."""
    from vnl_experiments.artifacts import audit_env

    return audit_env.main(args)


def cmd_adopt(args) -> int:
    """Hardlink artifacts from an older producer version onto the current ``spec_id``.

    For a run that trained on the *default* assets, the 2026-08-18 XML fix cannot change
    what ``produce`` would write, so recomputing the artifact under the new ``spec_id`` would
    burn GPU hours to reproduce identical bytes. Adopting instead hardlinks the existing file
    and writes a fresh sidecar that records where it came from.

    The predicate is checked per run, not asserted by the caller: a run whose trained assets
    differ from the defaults is **refused**, because for those the fix does change the output.
    """
    from vnl_experiments.artifacts.audit_env import classify

    store = Store()
    wanted = set(read_run_ids(args.runs))
    rows = {(r.kind, r.wandb_id, r.spec_id): r for r in classify()}

    adopted = refused = missing = skipped = 0
    for wandb_id in sorted(wanted):
        row = rows.get((args.kind, wandb_id, args.from_spec))
        if row is None:
            print(f"  {wandb_id}: no {args.kind}:{args.from_spec} in the store; skipped")
            missing += 1
            continue
        if row.verdict != "adoptable":
            print(f"  {wandb_id}: REFUSED ({row.verdict}"
                  f"{': ' + row.reason if row.reason else ''})")
            refused += 1
            continue

        source = store.lookup(args.kind, wandb_id, args.from_spec)
        spec = dict(source.spec)
        new_sid = get_producer(args.kind).spec_id(get_producer(args.kind).spec(**spec))
        if store.have(args.kind, wandb_id, new_sid):
            skipped += 1
            continue

        src_path = store.root / source.path
        dest = store.path_for(args.kind, wandb_id, new_sid, src_path.suffix)
        if args.dry_run:
            print(f"  would link {source.path} -> {dest.relative_to(store.root)}")
            adopted += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                os.link(src_path, dest)
            except OSError:            # different filesystem: fall back to a copy
                dest.write_bytes(src_path.read_bytes())
        store.record(args.kind, wandb_id, new_sid, dest, spec=spec,
                     producer={**source.producer, "adopted_from": args.from_spec,
                               "adopted_reason": "run trained on the default assets, so "
                                                 "the walker-XML fix cannot change the output"},
                     resolved={**source.resolved, **row.trained})
        adopted += 1

    print(f"\nadopted {adopted}, refused {refused}, already present {skipped}, "
          f"absent {missing}")
    if adopted and not args.dry_run:
        print(f"manifest reindexed: {len(store.reindex())} artifacts")
    return 1 if refused else 0


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vnl_experiments.artifacts", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_runs(p, required=True):
        p.add_argument("--runs", required=required,
                       help="runs.csv, a text file of ids, or a comma-separated list")

    def add_kind(p, required=True):
        p.add_argument("--kind", required=required, choices=KINDS)
        p.add_argument("--set", action="append", metavar="KEY=JSON",
                       help="override a spec field (repeatable)")

    p = sub.add_parser("plan", help="list runs missing an artifact")
    add_kind(p); add_runs(p); p.add_argument("--out")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("ensure", help="produce artifacts that are missing")
    add_kind(p); add_runs(p)
    p.add_argument("--override", action="store_true", help="re-produce even if present")
    p.add_argument("--workers", type=int, default=8,
                   help="parallelism, honoured only by I/O-bound producers")
    p.set_defaults(func=cmd_ensure)

    p = sub.add_parser("pull", help="rsync artifacts from the cluster store")
    add_kind(p, required=False); add_runs(p)
    p.add_argument("--remote", help="default: $VNL_CLUSTER_ARTIFACTS")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", help="where to write the rsync file list")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("verify", help="checksums, missing files, orphans")
    p.add_argument("--kind", choices=KINDS)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("reindex", help="rebuild the committed manifest from the sidecars")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("ls", help="summarise the manifest")
    p.add_argument("--kind", choices=KINDS)
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("import-legacy", help="adopt eval_results/ into the store")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_import_legacy)

    p = sub.add_parser("audit-env",
                       help="artifacts built on a different body than was trained")
    p.add_argument("--out-dir", help="write todo_*/adopt_* run lists + summary.txt here")
    p.add_argument("--by-analysis", action="store_true",
                   help="which analysis folders consumed a broken artifact")
    p.set_defaults(func=cmd_audit_env)

    p = sub.add_parser("adopt",
                       help="hardlink artifacts from an older producer version to the "
                            "current spec_id (only where provably identical)")
    p.add_argument("--kind", required=True, choices=KINDS)
    add_runs(p)
    p.add_argument("--from-spec", required=True, help="the spec_id to adopt from")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_adopt)

    args = parser.parse_args(argv)
    if getattr(args, "kind", None) and args.command in ("plan", "ensure") \
            and args.kind not in PRODUCERS:
        parser.error(f"no producer for kind {args.kind!r}; "
                     f"available: {sorted(PRODUCERS)}")
    return args.func(args)
