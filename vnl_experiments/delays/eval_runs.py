#!/usr/bin/env python3
"""Batch-evaluate every model that appears in a committed analysis.

For each run listed in ``eval_runs.txt`` this:

* locates its checkpoint (``downloaded_checkpoints/{name}`` or
  ``checkpoints/{name}``; missing → warn + skip),
* rebuilds the env (``AbsoluteImitation`` or base ``Imitation``) and the network
  (``RodentEncDecDelays`` or ``RodentForwardModel``) from the checkpoint's
  ``config.json``, and restores the latest step's weights + normalizer stats,
* evaluates the (deterministic) policy on **three datasets** — the 80% train
  split, the held-out 20% test split (both from the run's ``reference_data_path``
  with ``split(0.8, seed=0)``), and the new 32-clip eval set — recording episode
  reward, lifespan, per-reason termination rate, env error measures, and network
  metrics (forward-model MSE, KL, ...),
* counts parameters hierarchically, and writes one JSON per run keyed by
  ``wandb_id`` (skipped if it already exists unless ``--override``).

The measurement itself lives in :mod:`vnl_experiments.delays.evaluation`, shared
with the end-of-training eval in the training scripts, so records from either
producer are directly comparable. This script remains the authority for
cross-run comparisons: only it can re-evaluate the whole cohort under a single
version of the eval code.

Populate the run list (one-off, reads the analysis CSVs)::

    ../.venv/bin/python -m vnl_experiments.delays.eval_runs --populate

Evaluate (cluster: all checkpoints present; locally use --limit-clips)::

    ../.venv/bin/python -m vnl_experiments.delays.eval_runs
    ../.venv/bin/python -m vnl_experiments.delays.eval_runs \
        --run-list /tmp/test_runs.txt --limit-clips 16

Gather the ``eval.json`` files written by end-of-training evals into the
directory the analyses read from::

    ../.venv/bin/python -m vnl_experiments.delays.eval_runs --collect
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import gc
import glob
import json
import warnings
from pathlib import Path

import jax

# The evaluation core is shared with the training scripts' end-of-training eval
# and with analysis/implicit-forward-model/record_activations.py. These names are
# re-exported (rather than imported where used) so existing `from
# vnl_experiments.delays.eval_runs import ...` call sites keep working.
from vnl_experiments.delays.evaluation import (  # noqa: F401
    DEFAULT_NEW_EVAL_H5,
    NEW_EVAL_CLIP_LENGTH,
    REPO_ROOT,
    _make_env,
    build_datasets,
    eval_dataset,
    evaluate_networks,
    param_counts,
    parse_env_config,
    prepare_eval_config,
    resolve_env_class,
    run_metadata,
    split_clips,
)
# Network reconstruction + checkpoint loading are shared with eval_videos.py.
from vnl_experiments.delays.network_builders import build_network, load_network  # noqa: F401

DEFAULT_RUN_LIST = Path(__file__).resolve().parent / "eval_runs.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_results"
# Where the current analyses read from (eval_results/ also holds the previous
# result set and the activation h5s, so --collect targets the nested dir).
DEFAULT_COLLECT_DIR = REPO_ROOT / "eval_results" / "eval_results"
ANALYSIS_GLOB = str(REPO_ROOT / "analysis" / "*" / "data.csv")


# ---------------------------------------------------------------------------
# Run list
# ---------------------------------------------------------------------------

def populate_run_list(path: Path) -> None:
    """Write the run list from all analysis CSVs, deduped by ``wandb_id``."""
    csvs = sorted(glob.glob(ANALYSIS_GLOB))
    if not csvs:
        raise FileNotFoundError(f"No analysis CSVs matched {ANALYSIS_GLOB}")
    runs: dict[str, tuple[str, str]] = {}  # wandb_id -> (wandb_name, env_class)
    for c in csvs:
        with open(c) as f:
            for row in csv.DictReader(f):
                wid, wn = row.get("wandb_id"), row.get("wandb_name")
                if not wid or not wn:
                    continue
                env_class = (row.get("env") or "AbsoluteImitation").strip()
                runs.setdefault(wid, (wn, env_class))
    rows = sorted(runs.items(), key=lambda kv: kv[1][0])
    with open(path, "w") as f:
        f.write("# wandb_id\twandb_name\tenv_class\n")
        f.write(f"# {len(rows)} runs, deduped by wandb_id from {len(csvs)} analyses\n")
        for wid, (wn, env_class) in rows:
            f.write(f"{wid}\t{wn}\t{env_class}\n")
    print(f"Wrote {len(rows)} runs to {path}")


def read_run_list(path: Path) -> list[tuple[str, str, str]]:
    """Read ``(wandb_id, wandb_name, env_class)`` rows; env_class optional."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            wid, wn = parts[0], parts[1]
            env_class = parts[2] if len(parts) > 2 and parts[2] else ""
            out.append((wid, wn, env_class))
    return out


# ---------------------------------------------------------------------------
# Collecting inline (end-of-training) eval records
# ---------------------------------------------------------------------------

def collect_inline_evals(search_dirs: list[Path], output_dir: Path,
                         override: bool) -> None:
    """Copy ``{ckpt_dir}/eval.json`` records into ``output_dir/{wandb_id}.json``.

    The training scripts write their end-of-training eval next to the
    checkpoint, which is what gets rsynced back from the cluster. This gathers
    them into the flat, wandb_id-keyed directory the analyses glob over.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    copied, skipped, bad = 0, 0, 0
    for d in search_dirs:
        for src in sorted(d.glob("*/eval.json")):
            try:
                with open(src) as f:
                    record = json.load(f)
                wid = record["wandb_id"]
            except (json.JSONDecodeError, KeyError, OSError) as e:
                warnings.warn(f"Unreadable eval record {src}: {e!r}")
                bad += 1
                continue
            dst = output_dir / f"{wid}.json"
            if dst.exists() and not override:
                skipped += 1
                continue
            with open(dst, "w") as f:
                json.dump(record, f, indent=2)
            print(f"  {src.parent.name} -> {dst.name}")
            copied += 1
    print(f"\nCollected {copied} records into {output_dir} "
          f"({skipped} already present, {bad} unreadable)")


# ---------------------------------------------------------------------------
# Per-run orchestration
# ---------------------------------------------------------------------------

def evaluate_run(wid: str, wn: str, env_class_hint: str, ckpt_dir: Path,
                 new_eval_h5: Path, seed: int, limit_clips: int | None) -> dict:
    """Rebuild one run from its checkpoint and evaluate it."""
    with open(ckpt_dir / "config.json") as f:
        cfg_json = json.load(f)
    env_params = cfg_json["env_params"]
    net_params = cfg_json["net_params"]

    env_cls, default_fn = resolve_env_class(env_class_hint, env_params,
                                            default="AbsoluteImitation")
    base_cfg = parse_env_config(env_params, default_fn)

    # Build the network + load weights once, using the train env to size obs.
    # That env is then handed to evaluate_networks as the "train" dataset env.
    train_clips, test_clips = split_clips(base_cfg)
    train_env = _make_env(env_cls, prepare_eval_config(base_cfg), train_clips)
    loaded = load_network(ckpt_dir, net_params, train_env, seed)
    if loaded is None:
        return None
    nets, step = loaded

    return evaluate_networks(
        nets, env_cls, base_cfg,
        metadata=run_metadata(wid, wn, ckpt_dir, step, net_params,
                              env_cls.__name__),
        train_clips=train_clips, test_clips=test_clips, train_env=train_env,
        new_eval_h5=new_eval_h5, seed=seed, limit_clips=limit_clips,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--populate", action="store_true",
                   help="(Re)write the run list from analysis CSVs and exit.")
    p.add_argument("--collect", action="store_true",
                   help="Gather {ckpt_dir}/eval.json records written by "
                        "end-of-training evals into --collect-dir, then exit.")
    p.add_argument("--collect-dir", type=Path, default=DEFAULT_COLLECT_DIR)
    p.add_argument("--run-list", type=Path, default=DEFAULT_RUN_LIST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--checkpoint-dirs", nargs="+",
                   default=["downloaded_checkpoints", "checkpoints"],
                   help="Dirs (relative to repo root or absolute) searched for {name}.")
    p.add_argument("--new-eval-h5", type=Path, default=DEFAULT_NEW_EVAL_H5)
    p.add_argument("--override", action="store_true",
                   help="Recompute even if a result JSON already exists.")
    p.add_argument("--limit-clips", type=int, default=None,
                   help="Cap clips per split (local/memory-limited testing).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.populate:
        populate_run_list(args.run_list)
        return

    search_dirs = [
        Path(d) if Path(d).is_absolute() else REPO_ROOT / d
        for d in args.checkpoint_dirs
    ]

    if args.collect:
        collect_inline_evals(search_dirs, args.collect_dir, args.override)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = read_run_list(args.run_list)
    print(f"Loaded {len(runs)} runs from {args.run_list}")

    evaluated, skipped_existing, missing = [], [], []
    for wid, wn, env_class_hint in runs:
        out_path = args.output_dir / f"{wid}.json"
        if out_path.exists() and not args.override:
            skipped_existing.append(wn)
            continue

        ckpt_dir = next((d / wn for d in search_dirs if (d / wn).is_dir()), None)
        if ckpt_dir is None:
            warnings.warn(f"Checkpoint not found for {wn} ({wid}); skipping.")
            missing.append(wn)
            continue

        print(f"\n=== {wn} ({wid}) ===")
        try:
            result = evaluate_run(wid, wn, env_class_hint, ckpt_dir,
                                  args.new_eval_h5, args.seed, args.limit_clips)
            if result is None:
                missing.append(wn)
                continue
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  wrote {out_path}")
            evaluated.append(wn)
        except Exception as e:  # noqa: BLE001 — keep the batch going
            warnings.warn(f"Eval failed for {wn} ({wid}): {e!r}")
            missing.append(wn)
        finally:
            # Each run has a unique (static) env + network, so its compiled
            # executables — which bake the env's reference-clip arrays in as
            # constants — are never reused. Without eviction they accumulate
            # across runs until the GPU OOMs. Clear per run; nothing is lost.
            jax.clear_caches()
            gc.collect()

    print(f"\n=== Summary ===")
    print(f"  evaluated:        {len(evaluated)}")
    print(f"  skipped existing: {len(skipped_existing)}")
    print(f"  missing/failed:   {len(missing)}")
    if missing:
        print("  missing/failed runs:")
        for wn in missing:
            print(f"    - {wn}")


if __name__ == "__main__":
    main()
