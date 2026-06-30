"""Stage 2 of the implicit-forward-model probe: decode -> data.csv.

Reads the activation HDF5s written by ``record_activations.py`` (under
``eval_results/activations/``), fits the ridge linear decoders (``decode.py``),
and writes the committed snapshot:

* ``data.csv``          one row per (run, dataset, probe, target) with held-out R²;
* ``comparability.txt`` invariants across the three conditions, so the
  implicit-vs-explicit comparison is auditably fair (§4 of analysis/README.md).

This is the only stage besides ``record_activations.py`` that touches the data
sources (the HDF5s and each checkpoint's ``config.json``). ``plot.py`` reads only
``data.csv``.

Run from the repo root (after recording)::

    ../.venv/bin/python analysis/implicit-forward-model/extract.py --datasets old_eval
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import decode` resolves when run from repo root
from decode import decode_file  # noqa: E402

REPO_ROOT = HERE.parents[1]
ACT_DIR = REPO_ROOT / "eval_results" / "activations"
CKPT_DIRS = [REPO_ROOT / "downloaded_checkpoints", REPO_ROOT / "checkpoints"]

# Invariants that must match across conditions for the comparison to be fair.
# (delay_k / network_class are intentionally allowed to vary — that's the
# experimental contrast; the no_delay floor also differs by design.)
INVARIANT_NET_KEYS = ["latent_size", "kl_weight", "enc_hidden_sizes",
                      "dec_hidden_sizes", "efference_length", "normalize_obs"]
INVARIANT_ENV_KEYS = ["body_target_frame", "ctrl_dt", "clip_set", "rescale_factor"]


def read_run_list(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return rows


def find_ckpt(name: str) -> Path | None:
    return next((d / name for d in CKPT_DIRS if (d / name).is_dir()), None)


def comparability(run_list, datasets) -> str:
    """Report config invariants per condition (programmatic comparability)."""
    lines = ["Implicit-forward-model probe — comparability\n" + "=" * 44, ""]
    rows = []
    for name, cond in run_list:
        ckpt = find_ckpt(name)
        if ckpt is None:
            lines.append(f"  {cond:14s} {name}: CHECKPOINT MISSING")
            continue
        cfg = json.loads((ckpt / "config.json").read_text())
        net, env = cfg["net_params"], cfg["env_params"]
        step = max((int(p.name.split("_")[1]) for p in ckpt.glob("step_*")),
                   default=-1)
        rec = {"condition": cond, "run_name": name, "network_class":
               net.get("network_class"), "delay_k": net.get("delay_k"),
               "max_step": step}
        for k in INVARIANT_NET_KEYS:
            rec[k] = net.get(k)
        for k in INVARIANT_ENV_KEYS:
            rec[k] = env.get(k)
        rows.append(rec)

    df = pd.DataFrame(rows)
    lines.append(df.to_string(index=False))
    lines.append("")
    lines.append("Invariants (should be single-valued across conditions):")
    for k in INVARIANT_NET_KEYS + INVARIANT_ENV_KEYS + ["max_step"]:
        vals = df[k].astype(str).unique().tolist() if k in df else []
        flag = "OK " if len(vals) == 1 else "*** VARIES ***"
        lines.append(f"  {flag} {k}: {vals}")
    lines.append("")
    lines.append("Experimental contrast (expected to vary): condition, "
                 "network_class, and the delay sweep delay_k / efference_length "
                 "(delay_k=0 is the matched no-delay floor).")
    lines.append(f"Datasets decoded: {datasets}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["old_eval"],
                   choices=["train", "old_eval", "new_eval"])
    p.add_argument("--run-list", type=Path, default=HERE / "run_list.txt")
    p.add_argument("--act-dir", type=Path, default=ACT_DIR)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_list = read_run_list(args.run_list)
    all_rows = []
    for name, cond in run_list:
        for ds in args.datasets:
            h5 = args.act_dir / f"{name}__{ds}.h5"
            if not h5.exists():
                print(f"  missing {h5.name}; run record_activations.py first.")
                continue
            print(f"  decoding {h5.name} ...", flush=True)
            all_rows.extend(decode_file(h5, seed=args.seed))

    if not all_rows:
        print("No activation files found — nothing written.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["dataset", "condition", "target", "probe"]).reset_index(drop=True)
    (HERE / "data.csv").write_text(df.to_csv(index=False))
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")

    report = comparability(run_list, args.datasets)
    (HERE / "comparability.txt").write_text(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
