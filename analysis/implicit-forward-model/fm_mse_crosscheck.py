"""Cross-check: recompute the forward-prediction L2 from recorded activations.

The WandB analysis ``analysis/forward-loss-vs-architecture`` reads
``fm_pred_mse`` = mean squared error between the predictor output ``p̂`` and the
(normalised) current proprioception. That exact quantity is recoverable from our
recordings: ``p̂`` is the predictor's output leaf (``.../predictor/4``) and the
normalised current proprioception is the ``Normalizer`` leaf
(``1/state/proprioception``) — the same signal ForwardModel uses as its target.

For every forward-model-architecture run this prints, over the valid mask
(alive & delay-warmed steps):

* ``mse``       — mean((p̂ − current_norm)²), directly comparable to the logged
                  ``fm_pred_mse`` (a *direct*, no-decoder readout);
* ``r2_direct`` — 1 − mse / var(current_norm): how good p̂ is *as itself*;
* ``r2_decoded``— the linear-decodability of current proprio from p̂ (copied from
                  data.csv) — how much current-state info is *linearly present* in
                  p̂, regardless of whether it is output in identity form.

Run from the repo root::

    ../.venv/bin/python analysis/implicit-forward-model/fm_mse_crosscheck.py
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ACT_DIR = REPO_ROOT / "eval_results" / "activations"


def read_run_list():
    rows = []
    for line in (HERE / "run_list.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return rows


def _leaf(f, needle):
    found = {}
    f["activations"].visititems(
        lambda n, o: found.__setitem__(n, o[:]) if hasattr(o, "shape")
        and n.endswith(needle) else None)
    return next(iter(found.values())) if found else None


def main():
    df = pd.read_csv(HERE / "data.csv")
    print(f"{'condition':16s} {'delay':>5} {'mse':>8} {'r2_direct':>10} "
          f"{'r2_decoded(p̂)':>14}")
    print("-" * 60)
    for name, cond in read_run_list():
        if cond not in ("forward_model", "pg_forward_model"):
            continue
        h5 = ACT_DIR / f"{name}__old_eval.h5"
        if not h5.exists():
            continue
        with h5py.File(h5, "r") as f:
            dk = int(f.attrs["delay_k"])
            dones = f["dones"][:].astype(bool)
            phat = _leaf(f, "predictor/4").astype(np.float32)      # [T,N,P]
            cur = _leaf(f, "1/state/proprioception").astype(np.float32)
        mask = (~dones)
        mask[:dk] = False
        p = phat[mask]; c = cur[mask]
        mse = float(np.mean((p - c) ** 2))
        var = float(np.mean((c - c.mean(0)) ** 2))
        r2_direct = 1.0 - mse / var if var > 0 else float("nan")

        dec = df[(df.run_name == name) & (df.target == "proprio")
                 & (df.probe == "layer::3/action/1/predictor/4")]["test_r2"]
        r2_dec = float(dec.iloc[0]) if len(dec) else float("nan")
        print(f"{cond:16s} {dk:>5} {mse:>8.3f} {r2_direct:>10.3f} "
              f"{r2_dec:>14.3f}")


if __name__ == "__main__":
    main()
