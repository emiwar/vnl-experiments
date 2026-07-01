"""Linear-decoding helpers for the implicit-forward-model probe (stage 2 core).

Given the activations recorded by ``record_activations.py``, fit a **ridge linear
decoder** from each layer's activations to two targets and report held-out R²:

* ``proprio`` — the current (un-delayed) proprioception. How much current state is
  linearly present in the layer.
* ``delta``   — current minus delayed proprioception (delayed = current shifted
  by ``delay_k`` steps within a clip, matching the in-network ``Delay``). This is
  *exactly what a forward model must supply*: the part of the current state that
  is not already in the delayed input. Its decodability is the cleanest
  forward-model signature.

Two reference rows are emitted per file alongside the network layers:

* ``input::delayed_proprio`` — decode the targets from the delayed proprioception
  itself. This is the ``obs_(t-k) -> obs_t`` baseline: how far autocorrelation
  alone gets you. A layer beats the implicit-forward-model bar only if it decodes
  the targets *better than this*.
* ``input::current_proprio`` — decode from the current proprioception (a sanity
  ceiling: R²≈1 for ``proprio``, ≈1 for ``delta`` too since delta is a function
  of current+delayed — informative mainly as a pipeline check).

Decoder details
---------------
* **By-clip split** (not by-timestep): whole clips go to train or test, so the
  strong within-clip temporal autocorrelation can't leak the answer.
* **Ridge** with z-scored inputs; the penalty ``lambda`` is picked per layer from
  a small grid by an inner validation split carved out of the *train* clips, then
  the chosen model is refit on all train clips and scored once on the held-out
  test clips. R² is out-of-sample, baselined against the train-mean predictor.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

LAMBDA_GRID = (1.0, 10.0, 100.0, 1000.0)
DEFAULT_TEST_FRAC = 0.3
DEFAULT_VAL_FRAC = 0.2          # of the train clips, for lambda selection
DEFAULT_MAX_SAMPLES = 60_000    # cap rows per fit (subsampled reproducibly)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_recording(path: str | Path) -> dict:
    """Load an activation HDF5 into numpy arrays + attrs."""
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        target = f["target_proprio"][:].astype(np.float32)   # [T, N, P]
        dones = f["dones"][:].astype(bool)                   # [T, N]
        # Activations are nested groups mirroring the container tree; collect
        # every leaf dataset keyed by its full "/"-path under "activations".
        layers: dict[str, np.ndarray] = {}

        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                layers[name] = obj[:].astype(np.float32)     # [T, N, feat]

        f["activations"].visititems(_collect)
    return {"attrs": attrs, "target": target, "dones": dones, "layers": layers}


# ---------------------------------------------------------------------------
# Targets and validity mask
# ---------------------------------------------------------------------------

def make_targets(target: np.ndarray, delay_k: int) -> dict[str, np.ndarray]:
    """Return current proprio, the delayed version, and their delta ([T,N,P])."""
    delayed = np.zeros_like(target)
    if delay_k > 0:
        delayed[delay_k:] = target[:-delay_k]
    else:
        delayed[:] = target
    return {"current": target, "delayed": delayed, "delta": target - delayed}


def valid_mask(dones: np.ndarray, delay_k: int) -> np.ndarray:
    """Alive steps with the delayed buffer filled: ``[T, N]`` boolean."""
    T = dones.shape[0]
    alive = ~dones
    warm = np.zeros_like(dones)
    warm[delay_k:] = True
    return alive & warm


def efference_queue(actions: np.ndarray, eff: int) -> np.ndarray:
    """Reconstruct the efference copy from the recorded action leaf.

    At step ``t`` the queue the network sees is the last ``eff`` actions
    ``[a(t-1), ..., a(t-eff)]`` (zeros before the clip starts, matching
    ``EfferenceCopy``'s init). Returns ``[T, N, eff*A]`` (empty if ``eff == 0``).
    Each clip is one contiguous episode from t=0, so a plain time-shift per
    column is correct. Order within the queue is irrelevant to a linear decoder.
    """
    T, N, A = actions.shape
    if eff <= 0:
        return np.zeros((T, N, 0), dtype=actions.dtype)
    q = np.zeros((T, N, eff, A), dtype=actions.dtype)
    for j in range(1, eff + 1):
        q[j:, :, j - 1, :] = actions[: T - j]
    return q.reshape(T, N, eff * A)


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def _ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ymean = Y.mean(0)
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1], dtype=Xs.dtype)
    W = np.linalg.solve(A, Xs.T @ (Y - ymean))
    return (mu, sd, ymean, W)


def _ridge_predict(model, X):
    mu, sd, ymean, W = model
    return ((X - mu) / sd) @ W + ymean


def _r2(Y, pred, ymean) -> float:
    """Out-of-sample R², baselined against the train-mean predictor."""
    ss_res = float(((Y - pred) ** 2).sum())
    ss_tot = float(((Y - ymean) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# By-clip split + decode
# ---------------------------------------------------------------------------

def _split_clips(n_clips, test_frac, val_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_clips)
    n_test = max(1, int(round(test_frac * n_clips)))
    test = perm[:n_test]
    rest = perm[n_test:]
    n_val = max(1, int(round(val_frac * len(rest))))
    val, train = rest[:n_val], rest[n_val:]
    return set(train.tolist()), set(val.tolist()), set(test.tolist())


def _rows(X3, Y3, mask, clip_set, max_samples, seed):
    """Flatten [T,N,*] -> [rows, *] for clips in ``clip_set`` where mask is set."""
    T, N = mask.shape
    clip_ok = np.array([n in clip_set for n in range(N)])
    sel = mask & clip_ok[None, :]                       # [T, N]
    idx = np.argwhere(sel)                              # [rows, 2] (t, n)
    if len(idx) > max_samples:
        rng = np.random.default_rng(seed)
        idx = idx[rng.choice(len(idx), max_samples, replace=False)]
    t, n = idx[:, 0], idx[:, 1]
    return X3[t, n], Y3[t, n]


def decode(X3, Y3, mask, n_clips, *, test_frac=DEFAULT_TEST_FRAC,
           val_frac=DEFAULT_VAL_FRAC, max_samples=DEFAULT_MAX_SAMPLES,
           seed=0, lambdas=LAMBDA_GRID) -> dict:
    """Ridge-decode ``Y3`` from ``X3`` with a by-clip split. Returns metrics."""
    train, val, test = _split_clips(n_clips, test_frac, val_frac, seed)
    Xtr, Ytr = _rows(X3, Y3, mask, train, max_samples, seed + 1)
    Xva, Yva = _rows(X3, Y3, mask, val, max_samples, seed + 2)
    Xte, Yte = _rows(X3, Y3, mask, test, max_samples, seed + 3)
    if min(len(Xtr), len(Xva), len(Xte)) < X3.shape[-1] + 2:
        return {"test_r2": float("nan"), "lambda": float("nan"),
                "n_train": len(Xtr), "n_test": len(Xte), "n_features": X3.shape[-1]}

    best_lam, best_val = lambdas[0], -np.inf
    for lam in lambdas:
        m = _ridge_fit(Xtr, Ytr, lam)
        r2 = _r2(Yva, _ridge_predict(m, Xva), m[2])
        if r2 > best_val:
            best_val, best_lam = r2, lam

    # Refit on train+val with the chosen lambda; score once on test.
    Xtv = np.concatenate([Xtr, Xva]); Ytv = np.concatenate([Ytr, Yva])
    model = _ridge_fit(Xtv, Ytv, best_lam)
    test_r2 = _r2(Yte, _ridge_predict(model, Xte), model[2])
    return {"test_r2": test_r2, "val_r2": best_val, "lambda": best_lam,
            "n_train": len(Xtv), "n_test": len(Xte), "n_features": X3.shape[-1]}


def decode_file(path: str | Path, *, seed=0, **kw) -> list[dict]:
    """Decode every layer + the two input references in one recording file.

    Returns one row per (layer, target) with the recording's metadata attached.
    """
    rec = load_recording(path)
    attrs, target, dones, layers = (
        rec["attrs"], rec["target"], rec["dones"], rec["layers"])
    delay_k = int(attrs["delay_k"])
    n_clips = int(target.shape[1])
    tg = make_targets(target, delay_k)
    mask = valid_mask(dones, delay_k)

    # Network layers + the reference "input" probes.
    probes = {f"layer::{name}": arr for name, arr in layers.items()}
    probes["input::delayed_proprio"] = tg["delayed"]
    probes["input::current_proprio"] = tg["current"]

    # The actual input to the actor's decoder/predictor: the delayed
    # proprioception PLUS the efference copy (reconstructed from the recorded
    # action leaf). Decoding the target from this is the principled "layer 0"
    # baseline — the best a *linear* readout of the raw forward-model inputs can
    # do. Any deeper layer that beats it has added genuine (nonlinear / learned)
    # computation, not just a projection of its inputs.
    eff = int(attrs["efference_length"])
    action_leaf = next((arr for name, arr in layers.items()
                        if name.endswith("5/action")), None)
    if action_leaf is not None:
        queue = efference_queue(action_leaf, eff)
        probes["input::delayed_plus_efference"] = np.concatenate(
            [tg["delayed"], queue], axis=-1)

    targets = {"proprio": tg["current"], "delta": tg["delta"]}

    rows = []
    for probe_name, X3 in probes.items():
        for tname, Y3 in targets.items():
            res = decode(X3, Y3, mask, n_clips, seed=seed, **kw)
            rows.append({
                "run_name": str(attrs["run_name"]),
                "condition": str(attrs["condition"]),
                "network_class": str(attrs["network_class"]),
                "dataset": str(attrs["dataset"]),
                "delay_k": delay_k,
                "efference_length": int(attrs["efference_length"]),
                "step": int(attrs["step"]),
                "proprio_size": int(attrs["proprio_size"]),
                "probe": probe_name,
                "target": tname,
                **res,
            })
    return rows
