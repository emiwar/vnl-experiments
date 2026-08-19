"""Ridge linear decoding of recorded activations: how much of the current state is there?

Given the activations written by ``record_activations.py``, fit a **ridge linear decoder**
from each layer to two targets and report held-out R²:

* ``proprio`` -- the current (un-delayed) proprioception. How much current state is
  linearly present in the layer.
* ``delta``   -- current minus delayed proprioception (delayed = current shifted by
  ``delay_k`` steps within a clip, matching the in-network ``Delay``). This is *exactly what
  a forward model must supply*: the part of the current state not already in the delayed
  input, so its decodability is the cleanest forward-model signature. At ``delay_k == 0``
  the delta is identically zero and the R² is undefined -- reported as NaN with
  ``target_degenerate`` set, never as a silent gap.

Three reference probes are emitted alongside the network layers:

* ``input::delayed_proprio`` -- the ``obs_(t-k) -> obs_t`` autocorrelation floor.
* ``input::current_proprio`` -- ceiling / pipeline sanity check (R² ~ 1).
* ``input::delayed_plus_efference`` -- delayed proprioception **plus** the efference copy,
  i.e. the actual input to the actor. This is the principled "layer 0" baseline: the best a
  *linear* readout of the raw inputs can do. A deeper layer only demonstrates learned
  computation by beating it. (An earlier version of this analysis used the delayed-only
  floor and consequently overstated its result.)

Decoder details
---------------
* **By-clip split** (not by-timestep): whole clips go to train or test, so within-clip
  temporal autocorrelation cannot leak the answer.
* **Ridge** on z-scored inputs; ``lambda`` is picked per layer from a small grid on an inner
  validation split carved out of the *train* clips, then refit on train+val and scored once
  on the held-out test clips. R² is out-of-sample against the train-mean predictor.

Provenance: ported verbatim (arithmetic unchanged) from
``analysis/implicit-forward-model/decode.py``, which stays frozen with its analysis. The
additions here are lazy layer loading, store-path opening, explicit metadata, and the
degeneracy/validity reporting. ``linear_decoding_test.py`` pins the numbers against that
folder's committed ``data.csv`` -- if a change to this module breaks the old report's
figures, that test says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import h5py
import numpy as np

LAMBDA_GRID = (1.0, 10.0, 100.0, 1000.0)
DEFAULT_TEST_FRAC = 0.3
DEFAULT_VAL_FRAC = 0.2          # of the train clips, for lambda selection
DEFAULT_MAX_SAMPLES = 60_000    # cap rows per fit (subsampled reproducibly)

#: Bump when the arithmetic changes. Recorded in ``data.csv`` so a decoder change shows as
#: a diff on every row rather than a silent numeric shift.
DECODE_VERSION = 1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Recording:
    """One activation HDF5, with layers read on demand.

    The eager version of this loaded all ~31 leaves as float32 up front, about 4 GB per
    file. Reading one layer at a time keeps the peak under 1 GB, which is what makes
    decoding several recordings in parallel practical. ``target_proprio`` (~94 MB) and
    ``dones`` (~85 kB) are read eagerly because every fit needs them.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        self.attrs: dict[str, Any] = dict(self._file.attrs)
        self.target: np.ndarray = self._file["target_proprio"][:].astype(np.float32)
        self.dones: np.ndarray = self._file["dones"][:].astype(bool)
        names: list[str] = []
        self._file["activations"].visititems(
            lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
        self.layer_names: list[str] = names

    # -- context manager ----------------------------------------------------

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- accessors ----------------------------------------------------------

    def layer(self, name: str) -> np.ndarray:
        """One layer as ``[T, N, features]`` float32, read from disk on each call."""
        return self._file["activations"][name][:].astype(np.float32)

    @property
    def delay_k(self) -> int:
        return int(self.attrs["delay_k"])

    @property
    def efference_length(self) -> int:
        return int(self.attrs["efference_length"])

    @property
    def n_clips(self) -> int:
        return int(self.target.shape[1])

    def action_leaf_name(self) -> str | None:
        """The sampled-action leaf, needed to rebuild the efference queue.

        Matched by suffix because the path differs by architecture
        (``3/action/1/decoder/5/action`` vs ``3/action/1/5/action``). Asserted unique: two
        matches would mean silently rebuilding the queue from the wrong tensor.
        """
        hits = [n for n in self.layer_names if n.endswith("5/action")]
        if len(hits) > 1:
            raise ValueError(f"{self.path}: {len(hits)} leaves end in '5/action' ({hits}); "
                             f"cannot tell which is the sampled action")
        return hits[0] if hits else None


def open_recording(store, wandb_id: str, spec_id: str) -> Recording:
    """Open the activations artifact for one run from the artifact store."""
    entry = store.lookup("activations", wandb_id, spec_id)
    if entry is None:
        raise FileNotFoundError(
            f"no activations:{spec_id} for {wandb_id}. Produce or fetch it: "
            f"`python -m vnl_experiments.artifacts plan --kind activations --runs {wandb_id}`")
    return Recording(store.root / entry.path)


# ---------------------------------------------------------------------------
# Targets and validity
# ---------------------------------------------------------------------------

def make_targets(target: np.ndarray, delay_k: int) -> dict[str, np.ndarray]:
    """Return current proprio, the delayed version, and their delta (``[T,N,P]``)."""
    delayed = np.zeros_like(target)
    if delay_k > 0:
        delayed[delay_k:] = target[:-delay_k]
    else:
        delayed[:] = target
    return {"current": target, "delayed": delayed, "delta": target - delayed}


def degenerate_targets(delay_k: int) -> dict[str, bool]:
    """Which targets carry no signal at this delay, so their R² must be NaN."""
    return {"proprio": False, "delta": delay_k == 0}


def valid_mask(dones: np.ndarray, delay_k: int) -> np.ndarray:
    """Alive steps with the delayed buffer filled: ``[T, N]`` boolean."""
    alive = ~dones
    warm = np.zeros_like(dones)
    warm[delay_k:] = True
    return alive & warm


def valid_stats(dones: np.ndarray, delay_k: int) -> dict[str, float]:
    """How much usable data a recording actually has.

    Survival falls steeply with delay, so the number of rows behind each R² varies several
    fold across cells. R² is scale-free but its *estimation noise* is not, so this belongs
    in the committed data rather than being inferred from the delay.
    """
    mask = valid_mask(dones, delay_k)
    alive = ~dones
    return {"valid_rows": int(mask.sum()),
            "frac_valid": float(mask.mean()),
            "mean_lifespan": float(alive.sum(0).mean())}


def efference_queue(actions: np.ndarray, eff: int) -> np.ndarray:
    """Reconstruct the efference copy from the recorded action leaf.

    At step ``t`` the queue the network sees is the last ``eff`` actions
    ``[a(t-1), ..., a(t-eff)]`` (zeros before the clip starts, matching ``EfferenceCopy``'s
    init). Returns ``[T, N, eff*A]`` (empty if ``eff == 0``). Each clip is one contiguous
    episode from t=0, so a plain time-shift per column is correct. Order within the queue is
    irrelevant to a linear decoder.
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
    """Flatten ``[T,N,*] -> [rows, *]`` for clips in ``clip_set`` where mask is set."""
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
        return {"test_r2": float("nan"), "val_r2": float("nan"),
                "lambda": float("nan"), "n_train": len(Xtr), "n_test": len(Xte),
                "n_features": X3.shape[-1]}

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


# ---------------------------------------------------------------------------
# Whole-file decoding
# ---------------------------------------------------------------------------

def iter_probes(rec: Recording, targets: Mapping[str, np.ndarray]) -> Iterator[tuple]:
    """Yield ``(probe_name, X3)`` one at a time so only one layer is resident."""
    for name in rec.layer_names:
        yield f"layer::{name}", rec.layer(name)

    yield "input::delayed_proprio", targets["delayed"]
    yield "input::current_proprio", targets["current"]

    action_name = rec.action_leaf_name()
    if action_name is not None:
        queue = efference_queue(rec.layer(action_name), rec.efference_length)
        yield "input::delayed_plus_efference", np.concatenate(
            [targets["delayed"], queue], axis=-1)


def decode_file(path: str | Path, *, meta: Mapping[str, Any] | None = None,
                seed: int = 0, **kw) -> list[dict]:
    """Decode every layer plus the reference probes in one recording.

    ``meta`` overrides or adds row fields -- pass the run's ``condition``/``arm``/``budget``
    from the run index. Store-produced recordings carry ``attrs["condition"] == ""``
    (``ActivationsProducer`` passes ``ctx.get("condition", "")`` and the CLI never sets it),
    so relying on the file for it yields empty conditions and blank figures; that case
    raises here instead.
    """
    from vnl_experiments.probes import pathways

    meta = dict(meta or {})
    with Recording(path) as rec:
        condition = str(rec.attrs.get("condition") or "") or str(meta.get("condition") or "")
        if not condition:
            raise ValueError(
                f"{path}: attrs['condition'] is empty and no meta override was given. "
                f"Store-produced activations never carry a condition -- derive it from the "
                f"run index (fm_loss_weight / detach_prediction) and pass meta=.")

        delay_k = rec.delay_k
        tg = make_targets(rec.target, delay_k)
        mask = valid_mask(rec.dones, delay_k)
        degenerate = degenerate_targets(delay_k)
        targets = {"proprio": tg["current"], "delta": tg["delta"]}

        base = {"run_name": str(rec.attrs["run_name"]),
                "condition": condition,
                "network_class": str(rec.attrs["network_class"]),
                "dataset": str(rec.attrs["dataset"]),
                "delay_k": delay_k,
                "efference_length": rec.efference_length,
                "step": int(rec.attrs["step"]),
                "proprio_size": int(rec.attrs["proprio_size"]),
                "decode_seed": seed,
                "decode_version": DECODE_VERSION,
                "probe_set_version": pathways.PROBE_SET_VERSION}
        base.update(meta)

        rows = []
        for probe_name, X3 in iter_probes(rec, tg):
            pathway, stage_index, stage_label = pathways.resolve_stage(probe_name)
            for target_name, Y3 in targets.items():
                res = decode(X3, Y3, mask, rec.n_clips, seed=seed, **kw)
                rows.append({**base,
                             "probe": probe_name,
                             "pathway": pathway,
                             "stage_index": stage_index,
                             "stage_label": stage_label,
                             "target": target_name,
                             "target_degenerate": degenerate[target_name],
                             **res})
            del X3
    return rows
