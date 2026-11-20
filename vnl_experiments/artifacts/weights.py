"""Network weights as artifacts: one flat ``.npz`` per (run, checkpoint step).

A checkpoint step directory holds ``networks/`` (orbax), ``optimizer/`` (orbax) and
``metadata.pkl`` (pickled env and carry state). For analysing *weights* only the first is
needed, and it is small: 0.08--30 MB, against 100--860 MB for the pickle. So the
``weights`` artifact is ``networks/`` exported to one ``.npz``:

* Each key is the ``/``-joined nnx path of one variable, with the trailing ``value``
  dropped, e.g. ``afferents/arm_L/linear/kernel`` or ``normalizer/M2/arm_L``. Every
  non-``RngKey`` variable is there: ``Param``s, ``RngCount``s and the observation
  normaliser's running statistics, without which the input weights cannot be read.
* ``__config__`` holds the run's ``config.json`` (``env_params`` + ``net_params``) as a
  JSON string, so a file describes itself without the checkpoint directory.
* **Not** included: optimizer state, ``metadata.pkl``, and the nnx ``RngKey``
  variables (pickled in ``metadata.pkl``, since orbax cannot store new-style keys). None
  of them matter for reading weights or running the policy; they matter for resuming
  training, which this artifact does not support.

Identity. The spec is ``{"step": <int>}`` -- the concrete step, never ``"last"`` -- so
every checkpoint gets its own ``spec_id`` (``w-s0450007040-3fa1b2c4``) and the id depends
only on the step and the producer version. That makes the file names computable on the
laptop, which is what lets ``pull --steps 450007040`` fetch exactly one step, and lets an
eval artifact's ``resolved.checkpoint_step`` be joined to the weights it measured via
:func:`weights_spec_id`.

Staleness. A requeued run that rewinds rewrites ``step_*`` directories in place, so a
step exported mid-training can stop describing what is on disk. ``resolved.source_sha256``
is a hash of the exported ``networks/`` directory; ``ensure`` re-exports when it no
longer matches (:func:`is_stale`).

Reading::

    from vnl_experiments.artifacts.weights import load_weights, iter_weights
    w = load_weights("abc123")                 # the latest step held locally
    w.arrays["afferents/arm_L/linear/kernel"]  # numpy array
    for w in iter_weights("abc123"):           # every step held, in order
        ...
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from vnl_experiments.artifacts.store import Entry, Store, spec_id

KIND = "weights"
#: Bump when the meaning of the exported file changes (see producers.py).
VERSION = 1
EXT = ".npz"
CONFIG_KEY = "__config__"
STEP_DIR_PREFIX = "step_"


# --------------------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------------------


def weights_prefix(step: int) -> str:
    """Readable spec_id prefix. Zero-padded like the step dirs, so ``ls`` sorts by step."""
    return f"w-s{int(step):010d}"


def weights_spec(step: int) -> dict[str, Any]:
    return {"step": int(step)}


def weights_spec_id(step: int, version: int = VERSION) -> str:
    """The spec_id of the weights at ``step``, e.g. for joining an eval's checkpoint_step."""
    return spec_id(weights_prefix(step), weights_spec(step), version)


def step_from_spec_id(sid: str) -> int | None:
    """Inverse of :func:`weights_prefix` on a spec_id, or None if it is not one."""
    head = sid.split("-")
    if len(head) >= 2 and head[0] == "w" and head[1].startswith("s"):
        try:
            return int(head[1][1:])
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------------------
# step selection
# --------------------------------------------------------------------------------------


def checkpoint_steps(ckpt_dir: Path | str) -> list[int]:
    """Complete checkpoint steps in a run directory, ascending.

    Same rule as ``nnx_ppo.algorithms.checkpointing.latest_checkpoint``: only
    ``step_<int>`` directories count, so a half-written ``.tmp-step_*`` never does.
    A step additionally needs a ``networks/`` subdirectory to be exportable.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    steps = []
    for child in ckpt_dir.iterdir():
        if not child.name.startswith(STEP_DIR_PREFIX) or not child.is_dir():
            continue
        try:
            step = int(child.name[len(STEP_DIR_PREFIX):])
        except ValueError:
            continue
        if (child / "networks").is_dir():
            steps.append(step)
    return sorted(steps)


def step_dir(ckpt_dir: Path | str, step: int) -> Path:
    return Path(ckpt_dir) / f"{STEP_DIR_PREFIX}{int(step):010d}"


@dataclass(frozen=True)
class StepSelector:
    """Which checkpoints of a run to act on. Parse from the CLI with :meth:`parse`.

    * ``last`` -- the highest step (the default everywhere).
    * ``all`` -- every step.
    * ``every:N`` -- every N-th checkpoint counting from the first, plus the last, so
      a thinned series always ends at the trained network.
    * ``<int>[,<int>...]`` -- exactly these steps.
    """

    mode: str                       # "last" | "all" | "every" | "explicit"
    every: int = 1
    steps: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def parse(cls, text: str | None) -> "StepSelector":
        text = (text or "last").strip()
        if text in ("last", "all"):
            return cls(text)
        if text.startswith("every:"):
            n = int(text.split(":", 1)[1])
            if n < 1:
                raise ValueError(f"--steps {text!r}: N must be >= 1")
            return cls("every", every=n)
        try:
            steps = tuple(sorted({int(s) for s in text.split(",") if s.strip()}))
        except ValueError:
            raise ValueError(
                f"--steps {text!r}: expected last, all, every:N, or comma-separated "
                f"integer steps") from None
        if not steps:
            raise ValueError("--steps: no steps given")
        return cls("explicit", steps=steps)

    @property
    def is_explicit(self) -> bool:
        return self.mode == "explicit"

    def select(self, available: Sequence[int]) -> list[int]:
        """Apply to the steps a run has. Explicit steps a run lacks are dropped here;
        callers that need to report them compare against :attr:`steps`."""
        available = sorted(available)
        if not available:
            return []
        if self.mode == "last":
            return [available[-1]]
        if self.mode == "all":
            return available
        if self.mode == "every":
            picked = available[:: self.every]
            if picked[-1] != available[-1]:
                picked.append(available[-1])
            return picked
        have = set(available)
        return [s for s in self.steps if s in have]

    def __str__(self) -> str:
        if self.mode == "every":
            return f"every:{self.every}"
        if self.mode == "explicit":
            return ",".join(str(s) for s in self.steps)
        return self.mode


# --------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------


def directory_sha256(directory: Path | str) -> str:
    """Hash of every file under ``directory``: relative paths and contents, in order."""
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _key_name(entry: Any) -> str:
    """One element of a jax key path as a string."""
    for attr in ("key", "idx", "name"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)


def flat_key(path: Sequence[Any]) -> str:
    """``/``-joined key for a variable, without the trailing ``value`` nnx adds."""
    names = [p if isinstance(p, str) else _key_name(p) for p in path]
    if names and names[-1] == "value":
        names = names[:-1]
    return "/".join(str(n) for n in names)


def read_networks_dir(networks_dir: Path | str) -> dict[str, np.ndarray]:
    """Restore an orbax ``networks/`` checkpoint to a flat ``{path: array}`` dict.

    No network is built. Orbax refuses a restore with no target ("Topology mismatch"),
    but the checkpoint's own metadata tree has every shape and dtype, so it serves as
    the target.
    """
    import jax
    import orbax.checkpoint as ocp

    networks_dir = Path(networks_dir).resolve()
    checkpointer = ocp.StandardCheckpointer()
    try:
        meta = checkpointer.metadata(networks_dir)
        tree_meta = getattr(meta, "item_metadata", meta)
        tree_meta = getattr(tree_meta, "tree", tree_meta)
        target = jax.tree_util.tree_map(lambda m: np.zeros(m.shape, m.dtype), tree_meta)
        tree = checkpointer.restore(networks_dir, target)
    finally:
        checkpointer.close()

    out: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        key = flat_key(path)
        if key in out:
            raise ValueError(f"duplicate flattened key {key!r} in {networks_dir}")
        out[key] = np.asarray(leaf)
    return out


def export_step(ckpt_dir: Path | str, step: int, out_path: Path | str) -> dict[str, Any]:
    """Write the weights of one checkpoint step to ``out_path``; return ``resolved``."""
    ckpt_dir = Path(ckpt_dir)
    sdir = step_dir(ckpt_dir, step)
    networks_dir = sdir / "networks"
    if not networks_dir.is_dir():
        raise FileNotFoundError(f"no {networks_dir}")
    source_sha = directory_sha256(networks_dir)

    arrays = read_networks_dir(networks_dir)
    if CONFIG_KEY in arrays:
        raise ValueError(f"a network variable is named {CONFIG_KEY!r}")
    config_path = ckpt_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    out_path = Path(out_path)
    tmp = out_path.with_name(f".tmp-{out_path.name}")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays, **{CONFIG_KEY: np.array(json.dumps(config))})
    tmp.replace(out_path)

    net_params = config.get("net_params", {}) if isinstance(config, dict) else {}
    return {
        "checkpoint_step": int(step),
        "network_class": net_params.get("network_class"),
        "n_arrays": len(arrays),
        "n_params": int(sum(a.size for a in arrays.values())),
        "bytes_raw": int(sum(a.nbytes for a in arrays.values())),
        # A diverged run's weights go NaN (one NaN reward is enough; see NaNGuardWrapper)
        # and stay checkpointed that way. Surfaced so the manifest can filter on it.
        "n_nonfinite": int(sum(int((~np.isfinite(a)).sum()) for a in arrays.values()
                               if a.dtype.kind in "fc")),
        "has_config": bool(config),
        "source_sha256": source_sha,
        "source_mtime": networks_dir.stat().st_mtime,
    }


def is_stale(entry: Entry, ckpt_dir: Path | str | None) -> bool:
    """Whether the checkpoint this entry was exported from has since been rewritten.

    False when the checkpoint is not here to compare against (e.g. on the laptop): absence
    of the source is not evidence the artifact is wrong.
    """
    if ckpt_dir is None:
        return False
    networks_dir = step_dir(ckpt_dir, entry.spec["step"]) / "networks"
    if not networks_dir.is_dir():
        return False
    recorded = entry.resolved.get("source_sha256")
    return recorded is not None and recorded != directory_sha256(networks_dir)


# --------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------


@dataclass
class Weights:
    """One exported checkpoint: arrays by flat path, the run config, and the sidecar."""

    wandb_id: str
    step: int
    arrays: dict[str, np.ndarray]
    config: dict[str, Any]
    entry: Entry

    @property
    def net_params(self) -> dict[str, Any]:
        return self.config.get("net_params", {})

    @property
    def env_params(self) -> dict[str, Any]:
        return self.config.get("env_params", {})

    def subset(self, prefix: str) -> dict[str, np.ndarray]:
        """Arrays under one module, keyed relative to it: ``subset("afferents/arm_L")``."""
        prefix = prefix.rstrip("/") + "/"
        return {k[len(prefix):]: v for k, v in self.arrays.items() if k.startswith(prefix)}


def read_npz(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Arrays and config from an exported file. Never unpickles."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != CONFIG_KEY}
        config = json.loads(str(data[CONFIG_KEY])) if CONFIG_KEY in data.files else {}
    return arrays, config


def entries_for(wandb_id: str, store: Store | None = None,
                version: int | None = VERSION) -> list[Entry]:
    """The weights sidecars held locally for a run, ascending by step.

    Only the current producer ``version`` by default, so a future format change can
    never mix old and new exports in one series; pass ``version=None`` for all.
    """
    store = store or Store()
    directory = store.root / KIND / wandb_id
    if not directory.is_dir():
        return []
    out = []
    for meta in directory.glob("*.meta.json"):
        sid = meta.name[: -len(".meta.json")]
        entry = store.lookup(KIND, wandb_id, sid)
        if entry is None:
            continue
        if version is not None and entry.producer.get("version") != version:
            continue
        out.append(entry)
    return sorted(out, key=lambda e: int(e.spec["step"]))


def available_steps(wandb_id: str, store: Store | None = None) -> list[int]:
    """Steps whose weights are in the local store, ascending."""
    return [int(e.spec["step"]) for e in entries_for(wandb_id, store)]


def _load_entry(wandb_id: str, entry: Entry, store: Store) -> Weights:
    arrays, config = read_npz(store.root / entry.path)
    return Weights(wandb_id=wandb_id, step=int(entry.spec["step"]), arrays=arrays,
                   config=config, entry=entry)


def load_weights(wandb_id: str, step: int | str = "last",
                 store: Store | None = None) -> Weights:
    """The weights of one run at one step (``"last"``: the highest step held locally).

    "Last" means last *pulled*, not last trained: pull with ``--steps last`` again to
    refresh it for a run that was still training.
    """
    store = store or Store()
    entries = entries_for(wandb_id, store)
    if not entries:
        raise FileNotFoundError(
            f"no weights for {wandb_id} in {store.root}; pull them with\n"
            f"  python -m vnl_experiments.artifacts pull --kind weights --runs {wandb_id}")
    if step == "last":
        return _load_entry(wandb_id, entries[-1], store)
    for entry in entries:
        if int(entry.spec["step"]) == int(step):
            return _load_entry(wandb_id, entry, store)
    raise FileNotFoundError(
        f"no weights for {wandb_id} at step {step}; held: "
        f"{[int(e.spec['step']) for e in entries]}")


def iter_weights(wandb_id: str, steps: str | StepSelector = "all",
                 store: Store | None = None) -> Iterator[Weights]:
    """Every selected step held locally, ascending, loaded one at a time."""
    store = store or Store()
    selector = steps if isinstance(steps, StepSelector) else StepSelector.parse(steps)
    entries = {int(e.spec["step"]): e for e in entries_for(wandb_id, store)}
    for step in selector.select(list(entries)):
        yield _load_entry(wandb_id, entries[step], store)


# --------------------------------------------------------------------------------------
# back into a network
# --------------------------------------------------------------------------------------


def restore_into(networks: Any, weights: Weights | Mapping[str, np.ndarray]) -> None:
    """Overwrite a built network's variables (all but ``RngKey``) from exported weights.

    ``networks`` must have the architecture the weights came from, e.g. built with
    ``delays.network_builders.build_network(w.net_params, env, nnx.Rngs(0))``. Every
    variable must be matched exactly, in both directions, with the same shape;
    anything else raises with the offending paths listed, because a partial restore
    would leave a silently half-random network.
    """
    import jax.numpy as jnp
    from flax import nnx

    arrays = weights.arrays if isinstance(weights, Weights) else dict(weights)
    _, _, state = nnx.split(networks, nnx.RngKey, ...)
    flat = list(nnx.to_flat_state(state))
    targets = {flat_key(path): (path, var) for path, var in flat}

    missing = sorted(set(targets) - set(arrays))
    extra = sorted(set(arrays) - set(targets))
    shape_bad = sorted(k for k in set(targets) & set(arrays)
                       if tuple(np.shape(targets[k][1].get_value())) != tuple(arrays[k].shape))
    if missing or extra or shape_bad:
        def few(xs):
            return ", ".join(xs[:8]) + (f" (+{len(xs) - 8} more)" if len(xs) > 8 else "")
        raise ValueError(
            "weights do not match the network:"
            + (f"\n  missing from weights: {few(missing)}" if missing else "")
            + (f"\n  not in network: {few(extra)}" if extra else "")
            + (f"\n  shape mismatch: {few(shape_bad)}" if shape_bad else ""))

    new_flat = []
    for path, var in flat:
        value = arrays[flat_key(path)]
        new_flat.append((path, var.replace(jnp.asarray(value, dtype=var.get_value().dtype))))
    nnx.update(networks, nnx.from_flat_state(new_flat))
