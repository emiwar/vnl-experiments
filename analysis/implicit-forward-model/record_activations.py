#!/usr/bin/env python3
"""Stage 1 of the implicit-forward-model probe: record per-layer activations.

Question
--------
Do the standard enc-dec networks (delayed proprioception + efference copy, but
*no* explicit forward model) nonetheless build an **implicit** forward model? If
they do, their internal activations should let us linearly reconstruct the
*current, non-delayed* proprioception better than the delayed input alone — and
ideally approaching the explicit-forward-model network, whose ``predictor`` is
trained to output exactly that.

What this script does
---------------------
For each checkpoint in ``run_list.txt`` it rebuilds the env + network from
``config.json`` (reusing the verbatim builders in
``vnl_experiments.delays.eval_runs``), makes the network *recordable*
(:func:`nnx_ppo.networks.recording.with_recording`), and rolls out the
deterministic policy on the requested datasets — **one latched episode per clip,
reset at frame 0**, exactly like ``eval_runs._rollout``. Every step it stacks:

* ``extract_activations(out.metrics)``  — the per-layer activations, keyed by
  the module's structural path (e.g. ``action/0``, ``action/1/inner/...``);
* the **current** (un-delayed) flattened proprioception from ``env_state.obs``
  — this is the decoding *target* (the delay is applied *inside* the network,
  so the env's obs is the ground-truth current state);
* the pre-step ``done`` mask (so post-termination steps can be dropped).

Output: one HDF5 per (run, dataset) under ``eval_results/activations/`` (which
is git-ignored) — activations as float16, target proprio as float32, plus the
``delay_k`` / ``efference_length`` / clip bookkeeping needed by ``decode.py``.

Memory
------
Activations are materialised ``[n_steps, n_clips, Sigma units]`` on device. We
chunk over clips (``--clip-chunk``) and cap the rollout horizon (``--max-steps``)
so an 8 GB GPU stays comfortable; each chunk is pulled to host (float16) and the
JAX caches are cleared between runs. Check ``nvidia-smi`` before launching.

Usage (from repo root)::

    ../.venv/bin/python analysis/implicit-forward-model/record_activations.py \
        --datasets old_eval --limit-clips 8 --max-steps 200   # quick smoke test

    ../.venv/bin/python analysis/implicit-forward-model/record_activations.py \
        --datasets old_eval new_eval                          # full probe set
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import argparse
import functools
import gc
import json
import math
from pathlib import Path

import h5py
import jax
import jax.numpy as jp
import numpy as np
from flax import nnx
from jax import tree_util as jtu

from vnl_playground.tasks.reference_clips import ReferenceClips

from nnx_ppo.networks.recording import with_recording, extract_activations

# Reuse the env/network reconstruction verbatim — never duplicate it here.
from vnl_experiments.delays.eval_runs import (
    REPO_ROOT,
    DEFAULT_NEW_EVAL_H5,
    NEW_EVAL_CLIP_LENGTH,
    resolve_env_class,
    parse_env_config,
    build_network,
    load_network,
    _make_env,
)

HERE = Path(__file__).resolve().parent
DEFAULT_RUN_LIST = HERE / "run_list.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_results" / "activations"


# ---------------------------------------------------------------------------
# Run list
# ---------------------------------------------------------------------------

def read_run_list(path: Path) -> list[tuple[str, str, str]]:
    """Read ``name<TAB>condition<TAB>env_class`` rows (env_class optional)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            name = parts[0]
            condition = parts[1] if len(parts) > 1 else ""
            env_class = parts[2] if len(parts) > 2 and parts[2] else ""
            out.append((name, condition, env_class))
    return out


# ---------------------------------------------------------------------------
# Proprioception flattening (matches the network's Flattener: ravel + concat)
# ---------------------------------------------------------------------------

def _flatten_features(tree, batch: int):
    """Concatenate every leaf of ``tree`` into ``[batch, -1]`` (sorted by path)."""
    leaves = [
        jp.reshape(leaf, (batch, -1))
        for _, leaf in sorted(
            jtu.tree_leaves_with_path(tree),
            key=lambda kv: jtu.keystr(kv[0]),
        )
    ]
    return jp.concatenate(leaves, axis=-1)


# ---------------------------------------------------------------------------
# Recording rollout — deterministic, one latched episode per clip
# ---------------------------------------------------------------------------

def record_rollout(env, networks, clip_ids, n_steps: int, key):
    """Roll out one latched episode per clip, stacking activations + target.

    Mirrors ``eval_runs._rollout``'s reset/latch convention (reset at frame 0,
    ``done`` is monotonic) but returns per-step activations instead of reduced
    scalars. ``networks`` is wrapped with :func:`with_recording` (the original is
    untouched) and run in eval mode (deterministic actions).

    Returns ``(activations, target_proprio, dones)`` with leading dims
    ``[n_steps, n_clips, ...]``; ``target_proprio`` is the *current* (un-delayed)
    flattened proprioception of the obs that produced each step's activations.
    """
    rec = with_recording(networks)
    rec.eval()

    n_clips = int(clip_ids.shape[0])
    keys = jax.random.split(key, n_clips)
    env_states = jax.vmap(
        lambda k, c: env.reset(k, clip_idx=c, start_frame=0)
    )(keys, clip_ids)
    env_states = env_states.replace(done=env_states.done.astype(float))
    net_states = rec.initialize_state(n_clips)

    def step(env, networks, carry):
        env_state, net_state = carry
        out = networks(net_state, env_state.obs)
        # Target = current proprioception of THIS obs (pre-step, un-delayed).
        proprio = _flatten_features(env_state.obs["state"]["proprioception"], n_clips)
        next_env_state = jax.vmap(env.step)(env_state, out.output.actions)
        next_env_state = next_env_state.replace(
            done=jp.logical_or(next_env_state.done, env_state.done).astype(float)
        )
        ys = (extract_activations(out.metrics), proprio, env_state.done)
        return (next_env_state, out.next_state), ys

    step_scan = nnx.scan(
        functools.partial(step, env),
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=(nnx.Carry, 0),
        length=n_steps,
    )
    _, (activations, proprio, dones) = step_scan(rec, (env_states, net_states))
    return activations, proprio, dones


def _key_token(k) -> str:
    """A flat, slash-free token for one pytree path key."""
    for attr in ("key", "idx", "name"):  # DictKey / SequenceKey / GetAttrKey
        v = getattr(k, attr, None)
        if v is not None:
            return str(v)
    return str(k)


def _flatten_activations(act_tree) -> dict[str, np.ndarray]:
    """Flatten the activation pytree to ``{path: [T, N, feat]}`` (numpy float16).

    Each leaf keeps its ``[T, N]`` leading dims and is flattened over the rest.
    Path keys are joined with ``/`` so each leaf becomes a genuine nested HDF5
    group mirroring the network's container tree (e.g. ``action/0``,
    ``action/1/inner/predictor``). Non-floating leaves (e.g. an action sampler's
    integer fields) are skipped.
    """
    out: dict[str, np.ndarray] = {}
    for path, leaf in jtu.tree_leaves_with_path(act_tree):
        arr = np.asarray(leaf)
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        name = "/".join(_key_token(k) for k in path)
        t, n = arr.shape[0], arr.shape[1]
        out[name] = arr.reshape(t, n, -1).astype(np.float16)
    return out


# ---------------------------------------------------------------------------
# Per-run orchestration
# ---------------------------------------------------------------------------

def steps_for(clip_len_frames: int, frames_per_step: float, cap: int | None) -> int:
    n = int(math.ceil(clip_len_frames / frames_per_step)) + 2
    return n if cap is None else min(n, cap)


def record_run(name, condition, env_class_hint, ckpt_dir, datasets, new_eval_h5,
               seed, limit_clips, clip_chunk, max_steps, output_dir, override):
    with open(ckpt_dir / "config.json") as f:
        cfg_json = json.load(f)
    env_params = cfg_json["env_params"]
    net_params = cfg_json["net_params"]
    network_class = str(net_params.get("network_class", ""))
    delay_k = int(net_params.get("delay_k", 0) or 0)
    efference_length = int(net_params.get("efference_length", 0) or 0)

    env_cls, default_fn = resolve_env_class(env_class_hint, env_params,
                                            default="AbsoluteImitation")
    base_cfg = parse_env_config(env_params, default_fn)
    clip_length = int(base_cfg.clip_length)
    ctrl_dt = float(base_cfg.ctrl_dt)
    mocap_hz = int(base_cfg.mocap_hz)
    frames_per_step = ctrl_dt * mocap_hz

    all_clips = ReferenceClips(
        base_cfg.reference_data_path, clip_length, base_cfg.keep_clips_idx
    )
    train_clips, test_clips = all_clips.split()  # ratio=0.8, seed=0
    new_clips = ReferenceClips(str(new_eval_h5), NEW_EVAL_CLIP_LENGTH)

    spec = {
        "train": (train_clips, base_cfg, clip_length),
        "old_eval": (test_clips, base_cfg, clip_length),
    }
    new_cfg = parse_env_config(env_params, default_fn)
    new_cfg.clip_length = NEW_EVAL_CLIP_LENGTH
    spec["new_eval"] = (new_clips, new_cfg, NEW_EVAL_CLIP_LENGTH)

    # Build + load weights once (train env sizes the obs).
    train_env = _make_env(env_cls, base_cfg, train_clips)
    loaded = load_network(ckpt_dir, net_params, train_env, seed)
    if loaded is None:
        print(f"  could not load {name}; skipping.")
        return
    nets, step = loaded

    rollout_jit = nnx.jit(record_rollout, static_argnums=(0, 3))
    key = jax.random.key(seed)

    for ds in datasets:
        clips, cfg, clen = spec[ds]
        out_path = output_dir / f"{name}__{ds}.h5"
        if out_path.exists() and not override:
            print(f"  [{ds}] exists, skipping ({out_path.name})")
            continue

        env = train_env if clips is train_clips else _make_env(env_cls, cfg, clips)
        n_total = int(clips.qpos.shape[0])
        n_clips = n_total if limit_clips is None else min(n_total, limit_clips)
        n_steps = steps_for(clen, frames_per_step, max_steps)
        print(f"  [{ds}] {n_clips} clips x {n_steps} steps "
              f"({clen} frames, chunk={clip_chunk})")

        layer_chunks: dict[str, list[np.ndarray]] = {}
        proprio_chunks, done_chunks = [], []
        for lo in range(0, n_clips, clip_chunk):
            hi = min(lo + clip_chunk, n_clips)
            clip_ids = jp.arange(lo, hi)
            key, sub = jax.random.split(key)
            acts, proprio, dones = rollout_jit(env, nets, clip_ids, n_steps, sub)
            for lname, arr in _flatten_activations(acts).items():
                layer_chunks.setdefault(lname, []).append(arr)
            proprio_chunks.append(np.asarray(proprio, dtype=np.float32))
            done_chunks.append(np.asarray(dones, dtype=bool))
            del acts, proprio, dones
            gc.collect()
            print(f"    clips {lo}:{hi} done", flush=True)

        layers = {k: np.concatenate(v, axis=1) for k, v in layer_chunks.items()}
        target = np.concatenate(proprio_chunks, axis=1)       # [T, N, P]
        dones = np.concatenate(done_chunks, axis=1)           # [T, N]

        output_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs.update(dict(
                run_name=name, condition=condition, network_class=network_class,
                dataset=ds, delay_k=delay_k, efference_length=efference_length,
                step=int(step), proprio_size=int(target.shape[-1]),
                ctrl_dt=ctrl_dt, n_clips=int(target.shape[1]),
                n_steps=int(target.shape[0]),
            ))
            f.create_dataset("target_proprio", data=target, compression="gzip",
                             compression_opts=4)
            f.create_dataset("dones", data=dones, compression="gzip")
            g = f.create_group("activations")
            for lname, arr in layers.items():
                g.create_dataset(lname, data=arr, compression="gzip",
                                 compression_opts=4)
        n_layers = len(layers)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [{ds}] wrote {out_path.name}  ({n_layers} layers, {size_mb:.0f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-list", type=Path, default=DEFAULT_RUN_LIST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--checkpoint-dirs", nargs="+",
                   default=["downloaded_checkpoints", "checkpoints"])
    p.add_argument("--datasets", nargs="+", default=["old_eval"],
                   choices=["train", "old_eval", "new_eval"])
    p.add_argument("--new-eval-h5", type=Path, default=DEFAULT_NEW_EVAL_H5)
    p.add_argument("--limit-clips", type=int, default=None)
    p.add_argument("--clip-chunk", type=int, default=16,
                   help="Clips per device rollout (memory knob; an 8 GB GPU "
                        "handles 16-32, more for short --max-steps).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap the rollout horizon (memory knob; long clips lose "
                        "their tail, which is fine for decoding).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", action="store_true")
    args = p.parse_args()

    search_dirs = [
        Path(d) if Path(d).is_absolute() else REPO_ROOT / d
        for d in args.checkpoint_dirs
    ]
    runs = read_run_list(args.run_list)
    print(f"Loaded {len(runs)} runs from {args.run_list}; datasets={args.datasets}")

    for name, condition, env_hint in runs:
        ckpt_dir = next((d / name for d in search_dirs if (d / name).is_dir()), None)
        if ckpt_dir is None:
            print(f"\n=== {name}: checkpoint not found, skipping ===")
            continue
        print(f"\n=== {name}  (condition={condition}) ===")
        try:
            record_run(name, condition, env_hint, ckpt_dir, args.datasets,
                       args.new_eval_h5, args.seed, args.limit_clips,
                       args.clip_chunk, args.max_steps, args.output_dir,
                       args.override)
        except Exception as e:  # noqa: BLE001 — keep the batch going
            print(f"  FAILED for {name}: {e!r}")
        finally:
            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    main()
