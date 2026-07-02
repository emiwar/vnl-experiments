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

Populate the run list (one-off, reads the analysis CSVs)::

    ../.venv/bin/python -m vnl_experiments.delays.eval_runs --populate

Evaluate (cluster: all checkpoints present; locally use --limit-clips)::

    ../.venv/bin/python -m vnl_experiments.delays.eval_runs
    ../.venv/bin/python -m vnl_experiments.delays.eval_runs \
        --run-list /tmp/test_runs.txt --limit-clips 16
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
import math
import pickle
import warnings
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from flax import nnx
from jax import tree_util as jtu

from etils import epath

from vnl_playground.tasks.rodent.imitation import Imitation
from vnl_playground.tasks.rodent.imitation import default_config as imitation_default_config
from vnl_playground.tasks.reference_clips import ReferenceClips
from vnl_experiments.envs.absolute_imitation import (
    AbsoluteImitation,
    default_config as absolute_default_config,
)

from nnx_ppo.networks.adapter import PPOAdapter

from vnl_experiments.delays.forward_model import ForwardModel
# Network reconstruction + checkpoint loading are shared with eval_videos.py.
from vnl_experiments.delays.network_builders import build_network, load_network

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_LIST = Path(__file__).resolve().parent / "eval_runs.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_results"
DEFAULT_NEW_EVAL_H5 = (
    REPO_ROOT / "assets" / "art" / "2020_12_22_1" / "eval_clips_32x30s.h5"
)
ANALYSIS_GLOB = str(REPO_ROOT / "analysis" / "*" / "data.csv")
NEW_EVAL_CLIP_LENGTH = 1500  # 30 s @ 50 Hz; the new eval file's fixed clip length

_TERMINATION_REASONS = (
    "root_too_far",
    "root_too_rotated",
    "pose_error",
    "nan_termination",
)
_ERROR_KEYS = (
    "root_pos_distance",
    "root_angular_error",
    "joint_l2_error",
    "joint_vel_l2_error",
    "body_errors/total",
    "body_errors/end_eff_total",
)


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
# Env construction
# ---------------------------------------------------------------------------

def resolve_env_class(env_class_hint: str, env_params: dict, default: str):
    """Pick the env class: run-list hint → body_target_frame heuristic → default."""
    name = env_class_hint or ""
    if not name:
        if "body_target_frame" in env_params:
            name = "AbsoluteImitation"
        else:
            name = default
            warnings.warn(
                f"No env class for run and no body_target_frame in config; "
                f"defaulting to {default}."
            )
    if name == "Imitation":
        return Imitation, imitation_default_config
    if name == "AbsoluteImitation":
        return AbsoluteImitation, absolute_default_config
    raise ValueError(f"Unknown env class {name!r}")


def parse_env_config(env_params: dict, default_config_fn):
    """Reconstruct an imitation env config, keeping the run's clip_length.

    Mirrors ``eval_videos.parse_imitation_env_config`` but is env-class
    aware (``body_target_frame`` only applied when the config supports it) and
    does not force the video-eval clip_length/start_frame overrides.
    """
    cfg = default_config_fn()

    for field, conv in [
        ("ctrl_dt", float), ("sim_dt", float),
        ("naconmax", int), ("njmax", int),
        ("iterations", int), ("ls_iterations", int), ("noslip_iterations", int),
        ("mocap_hz", int), ("rescale_factor", float), ("clip_length", int),
    ]:
        if field in env_params:
            setattr(cfg, field, conv(env_params[field]))

    for field in ["solver", "mujoco_impl", "clip_set", "qvel_init", "body_target_frame"]:
        if field in env_params and field in cfg:
            setattr(cfg, field, env_params[field])

    if "torque_actuators" in env_params:
        v = env_params["torque_actuators"]
        cfg.torque_actuators = v if isinstance(v, bool) else v == "True"

    if "reward_terms" in env_params:
        for k, v in env_params["reward_terms"].items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    try:
                        cfg.reward_terms[k][sub_k] = float(sub_v)
                    except (KeyError, TypeError, ValueError):
                        pass
            else:
                try:
                    cfg.reward_terms[k]["weight"] = float(v)
                except (KeyError, TypeError, ValueError):
                    pass

    cfg.start_frame_range = [0, 1]

    # Always use the local XML paths (cluster paths in the checkpoint are stale).
    default = default_config_fn()
    cfg.reference_data_path = default.reference_data_path
    cfg.walker_xml_path = default.walker_xml_path
    cfg.arena_xml_path = default.arena_xml_path
    return cfg


# ---------------------------------------------------------------------------
# Parameter counts (hierarchical)
# ---------------------------------------------------------------------------

def _count_params(module) -> int:
    return int(sum(jax.tree.leaves(
        jax.tree.map(lambda x: x.size, nnx.state(module, nnx.Param))
    )))


def _param_tree(module) -> dict:
    """Nested dict of param counts mirroring the module's attribute hierarchy."""
    state = nnx.state(module, nnx.Param)
    tree: dict = {}
    for path, leaf in jtu.tree_leaves_with_path(state):
        keys = []
        for k in path:
            keys.append(getattr(k, "key", getattr(k, "idx", str(k))))
        node = tree
        for key in keys[:-1]:
            node = node.setdefault(str(key), {})
        node[str(keys[-1])] = int(leaf.size) if hasattr(leaf, "size") else int(
            np.asarray(leaf).size)
    return tree


def param_counts(nets, network_class: str) -> dict:
    """Semantic per-submodule counts + the full hierarchical tree."""
    out = {"total": _count_params(nets), "tree": _param_tree(nets)}
    adapter = next((l for l in nets.layers if isinstance(l, PPOAdapter)), None)
    if adapter is None:
        return out
    out["critic"] = _count_params(adapter.value)
    out["actor"] = _count_params(adapter.action)
    try:
        head = adapter.action.layers[0]          # Concat (EncDec) or Map (FM)
        out["encoder"] = _count_params(head.components["task_obs"])
        inner = adapter.action.layers[1].inner   # decoder (EncDec) or ForwardModel
        if isinstance(inner, ForwardModel):
            out["decoder"] = _count_params(inner.decoder)
            out["predictor"] = _count_params(inner.predictor)
        else:
            out["decoder"] = _count_params(inner)
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        warnings.warn(f"Could not extract semantic param groups: {e!r}")
    return out


# ---------------------------------------------------------------------------
# Deterministic per-clip rollout
# ---------------------------------------------------------------------------

def _rollout(env, networks, n_clips: int, n_steps: int, key):
    """One latched episode per clip (reset at frame 0). Returns raw per-clip arrays.

    ``n_steps`` is the number of *env steps* to scan. It must be large enough to
    traverse the whole reference clip: each env step advances the reference by
    ``ctrl_dt * mocap_hz`` frames (0.5 at the defaults), so covering a
    ``clip_length``-frame clip needs ``clip_length / (ctrl_dt * mocap_hz)`` steps
    — about twice ``clip_length``. All accumulators are masked by the pre-step
    ``done`` flag, so contributions after termination (and any post-termination
    NaNs) are dropped.
    """
    keys = jax.random.split(key, n_clips)
    clip_ids = jp.arange(n_clips)
    env_states = jax.vmap(
        lambda k, c: env.reset(k, clip_idx=c, start_frame=0)
    )(keys, clip_ids)
    env_states = env_states.replace(done=env_states.done.astype(float))
    net_states = networks.initialize_state(n_clips)

    probe = networks(net_states, env_states.obs)
    init_env_accum = jax.tree.map(jp.zeros_like, env_states.metrics)
    init_net_accum = jax.tree.map(jp.zeros_like, probe.metrics)

    def mask(done, x):
        return jp.where(
            done.reshape(done.shape + (1,) * (x.ndim - 1)), jp.zeros_like(x), x
        )

    def step(env, networks, carry):
        env_state, net_state, cuml_reward, lifespan, env_accum, net_accum = carry
        out = networks(net_state, env_state.obs)
        next_env_state = jax.vmap(env.step)(env_state, out.output.actions)
        next_env_state = next_env_state.replace(
            done=jp.logical_or(next_env_state.done, env_state.done).astype(float)
        )
        already_done = env_state.done
        step_reward = jax.tree.reduce(jp.add, next_env_state.reward)
        cuml_reward = cuml_reward + mask(already_done, step_reward)
        lifespan = lifespan + jp.where(next_env_state.done.astype(bool), 0.0, 1.0)
        env_accum = jax.tree.map(
            lambda c, m: c + mask(already_done, m), env_accum, next_env_state.metrics
        )
        net_accum = jax.tree.map(
            lambda c, m: c + mask(already_done, m), net_accum, out.metrics
        )
        return (next_env_state, out.next_state, cuml_reward, lifespan,
                env_accum, net_accum)

    import functools
    step_scan = nnx.scan(
        functools.partial(step, env),
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=nnx.Carry,
        length=n_steps,
    )
    init_carry = (
        env_states, net_states,
        jp.zeros(n_clips), jp.zeros(n_clips),
        init_env_accum, init_net_accum,
    )
    _, _, cuml_reward, lifespan, env_accum, net_accum = step_scan(networks, init_carry)
    return cuml_reward, lifespan, env_accum, net_accum


def _mean_std(x) -> dict:
    a = np.asarray(x, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def _flatten_net_metrics(net_accum: dict, denom: np.ndarray) -> dict:
    """Per-clip alive-mean of each net-metric leaf, reduced to a scalar."""
    out = {}
    for path, leaf in jtu.tree_leaves_with_path(net_accum):
        name = "/".join(
            str(getattr(k, "key", getattr(k, "idx", k))) for k in path
        )
        per_clip = np.asarray(leaf, dtype=float)
        # leaf shape [n_clips, ...]; average over clips after alive-normalisation
        per_clip = per_clip / denom.reshape(denom.shape + (1,) * (per_clip.ndim - 1))
        out[name] = float(per_clip.mean())
    return out


def eval_dataset(env, networks, n_clips: int, n_steps: int, ctrl_dt: float,
                 key, limit_clips: int | None) -> dict:
    n = n_clips if limit_clips is None else min(n_clips, limit_clips)
    eval_jit = nnx.jit(_rollout, static_argnums=(0, 2, 3))
    networks.eval()
    cuml_reward, lifespan, env_accum, net_accum = eval_jit(
        env, networks, n, n_steps, key
    )
    networks.train()

    cuml_reward = np.asarray(cuml_reward, dtype=float)
    lifespan = np.asarray(lifespan, dtype=float)
    env_accum = {k: np.asarray(v, dtype=float) for k, v in env_accum.items()}
    denom = np.maximum(lifespan, 1.0)

    # Termination flags are 0/1 per episode (a reason fires only at the end) →
    # masked SUM, then mean over clips = per-reason termination rate.
    term = {}
    for reason in _TERMINATION_REASONS:
        k = f"terminations/{reason}"
        term[reason] = float(env_accum[k].mean()) if k in env_accum else None
    any_key = "terminations/any"
    any_rate = float(env_accum[any_key].mean()) if any_key in env_accum else None
    term["any"] = any_rate
    term["survived"] = (1.0 - any_rate) if any_rate is not None else None

    # Errors and reward terms are per-step → masked sum ÷ lifespan, then over clips.
    errors = {}
    for k in _ERROR_KEYS:
        if k in env_accum:
            errors[k] = _mean_std(env_accum[k] / denom)
    reward_terms = {}
    for k, v in env_accum.items():
        if k.startswith("rewards/"):
            reward_terms[k[len("rewards/"):]] = _mean_std(v)  # episode totals

    return {
        "n_clips": int(n),
        "n_steps": int(n_steps),
        "episode_reward": _mean_std(cuml_reward),
        "reward_terms": reward_terms,
        "lifespan_steps": _mean_std(lifespan),
        "lifespan_s": _mean_std(lifespan * ctrl_dt),
        "termination_rate": term,
        "errors": errors,
        "net_metrics": _flatten_net_metrics(net_accum, denom),
    }


# ---------------------------------------------------------------------------
# Per-run orchestration
# ---------------------------------------------------------------------------

def _make_env(env_cls, cfg, clips):
    return env_cls(cfg, clips=clips)


def evaluate_run(wid: str, wn: str, env_class_hint: str, ckpt_dir: Path,
                 new_eval_h5: Path, seed: int, limit_clips: int | None) -> dict:
    with open(ckpt_dir / "config.json") as f:
        cfg_json = json.load(f)
    env_params = cfg_json["env_params"]
    net_params = cfg_json["net_params"]
    network_class = str(net_params.get("network_class", ""))

    env_cls, default_fn = resolve_env_class(env_class_hint, env_params,
                                            default="AbsoluteImitation")
    base_cfg = parse_env_config(env_params, default_fn)
    clip_length = int(base_cfg.clip_length)
    ctrl_dt = float(base_cfg.ctrl_dt)
    mocap_hz = int(base_cfg.mocap_hz)
    # Env steps needed to traverse a clip: clip_length is in *mocap frames* and
    # each env step advances the reference by ctrl_dt*mocap_hz frames.
    frames_per_step = ctrl_dt * mocap_hz

    def steps_for(clip_len_frames: int) -> int:
        return int(math.ceil(clip_len_frames / frames_per_step)) + 2

    # Datasets: train/test split of the run's reference data + the new eval set.
    all_clips = ReferenceClips(
        base_cfg.reference_data_path, clip_length, base_cfg.keep_clips_idx
    )
    train_clips, test_clips = all_clips.split()  # ratio=0.8, seed=0
    new_clips = ReferenceClips(str(new_eval_h5), NEW_EVAL_CLIP_LENGTH)

    datasets_spec = [
        ("train", train_clips, base_cfg, clip_length),
        ("old_eval", test_clips, base_cfg, clip_length),
    ]
    new_cfg = parse_env_config(env_params, default_fn)
    new_cfg.clip_length = NEW_EVAL_CLIP_LENGTH
    datasets_spec.append(("new_eval", new_clips, new_cfg, NEW_EVAL_CLIP_LENGTH))

    # Build the network + load weights once, using the train env to size obs.
    train_env = _make_env(env_cls, base_cfg, train_clips)
    loaded = load_network(ckpt_dir, net_params, train_env, seed)
    if loaded is None:
        return None
    nets, step = loaded

    result = {
        "wandb_id": wid,
        "wandb_name": wn,
        "checkpoint_dir": str(ckpt_dir),
        "step": step,
        "network_class": network_class,
        "env_class": env_cls.__name__,
        "delay_k": int(net_params.get("delay_k", 0)) if "delay_k" in net_params else None,
        "efference_length": (
            int(net_params["efference_length"]) if "efference_length" in net_params else None
        ),
        "fm_loss_weight": (
            float(net_params["fm_loss_weight"]) if "fm_loss_weight" in net_params else None
        ),
        "param_counts": param_counts(nets, network_class),
        "datasets": {},
    }

    key = jax.random.key(seed)
    for name, clips, cfg, clen in datasets_spec:
        env = train_env if clips is train_clips else _make_env(env_cls, cfg, clips)
        n_clips = int(clips.qpos.shape[0])
        n_steps = steps_for(clen)
        key, sub = jax.random.split(key)
        print(f"    [{name}] {min(n_clips, limit_clips or n_clips)} clips "
              f"x {n_steps} env steps ({clen} mocap frames)")
        result["datasets"][name] = eval_dataset(
            env, nets, n_clips, n_steps, ctrl_dt, sub, limit_clips
        )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--populate", action="store_true",
                   help="(Re)write the run list from analysis CSVs and exit.")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_dirs = [
        Path(d) if Path(d).is_absolute() else REPO_ROOT / d
        for d in args.checkpoint_dirs
    ]
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
