"""Shared evaluation core for the delays / forward-model rodent experiments.

One implementation, three callers:

* ``train_rodent_delays.py`` / ``train_rodent_forward_model.py`` call
  :func:`evaluate_networks` at the end of training with the in-memory networks
  and envs, and write the record to ``{ckpt_dir}/eval.json``;
* ``eval_runs.py`` rebuilds a network from a checkpoint and calls the same
  function, writing ``eval_results/{wandb_id}.json``;
* ``analysis/implicit-forward-model/record_activations.py`` reuses the env /
  config / dataset construction (:func:`build_datasets`) for its own
  activation-stacking rollout.

Keeping the measurement in one place is what makes results from the two
producers comparable — if the inline and batch paths ever diverge, JSONs of
mixed provenance become silently incomparable.

The evaluation protocol: for each dataset, one **latched** episode per clip
(reset at frame 0, deterministic policy), scanned for enough env steps to
traverse the whole reference clip, with every accumulator masked by the pre-step
``done`` flag so post-termination contributions (and NaNs) are dropped.
"""

import copy
import functools
import json
import math
import traceback
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jp
import numpy as np
from flax import nnx
from jax import tree_util as jtu

from vnl_playground.tasks.rodent.imitation import Imitation
from vnl_playground.tasks.rodent.imitation import default_config as imitation_default_config
from vnl_playground.tasks.reference_clips import ReferenceClips

from nnx_ppo.networks.adapter import PPOAdapter

from vnl_experiments.delays.forward_model import ForwardModel
from vnl_experiments.envs.absolute_imitation import (
    AbsoluteImitation,
    default_config as absolute_default_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEW_EVAL_H5 = (
    REPO_ROOT / "assets" / "art" / "2020_12_22_1" / "eval_clips_32x30s.h5"
)
NEW_EVAL_CLIP_LENGTH = 1500  # 30 s @ 50 Hz; the new eval file's fixed clip length

DATASET_NAMES = ("train", "old_eval", "new_eval")

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


def prepare_eval_config(cfg):
    """Copy an env config and apply the eval-only overrides.

    Always start episodes at frame 0 of the clip, so every clip is traversed
    from its beginning and results are deterministic. The copy matters for the
    inline (end-of-training) caller, whose ``env_config`` is a live object still
    referenced by the training env.
    """
    cfg = copy.deepcopy(cfg)
    cfg.start_frame_range = [0, 1]
    return cfg


def parse_env_config(env_params: dict, default_config_fn):
    """Reconstruct an imitation env config from a checkpoint's ``config.json``.

    Keeps the run's ``clip_length``. Env-class aware (``body_target_frame`` only
    applied when the config supports it) and does not force the video-eval
    clip_length/start_frame overrides.
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


def _make_env(env_cls, cfg, clips):
    return env_cls(cfg, clips=clips)


def split_clips(cfg):
    """The run's own 80/20 clip split (``split()`` defaults: ratio 0.8, seed 0)."""
    all_clips = ReferenceClips(
        cfg.reference_data_path, int(cfg.clip_length), cfg.keep_clips_idx
    )
    return all_clips.split()


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

class EvalDataset(NamedTuple):
    """One evaluation dataset: how to build its env, its clips, and the horizon.

    ``make_env`` is a thunk rather than a constructed env on purpose: each env
    holds a full MuJoCo Warp model plus its reference-clip arrays on device, so
    materialising all three at once is enough to OOM an 8 GB card. Callers build
    each env as they reach it and let the previous one go.
    """

    name: str
    make_env: Any  # () -> env
    clips: Any
    n_clips: int
    n_steps: int
    clip_length: int
    ctrl_dt: float


def build_datasets(
    env_cls,
    env_config,
    *,
    train_clips=None,
    test_clips=None,
    train_env=None,
    new_eval_h5: Path = DEFAULT_NEW_EVAL_H5,
    names: tuple[str, ...] = DATASET_NAMES,
    max_steps: int | None = None,
) -> list[EvalDataset]:
    """Build the train / old_eval / new_eval datasets for one run.

    ``train``/``old_eval`` are the 80/20 split of the run's own reference data
    (``split()`` defaults: ratio 0.8, seed 0) at the run's ``clip_length``;
    ``new_eval`` is the fixed 32-clip 30 s eval set at ``NEW_EVAL_CLIP_LENGTH``.

    ``train_clips``/``test_clips``/``train_env`` let a caller hand over objects
    it has already built (the training scripts have all three); when omitted they
    are constructed here. ``max_steps`` caps the rollout horizon.
    """
    base_cfg = prepare_eval_config(env_config)
    clip_length = int(base_cfg.clip_length)
    ctrl_dt = float(base_cfg.ctrl_dt)
    # Env steps needed to traverse a clip: clip_length is in *mocap frames* and
    # each env step advances the reference by ctrl_dt*mocap_hz frames.
    frames_per_step = ctrl_dt * int(base_cfg.mocap_hz)

    def steps_for(clip_len_frames: int) -> int:
        n = int(math.ceil(clip_len_frames / frames_per_step)) + 2
        return n if max_steps is None else min(n, max_steps)

    needs_train = train_clips is None and "train" in names
    needs_test = test_clips is None and "old_eval" in names
    if needs_train or needs_test:
        split_train, split_test = split_clips(base_cfg)
        train_clips = split_train if needs_train else train_clips
        test_clips = split_test if needs_test else test_clips

    out: list[EvalDataset] = []
    for name in names:
        if name == "train":
            clips, clen = train_clips, clip_length
            make_env = (
                (lambda e=train_env: e) if train_env is not None
                else functools.partial(_make_env, env_cls, base_cfg, clips)
            )
        elif name == "old_eval":
            clips, clen = test_clips, clip_length
            make_env = functools.partial(_make_env, env_cls, base_cfg, clips)
        elif name == "new_eval":
            clen = NEW_EVAL_CLIP_LENGTH
            new_cfg = prepare_eval_config(env_config)
            new_cfg.clip_length = clen
            clips = ReferenceClips(str(new_eval_h5), clen)
            make_env = functools.partial(_make_env, env_cls, new_cfg, clips)
        else:
            raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASET_NAMES}")
        out.append(EvalDataset(
            name=name, make_env=make_env, clips=clips,
            n_clips=int(clips.qpos.shape[0]), n_steps=steps_for(clen),
            clip_length=clen, ctrl_dt=ctrl_dt,
        ))
    return out


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
# Result records
# ---------------------------------------------------------------------------

def run_metadata(wandb_id: str, wandb_name: str, checkpoint_dir, step: int,
                 net_params: dict, env_class: str) -> dict:
    """The identity/hyperparameter half of a result record.

    Both producers build this the same way, so the JSON schema stays identical
    whether the record came from an inline (end-of-training) eval or from the
    batch re-evaluation in ``eval_runs.py``.
    """
    return {
        "wandb_id": wandb_id,
        "wandb_name": wandb_name,
        "checkpoint_dir": str(checkpoint_dir),
        "step": step,
        "network_class": str(net_params.get("network_class", "")),
        "env_class": env_class,
        "delay_k": (
            int(net_params["delay_k"]) if "delay_k" in net_params else None
        ),
        "efference_length": (
            int(net_params["efference_length"])
            if "efference_length" in net_params else None
        ),
        "fm_loss_weight": (
            float(net_params["fm_loss_weight"])
            if "fm_loss_weight" in net_params else None
        ),
    }


def evaluate_networks(
    nets,
    env_cls,
    env_config,
    *,
    metadata: dict,
    train_clips=None,
    test_clips=None,
    train_env=None,
    new_eval_h5: Path = DEFAULT_NEW_EVAL_H5,
    names: tuple[str, ...] = DATASET_NAMES,
    seed: int = 0,
    limit_clips: int | None = None,
) -> dict:
    """Evaluate ``nets`` on every dataset and return the full result record.

    Returns ``{**metadata, "param_counts": ..., "datasets": {name: metrics}}``.
    """
    datasets = build_datasets(
        env_cls, env_config, train_clips=train_clips, test_clips=test_clips,
        train_env=train_env, new_eval_h5=new_eval_h5, names=names,
    )

    result = {
        **metadata,
        "param_counts": param_counts(nets, metadata.get("network_class", "")),
        "datasets": {},
    }

    key = jax.random.key(seed)
    for ds in datasets:
        key, sub = jax.random.split(key)
        print(f"    [{ds.name}] {min(ds.n_clips, limit_clips or ds.n_clips)} clips "
              f"x {ds.n_steps} env steps ({ds.clip_length} mocap frames)")
        env = None  # release the previous dataset's env before building the next
        env = ds.make_env()
        result["datasets"][ds.name] = eval_dataset(
            env, nets, ds.n_clips, ds.n_steps, ds.ctrl_dt, sub, limit_clips
        )
    return result


def run_final_eval(
    nets,
    env_cls,
    env_config,
    *,
    ckpt_dir,
    wandb_id: str,
    wandb_name: str,
    step: int,
    net_params: dict,
    train_env=None,
    train_clips=None,
    test_clips=None,
    new_eval_h5: Path = DEFAULT_NEW_EVAL_H5,
    seed: int = 0,
    limit_clips: int | None = None,
    summary_fn=None,
) -> dict | None:
    """End-of-training eval: evaluate, write ``{ckpt_dir}/eval.json``, log a summary.

    Written next to the checkpoint so the record rides along with the
    cluster→laptop rsync; ``eval_runs.py --collect`` gathers these into the flat,
    wandb_id-keyed directory the analyses read.

    **Never raises.** A 600M-step training run must not be lost to a failed
    evaluation, so problems are reported and swallowed. ``summary_fn`` (e.g.
    ``wandb.run.summary.update``) receives :func:`flat_summary` of the record.
    """
    try:
        record = evaluate_networks(
            nets, env_cls, env_config,
            metadata=run_metadata(wandb_id, wandb_name, ckpt_dir, step,
                                  net_params, env_cls.__name__),
            train_env=train_env, train_clips=train_clips, test_clips=test_clips,
            new_eval_h5=new_eval_h5, seed=seed, limit_clips=limit_clips,
        )
        out_path = Path(ckpt_dir) / "eval.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Final eval written to {out_path}")

        if summary_fn is not None:
            summary_fn(flat_summary(record))
        return record
    except Exception as e:  # noqa: BLE001 — never lose a finished training run
        traceback.print_exc()
        warnings.warn(f"Final eval failed ({e!r}); training results are unaffected.")
        return None


def flat_summary(record: dict, prefix: str = "final_eval") -> dict[str, float]:
    """Flatten a result record to scalar ``{prefix}/...`` keys for wandb.

    Only the headline numbers — enough to compare runs from the wandb API
    without the local JSONs. The full record stays in ``eval.json``.
    """
    out: dict[str, float] = {}
    counts = record.get("param_counts", {})
    for key in ("total", "actor", "critic", "encoder", "decoder", "predictor"):
        if key in counts:
            out[f"{prefix}/params/{key}"] = float(counts[key])

    for ds_name, ds in record.get("datasets", {}).items():
        base = f"{prefix}/{ds_name}"
        for stat in ("episode_reward", "lifespan_steps", "lifespan_s"):
            if stat in ds:
                out[f"{base}/{stat}/mean"] = ds[stat]["mean"]
                out[f"{base}/{stat}/std"] = ds[stat]["std"]
        for reason, v in ds.get("termination_rate", {}).items():
            if v is not None:
                out[f"{base}/termination_rate/{reason}"] = float(v)
        for name, v in ds.get("errors", {}).items():
            out[f"{base}/errors/{name}"] = v["mean"]
        for name, v in ds.get("reward_terms", {}).items():
            out[f"{base}/reward_terms/{name}"] = v["mean"]
        for name, v in ds.get("net_metrics", {}).items():
            out[f"{base}/net_metrics/{name}"] = float(v)
    return out
