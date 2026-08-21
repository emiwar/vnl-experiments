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
from vnl_experiments.envs.config_io import resolve_local_xml_paths

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

    # Repair the asset paths: the cluster paths in the checkpoint are stale here. Only the
    # *directory* is stale though -- keep the run's own XML files, or an offline eval
    # silently simulates a different body than the run trained on. See config_io.
    default = default_config_fn()
    cfg.reference_data_path = default.reference_data_path
    resolve_local_xml_paths(cfg, env_params, default)
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
    """Semantic per-submodule counts + the full hierarchical tree.

    An architecture may supply its own grouping via ``Architecture.param_groups``
    (see ``network_builders``); otherwise the generic positional introspection
    below applies. The two original architectures deliberately have **no** hook,
    so they keep taking the generic path and their eval-record bytes -- and hence
    every stored artifact ``spec_id`` -- stay identical.
    """
    # Local import: network_builders' param-group hooks import back from here.
    from vnl_experiments.delays.network_builders import get_architecture

    out = {"total": _count_params(nets), "tree": _param_tree(nets)}
    adapter = next((l for l in nets.layers if isinstance(l, PPOAdapter)), None)
    if adapter is None:
        return out
    out["critic"] = _count_params(adapter.value)
    out["actor"] = _count_params(adapter.action)

    arch = get_architecture(network_class)
    if arch is not None and arch.param_groups is not None:
        try:
            out.update(arch.param_groups(nets))
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            warnings.warn(f"Could not extract semantic param groups: {e!r}")
        return out

    try:
        head = adapter.action.layers[0]          # Concat (EncDec) or Map (FM)
        # "task_obs" is absent when the intention stream is ablated away
        # (dec_use_intention=False) -- that run has no encoder to count, but its
        # decoder still has to be reported.
        if "task_obs" in head.components:
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

def _rollout(env, networks, n_clips: int, n_steps: int, key,
             action_noise: float | None = None):
    """One latched episode per clip (reset at frame 0). Returns raw per-clip arrays.

    ``n_steps`` is the number of *env steps* to scan. It must be large enough to
    traverse the whole reference clip: each env step advances the reference by
    ``ctrl_dt * mocap_hz`` frames (0.5 at the defaults), so covering a
    ``clip_length``-frame clip needs ``clip_length / (ctrl_dt * mocap_hz)`` steps
    — about twice ``clip_length``. All accumulators are masked by the pre-step
    ``done`` flag, so contributions after termination (and any post-termination
    NaNs) are dropped.

    ``action_noise`` is the std of a fixed Gaussian perturbation added to the
    action *after* the network has produced it (post-tanh, so in the same units
    as the actuator range, then clipped back into ``[-1, 1]``). It is deliberately
    injected outside the network: the efference copy therefore queues the
    *intended* action while the body executes the perturbed one, i.e. the noise is
    unobserved motor noise rather than a wider policy distribution. ``None`` (or
    ``0.0``) leaves the action untouched and consumes no randomness, so the
    default path is bit-identical to what this function did before the noise
    option existed.
    """
    # `fold_in` rather than a split, so the per-clip reset keys below are exactly
    # the ones a noise-free eval has always used.
    noise_key = jax.random.fold_in(key, 1)
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
        (env_state, net_state, noise_key, cuml_reward, lifespan,
         env_accum, net_accum) = carry
        out = networks(net_state, env_state.obs)
        actions = out.output.actions
        if action_noise:
            noise_key, sub = jax.random.split(noise_key)
            # One key per leaf: `actions` is a single array for the rodent nets but
            # a pytree for a multi-sampler bank, whose leaves must not share noise.
            leaves, treedef = jax.tree.flatten(actions)
            actions = jax.tree.unflatten(treedef, [
                jp.clip(a + action_noise * jax.random.normal(k, a.shape), -1.0, 1.0)
                for a, k in zip(leaves, jax.random.split(sub, len(leaves)))
            ])
        next_env_state = jax.vmap(env.step)(env_state, actions)
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
        return (next_env_state, out.next_state, noise_key, cuml_reward, lifespan,
                env_accum, net_accum)

    step_scan = nnx.scan(
        functools.partial(step, env),
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=nnx.Carry,
        length=n_steps,
    )
    init_carry = (
        env_states, net_states, noise_key,
        jp.zeros(n_clips), jp.zeros(n_clips),
        init_env_accum, init_net_accum,
    )
    _, _, _, cuml_reward, lifespan, env_accum, net_accum = step_scan(
        networks, init_carry
    )
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
                 key, limit_clips: int | None,
                 action_noise: float | None = None) -> dict:
    n = n_clips if limit_clips is None else min(n_clips, limit_clips)
    # `action_noise` is static so the `if action_noise` branch in `_rollout`
    # resolves at trace time.
    eval_jit = nnx.jit(_rollout, static_argnums=(0, 2, 3, 5))
    networks.eval()
    cuml_reward, lifespan, env_accum, net_accum = eval_jit(
        env, networks, n, n_steps, key, action_noise
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
                 net_params: dict, env_class: str,
                 action_noise: float | None = None) -> dict:
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
        # A property of the *measurement*, not of the run: the std of the fixed
        # Gaussian perturbation added to the executed action. None = the ordinary
        # noise-free eval.
        "action_noise": None if action_noise is None else float(action_noise),
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
    action_noise: float | None = None,
) -> dict:
    """Evaluate ``nets`` on every dataset and return the full result record.

    Returns ``{**metadata, "param_counts": ..., "datasets": {name: metrics}}``.

    ``action_noise`` (see :func:`_rollout`) perturbs the executed action; the
    default of ``None`` is the ordinary noise-free evaluation.
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
            env, nets, ds.n_clips, ds.n_steps, ds.ctrl_dt, sub, limit_clips,
            action_noise,
        )
    return result


def latest_checkpoint_step(ckpt_dir) -> int | None:
    """Step of the newest ``step_*`` checkpoint in ``ckpt_dir``, or None."""
    steps = []
    for d in Path(ckpt_dir).glob("step_*"):
        if d.is_dir():
            try:
                steps.append(int(d.name.removeprefix("step_")))
            except ValueError:
                continue
    return max(steps) if steps else None


def warn_if_step_not_checkpointed(ckpt_dir, step: int) -> None:
    """Warn when the newest checkpoint on disk is behind the evaluated weights."""
    latest = latest_checkpoint_step(ckpt_dir)
    if latest is None:
        warnings.warn(
            f"No checkpoint found in {ckpt_dir}; the eval record at step {step} "
            f"cannot be reproduced by eval_runs.py."
        )
    elif latest != step:
        warnings.warn(
            f"Evaluating the in-memory network at step {step}, but the newest "
            f"checkpoint is step {latest} ({step - latest} steps behind) — "
            f"total_steps is not a multiple of checkpoint_every_steps. A later "
            f"eval_runs.py pass will restore step {latest} and so may differ "
            f"slightly from this record."
        )


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

    This evaluates the **in-memory** network. Training does not write an extra
    checkpoint when it finishes, so if ``step`` is not on the
    ``checkpoint_every_steps`` grid the newest checkpoint on disk is slightly
    behind these weights and a later ``eval_runs.py`` pass would restore that
    older one instead. In practice total_steps is a multiple of the checkpoint
    interval and the two coincide; when they don't, we warn and carry on — the
    discrepancy is a fraction of a percent of training.

    **Never raises.** A 600M-step training run must not be lost to a failed
    evaluation, so problems are reported and swallowed. ``summary_fn`` (e.g.
    ``wandb.run.summary.update``) receives :func:`flat_summary` of the record.
    """
    warn_if_step_not_checkpointed(ckpt_dir, step)
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
