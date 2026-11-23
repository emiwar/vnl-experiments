"""Record unit activations of a walker-terrain policy on fixed courses, as HDF5.

The dm_control counterpart of :mod:`vnl_experiments.delays.record_activations`, which is
rodent-only (mocap clip "datasets", ``clip_idx`` resets, ``obs["state"]["proprioception"]``
targets). What carries over unchanged is the network side -- nnx-ppo's
``with_recording``/``extract_activations`` work on any ``StatefulModule`` tree, and the
activation flattening is imported from the rodent module -- and the file layout, which
stays readable by :class:`vnl_experiments.probes.linear_decoding.Recording`.

What replaces a "dataset" is a **course set** (:mod:`vnl_experiments.envs.walker_courses`):
episode ``i`` walks course ``i mod len(set)`` via ``WalkerTerrain.reset_on_course``, so
every policy walks the same ground whatever its speed. The episodes are vmapped in one
rollout of ``n_steps`` (the run's ``episode_length``) on the bare eval env -- no
``EpisodeWrapper``, no reward scaling -- so every episode runs its full length from t=0.

Row ``t`` of every per-step array describes the same moment: the state ``s_t`` the
network saw (``target_proprio`` = its observation, ``qpos``, ``qvel``, feet, ground,
block sensor), the activations it produced, the action ``a_t`` it took, and the reward
``r_t`` and reward terms of the transition that action caused.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import jax
import jax.numpy as jp
import numpy as np
from flax import nnx

from nnx_ppo.networks.recording import extract_activations, with_recording

from vnl_experiments.delays.record_activations import _flatten_activations
from vnl_experiments.envs import walker_courses
from vnl_experiments.envs.walker_terrain import PARK_X, WalkerTerrain

#: The walker's feet, whose ground height and position are recorded.
FEET = ("left_foot", "right_foot")
REWARD_TERMS = ("reward/standing", "reward/upright", "reward/stand", "reward/move")


# --------------------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------------------


def build_env(task: str, env_params: Mapping[str, Any], *, impl: str | None = None):
    """The run's eval env, built exactly as training builds it (servo included).

    ``impl`` overrides the physics backend; only for testing where Warp is unavailable --
    a recording made that way is not the run's physics.
    """
    from vnl_experiments.config.env_builder import build_env_config
    from vnl_experiments.envs import registry

    spec = registry.get(task)
    cfg = build_env_config(spec.default_config, dict(env_params))
    if impl is not None:
        cfg.impl = impl
    env = spec.build(cfg, for_eval=True)
    if not isinstance(env, WalkerTerrain):
        raise ValueError(f"task {task!r} builds {type(env).__name__}, not a WalkerTerrain; "
                         f"fixed-course recording only applies to the walker-terrain task")
    return env


def build_network(net_params: Mapping[str, Any], env, arrays: Mapping[str, np.ndarray],
                  seed: int = 0):
    """Rebuild the run's network and load ``arrays`` (flat weights) into it."""
    from vnl_experiments.artifacts.weights import restore_into
    from vnl_experiments.delays.network_builders import build_network as _build

    nets = _build(dict(net_params), env, nnx.Rngs(seed))
    if nets is None:
        raise ValueError(f"no architecture for network_class "
                         f"{net_params.get('network_class')!r}")
    restore_into(nets, arrays)
    return nets


def load_checkpoint_step(ckpt_dir: Path | str, step: int):
    """``(config, arrays, source_sha256)`` for one checkpoint step, without the pickle."""
    from vnl_experiments.artifacts import weights as W

    ckpt_dir = Path(ckpt_dir)
    available = W.checkpoint_steps(ckpt_dir)
    if int(step) not in available:
        raise FileNotFoundError(
            f"step {step} is not among {ckpt_dir.name}'s checkpoints "
            f"({len(available)} held, last {available[-1] if available else None})")
    networks_dir = W.step_dir(ckpt_dir, step) / "networks"
    config = json.loads((ckpt_dir / "config.json").read_text())
    return config, W.read_networks_dir(networks_dir), W.directory_sha256(networks_dir)


# --------------------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------------------


def record_rollout(env: WalkerTerrain, networks, course_x, course_top, n_steps: int, key):
    """One vmapped rollout, episode ``i`` on course ``i``; returns ``[T, N, ...]`` arrays."""
    rec = with_recording(networks)
    rec.eval()  # deterministic actions

    n = int(course_x.shape[0])
    env_states = jax.vmap(env.reset_on_course)(jax.random.split(key, n), course_x, course_top)
    env_states = env_states.replace(done=env_states.done.astype(float))
    net_states = rec.initialize_state(n)
    foot_ids = jp.array([env.mj_model.body(f).id for f in FEET])
    torso_id = env._torso_id

    def describe(data):
        """Per-episode facts about one state (vmapped over episodes by the caller)."""
        feet = data.xpos[foot_ids][:, [0, 2]]                              # [2, (x, z)]
        ground = jax.vmap(lambda x: env._ground_height_under(data, x))(feet[:, 0])
        return dict(
            qpos=data.qpos, qvel=data.qvel, foot_pos=feet, foot_ground_height=ground,
            block_sensor=env._terrain_obs(data),
            torso_x=data.xpos[torso_id, 0],
            n_parked=jp.sum(data.mocap_pos[:, 0] <= PARK_X / 2),
        )

    def step(env, networks, carry):
        env_state, net_state = carry
        out = networks(net_state, env_state.obs)
        actions = out.output.actions
        nxt = jax.vmap(env.step)(env_state, actions)
        nxt = nxt.replace(done=jp.logical_or(nxt.done, env_state.done).astype(float))
        acts = jax.tree.map(
            lambda x: x.astype(jp.float16) if jp.issubdtype(x.dtype, jp.floating) else x,
            extract_activations(out.metrics),
        )
        ys = dict(
            activations=acts,
            obs=env_state.obs,
            actions=actions,
            reward=nxt.reward,
            reward_terms=jp.stack([nxt.metrics[k] for k in REWARD_TERMS], axis=-1),
            done=env_state.done,
            **jax.vmap(describe)(env_state.data),
        )
        return (nxt, out.next_state), ys

    step_scan = nnx.scan(
        functools.partial(step, env),
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=(nnx.Carry, 0),
        length=n_steps,
    )
    _, ys = step_scan(rec, (env_states, net_states))
    return ys


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


def record(out_path: Path | str, *, task: str, config: Mapping[str, Any],
           arrays: Mapping[str, np.ndarray], step: int, course_set: str,
           n_episodes: int = 3, seed: int = 0, n_steps: int | None = None,
           run_name: str = "", impl: str | None = None) -> dict[str, Any]:
    """Record one (network, course set) and write the HDF5; return summary facts.

    Raises if any episode ran off the end of its course: blocks would have parked, and
    the episode would have finished on terrain the course does not describe.
    """
    env_params, net_params = config["env_params"], config["net_params"]
    env = build_env(task, env_params, impl=impl)
    courses = walker_courses.courses_for_episodes(course_set, n_episodes)
    for c in courses:
        env.check_course(c.x, c.top)
    xs, tops = walker_courses.stack_courses(courses)
    nets = build_network(net_params, env, arrays, seed)
    n_steps = int(n_steps or env_params.get("episode_length", 1000))

    rollout = nnx.jit(record_rollout, static_argnums=(0, 4))
    ys = rollout(env, nets, jp.asarray(xs), jp.asarray(tops), n_steps, jax.random.key(seed))
    ys = jax.tree.map(np.asarray, ys)

    parked = ys["n_parked"].max(axis=0)
    if parked.any():
        bad = [courses[i].name for i in np.flatnonzero(parked)]
        raise RuntimeError(f"episode(s) on {bad} ran off the end of the course "
                           f"(max torso x {ys['torso_x'].max(axis=0).round(1).tolist()} m); "
                           f"regenerate longer courses")

    layers = _flatten_activations(ys["activations"])
    max_torso_x = ys["torso_x"].max(axis=0)
    obs = ys["obs"].astype(np.float32)
    dones = ys["done"].astype(bool)

    out_path = Path(out_path)
    tmp = out_path.with_name(f".tmp-{out_path.name}")
    gz = dict(compression="gzip", compression_opts=4)
    with h5py.File(tmp, "w") as f:
        f.attrs.update(dict(
            # The rodent recording's attrs, for probes.linear_decoding.Recording.
            run_name=run_name, condition="", network_class=str(net_params["network_class"]),
            dataset=course_set, delay_k=int(net_params.get("delay_k", 0) or 0),
            efference_length=int(net_params.get("efference_length", 0) or 0),
            step=int(step), proprio_size=int(obs.shape[-1]),
            ctrl_dt=float(env_params.get("ctrl_dt", env.dt)), n_clips=int(obs.shape[1]),
            n_steps=int(obs.shape[0]),
            # dm_control-specific.
            task=task, course_names=[c.name for c in courses],
            course_sha256=walker_courses.set_sha256(course_set),
            terrain_observed=bool(env._terrain.observed),
            impl=str(env_params.get("impl") if impl is None else impl),
            foot_names=list(FEET), reward_terms=list(REWARD_TERMS),
            foot_pos_axes=["x", "z"],
            block_sensor_layout="[dist0, top0, dist1, top1] of the next obs_blocks blocks",
        ))
        f.create_dataset("target_proprio", data=obs, **gz)
        f.create_dataset("dones", data=dones, compression="gzip")
        for name in ("qpos", "qvel", "actions", "reward", "reward_terms", "foot_pos",
                     "foot_ground_height", "block_sensor"):
            f.create_dataset(name, data=ys[name].astype(np.float32), **gz)
        f.create_dataset("max_torso_x", data=max_torso_x.astype(np.float32))
        f.create_dataset("course_x", data=xs)
        f.create_dataset("course_top", data=tops)
        g = f.create_group("activations")
        for lname, arr in layers.items():
            g.create_dataset(lname, data=arr, **gz)
    tmp.replace(out_path)

    return {
        "checkpoint_step": int(step),
        "task": task,
        "network_class": str(net_params["network_class"]),
        "course_set": course_set,
        "course_names": [c.name for c in courses],
        "n_episodes": int(obs.shape[1]),
        "n_steps": int(obs.shape[0]),
        "n_layers": len(layers),
        "n_units": int(sum(a.shape[-1] for a in layers.values())),
        "terrain_observed": bool(env._terrain.observed),
        "max_torso_x": [round(float(v), 2) for v in max_torso_x],
        # Unscaled (the eval env has no RewardScalingWrapper): per-episode return.
        "mean_return": float(ys["reward"].sum(axis=0).mean()),
        "any_done": bool(dones.any()),
    }
