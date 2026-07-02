#!/usr/bin/env python3
"""Render evaluation-set videos for checkpoints trained with the delays study.

For each checkpoint in ``CHECKPOINTS``:
  - Resolves the env class from the saved config (``AbsoluteImitation`` when
    ``body_target_frame`` is present, else base ``Imitation``) and rebuilds the
    network (enc-dec *or* forward-model) from ``config.json``.
  - Runs one deterministic rollout on each of the first ``N_EVAL_CLIPS`` clips of
    the curated eval dataset (``eval_clips_32x30s.h5``, 30 s / 1500-frame clips).
  - Writes to ``eval_videos/{checkpoint_name}/``:
      stats.json   — step, per-clip + overall reward / resets / time-alive / errors
      rollout.h5   — rollout_qpos and reference_qpos (all clips, at video frame rate)
      rollout.mp4  — 1920x1200 Camera-4-matched render with a logistic fade between
                     clips (streamed frame-by-frame; peak RAM ≈ one frame)

Two behaviour toggles are compile-time constants:
  * ``AUTO_RESET_ON_TERMINATION`` — when True the rollout snaps back to the
    reference timeline on termination (video stays reference-locked); when False
    the simulation free-runs past a termination so you can watch the policy fail.
  * ``RENDER_GHOST`` — reserved; the streaming render path shows only the policy
    rodent (no ghost target).

Usage:
    .venv/bin/python -m vnl_experiments.delays.eval_videos
"""

import gc
import json
import math
import os
import types as _types
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import cv2
import h5py
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import scipy.io as spio
from flax import nnx
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from etils import epath

from vnl_playground.tasks.rodent.imitation import (
    Imitation,
    default_config as imitation_default_config,
)
from vnl_experiments.envs.absolute_imitation import (
    AbsoluteImitation,
    default_config as absolute_default_config,
)

from nnx_ppo.algorithms.rollout import SlimData, SlimState

from vnl_experiments.delays.network_builders import build_network, load_network

# ---------------------------------------------------------------------------
# Hard-coded settings — edit before running
# ---------------------------------------------------------------------------

# The loadable June-2026 checkpoints under downloaded_checkpoints/ (both network
# types). Omitted:
#   * RodentEncDec_delay5_eff5-20260622-063455 (no step_* directory).
#   * RodentEncDec_delay0_eff0-20260601-105420 / -20260603-083745 — the two oldest
#     runs use an older network layout (imitation_target naming / different
#     normalizer) that build_delay_network no longer reproduces; not supported.
CHECKPOINTS = [
    "downloaded_checkpoints/RodentEncDec_delay0_eff0-20260629-090548",
    "downloaded_checkpoints/RodentEncDec_delay5_eff5-20260623-085807",
    "downloaded_checkpoints/RodentEncDec_delay10_eff10-20260611-120848",
    "downloaded_checkpoints/RodentEncDec_delay10_eff10-20260629-094431",
    "downloaded_checkpoints/RodentEncDec_delay20_eff20-20260626-084448",
    "downloaded_checkpoints/RodentForwardModel_delay0_eff0-20260619-032834",
    "downloaded_checkpoints/RodentForwardModel_delay5_eff5-20260619-033243",
    "downloaded_checkpoints/RodentForwardModel_delay10_eff10-20260619-033822",
    "downloaded_checkpoints/RodentForwardModel_delay20_eff20-20260619-034701",
    "downloaded_checkpoints/RodentForwardModel_delay0_eff0_nodetach-20260630-083013",
    "downloaded_checkpoints/RodentForwardModel_delay5_eff5_nodetach-20260630-083050",
    "downloaded_checkpoints/RodentForwardModel_delay10_eff10_nodetach-20260630-083050",
    "downloaded_checkpoints/RodentForwardModel_delay20_eff20_nodetach-20260630-083014",
]

# When True, reset to the scheduled reference frame on termination (timeline
# stays locked to the reference video). When False, keep simulating past the
# termination so the policy's failure mode is visible. Free-run outputs get a
# "_noreset" filename suffix so they sit alongside the reset outputs.
AUTO_RESET_ON_TERMINATION = False
OUTPUT_SUFFIX = "" if AUTO_RESET_ON_TERMINATION else "_noreset"

# Reserved: the streaming render path renders only the policy rodent.
RENDER_GHOST = False

N_EVAL_CLIPS = 4          # render the first N clips of the eval dataset
FADE_FRAMES = 25          # ~0.5 s logistic fade written between clips

# Reference session directory (only used for the Camera-4 calibration .mat).
REFERENCE_DIR = "assets/art/2020_12_22_1"
# The curated eval dataset (32 x 30 s clips; fixed 1500-frame clip length).
NEW_EVAL_H5 = "assets/art/2020_12_22_1/eval_clips_32x30s.h5"

OUTPUT_DIR = "eval_videos"
VIDEO_HEIGHT = 1200
VIDEO_WIDTH = 1920
FPS = 50
CAMERA_NAME = "CalibCamera"
SEED = 0

# ---------------------------------------------------------------------------
# Camera calibration (Camera 4 from the DANNCE rig)
# ---------------------------------------------------------------------------

_DANNCE_IMAGE_HEIGHT = 1200


def load_camera_calibration(mat_path: str):
    mat = spio.loadmat(mat_path, squeeze_me=True)
    return _types.SimpleNamespace(
        K=mat["K"], r=mat["r"], t=mat["t"],
        RDistort=mat["RDistort"], TDistort=mat["TDistort"],
    )


def convert_camera(cam, name: str = CAMERA_NAME) -> dict:
    rot = R.from_matrix(cam.r.T)
    eul = rot.as_euler("zyx")
    eul[2] += np.pi
    quat = R.from_euler("zyx", eul).as_quat()
    quat = quat[np.array([3, 0, 1, 2])]
    quat[0] *= -1
    fovy = 2 * np.arctan(_DANNCE_IMAGE_HEIGHT / (2 * cam.K[1, 1])) / (2 * np.pi) * 360
    pos = (-cam.t.reshape(1, 3) @ cam.r.T / 1000).squeeze()
    return {"name": name, "pos": pos, "quat": quat, "fovy": fovy}


def build_render_model(env, camera_kwargs: dict) -> mujoco.MjModel:
    """Compile a render-only MjModel with the calibrated camera added.

    Works on a *copy* of ``env._spec`` so the env's own spec/mjx model signatures
    stay in sync (``data.bind`` compares them). The camera is render-only and does
    not affect physics.
    """
    spec = env._spec.copy()
    cam = spec.worldbody.add_camera()
    cam.name = camera_kwargs["name"]
    cam.pos = np.array(camera_kwargs["pos"])
    cam.quat = np.array(camera_kwargs["quat"])
    cam.fovy = float(camera_kwargs["fovy"])
    mj_model = spec.compile()
    mj_model.vis.global_.offwidth = VIDEO_WIDTH
    mj_model.vis.global_.offheight = VIDEO_HEIGHT
    return mj_model


# ---------------------------------------------------------------------------
# Environment config parsing
# ---------------------------------------------------------------------------

def resolve_env_class(env_params: dict):
    """Pick the env class from the saved config.

    ``AbsoluteImitation`` writes a ``body_target_frame`` field; base ``Imitation``
    does not. Returns ``(EnvClass, default_config_fn)``. Both share obs
    keys/shapes, so the network builders are unaffected by the choice.
    """
    if "body_target_frame" in env_params:
        return AbsoluteImitation, absolute_default_config
    return Imitation, imitation_default_config


def parse_imitation_env_config(env_params: dict, reference_h5: str,
                               clip_length: int, default_config_fn):
    """Reconstruct an (Absolute)Imitation config from the saved env_params dict."""
    cfg = default_config_fn()

    for field, conv in [
        ("ctrl_dt", float),
        ("sim_dt", float),
        ("naconmax", int),
        ("njmax", int),
        ("iterations", int),
        ("ls_iterations", int),
        ("noslip_iterations", int),
        ("mocap_hz", int),
        ("rescale_factor", float),
    ]:
        if field in env_params:
            setattr(cfg, field, conv(env_params[field]))

    for field in ["solver", "mujoco_impl", "clip_set", "qvel_init",
                  "body_target_frame"]:
        if field in env_params:
            setattr(cfg, field, env_params[field])

    for field in ["torque_actuators"]:
        if field in env_params:
            val = env_params[field]
            setattr(cfg, field, val if isinstance(val, bool) else val == "True")

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

    # Eval overrides: read the curated eval dataset, one clip per 1500 frames.
    cfg.reference_data_path = epath.Path(reference_h5)
    cfg.clip_length = clip_length
    cfg.start_frame_range = [0, 1]
    # Reduce pre-allocation sizes — a single env only needs a fraction.
    cfg.naconmax = min(int(cfg.naconmax), 2048)
    cfg.njmax = min(int(cfg.njmax), 256)
    # Reset clip_set to "all" so the loaded 32-clip file isn't filtered by a
    # (possibly stale) trained clip_set; we index clips explicitly anyway.
    with cfg.ignore_type():
        cfg.clip_set = "all"

    # Keep local XML paths (training cluster paths are invalid here).
    default = default_config_fn()
    cfg.walker_xml_path = default.walker_xml_path
    cfg.arena_xml_path = default.arena_xml_path

    return cfg


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _slim(env_state) -> SlimState:
    return SlimState(
        data=SlimData(
            qpos=env_state.data.qpos,
            qvel=env_state.data.qvel,
            time=env_state.data.time,
            mocap_pos=env_state.data.mocap_pos,
            mocap_quat=env_state.data.mocap_quat,
            xfrc_applied=env_state.data.xfrc_applied,
        ),
        done=env_state.done,
        info=env_state.info,
        metrics=env_state.metrics,
    )


def rollout_collect_stats(env, networks, n_steps: int, key, clip_idx, auto_reset: bool):
    """Deterministic single-clip rollout collecting per-step error metrics.

    ``clip_idx`` (a traced array) selects the reference clip; the rollout starts
    at frame 0. When ``auto_reset`` is True and a termination fires, the env is
    reset to the *deterministic global schedule* ``floor(step * ctrl_dt *
    mocap_hz)`` within the same clip, so the rendered rollout stays locked to the
    reference timeline (see the git history of this file for why the env's own
    ``current_frame`` is unsuitable). When ``auto_reset`` is False the simulation
    is never reset — it free-runs past the termination.

    Returns:
        stacked_states  : SlimState with leading dim n_steps (pre-step states)
        reset_mask      : bool (n_steps,), True at reset steps (all False if not auto_reset)
        per_step_reward : float (n_steps,)
        root_errors     : float (n_steps,)
        end_eff_errors  : float (n_steps,)
    """
    key, key2 = jax.random.split(key)
    env_state = env.reset(key, clip_idx=clip_idx, start_frame=jp.array(0))
    net_state = networks.initialize_state(1)
    net_state = jax.tree.map(lambda x: x[0], net_state)

    # Reference frames advanced per control step (0.5 for ctrl_dt=0.01, 50 Hz).
    frames_per_step = float(env._config.ctrl_dt * env._config.mocap_hz)

    def step_fn(networks, carry):
        env_state, net_state, rng, step = carry

        obs_batched = jax.tree.map(lambda x: x[None], env_state.obs)
        net_state_batched = jax.tree.map(lambda x: x[None], net_state)
        result = networks(net_state_batched, obs_batched)
        next_net_state = jax.tree.map(lambda x: x[0], result.next_state)
        action = jax.tree.map(lambda x: x[0], result.output.actions)
        next_env_state = env.step(env_state, action)

        reset_happened = next_env_state.done.astype(bool)

        # Capture reward/metrics from the post-step, pre-reset state.
        step_reward = sum(jax.tree.leaves(next_env_state.reward))
        root_err = next_env_state.metrics["root_pos_distance"]
        end_eff_err = next_env_state.metrics["body_errors/end_eff_total"]

        if auto_reset:
            scheduled_frame = jp.floor(
                (step + 1).astype(jp.float32) * frames_per_step
            ).astype(jp.int32)

            def do_reset(rng):
                return env.reset(rng, clip_idx=clip_idx, start_frame=scheduled_frame)

            next_env_state = jax.lax.cond(
                reset_happened, do_reset, lambda rng: next_env_state, rng
            )
            next_net_state = jax.lax.cond(
                reset_happened, networks.reset_state, lambda x: x, next_net_state
            )
        # else: free-run — leave next_env_state / next_net_state as stepped.

        (new_rng,) = jax.random.split(rng, 1)

        return (next_env_state, next_net_state, new_rng, step + 1), (
            _slim(env_state),
            reset_happened,
            step_reward,
            root_err,
            end_eff_err,
        )

    scan_fn = nnx.scan(
        step_fn,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=(nnx.Carry, 0),
        length=n_steps,
    )

    init_carry = (env_state, net_state, key2, jp.array(0, jp.int32))
    _, (stacked_states, reset_mask, per_step_reward, root_errors, end_eff_errors) = (
        scan_fn(networks, init_carry)
    )
    return stacked_states, reset_mask, per_step_reward, root_errors, end_eff_errors


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(
    per_step_reward: np.ndarray,
    root_errors: np.ndarray,
    end_eff_errors: np.ndarray,
    done_mask: np.ndarray,
    ctrl_dt: float,
    n_steps: int,
    auto_reset: bool,
) -> dict:
    """Per-clip stats. ``done_mask`` is the per-step termination flag.

    With ``auto_reset`` the mask marks resets, so it reports the reset count and
    the average time alive per episode over the full horizon. Without auto_reset
    the mask stays true after the first termination (free-run), so instead it
    reports whether/when the clip terminated and the tracking error *before* that
    first termination (post-failure flailing would otherwise dominate).
    """
    done = np.asarray(done_mask).astype(bool)
    stats = {"total_reward": float(np.sum(per_step_reward))}
    if auto_reset:
        n_resets = int(np.sum(done))
        stats["n_resets"] = n_resets
        stats["avg_time_alive_s"] = (n_steps / (n_resets + 1)) * ctrl_dt
        r_err, e_err = root_errors, end_eff_errors
    else:
        terminated = bool(done.any())
        first = int(np.argmax(done)) if terminated else n_steps
        stats["terminated"] = int(terminated)
        stats["time_alive_s"] = first * ctrl_dt
        window = slice(0, max(first, 1))  # tracking quality before first failure
        r_err, e_err = root_errors[window], end_eff_errors[window]
    stats["root_error_mean"] = float(np.mean(r_err))
    stats["root_error_std"] = float(np.std(r_err))
    stats["end_eff_error_mean"] = float(np.mean(e_err))
    stats["end_eff_error_std"] = float(np.std(e_err))
    return stats


def aggregate_stats(per_clip: list, auto_reset: bool) -> dict:
    """Combine per-clip stats into an overall summary."""
    err_keys = ["root_error_mean", "root_error_std",
                "end_eff_error_mean", "end_eff_error_std"]
    out = {"total_reward": float(sum(s["total_reward"] for s in per_clip))}
    for k in err_keys:
        out[k] = float(np.mean([s[k] for s in per_clip]))
    if auto_reset:
        out["n_resets"] = int(sum(s["n_resets"] for s in per_clip))
        out["avg_time_alive_s"] = float(np.mean([s["avg_time_alive_s"] for s in per_clip]))
    else:
        out["frac_terminated"] = float(np.mean([s["terminated"] for s in per_clip]))
        out["mean_time_alive_s"] = float(np.mean([s["time_alive_s"] for s in per_clip]))
    return out


# ---------------------------------------------------------------------------
# H5 saving
# ---------------------------------------------------------------------------

def save_h5(
    out_path: Path,
    clip_qpos: list,
    frame_skip: int,
    reference_h5_path: str,
    clip_length: int,
    n_eval_clips: int,
) -> None:
    """Save concatenated rollout + reference qpos at video frame rate.

    Each rendered frame is every ``frame_skip``-th env step; at the defaults this
    equals one reference frame, so rollout and reference rows align 1:1.
    """
    rollout_qpos = np.concatenate([q[::frame_skip] for q in clip_qpos], axis=0)

    with h5py.File(reference_h5_path, "r") as f:
        ref_flat = f["qpos"][: n_eval_clips * clip_length]
    ref = ref_flat.reshape(n_eval_clips, clip_length, -1)
    n_ref = rollout_qpos.shape[0] // n_eval_clips
    reference_qpos = np.concatenate([ref[c][:n_ref] for c in range(n_eval_clips)], axis=0)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("rollout_qpos", data=rollout_qpos)
        f.create_dataset("reference_qpos", data=np.array(reference_qpos))
    print(f"  Saved {out_path}  (rollout_qpos {rollout_qpos.shape}, "
          f"reference_qpos {reference_qpos.shape})")


# ---------------------------------------------------------------------------
# Video rendering (streamed frame-by-frame; peak RAM ~ one frame)
# ---------------------------------------------------------------------------

def render_clips_video(
    env,
    clip_qpos: list,
    clip_qvel: list,
    frame_skip: int,
    camera_kwargs: dict,
    output_path: Path,
    fps: int,
) -> None:
    """Render all clips into one mp4, with a logistic fade between clips.

    A single Renderer + VideoWriter are reused for the whole video and each frame
    is written and discarded immediately, so memory stays bounded to ~one frame
    (a 1920x1200x3 frame is ~6.6 MiB; a full clip would otherwise be ~10 GiB).
    """
    mj_model = build_render_model(env, camera_kwargs)  # has the CalibCamera
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    os.makedirs(output_path.parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))

    n_clips = len(clip_qpos)
    total_frames = sum(len(range(0, len(q), frame_skip)) for q in clip_qpos)
    print(f"  Rendering {total_frames} frames at {VIDEO_WIDTH}x{VIDEO_HEIGHT} "
          f"(+{FADE_FRAMES} fade frames x {n_clips - 1} boundaries)…", flush=True)

    for ci, (qpos_all, qvel_all) in enumerate(zip(clip_qpos, clip_qvel)):
        last_frame = None
        idxs = range(0, len(qpos_all), frame_skip)
        for t in tqdm(idxs, desc=f"  Clip {ci}", leave=False):
            mj_data.qpos = qpos_all[t]
            mj_data.qvel = qvel_all[t]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=CAMERA_NAME)
            last_frame = renderer.render()
            writer.write(cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))

        # Logistic fade-out between clips (not after the final clip). Formula
        # copied from Imitation.render (vnl-playground .../rodent/imitation.py:587-593).
        if ci < n_clips - 1 and FADE_FRAMES > 0 and last_frame is not None:
            for t in range(FADE_FRAMES):
                rel_t = t / FADE_FRAMES
                fade_factor = 1 / (1 + np.exp(10 * (rel_t - 0.5)))
                faded = (last_frame * fade_factor).astype(np.uint8)
                writer.write(cv2.cvtColor(faded, cv2.COLOR_RGB2BGR))

    writer.release()
    renderer.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_calib_mat(reference_dir: str) -> str:
    path = Path(reference_dir) / "calibration" / "hires_cam4_params.mat"
    if not path.exists():
        raise FileNotFoundError(f"Camera calibration not found: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CHECKPOINTS:
        print("No checkpoints listed in CHECKPOINTS. Edit the script to add paths.")
        return

    # Eval dataset: read the fixed per-clip frame count from the file's metadata.
    with h5py.File(NEW_EVAL_H5, "r") as f:
        clip_length = int(f.attrs["n_frames_per_clip"])
        n_available = int(f.attrs.get("n_clips", f["qpos"].shape[0] // clip_length))
    n_eval_clips = min(N_EVAL_CLIPS, n_available)
    print(f"Eval dataset: {NEW_EVAL_H5}  ({n_available} clips x {clip_length} frames); "
          f"rendering the first {n_eval_clips}.")

    calib_path = _find_calib_mat(REFERENCE_DIR)
    print(f"Loading Camera 4 calibration: {calib_path}")
    cam = load_camera_calibration(calib_path)
    camera_kwargs = convert_camera(cam, name=CAMERA_NAME)
    print(f"  fovy={camera_kwargs['fovy']:.2f}°  "
          f"pos={np.array(camera_kwargs['pos']).round(3)}")
    print(f"  AUTO_RESET_ON_TERMINATION={AUTO_RESET_ON_TERMINATION}")

    rollout_fn = nnx.jit(rollout_collect_stats, static_argnums=(0, 2, 5))

    for ckpt_path in tqdm(CHECKPOINTS, desc="Checkpoints"):
        print(f"\n=== {ckpt_path} ===")
        ckpt_dir = Path(ckpt_path)
        ckpt_name = ckpt_dir.name

        config_path = ckpt_dir / "config.json"
        if not config_path.exists():
            print(f"  [skip] no config.json at {config_path}")
            continue

        try:
            with open(config_path) as f:
                cfg_json = json.load(f)
            env_params = cfg_json["env_params"]
            net_params = cfg_json["net_params"]
            ctrl_dt = float(env_params.get("ctrl_dt", 0.01))
            mocap_hz = int(env_params.get("mocap_hz", 50))
            frames_per_step = ctrl_dt * mocap_hz
            n_steps = round(clip_length / frames_per_step)
            frame_skip = max(1, round(1.0 / (FPS * ctrl_dt)))
            print(f"  ctrl_dt={ctrl_dt}s  frames/step={frames_per_step}  "
                  f"n_steps={n_steps}  frame_skip={frame_skip}")

            env_class, default_config_fn = resolve_env_class(env_params)
            print(f"  env_class={env_class.__name__}  "
                  f"network_class={net_params.get('network_class')}")

            print("  Building environment…", flush=True)
            env_cfg = parse_imitation_env_config(
                env_params, NEW_EVAL_H5, clip_length, default_config_fn
            )
            env = env_class(env_cfg)

            print("  Loading checkpoint…", flush=True)
            loaded = load_network(ckpt_dir, net_params, env, seed=SEED)
            if loaded is None:
                print(f"  [skip] could not load network for {ckpt_name}")
                continue
            network, step = loaded
            network.eval()
            print(f"  step={step}")

            # Per-clip rollouts.
            clip_qpos, clip_qvel = [], []
            per_clip_stats = []
            for c in range(n_eval_clips):
                key = jax.random.key(SEED + c)
                stacked, done_mask, rewards, root_err, ee_err = rollout_fn(
                    env, network, n_steps, key, jp.array(c), AUTO_RESET_ON_TERMINATION
                )
                clip_qpos.append(np.array(stacked.data.qpos))
                clip_qvel.append(np.array(stacked.data.qvel))
                s = compute_stats(
                    np.array(rewards), np.array(root_err), np.array(ee_err),
                    np.array(done_mask), ctrl_dt, n_steps, AUTO_RESET_ON_TERMINATION,
                )
                s["clip_idx"] = c
                per_clip_stats.append(s)
                if AUTO_RESET_ON_TERMINATION:
                    print(f"    clip {c}: reward={s['total_reward']:.1f}  "
                          f"resets={s['n_resets']}  alive={s['avg_time_alive_s']:.1f}s  "
                          f"root_err={s['root_error_mean']:.4f}")
                else:
                    print(f"    clip {c}: reward={s['total_reward']:.1f}  "
                          f"terminated={s['terminated']}  alive={s['time_alive_s']:.1f}s  "
                          f"root_err(pre-fail)={s['root_error_mean']:.4f}")

            overall = aggregate_stats(per_clip_stats, AUTO_RESET_ON_TERMINATION)

            out_dir = Path(OUTPUT_DIR) / ckpt_name
            out_dir.mkdir(parents=True, exist_ok=True)

            stats = {
                "checkpoint": ckpt_name,
                "step": step,
                "network_class": net_params.get("network_class"),
                "env_class": env_class.__name__,
                "auto_reset_on_termination": AUTO_RESET_ON_TERMINATION,
                "n_eval_clips": n_eval_clips,
                "clip_length": clip_length,
                "ctrl_dt": ctrl_dt,
                "overall": overall,
                "per_clip": per_clip_stats,
            }
            stats_path = out_dir / f"stats{OUTPUT_SUFFIX}.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            alive = (overall["avg_time_alive_s"] if AUTO_RESET_ON_TERMINATION
                     else overall["mean_time_alive_s"])
            print(f"  Saved {stats_path}  "
                  f"(overall reward={overall['total_reward']:.1f}, "
                  f"alive={alive:.1f}s)")

            save_h5(out_dir / f"rollout{OUTPUT_SUFFIX}.h5", clip_qpos, frame_skip,
                    NEW_EVAL_H5, clip_length, n_eval_clips)
            render_clips_video(env, clip_qpos, clip_qvel, frame_skip,
                               camera_kwargs, out_dir / f"rollout{OUTPUT_SUFFIX}.mp4", FPS)
            print(f"  Done → {out_dir}/")
        except Exception as e:  # noqa: BLE001 — one bad checkpoint must not kill the batch
            print(f"  [skip] {ckpt_name}: {type(e).__name__}: {e}")
        finally:
            # Guard against JIT-cache / device-memory accumulation across runs
            # (each checkpoint bakes its env's reference arrays into the program).
            jax.clear_caches()
            gc.collect()

    print("\nAll checkpoints evaluated.")


if __name__ == "__main__":
    main()
