#!/usr/bin/env python3
"""Evaluate checkpoints trained with delays/train_rodent_delays.py.

For each checkpoint:
  - Reconstructs the Imitation env and enc-dec network from config.json
  - Runs a 10-minute video-synced rollout on the 2020_12_22_1 reference clip
  - Writes to eval_delays/{checkpoint_name}/:
      stats.json   — total reward, resets, avg time alive, root / end-eff errors
      rollout.h5   — rollout_qpos and reference_qpos at video frame rate
      rollout.mp4  — 1920x1200 Camera-4-matched render, no ghost, no labels

Usage:
    .venv/bin/python -m vnl_experiments.delays.eval_rodent_delays
"""

import json
import os
import pickle
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import scipy.io as spio
import types as _types
import cv2
import h5py
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from flax import nnx
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from etils import epath
from vnl_experiments.envs.absolute_imitation import AbsoluteImitation, default_config

from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state
from nnx_ppo.algorithms.rollout import SlimData, SlimState
from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Concat, Sequential
from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.utils import Filter, Flattener
from nnx_ppo.networks.variational import VariationalBottleneck

from vnl_experiments.delays.efference_copy import EfferenceCopy

# ---------------------------------------------------------------------------
# Hard-coded settings — edit before running
# ---------------------------------------------------------------------------

CHECKPOINTS = [
#    "downloaded_checkpoints/RodentEncDec_delay0_eff0-20260601-105420",
    "downloaded_checkpoints/RodentEncDec_delay0_eff0-20260603-083745",
    "downloaded_checkpoints/RodentEncDec_delay10_eff10-20260611-120848",
#    "checkpoints/RodentEncDec_delay0_eff0-20260603-083745",
#    "checkpoints/RodentEncDec_delay1_eff1-20260603-083843",
#    "checkpoints/RodentEncDec_delay2_eff2-20260603-084107",
#    "checkpoints/RodentEncDec_delay3_eff3-20260603-084626",
#    "checkpoints/RodentEncDec_delay4_eff4-20260603-084640",
#    "checkpoints/RodentEncDec_delay5_eff5-20260603-084658",
#    "checkpoints/RodentEncDec_delay6_eff6-20260603-084705",
#    "checkpoints/RodentEncDec_delay7_eff7-20260603-084705",
#    "checkpoints/RodentEncDec_delay8_eff8-20260603-084925",
#    "checkpoints/RodentEncDec_delay9_eff9-20260603-085005",
#    "checkpoints/RodentEncDec_delay10_eff10-20260603-085005",
]

# Reference session directory. Must contain:
#   transform_art_*.h5              (STAC output, single continuous clip)
#   calibration/hires_cam4_params.mat
REFERENCE_DIR = "assets/art/2020_12_22_1"

SAMPLE_MINUTES = 10
OUTPUT_DIR = "eval_delays"
VIDEO_HEIGHT = 1200
VIDEO_WIDTH = 1920
FPS = 50

# ---------------------------------------------------------------------------
# Camera calibration and rendering
# (Inlined from tools/camera_matched_rollout.py to avoid its import chain)
# ---------------------------------------------------------------------------

_DANNCE_IMAGE_HEIGHT = 1200


def load_camera_calibration(mat_path: str):
    mat = spio.loadmat(mat_path, squeeze_me=True)
    return _types.SimpleNamespace(
        K=mat["K"], r=mat["r"], t=mat["t"],
        RDistort=mat["RDistort"], TDistort=mat["TDistort"],
    )


def convert_camera(cam, name: str = "CalibCamera") -> dict:
    rot = R.from_matrix(cam.r.T)
    eul = rot.as_euler("zyx")
    eul[2] += np.pi
    quat = R.from_euler("zyx", eul).as_quat()
    quat = quat[np.array([3, 0, 1, 2])]
    quat[0] *= -1
    fovy = 2 * np.arctan(_DANNCE_IMAGE_HEIGHT / (2 * cam.K[1, 1])) / (2 * np.pi) * 360
    pos = (-cam.t.reshape(1, 3) @ cam.r.T / 1000).squeeze()
    return {"name": name, "pos": pos, "quat": quat, "fovy": fovy}


def render_with_calib_camera(env, trajectory: list, camera_kwargs: dict,
                              height: int, width: int) -> list:
    spec = env._spec.copy()
    cam = spec.worldbody.add_camera()
    cam.name = camera_kwargs["name"]
    cam.pos = np.array(camera_kwargs["pos"])
    cam.quat = np.array(camera_kwargs["quat"])
    cam.fovy = float(camera_kwargs["fovy"])
    mj_model = spec.compile()
    mj_model.vis.global_.offwidth = width
    mj_model.vis.global_.offheight = height
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    frames = []
    for state in trajectory:
        mj_data.qpos = np.array(state.data.qpos)
        mj_data.qvel = np.array(state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=camera_kwargs["name"])
        frames.append(renderer.render().copy())
    return frames


# ---------------------------------------------------------------------------
# Environment config parsing
# ---------------------------------------------------------------------------

def parse_imitation_env_config(env_params: dict, reference_h5: str, total_frames: int):
    """Reconstruct an Imitation config from the saved env_params dict."""
    cfg = default_config()

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
        ("clip_length", int),
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

    # Eval overrides
    cfg.reference_data_path = epath.Path(reference_h5)
    cfg.clip_length = total_frames
    cfg.start_frame_range = [0, 1]
    # Reduce pre-allocation sizes — single env only needs a fraction
    cfg.naconmax = min(cfg.naconmax, 2048)
    cfg.njmax = min(cfg.njmax, 256)

    # Keep local XML paths (training cluster paths are invalid here)
    default = default_config()
    cfg.walker_xml_path = default.walker_xml_path
    cfg.arena_xml_path = default.arena_xml_path

    return cfg


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def _parse_net_params(raw: dict) -> dict:
    """Convert JSON string values to proper Python types (matching checkpoint_utils)."""
    result = {}
    for k, v in raw.items():
        if isinstance(v, list):
            result[k] = [int(x) for x in v]
        elif v == "True":
            result[k] = True
        elif v == "False":
            result[k] = False
        elif v == "None":
            result[k] = None
        else:
            try:
                result[k] = int(v)
            except (ValueError, TypeError):
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    result[k] = v
    return result


def build_delay_network(net_params: dict, env: AbsoluteImitation, rngs: nnx.Rngs):
    """Reconstruct the enc-dec delay network from saved net_params."""
    p = _parse_net_params({k: v for k, v in net_params.items()
                           if k != "network_class"})

    obs_size = env.non_flattened_observation_size
    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0]))
    action_size = env.action_size

    delay_k = int(p.get("delay_k", 0))
    efference_length = int(p.get("efference_length", delay_k))

    enc_hidden = list(p.get("enc_hidden_sizes", [512] * 4))
    dec_hidden = list(p.get("dec_hidden_sizes", [512] * 4))
    critic_hidden = list(p.get("critic_hidden_sizes", [1024] * 2))
    latent_size = int(p.get("latent_size", 16))
    kl_weight = float(p.get("kl_weight", 0.01))
    latent_min_std = float(p.get("latent_min_std", 0.01))
    entropy_weight = float(p.get("entropy_weight", 1e-2))
    min_std = float(p.get("min_std", 1e-1))
    std_scale = float(p.get("std_scale", 1.0))
    normalize_obs = bool(p.get("normalize_obs", True))
    initializer_scale = float(p.get("initializer_scale", 1.0))

    activation_name = str(p.get("activation", "swish"))
    activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[activation_name]

    kernel_init = nnx.initializers.variance_scaling(
        initializer_scale, "fan_in", "uniform"
    )

    enc_sizes = [task_obs_size] + enc_hidden + [latent_size * 2]
    decoder_in = latent_size + proprio_size + efference_length * action_size
    dec_sizes = [decoder_in] + dec_hidden + [action_size * 2]
    critic_sizes = [task_obs_size + proprio_size] + critic_hidden + [1]

    encoder_branch = Sequential([
        Flattener(),
        *make_mlp_layers(enc_sizes, rngs, activation,
                         activation_last_layer=False, kernel_init=kernel_init),
        VariationalBottleneck(latent_size, rngs, kl_weight, latent_min_std),
    ])

    proprio_branch_layers = [Flattener()]
    if delay_k > 0:
        proprio_branch_layers.append(Delay(jp.zeros(proprio_size), k_steps=delay_k))
    proprio_branch = Sequential(proprio_branch_layers)

    decoder = Sequential([
        *make_mlp_layers(dec_sizes, rngs, activation,
                         activation_last_layer=False, kernel_init=kernel_init),
        NormalTanhSampler(rngs, entropy_weight=entropy_weight,
                          min_std=min_std, std_scale=std_scale),
    ])

    actor = Sequential([
        Concat(task_obs=encoder_branch, proprioception=proprio_branch),
        EfferenceCopy(inner=decoder, sample_action=jp.zeros(action_size),
                      queue_length=efference_length),
    ])

    critic = Sequential([
        Flattener(),
        *make_mlp_layers(critic_sizes, rngs, activation,
                         activation_last_layer=False, kernel_init=kernel_init),
    ])

    adapter = PPOAdapter(action=actor, value=critic)

    pre = Flattener(preserve_levels=2)
    lift = Filter({
        "task_obs": ("state", "task_obs"),
        "proprioception": ("state", "proprioception"),
    })

    if normalize_obs:
        normalizer_shape = {
            "state": {
                "task_obs": task_obs_size,
                "proprioception": proprio_size,
            }
        }
        return Sequential([pre, Normalizer(normalizer_shape), lift, adapter])
    return Sequential([pre, lift, adapter])


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_delays_checkpoint(ckpt_dir: "str | Path", env: AbsoluteImitation, seed: int = 0):
    """Load network weights from a delays checkpoint directory."""
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)

    net_params = cfg["net_params"]

    step_dirs = sorted(ckpt_dir.glob("step_*/"))
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directory found in {ckpt_dir}")
    step_dir = step_dirs[-1]
    print(f"  Loading step: {step_dir.name}")

    rngs = nnx.Rngs(seed)
    nets = build_delay_network(net_params, env, rngs)

    with open(step_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    ppo_cfg = meta.get("config")
    ppo_cfg = ppo_cfg.ppo if ppo_cfg is not None else None

    training_state = new_training_state(
        env, nets, n_envs=1, seed=seed,
        learning_rate=ppo_cfg.learning_rate if ppo_cfg else 1e-4,
        gradient_clipping=ppo_cfg.gradient_clipping if ppo_cfg else 1.0,
        weight_decay=ppo_cfg.weight_decay if ppo_cfg else None,
    )
    load_checkpoint(str(step_dir), training_state.networks, training_state.optimizer)
    return training_state.networks


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


def rollout_collect_stats(env, networks, n_steps: int, key):
    """Video-synced rollout that also collects per-step error metrics.

    On episode termination the env is reset to the reference frame dictated by
    a *deterministic global schedule* (``floor(step * ctrl_dt * mocap_hz)``)
    rather than to the env's own (possibly stalled) ``current_frame``. This
    keeps the rendered rollout locked to the reference timeline so it can be
    placed side-by-side with the real-rodent video without drift: step ``k``
    always depicts reference frame ``floor(k * ctrl_dt * mocap_hz)``.

    Using the env's ``current_frame`` instead would freeze the timeline if a
    frame is un-trackable: a 1-step episode advances time by only
    ``ctrl_dt*mocap_hz`` (< 1 frame), so ``floor`` never increments and every
    subsequent reset returns to the same frame — desyncing the video. The
    global schedule advances regardless, so resets can never stall it.

    Returns:
        stacked_states: SlimState with leading dim n_steps
        reset_mask:     bool array (n_steps,), True at reset steps
        per_step_reward: float array (n_steps,)
        root_errors:    float array (n_steps,)
        end_eff_errors: float array (n_steps,)
    """
    key, key2 = jax.random.split(key)
    env_state = env.reset(key)
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

        # Capture reward/metrics from the post-step, pre-reset state so they
        # reflect the policy's actual tracking rather than the reset pose.
        step_reward = sum(jax.tree.leaves(next_env_state.reward))
        root_err = next_env_state.metrics["root_pos_distance"]
        end_eff_err = next_env_state.metrics["body_errors/end_eff_total"]

        # Reset to the scheduled reference frame for the *next* rollout step,
        # so the rendered timeline stays aligned with the reference video.
        scheduled_frame = jp.floor(
            (step + 1).astype(jp.float32) * frames_per_step
        ).astype(jp.int32)

        def do_reset(rng):
            return env.reset(rng, clip_idx=jp.array(0), start_frame=scheduled_frame)

        next_env_state = jax.lax.cond(
            reset_happened, do_reset, lambda rng: next_env_state, rng
        )
        next_net_state = jax.lax.cond(
            reset_happened, networks.reset_state, lambda x: x, next_net_state
        )

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
    reset_mask: np.ndarray,
    ctrl_dt: float,
    n_steps: int,
) -> dict:
    n_resets = int(np.sum(reset_mask))
    n_episodes = n_resets + 1
    avg_time_alive_s = (n_steps / n_episodes) * ctrl_dt
    return {
        "total_reward": float(np.sum(per_step_reward)),
        "n_resets": n_resets,
        "avg_time_alive_s": avg_time_alive_s,
        "root_error_mean": float(np.mean(root_errors)),
        "root_error_std": float(np.std(root_errors)),
        "end_eff_error_mean": float(np.mean(end_eff_errors)),
        "end_eff_error_std": float(np.std(end_eff_errors)),
    }


# ---------------------------------------------------------------------------
# H5 saving
# ---------------------------------------------------------------------------

def save_h5(
    output_dir: Path,
    stacked_states: SlimState,
    frame_skip: int,
    reference_h5_path: str,
    n_video_frames: int,
) -> None:
    rollout_qpos = np.array(stacked_states.data.qpos[::frame_skip])  # (n_video_frames, q)
    with h5py.File(reference_h5_path, "r") as f:
        reference_qpos = f["qpos"][:n_video_frames]

    out_path = output_dir / "rollout.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("rollout_qpos", data=rollout_qpos)
        f.create_dataset("reference_qpos", data=np.array(reference_qpos))
    print(f"  Saved {out_path}  (rollout_qpos {rollout_qpos.shape}, "
          f"reference_qpos {reference_qpos.shape})")


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

def render_video(
    env,
    stacked_states: SlimState,
    n_steps: int,
    frame_skip: int,
    camera_kwargs: dict,
    output_path: Path,
    fps: int,
) -> None:
    n_frames = len(range(0, n_steps, frame_skip))
    print(f"  Rendering {n_frames} frames at {VIDEO_WIDTH}x{VIDEO_HEIGHT}…", flush=True)

    spec = env._spec.copy()
    cam = spec.worldbody.add_camera()
    cam.name = camera_kwargs["name"]
    cam.pos = np.array(camera_kwargs["pos"])
    cam.quat = np.array(camera_kwargs["quat"])
    cam.fovy = float(camera_kwargs["fovy"])
    mj_model = spec.compile()
    mj_model.vis.global_.offwidth = VIDEO_WIDTH
    mj_model.vis.global_.offheight = VIDEO_HEIGHT
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    os.makedirs(output_path.parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))

    for t in tqdm(range(0, n_steps, frame_skip), desc="  Rendering", leave=False):
        state = jax.tree.map(lambda x: x[t], stacked_states)
        mj_data.qpos = np.array(state.data.qpos)
        mj_data.qvel = np.array(state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=camera_kwargs["name"])
        frame = renderer.render()
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_reference_h5(reference_dir: str) -> str:
    ref = Path(reference_dir)
    matches = sorted(ref.glob("transform_art_*.h5"))
    if not matches:
        raise FileNotFoundError(
            f"No transform_art_*.h5 found in {reference_dir}"
        )
    return str(matches[0])


def _find_calib_mat(reference_dir: str) -> str:
    ref = Path(reference_dir)
    path = ref / "calibration" / "hires_cam4_params.mat"
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

    # Reference data
    reference_h5 = _find_reference_h5(REFERENCE_DIR)
    calib_path = _find_calib_mat(REFERENCE_DIR)

    with h5py.File(reference_h5, "r") as f:
        total_frames = f["qpos"].shape[0]
    print(f"Reference H5: {reference_h5}  ({total_frames} frames, "
          f"{total_frames / FPS / 60:.1f} min)")

    n_video_frames = SAMPLE_MINUTES * 60 * FPS
    if n_video_frames > total_frames:
        raise ValueError(
            f"Requested {SAMPLE_MINUTES} min = {n_video_frames} frames "
            f"but H5 only has {total_frames} frames"
        )

    print(f"Loading Camera 4 calibration: {calib_path}")
    cam = load_camera_calibration(calib_path)
    camera_kwargs = convert_camera(cam, name="CalibCamera")
    print(f"  fovy={camera_kwargs['fovy']:.2f}°  "
          f"pos={np.array(camera_kwargs['pos']).round(3)}")

    for ckpt_path in tqdm(CHECKPOINTS, desc="Checkpoints"):
        print(f"\n=== {ckpt_path} ===")
        ckpt_dir = Path(ckpt_path)
        ckpt_name = ckpt_dir.name

        with open(ckpt_dir / "config.json") as f:
            cfg_json = json.load(f)
        env_params = cfg_json["env_params"]
        ctrl_dt = float(env_params.get("ctrl_dt", 0.01))
        frame_skip = max(1, round(1.0 / (FPS * ctrl_dt)))
        n_steps = n_video_frames * frame_skip
        print(f"  ctrl_dt={ctrl_dt}s  frame_skip={frame_skip}  n_steps={n_steps}")

        # Build env
        print("  Building environment…", flush=True)
        env_cfg = parse_imitation_env_config(env_params, reference_h5, total_frames)
        env = AbsoluteImitation(env_cfg)

        # Load network
        print("  Loading checkpoint…", flush=True)
        network = load_delays_checkpoint(ckpt_dir, env, seed=0)
        network.eval()

        # Rollout
        print(f"  Running rollout ({n_steps} steps)…", flush=True)
        rollout_fn = nnx.jit(rollout_collect_stats, static_argnums=(0, 2))
        key = jax.random.key(0)
        stacked, reset_mask, rewards, root_errors, end_eff_errors = rollout_fn(
            env, network, n_steps, key
        )
        reset_mask_np = np.array(reset_mask)
        print(f"  Resets: {int(np.sum(reset_mask_np))}")

        # Stats
        stats = compute_stats(
            np.array(rewards),
            np.array(root_errors),
            np.array(end_eff_errors),
            reset_mask_np,
            ctrl_dt,
            n_steps,
        )
        print(f"  total_reward={stats['total_reward']:.1f}  "
              f"n_resets={stats['n_resets']}  "
              f"avg_alive={stats['avg_time_alive_s']:.1f}s  "
              f"root_err={stats['root_error_mean']:.4f}±{stats['root_error_std']:.4f}  "
              f"end_eff_err={stats['end_eff_error_mean']:.4f}±{stats['end_eff_error_std']:.4f}")

        # Output folder
        out_dir = Path(OUTPUT_DIR) / ckpt_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save stats
        stats_path = out_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved {stats_path}")

        # Save H5
        save_h5(out_dir, stacked, frame_skip, reference_h5, n_video_frames)

        # Render video
        render_video(env, stacked, n_steps, frame_skip, camera_kwargs,
                     out_dir / "rollout.mp4", FPS)

        print(f"  Done → {out_dir}/")

    print("\nAll checkpoints evaluated.")


if __name__ == "__main__":
    main()
