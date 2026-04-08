#!/usr/bin/env python3
"""Camera-matched rollout visualization.

Renders a side-by-side video of the original rat footage and one or more
imitation rollouts, with the MuJoCo camera matched to the real camera used to
film the rat.

Camera calibration files are the per-camera MATLAB files produced by DANNCE
(e.g. hires_cam4_params.mat). The reference H5 file is the STAC output for the
recording session, treated as a single continuous clip.

Usage:
    .venv/bin/python vnl_experiments/tools/camera_matched_rollout.py \\
        --checkpoint downloaded_checkpoints/my_run \\
        --calibration assets/art/2020_12_21_1/calibration/hires_cam4_params.mat \\
        --video assets/art/2020_12_21_1/videos/Camera4/0.mp4 \\
        --reference_h5 assets/art/2020_12_21_1/transform_art_2020_12_22_1.h5 \\
        --video_offset 0 \\
        --start_time 0 --end_time 10 \\
        --n_rollouts 3 \\
        --output /tmp/rollout_comparison.mp4
"""

import argparse
import json
import os
import types
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import cv2
import h5py
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import scipy.io as spio
from flax import nnx
from etils import epath
from scipy.spatial.transform import Rotation as R

from nnx_ppo.algorithms import rollout
from nnx_ppo.algorithms.rollout import SlimData, SlimState
from vnl_playground.tasks.modular_rodent.imitation_v4 import ModularImitation_v4
from vnl_experiments.tools.checkpoint_utils import (
    build_network,
    load_network_from_checkpoint,
    parse_env_config,
)


# Standard DANNCE rig image height used for fovy calculation
_DANNCE_IMAGE_HEIGHT = 1200


# ---------------------------------------------------------------------------
# Rollout with video-synced resets
# ---------------------------------------------------------------------------

_RESET_FADE_FRAMES = 15  # video frames over which the RESET label fades


def _slim(env_state) -> SlimState:
    return SlimState(
        data=SlimData(
            qpos=env_state.data.qpos,
            qvel=env_state.data.qvel,
            time=env_state.data.time,
        ),
        done=env_state.done,
        info=env_state.info,
        metrics=env_state.metrics,
    )


def rollout_video_synced(env, networks, n_steps: int, key):
    """Scan-based rollout that resets to the current video frame on termination.

    When the episode ends, `env.reset` is called with `start_frame` set to
    `metrics["current_frame"]` of the just-terminated state, keeping the
    reference target in sync with the real video.

    Returns:
        stacked_states: SlimState with leading dimension n_steps
        reset_mask: bool array of shape (n_steps,), True where a reset occurred
        total_reward: scalar total reward
    """
    key, key2 = jax.random.split(key)
    env_state = env.reset(key)
    net_state = networks.initialize_state(1)
    net_state = jax.tree.map(lambda x: x[0], net_state)

    def step_fn(networks, carry):
        env_state, net_state, cumulative_reward, rng = carry

        obs_batched = jax.tree.map(lambda x: x[None], env_state.obs)
        net_state_batched = jax.tree.map(lambda x: x[None], net_state)
        next_net_state, network_output = networks(net_state_batched, obs_batched)
        next_net_state = jax.tree.map(lambda x: x[0], next_net_state)
        action = jax.tree.map(lambda x: x[0], network_output.actions)

        next_env_state = env.step(env_state, action)
        reward_sum = sum(jax.tree.leaves(next_env_state.reward))

        reset_happened = next_env_state.done.astype(bool)
        current_frame = next_env_state.metrics["current_frame"]

        def do_reset(rng):
            return env.reset(rng, clip_idx=jp.array(0), start_frame=current_frame)

        next_env_state = jax.lax.cond(
            reset_happened, do_reset, lambda rng: next_env_state, rng
        )
        next_net_state = jax.lax.cond(
            reset_happened, networks.reset_state, lambda x: x, next_net_state
        )

        (new_rng,) = jax.random.split(rng, 1)
        return (
            next_env_state,
            next_net_state,
            cumulative_reward + reward_sum,
            new_rng,
        ), (_slim(env_state), reset_happened)

    scan_fn = nnx.scan(
        step_fn,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=(nnx.Carry, 0),
        length=n_steps,
    )

    init_carry = (env_state, net_state, jp.array(0.0), key2)
    (_, _, total_reward, _), (stacked_states, reset_mask) = scan_fn(
        networks, init_carry
    )
    return stacked_states, reset_mask, total_reward


def overlay_reset_indicators(
    frames: list[np.ndarray],
    reset_mask: np.ndarray,
    frame_skip: int,
    fade_frames: int = _RESET_FADE_FRAMES,
) -> list[np.ndarray]:
    """Overlay a fading 'RESET' label on frames following a reset event.

    Args:
        frames: Rendered video frames (RGB uint8), one per video frame.
        reset_mask: Boolean array of shape (n_steps,), True at reset steps.
        frame_skip: Policy steps per video frame (used to map steps → frames).
        fade_frames: Number of video frames over which the label fades out.
    """
    reset_video_frames = set(
        int(s) // frame_skip for s in np.where(reset_mask)[0]
    )
    for vf_idx, frame in enumerate(frames):
        for rf in reset_video_frames:
            dist = vf_idx - rf
            if 0 <= dist < fade_frames:
                alpha = 1.0 - dist / fade_frames
                color = (int(255 * alpha), int(50 * alpha), int(50 * alpha))
                cv2.putText(
                    frame, "RESET", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA,
                )
                break
    return frames


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------

def load_camera_calibration(mat_path: str):
    """Load one camera's calibration from a hires_cam{N}_params.mat file.

    Returns a SimpleNamespace with attributes:
        K         (3, 3)  intrinsic matrix
        r         (3, 3)  rotation matrix
        t         (3,)    translation vector (in mm)
        RDistort  (2,)    radial distortion coefficients
        TDistort  (2,)    tangential distortion coefficients
    """
    mat = spio.loadmat(mat_path, squeeze_me=True)
    return types.SimpleNamespace(
        K=mat["K"],
        r=mat["r"],
        t=mat["t"],
        RDistort=mat["RDistort"],
        TDistort=mat["TDistort"],
    )


def convert_camera(cam, name: str = "CalibCamera") -> dict:
    """Convert DANNCE camera parameters to MuJoCo camera kwargs.

    Ports the conversion from Diego's stac/stac/viz.py:convert_cameras().
    MATLAB/DANNCE uses a different X-axis convention from MuJoCo; we correct
    this by adding π to the Z Euler angle before converting to a quaternion.

    Args:
        cam: SimpleNamespace with .K, .r, .t, .RDistort, .TDistort
        name: Camera name in the MuJoCo model

    Returns:
        dict with keys 'name', 'pos', 'quat', 'fovy' ready for MjsCamera
    """
    rot = R.from_matrix(cam.r.T)
    eul = rot.as_euler("zyx")
    eul[2] += np.pi                             # MATLAB → MuJoCo X-axis flip
    quat = R.from_euler("zyx", eul).as_quat()  # scipy convention: [x, y, z, w]
    quat = quat[np.array([3, 0, 1, 2])]        # reorder to MuJoCo: [w, x, y, z]
    quat[0] *= -1                               # negate scalar component

    fovy = (
        2 * np.arctan(_DANNCE_IMAGE_HEIGHT / (2 * cam.K[1, 1])) / (2 * np.pi) * 360
    )
    pos = (-cam.t.reshape(1, 3) @ cam.r.T / 1000).squeeze()  # mm → m

    return {"name": name, "pos": pos, "quat": quat, "fovy": fovy}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_with_calib_camera(
    env: ModularImitation_v4,
    trajectory: list,
    camera_kwargs: dict,
    height: int,
    width: int,
) -> list[np.ndarray]:
    """Render a trajectory with a calibrated camera injected into the MjSpec.

    Copies env._spec (same pattern as imitation_v3.py render()), adds the
    calibrated camera, compiles, then renders each SlimState in trajectory.

    Args:
        env: Constructed ModularImitation_v4 environment
        trajectory: List of SlimState from rollout.unstack_trajectory()
        camera_kwargs: Dict from convert_camera() with name/pos/quat/fovy
        height: Output frame height in pixels
        width: Output frame width in pixels

    Returns:
        List of uint8 RGB arrays of shape (height, width, 3)
    """
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
# Real video extraction
# ---------------------------------------------------------------------------

def extract_real_frames(
    video_path: str,
    first_frame: int,
    n_frames: int,
    cam,
    height: int,
    width: int,
) -> list[np.ndarray]:
    """Extract, undistort, and resize frames from the real video.

    Args:
        video_path: Path to the .mp4 file
        first_frame: Index of the first frame to extract (= video_offset + start_frame)
        n_frames: Number of frames to extract
        cam: Camera calibration SimpleNamespace (for undistortion)
        height: Output frame height
        width: Output frame width

    Returns:
        List of uint8 RGB arrays of shape (height, width, 3)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    # MATLAB K is stored in column-major order relative to OpenCV convention
    K = cam.K.T
    dist_coeffs = np.concatenate([cam.RDistort.flatten(), cam.TDistort.flatten()])

    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        undistorted = cv2.undistort(frame_rgb, K, dist_coeffs, None, K)
        resized = cv2.resize(undistorted, (width, height))
        frames.append(resized)

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------

def write_side_by_side_video(
    real_frames: list[np.ndarray],
    rollout_frames_list: list[list[np.ndarray]],
    output_path: str,
    fps: int,
) -> None:
    """Write a horizontally concatenated comparison video.

    Layout: real video | rollout 1 | rollout 2 | ...

    Args:
        real_frames: Extracted real video frames (RGB uint8)
        rollout_frames_list: One list of frames per rollout (RGB uint8)
        output_path: Where to write the output .mp4
        fps: Output frame rate
    """
    n_frames = min(len(real_frames), min(len(f) for f in rollout_frames_list))
    h, w = real_frames[0].shape[:2]
    n_panels = 1 + len(rollout_frames_list)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * n_panels, h))

    labels = ["Real"] + [f"Rollout {i + 1}" for i in range(len(rollout_frames_list))]

    for i in range(n_frames):
        panels = [real_frames[i]] + [rf[i] for rf in rollout_frames_list]
        for panel, label in zip(panels, labels):
            cv2.putText(
                panel, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
            )
        row = np.concatenate(panels, axis=1)
        writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Wrote {n_frames} frames → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Camera-matched rollout visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Run directory containing config.json")
    parser.add_argument("--calibration", required=True,
                        help="Path to hires_cam{N}_params.mat")
    parser.add_argument("--video", required=True,
                        help="Path to the raw camera .mp4")
    parser.add_argument("--reference_h5", required=True,
                        help="Session STAC .h5 file (single continuous recording)")
    parser.add_argument("--video_offset", type=int, default=0,
                        help="Real video frame index corresponding to H5 frame 0")
    parser.add_argument("--start_time", type=float, required=True,
                        help="Start time within the H5 recording (seconds)")
    parser.add_argument("--end_time", type=float, required=True,
                        help="End time within the H5 recording (seconds)")
    parser.add_argument("--n_rollouts", type=int, default=3,
                        help="Number of rollouts to show side-by-side")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--height", type=int, default=480,
                        help="Output frame height in pixels")
    parser.add_argument("--width", type=int, default=640,
                        help="Output frame width in pixels")
    parser.add_argument("--fps", type=int, default=50,
                        help="Recording frame rate (used to convert time↔frames)")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    start_frame = int(args.start_time * args.fps)
    end_frame = int(args.end_time * args.fps)
    n_video_frames = end_frame - start_frame
    if n_video_frames <= 0:
        raise ValueError(f"end_time must be > start_time (got {args.start_time}–{args.end_time})")

    # ------------------------------------------------------------------
    # Camera calibration
    # ------------------------------------------------------------------
    print(f"Loading calibration: {args.calibration}")
    cam = load_camera_calibration(args.calibration)
    camera_kwargs = convert_camera(cam, name="CalibCamera")
    print(f"  fovy = {camera_kwargs['fovy']:.2f}°  pos = {np.array(camera_kwargs['pos']).round(3)}")

    # ------------------------------------------------------------------
    # Reference H5: read total frame count to set clip_length
    # ------------------------------------------------------------------
    with h5py.File(args.reference_h5, "r") as f:
        total_frames = f["qpos"].shape[0]
    print(f"Reference H5: {total_frames} frames ({total_frames / args.fps / 60:.1f} min)")
    if end_frame > total_frames:
        raise ValueError(
            f"end_time={args.end_time}s → frame {end_frame} exceeds H5 length {total_frames}"
        )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    print("Loading environment…")
    ckpt_dir = Path(args.checkpoint)
    with open(ckpt_dir / "config.json") as f:
        cfg_json = json.load(f)

    config = parse_env_config(cfg_json.get("env_params", {}))
    config.reference_data_path = epath.Path(args.reference_h5)
    config.clip_length = total_frames   # treat whole file as one clip; clip_set="all" → 1 clip
    config.start_frame_range = [start_frame, start_frame + 1]

    env = ModularImitation_v4(config)

    # ctrl_dt may be faster than the mocap fps — compute the integer step ratio
    frame_skip = max(1, round(1.0 / (args.fps * config.ctrl_dt)))
    n_steps = n_video_frames * frame_skip
    print(f"ctrl_dt={config.ctrl_dt}s → frame_skip={frame_skip}, n_steps={n_steps}")

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    network = load_network_from_checkpoint(ckpt_dir, env, seed=args.seed)
    network.eval()

    # ------------------------------------------------------------------
    # Rollouts (reset to current video frame on termination)
    # ------------------------------------------------------------------
    rollout_fn = nnx.jit(rollout_video_synced, static_argnums=(0, 2))
    base_key = jax.random.key(args.seed)
    all_trajectories = []
    all_reset_masks = []

    for i in range(args.n_rollouts):
        print(f"  Rollout {i + 1}/{args.n_rollouts}…", flush=True)
        key = jax.random.fold_in(base_key, i)
        stacked, reset_mask, total_reward = rollout_fn(env, network, n_steps, key)
        # Subsample every frame_skip-th step to match mocap fps
        traj = [
            jax.tree.map(lambda x: x[t], stacked)
            for t in range(0, n_steps, frame_skip)
        ]
        all_trajectories.append(traj)
        all_reset_masks.append(np.array(reset_mask))
        n_resets = int(np.sum(reset_mask))
        print(f"    total reward = {float(total_reward):.2f}  resets = {n_resets}")

    network.train()

    # ------------------------------------------------------------------
    # Render rollouts and overlay reset indicators
    # ------------------------------------------------------------------
    all_rollout_frames = []
    for i, (traj, reset_mask) in enumerate(zip(all_trajectories, all_reset_masks)):
        print(f"  Rendering rollout {i + 1}/{args.n_rollouts}…", flush=True)
        frames = render_with_calib_camera(
            env, traj, camera_kwargs, args.height, args.width
        )
        overlay_reset_indicators(frames, reset_mask, frame_skip)
        all_rollout_frames.append(frames)

    # ------------------------------------------------------------------
    # Extract real video frames
    # ------------------------------------------------------------------
    first_video_frame = args.video_offset + start_frame
    print(f"Extracting real video frames {first_video_frame}–{first_video_frame + n_video_frames}…")
    real_frames = extract_real_frames(
        args.video, first_video_frame, n_video_frames, cam, args.height, args.width
    )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    write_side_by_side_video(real_frames, all_rollout_frames, args.output, args.fps)


if __name__ == "__main__":
    main()
