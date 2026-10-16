#!/usr/bin/env python3
"""Render single rodent poses as transparent PNGs (no floor, no sky), for slides.

Poses are rows of an artifact video's ``reference_qpos`` (STAC fits of the mocap),
drawn on the run's own body (see ``render_reference_video.env_params_from_index``).
The floor plane is deleted from the spec rather than hidden, so it neither shows nor
catches shadows or reflections. Alpha comes from MuJoCo's segmentation pass; both
passes are rendered at ``SUPERSAMPLE``x and averaged down, which antialiases the
silhouette. The image is cropped to the body plus ``PAD_PX``.

Run:
    cd vnl-experiments && ../.venv/bin/python -m \
        vnl_experiments.video_editing.render_pose_png 594219y6 --frames 2332 1469  # rearing, running
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
from pathlib import Path

import cv2
import h5py
import mujoco
import numpy as np

from vnl_experiments.delays import eval_videos
from vnl_experiments.video_editing.make_collage import (
    ARTIFACT_VIDEO_DIR, EVAL_DIR, VIDEO_SPEC_V3)
from vnl_experiments.video_editing.render_reference_video import (
    TARGET_RGBA, env_params_from_index, recolour)

OUT_W, OUT_H = 1600, 1200
SUPERSAMPLE = 2
PAD_PX = 24


def build_model(wandb_id: str, rgba=None) -> mujoco.MjModel:
    env_params = env_params_from_index(wandb_id)
    env_class, default_config_fn = eval_videos.resolve_env_class(env_params)
    cfg = eval_videos.parse_imitation_env_config(
        env_params, eval_videos.NEW_EVAL_H5, 1500, default_config_fn)
    spec = env_class(cfg)._spec.copy()
    for geom in list(spec.geoms):  # the floor, which the arena puts in a `floor` body
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(geom)
    model = spec.compile()
    model.vis.global_.offwidth = OUT_W * SUPERSAMPLE
    model.vis.global_.offheight = OUT_H * SUPERSAMPLE
    if rgba is not None:
        recolour(model, rgba)
    return model


def _body(model, name: str) -> int:
    """Body id by its XML name; the arena attaches the walker as ``<name>-rodent``."""
    return next(i for i in range(model.nbody)
                if model.body(i).name.split("-")[0] == name)


def render_pose(model, qpos, azimuth_offset: float, elevation: float,
                distance: float) -> np.ndarray:
    """One RGBA image of ``qpos``, cropped to the body.

    The camera azimuth is ``azimuth_offset`` from the body's heading, taken as the
    horizontal pelvis -> skull direction rather than from the root quaternion, whose
    yaw is meaningless once the animal rears up (90 = side-on, left flank).
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    dx, dy = (data.xpos[_body(model, "skull")] - data.xpos[_body(model, "pelvis")])[:2]
    azimuth = np.degrees(np.arctan2(dy, dx)) + azimuth_offset

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.subtree_com[model.jnt_bodyid[0]]  # joint 0: the root freejoint
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance

    w, h = OUT_W * SUPERSAMPLE, OUT_H * SUPERSAMPLE
    with mujoco.Renderer(model, height=h, width=w) as r:
        r.update_scene(data, camera=cam)
        rgb = r.render().astype(np.float32)
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam)
        alpha = (r.render()[..., 0] >= 0).astype(np.float32)

    # Premultiply before downsampling so background colour cannot bleed into edges.
    rgb = cv2.resize(rgb * alpha[..., None], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    rgb = np.where(alpha[..., None] > 0, rgb / np.maximum(alpha[..., None], 1e-6), 0)
    rgba = np.dstack([np.clip(rgb, 0, 255), alpha * 255]).round().astype(np.uint8)

    ys, xs = np.nonzero(rgba[..., 3])
    if not len(xs):
        raise ValueError("body is out of frame; increase --distance")
    y0, y1 = max(ys.min() - PAD_PX, 0), min(ys.max() + PAD_PX + 1, OUT_H)
    x0, x1 = max(xs.min() - PAD_PX, 0), min(xs.max() + PAD_PX + 1, OUT_W)
    return rgba[y0:y1, x0:x1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wandb_id")
    p.add_argument("--spec", default=VIDEO_SPEC_V3)
    p.add_argument("--frames", type=int, nargs="+", required=True,
                   help="row indices into the video's reference_qpos")
    p.add_argument("--azimuth-offset", type=float, default=60.0,
                   help="camera azimuth relative to the body heading (deg); "
                        "90 = side-on, 60 = three-quarter front")
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--distance", type=float, default=0.32)
    p.add_argument("--target-colour", action="store_true",
                   help="amber, as the target in the overview video (default teal)")
    p.add_argument("--out-dir", type=Path, default=EVAL_DIR / "poses")
    args = p.parse_args()

    with h5py.File(ARTIFACT_VIDEO_DIR / args.wandb_id / f"{args.spec}.h5", "r") as f:
        ref_qpos = f["reference_qpos"][:]
    model = build_model(args.wandb_id, TARGET_RGBA if args.target_colour else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for frame in args.frames:
        img = render_pose(model, ref_qpos[frame], args.azimuth_offset, args.elevation,
                          args.distance)
        out = args.out_dir / f"rodent_{args.wandb_id}_frame{frame}.png"
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        print(f"Wrote {out}  ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
