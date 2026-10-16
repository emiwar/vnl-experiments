#!/usr/bin/env python3
"""Render the STAC reference poses (the imitation *target*) beside an artifact video.

Every ``video`` artifact (``artifacts/video/<wandb_id>/<spec>.mp4``) has an ``.h5``
sidecar holding ``reference_qpos`` -- the STAC fit of the mocap, one row per video
frame. This script renders those poses kinematically (``mj_forward`` only, no
physics) with the same body, the same Camera-4-matched view and the same
``CLIP_FRAMES + FADE_FRAMES`` layout as the rollout, so the result can be tiled
frame-for-frame next to it and next to the raw footage.

The body is rebuilt from the run's env params in the local WandB index, so the
target is drawn on exactly the walker XML / rescale the policy was trained on. It is
recoloured (``--colour``) so a viewer can tell "target" from "simulation" at a glance.

Run:
    cd vnl-experiments && ../.venv/bin/python -m \
        vnl_experiments.video_editing.render_reference_video 594219y6
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
from pathlib import Path

import h5py
import mujoco
import numpy as np
from tqdm import tqdm

from vnl_experiments.delays import eval_videos
from vnl_experiments.video_editing.hq_video import HQVideoWriter
from vnl_experiments.video_editing.make_collage import (
    ARTIFACT_VIDEO_DIR, CLIP_FRAMES, EVAL_DIR, FADE_FRAMES, VIDEO_SPEC_V3, fade_factor)
from vnl_experiments.wandb_utils import index

# Warm amber, to contrast with the teal the walker XML gives the simulated body.
TARGET_RGBA = (0.92, 0.6, 0.25, 1.0)


def env_params_from_index(wandb_id: str) -> dict:
    """The run's ``env_params`` as the nested dict a checkpoint ``config.json`` holds."""
    df = index.load()
    row = df[df.wandb_id == wandb_id]
    if row.empty:
        raise KeyError(f"{wandb_id} not in the local WandB index; run `index sync`")
    nested: dict = {}
    for key, value in row.iloc[0].items():
        if not key.startswith("env_params.") or value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, np.generic):  # np.True_ is not a bool to the parser
            value = value.item()
        *path, leaf = key.split(".")[1:]
        node = nested
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = value
    return nested


def recolour(model: mujoco.MjModel, rgba) -> None:
    """Paint the walker's teal geoms ``rgba``, leaving eyes, teeth and the floor."""
    for g in range(model.ngeom):
        r, gr, _, _ = model.geom_rgba[g]
        if model.geom_bodyid[g] != 0 and r < 0.1 and gr > 0.4:
            model.geom_rgba[g] = rgba
            model.geom_matid[g] = -1


def render_reference(wandb_id: str, spec: str, output: Path, rgba=TARGET_RGBA) -> None:
    h5_path = ARTIFACT_VIDEO_DIR / wandb_id / f"{spec}.h5"
    with h5py.File(h5_path, "r") as f:
        ref_qpos = f["reference_qpos"][:]
    n_clips = len(ref_qpos) // CLIP_FRAMES
    assert n_clips * CLIP_FRAMES == len(ref_qpos), ref_qpos.shape

    env_params = env_params_from_index(wandb_id)
    env_class, default_config_fn = eval_videos.resolve_env_class(env_params)
    cfg = eval_videos.parse_imitation_env_config(
        env_params, eval_videos.NEW_EVAL_H5, CLIP_FRAMES, default_config_fn)
    print(f"{wandb_id}: {env_class.__name__}, walker {Path(cfg.walker_xml_path).name}, "
          f"rescale {cfg.rescale_factor}")
    env = env_class(cfg)

    model = eval_videos.build_render_model(env, eval_videos.prepare_camera())
    recolour(model, rgba)
    data = mujoco.MjData(model)
    w, h = eval_videos.VIDEO_WIDTH, eval_videos.VIDEO_HEIGHT
    renderer = mujoco.Renderer(model, height=h, width=w)

    output.parent.mkdir(parents=True, exist_ok=True)
    with HQVideoWriter(output, w, h, eval_videos.FPS, pix_fmt="rgb24") as writer:
        for c in range(n_clips):
            frame = None
            for qpos in tqdm(ref_qpos[c * CLIP_FRAMES:(c + 1) * CLIP_FRAMES],
                             desc=f"clip {c}", leave=False):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=eval_videos.CAMERA_NAME)
                frame = renderer.render()
                writer.write(frame)
            if c < n_clips - 1:  # same inter-clip fade as the rollout render
                for t in range(FADE_FRAMES):
                    writer.write((frame * fade_factor(t, FADE_FRAMES)).astype(np.uint8))
    renderer.close()
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wandb_id")
    p.add_argument("--spec", default=VIDEO_SPEC_V3)
    p.add_argument("--output", type=Path, default=None,
                   help="default: eval_videos/reference_<wandb_id>_<spec>.mp4")
    args = p.parse_args()
    output = args.output or EVAL_DIR / f"reference_{args.wandb_id}_{args.spec}.mp4"
    render_reference(args.wandb_id, args.spec, output)


if __name__ == "__main__":
    main()
