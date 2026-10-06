"""Illustrative stills of the six tasks, transparent background, no floor.

Renders each task's MuJoCo model (from the same ``mujoco_playground`` registry the
training runs load) in a hand-set pose that shows what the task is, with the floor/walls
hidden and every pixel that hits no geom made transparent. Supersampled 2x and
downsampled with premultiplied alpha so the silhouette edges stay smooth.

    ../.venv/bin/python analysis/dm_control_suite/three-arm-overview/illustrations/render.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
# Loading a playground env imports JAX; keep it off the GPU (rendering is CPU MuJoCo).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco_playground
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

SIZE = 1000          # output pixels (square)
SUPERSAMPLE = 2
HIDDEN_GROUP = 5     # floors and walls are moved here and this group is switched off


@dataclass
class Shot:
    task: str
    out: str
    lookat: tuple
    distance: float
    azimuth: float
    elevation: float
    qpos: dict = field(default_factory=dict)          # joint name -> value
    geom_pos: dict = field(default_factory=dict)      # geom name -> xyz (e.g. targets)
    mocap_pos: dict = field(default_factory=dict)     # mocap body name -> xyz
    # mocap body name -> (geom name, xyz offset): placed relative to a posed geom
    mocap_near: dict = field(default_factory=dict)
    tint: dict = field(default_factory=dict)          # geom-name prefix -> rgb multiplier


def set_pose(model, data, shot: Shot) -> None:
    mujoco.mj_resetData(model, data)
    for name, value in shot.qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, (shot.task, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    for name, pos in shot.geom_pos.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, (shot.task, name)
        model.geom_pos[gid] = pos
    for name, pos in shot.mocap_pos.items():
        data.mocap_pos[model.body(name).mocapid[0]] = pos
    for prefix, factor in shot.tint.items():
        for g in range(model.ngeom):
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix):
                mat = model.geom_matid[g]
                base = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
                model.geom_rgba[g] = np.r_[base[:3] * factor, base[3]]
                model.geom_matid[g] = -1
    mujoco.mj_forward(model, data)
    for name, (geom, offset) in shot.mocap_near.items():
        data.mocap_pos[model.body(name).mocapid[0]] = data.geom(geom).xpos + offset
    mujoco.mj_forward(model, data)


def render(shot: Shot) -> Image.Image:
    env = mujoco_playground.registry.load(shot.task)
    model = env.mj_model
    hidden = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    model.geom_group[hidden] = HIDDEN_GROUP

    px = SIZE * SUPERSAMPLE
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, px)
    model.vis.global_.offheight = max(model.vis.global_.offheight, px)
    data = mujoco.MjData(model)
    set_pose(model, data, shot)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = shot.lookat
    cam.distance = shot.distance
    cam.azimuth = shot.azimuth
    cam.elevation = shot.elevation

    opt = mujoco.MjvOption()
    opt.geomgroup[HIDDEN_GROUP] = 0
    opt.sitegroup[:] = 0

    with mujoco.Renderer(model, px, px) as r:
        r.update_scene(data, camera=cam, scene_option=opt)
        rgb = r.render().astype(np.float32) / 255.0
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam, scene_option=opt)
        seg = r.render()[..., 0]
    alpha = (seg >= 0).astype(np.float32)

    # Premultiplied-alpha box downsample.
    def down(a):
        return a.reshape(SIZE, SUPERSAMPLE, SIZE, SUPERSAMPLE, -1).mean(axis=(1, 3))
    a = down(alpha[..., None])
    c = down(rgb * alpha[..., None]) / np.maximum(a, 1e-6)
    rgba = np.concatenate([c, a], axis=-1)
    img = Image.fromarray((np.clip(rgba, 0, 1) * 255).round().astype(np.uint8))
    return img.crop(img.getbbox())  # trim transparent margins


SHOTS = [
    # Pole partway through the swing-up (hinge 0 = upright).
    Shot("CartpoleSwingup", "cartpole.png", lookat=(0, 0, 1.0), distance=4.0,
         azimuth=90, elevation=-5, qpos={"slider": 0.2, "hinge_1": 0.9}),
    # Top-down: arm reaching towards the red target.
    Shot("ReacherHard", "reacher.png", lookat=(0, 0, 0), distance=0.8,
         azimuth=90, elevation=-90, qpos={"shoulder": 0.6, "wrist": 1.3},
         mocap_near={"target": ("finger", (0.05, 0.06, 0.0))}),
    # Ball swinging up towards the cup on its string.
    Shot("BallInCup", "ball_in_cup.png", lookat=(0, 0, 0.3), distance=1.6,
         azimuth=90, elevation=-5, qpos={"ball_x": 0.22, "ball_z": 0.12}),
    # Mid-bound.
    Shot("CheetahRun", "cheetah.png", lookat=(0, 0, 0.5), distance=2.5,
         azimuth=90, elevation=-5,
         qpos={"rooty": -0.1, "bthigh": 0.6, "bshin": -0.5, "bfoot": -0.3,
               "fthigh": -0.7, "fshin": 0.7, "ffoot": 0.3}),
    # Mid-stride; left leg darkened so the two legs read apart.
    Shot("WalkerWalk", "walker.png", lookat=(0, 0, 1.0), distance=3.0,
         azimuth=90, elevation=-5,
         qpos={"rooty": 0.05, "right_hip": 0.55, "right_knee": -0.25, "right_ankle": 0.1,
               "left_hip": -0.3, "left_knee": -0.9, "left_ankle": 0.4},
         tint={"left_": 0.7}),
    # Mid-stride, arms swinging.
    Shot("HumanoidWalk", "humanoid.png", lookat=(0, 0, 1.0), distance=3.0,
         azimuth=120, elevation=-5,
         qpos={"right_hip_y": -0.6, "right_knee": -0.3, "left_hip_y": 0.25,
               "left_knee": -0.9, "left_ankle_y": 0.3,
               "right_shoulder1": 0.8, "right_shoulder2": -0.8, "right_elbow": -0.3,
               "left_shoulder1": -0.8, "left_shoulder2": 0.8, "left_elbow": -0.3,
               "abdomen_y": -0.1}),
]


def main() -> None:
    for shot in SHOTS:
        img = render(shot)
        img.save(HERE / shot.out)
        print(f"{shot.out}: {img.size}")


if __name__ == "__main__":
    main()
