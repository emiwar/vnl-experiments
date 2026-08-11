"""Controlled local benchmark: how much slower is the new (almost-full-collision) XML?

Run from the repo root (needs a free local GPU)::

    ../.venv/bin/python analysis/collision-model-xml/benchmark_xml.py

Why this exists
---------------
The WandB throughput numbers answer Q5 only indirectly: runs were scheduled across
A100 and H200 nodes (~1.6x apart), and even within one GPU model the node-to-node
spread is a few percent, which is the same size as the effect we are trying to
measure. This script measures the *same* thing under fully controlled conditions:
one GPU, one process, back-to-back, identical env config, only ``walker_xml_path``
changed.

It measures the **environment step alone** (no network, no PPO update), which is the
quantity the XML can actually affect. Two action regimes are timed because contact
count -- and hence solver work -- is behaviour-dependent:

  ``zeros``  : null action; the body settles/collapses onto the floor (contact-heavy)
  ``random`` : uniform random actions in [-1, 1] (flailing, also contact-heavy)

Neither reproduces a trained policy's contact statistics exactly, so read the result
as "the physics cost of the extra collision geometry", not as a prediction of the
end-to-end training slowdown (the network forward/backward pass is unchanged by the
XML and dilutes any physics difference). The ``random`` regime in particular drives the
body far outside anything a trained policy visits and overflows ``njmax=256`` (MuJoCo
Warp prints "nefc overflow"), so ``zeros`` is the more meaningful of the two.

``mean_ncon`` is measured separately on the **CPU** model (MuJoCo Warp does not expose a
usable per-env contact count through the batched state), by dropping the body from its
reset pose and stepping it with zero control.

Writes ``benchmark.csv`` (committed) with one row per (xml, control mode, action
regime) and prints a model-size summary.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd

from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.reference_clips import ReferenceClips

from vnl_experiments.envs.absolute_imitation import (
    AbsoluteImitation,
    default_config as absolute_default_config,
)

HERE = Path(__file__).resolve().parent

XMLS = {
    "old": consts.RODENT_XML_PATH,                    # rodent.xml, sparse collisions
    "new": consts.RODENT_NO_TAIL_COLLISION_XML,       # almost-full collisions
}


def build_env(xml_path, torque_actuators: bool, n_envs: int):
    """Training env config, with only the walker XML / actuator mode changed."""
    cfg = absolute_default_config()
    cfg.solver = "newton"
    cfg.reward_terms["bodies_pos"]["weight"] = 0.0
    cfg.reward_terms["joints_vel"]["weight"] = 0.0
    cfg.mujoco_impl = "warp"
    cfg.njmax = 256
    # The training config uses naconmax = 32 * 4096 for 4096 envs; keep the same
    # per-env contact budget at the smaller local batch size.
    cfg.naconmax = 32 * n_envs
    cfg.ctrl_dt = 0.01
    cfg.body_target_frame = "current_root"
    cfg.torque_actuators = torque_actuators
    cfg.walker_xml_path = xml_path
    clips = ReferenceClips(cfg.reference_data_path, cfg.clip_length, cfg.keep_clips_idx)
    train_clips, _ = clips.split()
    return AbsoluteImitation(cfg, clips=train_clips)


def model_stats(env) -> dict:
    m = env.mj_model
    contype = m.geom_contype
    conaffinity = m.geom_conaffinity
    collidable = int(np.sum((contype != 0) | (conaffinity != 0)))
    return {
        "ngeom": int(m.ngeom),
        "collidable_geoms": collidable,
        "nbody": int(m.nbody),
        "nu": int(m.nu),
        "nq": int(m.nq),
    }


def cpu_mean_ncon(env, n_steps: int = 300) -> float:
    """Mean active contact count on the CPU model, zero control, from the reset pose."""
    import mujoco

    m = env.mj_model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.ctrl[:] = 0.0
    ncon = []
    for _ in range(n_steps):
        mujoco.mj_step(m, d)
        ncon.append(d.ncon)
    return float(np.mean(ncon))


def time_env(env, n_envs: int, n_steps: int, action_mode: str, seed: int = 0):
    """Return steps_per_second for a scanned rollout of n_steps."""
    reset = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    state = reset(keys)

    nu = env.action_size

    def body(carry, key):
        st = carry
        if action_mode == "zeros":
            act = jp.zeros((n_envs, nu))
        else:
            act = jax.random.uniform(key, (n_envs, nu), minval=-1.0, maxval=1.0)
        st = jax.vmap(env.step)(st, act)
        return st, None

    @jax.jit
    def rollout(st, key):
        return jax.lax.scan(body, st, jax.random.split(key, n_steps))

    # Warm-up (compile + first-touch), then time.
    state, _ = rollout(state, jax.random.PRNGKey(1))
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    state, _ = rollout(state, jax.random.PRNGKey(2))
    jax.block_until_ready(state)
    dt = time.perf_counter() - t0

    return n_envs * n_steps / dt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-envs", type=int, default=512)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    rows = []
    for torque in (True, False):
        for action_mode in ("zeros", "random"):
            for xml_name, xml_path in XMLS.items():
                env = build_env(xml_path, torque, args.n_envs)
                stats = model_stats(env)
                mean_ncon = cpu_mean_ncon(env)
                sps = [
                    time_env(env, args.n_envs, args.n_steps, action_mode, seed=rep)
                    for rep in range(args.repeats)
                ]
                row = {
                    "xml": xml_name,
                    "walker_xml": Path(str(xml_path)).name,
                    "control_mode": "torque" if torque else "position",
                    "action_mode": action_mode,
                    "n_envs": args.n_envs,
                    "n_steps": args.n_steps,
                    "repeats": args.repeats,
                    "env_sps_median": float(np.median(sps)),
                    "env_sps_min": float(np.min(sps)),
                    "env_sps_max": float(np.max(sps)),
                    "mean_ncon": mean_ncon,
                    **stats,
                }
                rows.append(row)
                print(f"{xml_name:3s} {row['control_mode']:8s} {action_mode:6s} "
                      f"sps={row['env_sps_median']:10.0f} "
                      f"(min {row['env_sps_min']:.0f} max {row['env_sps_max']:.0f}) "
                      f"collidable_geoms={stats['collidable_geoms']} "
                      f"mean_ncon={mean_ncon:.1f}")
                del env

    df = pd.DataFrame(rows)
    ratio = (
        df.pivot_table(index=["control_mode", "action_mode"], columns="xml",
                       values="env_sps_median")
        .assign(slowdown=lambda d: d["old"] / d["new"])
    )
    print("\nSlowdown factor (old sps / new sps; >1 means the new XML is slower):")
    print(ratio)

    (HERE / "benchmark.csv").write_text(df.to_csv(index=False))
    print(f"\nWrote {HERE / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
