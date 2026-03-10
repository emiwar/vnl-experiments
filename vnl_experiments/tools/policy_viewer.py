 #!/usr/bin/env python3
"""Interactive inspector for VNL checkpoint debugging.

Runs the trained policy on the ModularImitation environment using the
MuJoCo Warp backend, and displays it in the native MuJoCo passive viewer
with live per-module reward overlays and an optional ghost reference body.

Usage:
    .venv/bin/python vnl_experiments/tools/policy_viewer.py \\
        --checkpoint downloaded_checkpoints/my_run/step_0000050000 \\
        --hidden_size 512 \\
        [--clip_id 0] [--no_ghost]

Keyboard controls (in the viewer window):
    Space       Play / pause
    →           Step one frame (when paused) / next clip (when playing)
    ←           Previous clip
    R           Reset episode
    G           Toggle ghost reference body
    , / .       Slower / faster playback  (0.25x → 0.5x → 1x → 2x)
"""

import argparse
import os
import pickle
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from flax import nnx

from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state
from vnl_playground.tasks.modular_rodent.imitation import ModularImitation, default_config
from vnl_playground.tasks.modular_rodent import consts
from vnl_experiments.networks.nervenet_style import NerveNetNetwork

# ---------------------------------------------------------------------------
# Viewer global state (written from key callback, read from main loop)
# ---------------------------------------------------------------------------
_SPEEDS = [0.25, 0.5, 1.0, 2.0]

# GLFW key codes for arrow keys
_KEY_RIGHT = 262
_KEY_LEFT  = 263

_S: dict = {
    "running": True,
    "reset": False,
    "step": False,     # single-step requested (right arrow while paused)
    "ghost": True,
    "clip_delta": 0,   # +1 or -1 requested by arrow keys
    "speed_idx": 2,    # index into _SPEEDS (default 1.0x)
}


def _key_callback(key: int) -> None:
    if key == 32:                       # Space — play/pause
        _S["running"] = not _S["running"]
    elif key in (ord("R"), ord("r")):   # R — reset episode
        _S["reset"] = True
    elif key in (ord("G"), ord("g")):   # G — toggle ghost
        _S["ghost"] = not _S["ghost"]
    elif key == _KEY_LEFT:              # ← — previous clip
        _S["clip_delta"] = -1
    elif key == _KEY_RIGHT:
        if _S["running"]:               # → (playing) — next clip
            _S["clip_delta"] = +1
        else:                           # → (paused) — single step
            _S["step"] = True
    elif key == 44:                     # , — slower
        _S["speed_idx"] = max(0, _S["speed_idx"] - 1)
    elif key == 46:                     # . — faster
        _S["speed_idx"] = min(len(_SPEEDS) - 1, _S["speed_idx"] + 1)




# ---------------------------------------------------------------------------
# Ghost rendering — skeleton drawn in user_scn
# ---------------------------------------------------------------------------

def _freejoint_body_id(m: mujoco.MjModel) -> int:
    """Return the body ID that owns the freejoint (the floating base), or -1."""
    for jnt_id in range(m.njnt):
        if m.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(m.jnt_bodyid[jnt_id])
    return -1


def _update_ghost(
    scn: mujoco.MjvScene,
    m: mujoco.MjModel,
    ghost_d: mujoco.MjData,
    ref_qpos: np.ndarray,
    floating_base_id: int,
) -> None:
    """Set reference pose in ghost_d, run kinematics, draw bones in user_scn."""
    ghost_d.qpos[:] = ref_qpos
    mujoco.mj_kinematics(m, ghost_d)

    geom_idx = 0
    for body_id in range(1, m.nbody):
        if geom_idx >= scn.maxgeom - 1:
            break
        parent_id = int(m.body_parentid[body_id])
        # Skip capsules whose parent is worldbody (origin) or the floating
        # base body (CoM position, not an anatomical location).
        if parent_id == 0 or parent_id == floating_base_id:
            continue
        child_pos = ghost_d.xpos[body_id]
        parent_pos = ghost_d.xpos[parent_id]

        dist = float(np.linalg.norm(child_pos - parent_pos))
        if dist < 1e-4:
            continue

        mujoco.mjv_connector(
            scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.004,
            child_pos,
            parent_pos,
        )
        scn.geoms[geom_idx].rgba[:] = [1.0, 1.0, 0.2, 0.35]  # translucent yellow
        geom_idx += 1

    scn.ngeom = geom_idx


def _clear_ghost(scn: mujoco.MjvScene) -> None:
    scn.ngeom = 0


# ---------------------------------------------------------------------------
# Text overlay helpers
# ---------------------------------------------------------------------------

def _format_overlay(
    clip_id: int,
    clip_name: str,
    frame: int,
    n_frames: int,
    speed: float,
    ghost_on: bool,
    paused: bool,
    metrics: dict,
) -> tuple[str, str]:
    """Return (left_col, right_col) strings for viewer.set_texts()."""
    # Left column: per-module reward bars
    left_lines = ["--- Rewards ---"]
    total = 0.0
    for mod in list(consts.MODULES) + ["root"]:
        key = f"rewards/{mod}"
        if key in metrics:
            v = float(metrics[key])
            if np.isnan(v):
                continue
            total += v
            filled = min(int(v * 8), 8)
            bar = "\u2588" * filled + "\u2591" * (8 - filled)
            left_lines.append(f"{mod:<8} {v:.3f}  {bar}")
    left_lines.append(f"{'total':<8} {total:.3f}")

    # Right column: session info + key hints
    clip_label = f"{clip_name}" if clip_name else str(clip_id)
    status = "PAUSED" if paused else f"{speed:.2f}x"
    right_lines = [
        f"Clip  {clip_id} ({clip_label})",
        f"Frame {frame}/{n_frames}",
        f"Status {status}",
        f"Ghost {'ON ' if ghost_on else 'OFF'}",
        "",
        "Space  play/pause",
        "\u2192  next clip / step (paused)",
        "\u2190  prev clip",
        ",  / .  speed down / up",
        "R  reset    G  ghost",
    ]
    return "\n".join(left_lines), "\n".join(right_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VNL checkpoint inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a step checkpoint directory, e.g. downloaded_checkpoints/my_run/step_0000050000",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512,
        help="NerveNet hidden size — must match the checkpoint (default: 512)",
    )
    parser.add_argument(
        "--clip_id", type=int, default=0,
        help="Initial reference clip index (default: 0)",
    )
    parser.add_argument(
        "--no_ghost", action="store_true",
        help="Start with ghost reference body disabled",
    )
    args = parser.parse_args()

    _S["ghost"] = not args.no_ghost
    clip_id: int = args.clip_id

    # -----------------------------------------------------------------------
    # Environment  (Warp backend)
    # -----------------------------------------------------------------------
    print("Initialising environment (Warp backend)…")
    config = default_config()
    config.naconmax = 1024
    config.njmax = 1024
    config.torque_actuators = True
    config.reward_terms["root_pos_scale"] = 0.05
    config.reward_terms["limb_pos_exp_scale"] = 0.015
    config.reward_terms["joint_exp_scale"] = 0.1
    config.solver = "newton"
    config.iterations = 50
    config.ls_iterations = 50
    config.sim_dt = 0.002

    env = ModularImitation(config)

    n_clips: int = int(env.reference_clips.qpos.shape[0])
    n_frames: int = int(config.clip_length)
    print(f"  {n_clips} reference clips, {n_frames} frames each")

    # -----------------------------------------------------------------------
    # Network
    # -----------------------------------------------------------------------
    obs_sizes = {
        k: jp.squeeze(jax.tree.reduce(jp.add, o))
        for k, o in env.non_flattened_observation_size.items()
    }
    action_sizes = env.action_size  # dict {module: n_actuators}

    network = NerveNetNetwork(
        obs_sizes, action_sizes, args.hidden_size, rngs=nnx.Rngs(0)
    )

    # -----------------------------------------------------------------------
    # Load checkpoint weights
    # -----------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    # Peek at the saved TrainConfig so we can reconstruct a matching training
    # state (correct optimizer chain, learning rate, etc.) without hard-coding.
    with open(os.path.join(args.checkpoint, "metadata.pkl"), "rb") as _f:
        _meta = pickle.load(_f)
    _ppo_cfg = _meta["config"].ppo if _meta.get("config") is not None else None

    training_state = new_training_state(
        env, network, n_envs=1, seed=0,
        learning_rate=_ppo_cfg.learning_rate if _ppo_cfg else 1e-4,
        gradient_clipping=_ppo_cfg.gradient_clipping if _ppo_cfg else 1.0,
        weight_decay=_ppo_cfg.weight_decay if _ppo_cfg else None,
    )
    ckpt = load_checkpoint(args.checkpoint, training_state.networks, training_state.optimizer)
    network = ckpt["training_state"].networks
    print(f"  Loaded step {ckpt['step']}")
    print(f"  Normalizer counter: {float(network.normalizer.counter.get_value()):.0f}")
    network.eval()  # deterministic actions (mean instead of sample)

    # -----------------------------------------------------------------------
    # JIT-compile network + env step + env reset
    # -----------------------------------------------------------------------
    @nnx.jit
    def _run_network(net, net_state, obs):
        # The single-world Warp env returns unbatched obs leaves (e.g. shape
        # (3,), (1,)).  Flattener.reshape((a.shape[0], -1)) expects a leading
        # batch axis, so we add one here and squeeze it back off the actions.
        obs_b = jax.tree.map(lambda x: x[None], obs)
        net_state, out = net(net_state, obs_b)
        actions = jax.tree.map(lambda x: x[0], out.actions)
        return net_state, out.replace(actions=actions)

    _env_step  = jax.jit(env.step)
    _env_reset = jax.jit(env.reset)#, static_argnames=["clip_idx"])

    # -----------------------------------------------------------------------
    # CPU model / data for rendering
    # -----------------------------------------------------------------------
    m: mujoco.MjModel = env.mj_model
    d: mujoco.MjData = mujoco.MjData(m)
    ghost_d: mujoco.MjData = mujoco.MjData(m)
    floating_base_id: int = _freejoint_body_id(m)

    # -----------------------------------------------------------------------
    # Initial environment state
    # -----------------------------------------------------------------------
    rng = jax.random.PRNGKey(0)
    rng, sub = jax.random.split(rng)
    print(f"Resetting to clip {clip_id}…")
    env_state = _env_reset(sub, clip_idx=clip_id)
    net_state = network.initialize_state(batch_size=())

    # Eagerly trigger JIT compilation before opening the viewer so the
    # first interactive frame is not delayed by a multi-second compile.
    print("JIT-compiling network and env step (this may take a minute)…")
    net_state_tmp, net_out_tmp = _run_network(network, net_state, env_state.obs)
    env_state_tmp = _env_step(env_state, net_out_tmp.actions)
    # Discard the trial step — reset to a clean state
    rng, sub = jax.random.split(rng)
    env_state = _env_reset(sub, clip_idx=clip_id)
    net_state = network.initialize_state(batch_size=())
    print("Compilation done.")

    def _sync_viewer(env_state):
        from mujoco import mjx
        mjx.get_data_into(d, m, env_state.data)

    _sync_viewer(env_state)

    # -----------------------------------------------------------------------
    # Viewer
    # -----------------------------------------------------------------------
    viewer = mujoco.viewer.launch_passive(
        m, d,
        key_callback=_key_callback,
        show_left_ui=True,
        show_right_ui=True,
    )

    print("\nViewer opened.")
    print("  Space   play/pause")
    print("  →       next clip (playing) / step one frame (paused)")
    print("  ←       previous clip")
    print("  R       reset episode")
    print("  G       toggle ghost")
    print("  ,/.     slower/faster")
    print()

    try:
        with viewer:
            while viewer.is_running():
                t0 = time.perf_counter()

                # --- Clip change ---------------------------------------------
                if _S["clip_delta"] != 0:
                    clip_id = (clip_id + _S["clip_delta"]) % n_clips
                    _S["clip_delta"] = 0
                    _S["reset"] = True
                    print(f"Clip → {clip_id}")

                # --- Episode reset -------------------------------------------
                if _S["reset"]:
                    _S["reset"] = False
                    rng, sub = jax.random.split(rng)
                    env_state = _env_reset(sub, clip_idx=clip_id)
                    net_state = network.initialize_state(batch_size=())
                    _sync_viewer(env_state)
                    if viewer.user_scn is not None:
                        _clear_ghost(viewer.user_scn)
                    viewer.sync()
                    continue

                # --- Simulation step (running or single-step) ----------------
                do_step = _S["running"] or _S["step"]
                _S["step"] = False

                if do_step:
                    net_state, net_out = _run_network(network, net_state, env_state.obs)
                    env_state = _env_step(env_state, net_out.actions)
                    _sync_viewer(env_state)

                    # Auto-reset at episode end
                    if float(env_state.done) > 0.5:
                        _S["reset"] = True

                # --- Ghost ---------------------------------------------------
                if viewer.user_scn is not None:
                    if _S["ghost"]:
                        ref_clip = int(env_state.info.get("reference_clip", clip_id))
                        ref_frame = int(env_state.metrics.get("current_frame", 0))
                        ref_frame = min(ref_frame, n_frames - 1)
                        ref_qpos = np.array(
                            env.reference_clips.qpos[ref_clip, ref_frame]
                        )
                        _update_ghost(viewer.user_scn, m, ghost_d, ref_qpos, floating_base_id)
                    else:
                        _clear_ghost(viewer.user_scn)

                # --- Text overlay --------------------------------------------
                clip_name = ""
                if env.reference_clips.clip_names is not None:
                    clip_name = str(env.reference_clips.clip_names[clip_id])
                cur_frame = int(env_state.metrics.get("current_frame", 0))
                speed = _SPEEDS[_S["speed_idx"]]
                left_txt, right_txt = _format_overlay(
                    clip_id, clip_name, cur_frame, n_frames,
                    speed, _S["ghost"], not _S["running"], env_state.metrics,
                )
                viewer.set_texts([
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        left_txt,
                        right_txt,
                    ),
                ])

                viewer.sync()

                # --- Real-time pacing ----------------------------------------
                if _S["running"]:
                    elapsed = time.perf_counter() - t0
                    target_dt = config.ctrl_dt / speed
                    remaining = target_dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
    #except KeyboardInterrupt:
    #    pass
    finally:
        # JAX/Warp GPU contexts and the viewer's background thread prevent
        # normal Python exit; os._exit() terminates all threads immediately.
        os._exit(0)


if __name__ == "__main__":
    main()