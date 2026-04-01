#!/usr/bin/env python3
"""Gamepad-controlled interactive viewer for VNL checkpoints.

Loads a trained policy (MLPModularNetwork, NerveNetNetwork_v3, or
RecurrentModularNetwork) trained with reveal_targets='root_only' and lets
the user drive the rodent with a gamepad.

Controls:
    Right Stick Y   Walk forward / backward
    Right Stick X   Turn left / right
    Right Trigger   Rear up
    B button        Reset episode
    Space           Play / pause
    R               Reset episode
    , / .           Slower / faster playback  (0.25x → 0.5x → 1x → 2x)

Usage:
    .venv/bin/python vnl_experiments/tools/gamepad_viewer.py \\
        --checkpoint path/to/run/  \\
        [--max_speed 1.0] [--max_yaw_rate 3.0] [--debug_axes]

Gamepad axis mapping (constants at top of file — adjust for your controller):
    RS_X_AXIS = 2   Right Stick X
    RS_Y_AXIS = 3   Right Stick Y
    RT_AXIS   = 5   Right Trigger
    BTN_B     = 1   B button
"""

import argparse
import json
import os
import pickle
import struct
import time


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import glfw
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from flax import nnx

from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state
from vnl_experiments.envs.joystick_env import JoystickEnv
from vnl_experiments.tools.checkpoint_utils import (
    build_network,
    parse_env_config,
    parse_net_params,
)

# ---------------------------------------------------------------------------
# Gamepad axis / button constants — remap here for your controller
# Run gamepad_test.py first to find the right indices.
# ---------------------------------------------------------------------------
RS_X_AXIS   = 3   # Right Stick X   (left=-1, right=+1)
RS_Y_AXIS   = 4   # Right Stick Y   (forward=-1 on most drivers → negated below)
RT_AXIS     = 5   # Right Trigger   (-1=released, +1=fully pressed)
BTN_B       = 1   # B button index

# ---------------------------------------------------------------------------
# Viewer playback state (written from key callback, read from main loop)
# ---------------------------------------------------------------------------
_SPEEDS = [0.25, 0.5, 1.0, 2.0]

_KEY_COMMA = 44
_KEY_DOT   = 46

_S: dict = {
    "running":   True,
    "reset":     False,
    "step":      False,
    "speed_idx": 2,   # 1.0x
}


def _key_callback(key: int) -> None:
    if key == 32:                        # Space — play/pause
        _S["running"] = not _S["running"]
    elif key in (ord("R"), ord("r")):    # R — reset
        _S["reset"] = True
    elif key == _KEY_COMMA:              # , — slower
        _S["speed_idx"] = max(0, _S["speed_idx"] - 1)
    elif key == _KEY_DOT:                # . — faster
        _S["speed_idx"] = min(len(_SPEEDS) - 1, _S["speed_idx"] + 1)
    elif key == 262:                     # → (paused) — single step
        if not _S["running"]:
            _S["step"] = True

def read_joystick() -> tuple[bool, jp.ndarray, bool]:
    pointer, len = glfw.get_joystick_axes(0)
    if len == 0:
        return False, jp.zeros(3), False
    axes = np.ctypeslib.as_array(pointer, (len,)).copy()
    pointer, len = glfw.get_joystick_buttons(0)
    buttons = np.ctypeslib.as_array(pointer, (len,)).copy()

    stick_x = axes[RS_X_AXIS]
    stick_y = -axes[RS_Y_AXIS]  # negate: forward = −1 on most drivers
    rt_raw  = axes[RT_AXIS]
    rt      = (rt_raw + 1.0) / 2.0  # map [−1, 1] → [0, 1]
    reset_button = buttons[BTN_B]
    connected = True
    return connected, jp.array([stick_x, stick_y, rt]), reset_button


# ---------------------------------------------------------------------------
# HUD text overlay
# ---------------------------------------------------------------------------

def _format_overlay(
    connected: bool,
    joystick: np.ndarray,
    step: int,
    speed: float,
    paused: bool,
) -> tuple[str, str]:
    """Return (left_col, right_col) strings for viewer.set_texts()."""
    if connected:
        stick_x, stick_y, rt = float(joystick[0]), float(joystick[1]), float(joystick[2])
        left_lines = [
            "--- Gamepad ---",
            f"fwd  {stick_y:+.2f}",
            f"yaw  {stick_x:+.2f}",
            f"RT   {rt:.2f}",
        ]
    else:
        left_lines = [
            "GAMEPAD",
            "DISCONNECTED",
        ]

    status = "PAUSED" if paused else f"{speed:.2f}x"
    right_lines = [
        f"Step   {step}",
        f"Status {status}",
        "",
        "Space  play/pause",
        ",  / .  speed down/up",
        "R  or  B  reset",
    ]
    return "\n".join(left_lines), "\n".join(right_lines)


# ---------------------------------------------------------------------------
# Config / network loading helpers
# ---------------------------------------------------------------------------

def _load_run_config(checkpoint_path: str):
    """Load config.json from the run directory (parent of the step dir)."""
    run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        # checkpoint_path might already be the run dir (no step subdir given)
        cfg_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found near {checkpoint_path}")
    with open(cfg_path) as f:
        raw = json.load(f)
    return raw


def _find_step_dir(checkpoint_path: str) -> str:
    """Return the latest step_* subdir, or checkpoint_path itself if it is one."""
    if os.path.basename(checkpoint_path).startswith("step_"):
        return checkpoint_path
    step_dirs = sorted(
        d for d in os.listdir(checkpoint_path)
        if d.startswith("step_") and os.path.isdir(os.path.join(checkpoint_path, d))
    )
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directory found in {checkpoint_path}")
    return os.path.join(checkpoint_path, step_dirs[-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gamepad viewer for VNL checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to run directory (containing config.json) or a step_* subdir.",
    )
    parser.add_argument("--max_speed",    type=float, default=1.0,
                        help="Max forward speed (m/s).  Default: 1.0")
    parser.add_argument("--max_yaw_rate", type=float, default=3.0,
                        help="Max yaw rate (rad/s).  Default: 3.0")
    parser.add_argument("--max_rearing_z", type=float, default=0.15,
                        help="Max rearing z-offset (m).  Default: 0.15")
    parser.add_argument("--time_ahead",   type=float, default=0.1,
                        help="Future target time horizon (s).  Default: 0.1")
    parser.add_argument("--current_frac", type=float, default=0.1,
                        help="Fraction of future used for current target.  Default: 0.1")
    parser.add_argument("--debug_axes", action="store_true",
                        help="Print raw joystick axis/button values each frame.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    raw_cfg    = _load_run_config(args.checkpoint)
    net_params = raw_cfg.get("net_params", {})
    env_params = {
        k: v for k, v in raw_cfg.get("env_params", {}).items()
        if not k.endswith("_path")
    }

    network_class_str = str(net_params.get("network_class", ""))

    # -----------------------------------------------------------------------
    # Environment  (same physics as training checkpoint)
    # -----------------------------------------------------------------------
    print("Initialising environment…")
    config = parse_env_config(env_params)
    config.naconmax = 1024
    config.njmax    = 1024

    env = JoystickEnv(
        config,
        max_speed=args.max_speed,
        max_yaw_rate=args.max_yaw_rate,
        max_rearing_z=args.max_rearing_z,
        time_ahead=args.time_ahead,
        current_frac=args.current_frac,
    )
    print("  Environment ready.")

    # -----------------------------------------------------------------------
    # Network
    # -----------------------------------------------------------------------
    print("Building network…")
    network = build_network(net_params, env, rngs=nnx.Rngs(0))
    net_cls_name = network_class_str.rsplit(".", 1)[-1].rstrip("'>")
    print(f"  Network class: {net_cls_name}")

    # -----------------------------------------------------------------------
    # Load checkpoint weights
    # -----------------------------------------------------------------------
    step_dir = _find_step_dir(args.checkpoint)
    print(f"Loading checkpoint: {step_dir}")
    with open(os.path.join(step_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    ppo_cfg = meta.get("config")
    ppo_cfg = ppo_cfg.ppo if ppo_cfg is not None else None

    training_state = new_training_state(
        env, network, n_envs=1, seed=0,
        learning_rate=ppo_cfg.learning_rate if ppo_cfg else 1e-4,
        gradient_clipping=ppo_cfg.gradient_clipping if ppo_cfg else 1.0,
        weight_decay=ppo_cfg.weight_decay if ppo_cfg else None,
    )
    ckpt = load_checkpoint(step_dir, training_state.networks, training_state.optimizer)
    network = ckpt["training_state"].networks
    print(f"  Loaded step {ckpt['step']}")
    network.eval()

    # -----------------------------------------------------------------------
    # JIT compilation
    # -----------------------------------------------------------------------
    @nnx.jit
    def _run_network(net, net_state, obs):
        obs_b = jax.tree.map(lambda x: x[None], obs)
        net_state, out = net(net_state, obs_b)
        actions = jax.tree.map(lambda x: x[0], out.actions)
        return net_state, out.replace(actions=actions)

    _env_step  = jax.jit(lambda state, action, joy: env.step(state, action, joy))
    _env_reset = jax.jit(env.reset)

    rng = jax.random.PRNGKey(0)
    rng, sub = jax.random.split(rng)
    env_state = _env_reset(sub)
    net_state = network.initialize_state(batch_size=())

    print("JIT-compiling…")
    _net_state_tmp, _net_out_tmp = _run_network(network, net_state, env_state.obs)
    _env_state_tmp = _env_step(env_state, _net_out_tmp.actions, jp.zeros(3))
    rng, sub = jax.random.split(rng)
    env_state = _env_reset(sub)
    net_state = network.initialize_state(batch_size=())
    print("Compilation done.")

    # -----------------------------------------------------------------------
    # CPU model / data for rendering
    # -----------------------------------------------------------------------
    m: mujoco.MjModel = env.mj_model
    d: mujoco.MjData  = mujoco.MjData(m)

    def _sync_viewer(state):
        from mujoco import mjx as _mjx
        _mjx.get_data_into(d, m, state.data)

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

    ctrl_dt = float(config.ctrl_dt)
    print("\nViewer opened.")
    print("  Right Stick Y   walk forward / backward")
    print("  Right Stick X   turn")
    print("  Right Trigger   rear up")
    print("  B / R           reset episode")
    print("  Space           play / pause")
    print("  , / .           slower / faster")
    if args.debug_axes:
        print("  [debug_axes enabled — raw axis values printed below]")
    print()

    prev_btn_b = False

    try:
        with viewer:
            while viewer.is_running():
                t0 = time.perf_counter()

                # --- Read gamepad ----------------------------------------
                connected, joystick, reset = read_joystick()

                # B button — trigger reset on press (edge detect)
                if reset:
                    _S["reset"] = True

                # --- Episode reset ----------------------------------------
                if _S["reset"]:
                    _S["reset"] = False
                    rng, sub = jax.random.split(rng)
                    env_state = _env_reset(sub)
                    net_state = network.initialize_state(batch_size=())
                    _sync_viewer(env_state)
                    viewer.sync()
                    continue

                # --- Simulation step -------------------------------------
                do_step = _S["running"] or _S["step"]
                _S["step"] = False

                if do_step:
                    net_state, net_out = _run_network(network, net_state, env_state.obs)
                    env_state = _env_step(env_state, net_out.actions, joystick)
                    _sync_viewer(env_state)

                    if float(env_state.done) > 0.5:
                        rng, sub = jax.random.split(rng)
                        env_state = _env_reset(sub)
                        net_state = network.initialize_state(batch_size=())

                # --- Text overlay ----------------------------------------
                step_count = int(env_state.info.get("step", 0))
                speed = _SPEEDS[_S["speed_idx"]]
                left_txt, right_txt = _format_overlay(
                    connected, np.asarray(joystick),
                    step_count, speed, not _S["running"],
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

                # --- Real-time pacing -------------------------------------
                if _S["running"]:
                    elapsed  = time.perf_counter() - t0
                    target   = ctrl_dt / speed
                    remaining = target - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
