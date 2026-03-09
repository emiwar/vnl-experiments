#!/usr/bin/env python3
"""Interactive inspector for VNL checkpoint debugging.

Runs the trained policy on the ModularImitation environment using the
MuJoCo Warp backend, and displays it in the native MuJoCo passive viewer
with live per-module reward overlays and an optional ghost reference body.

Usage:
    .venv/bin/python vnl-experiments/tools/policy_viewer.py \\
        --checkpoint runs/my_run/step_0000050000 \\
        --hidden_size 256 \\
        [--clip_id 0] [--no_ghost]

Keyboard controls (in the viewer window):
    Space       Play / pause
    R           Reset episode
    G           Toggle ghost reference body
    [ / ]       Previous / next clip
    , / .       Slower / faster playback  (0.25x → 0.5x → 1x → 2x)
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np
from flax import nnx

# ---------------------------------------------------------------------------
# Path setup: allow running from any directory
# ---------------------------------------------------------------------------
#_SCRIPT_DIR = Path(__file__).resolve().parent
#_VNL_DIR = _SCRIPT_DIR.parent.parent  # …/vnl/

#for _p in ["vnl-playground", "nnx-ppo", "vnl-experiments"]:
#    _d = str(_VNL_DIR / _p)
#    if _d not in sys.path:
#        sys.path.insert(0, _d)

from vnl_playground.tasks.modular_rodent.imitation import ModularImitation, default_config
from vnl_playground.tasks.modular_rodent import consts
from vnl_experiments.networks.nervenet_style import NerveNetNetwork 

# ---------------------------------------------------------------------------
# Viewer global state (written from key callback, read from main loop)
# ---------------------------------------------------------------------------
_SPEEDS = [0.25, 0.5, 1.0, 2.0]

_S: dict = {
    "running": True,
    "reset": False,
    "ghost": True,
    "clip_delta": 0,   # +1 or -1 requested by [ / ] keys
    "speed_idx": 2,    # index into _SPEEDS (default 1.0x)
}


def _key_callback(key: int) -> None:
    if key == 32:                       # Space — play/pause
        _S["running"] = not _S["running"]
    elif key in (ord("R"), ord("r")):   # R — reset episode
        _S["reset"] = True
    elif key in (ord("G"), ord("g")):   # G — toggle ghost
        _S["ghost"] = not _S["ghost"]
    elif key == 91:                     # [ — previous clip
        _S["clip_delta"] = -1
    elif key == 93:                     # ] — next clip
        _S["clip_delta"] = +1
    elif key == 44:                     # , — slower
        _S["speed_idx"] = max(0, _S["speed_idx"] - 1)
    elif key == 46:                     # . — faster
        _S["speed_idx"] = min(len(_SPEEDS) - 1, _S["speed_idx"] + 1)


# ---------------------------------------------------------------------------
# Checkpoint loading (network weights only — no optimizer needed)
# ---------------------------------------------------------------------------

def load_network_weights(path: str, network) -> dict:
    """Restore network weights from an orbax + pickle checkpoint.

    Unlike load_checkpoint() in checkpointing.py, this does not require
    a matching optimizer instance.  Only the network parameters and the
    saved network_states carry are restored.
    """
    import orbax.checkpoint as ocp

    path = os.path.abspath(path)

    _, _, abstract_non_key = nnx.split(network, nnx.RngKey, ...)
    abstract_non_key = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), abstract_non_key
    )

    # Restore the full checkpoint as a raw nested dict (no template), then
    # extract only the leaves present in our template via tree_map_with_path.
    # This silently drops extra scalar entries (e.g. `in_features` stored as
    # NNX Variables in older Flax versions) that are absent from the current
    # network template without triggering orbax's strict tree-structure check.
    checkpointer = ocp.StandardCheckpointer()
    try:
        raw = checkpointer.restore(os.path.join(path, "networks"))
    finally:
        checkpointer.close()

    def _get(d, key_path):
        """Navigate nested dict by a JAX tree_map_with_path key path.

        The raw orbax restore is always a plain nested dict, so both DictKey
        and GetAttrKey entries are resolved via dict lookup (GetAttrKey is used
        for Variable `.value` attributes in the NNX State pytree, but in the
        raw restore these are just keys named 'value').
        """
        for entry in key_path:
            if hasattr(entry, "key"):    # DictKey
                d = d[entry.key]
            elif hasattr(entry, "name"):  # GetAttrKey (e.g. Variable.value)
                d = d[entry.name]
            else:                         # SequenceKey
                d = d[entry.idx]
        return d

    restored_non_key = jax.tree_util.tree_map_with_path(
        lambda key_path, _: np.array(_get(raw, key_path)),
        abstract_non_key,
    )

    with open(os.path.join(path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    full_net_state = nnx.merge_state(
        restored_non_key, metadata["networks_rng_key_state"]
    )
    nnx.update(network, full_net_state)

    return {
        "network_states": metadata.get("network_states", ()),
        "step": metadata.get("step", 0),
        "config": metadata.get("config", None),
    }


# ---------------------------------------------------------------------------
# Ghost rendering — skeleton drawn in user_scn
# ---------------------------------------------------------------------------

def _update_ghost(
    scn: mujoco.MjvScene,
    m: mujoco.MjModel,
    ghost_d: mujoco.MjData,
    ref_qpos: np.ndarray,
) -> None:
    """Set reference pose in ghost_d, run kinematics, draw bones in user_scn."""
    ghost_d.qpos[:] = ref_qpos
    mujoco.mj_kinematics(m, ghost_d)

    geom_idx = 0
    for body_id in range(1, m.nbody):
        if geom_idx >= scn.maxgeom - 1:
            break
        parent_id = int(m.body_parentid[body_id])
        child_pos = ghost_d.xpos[body_id]
        parent_pos = ghost_d.xpos[parent_id]

        dist = float(np.linalg.norm(child_pos - parent_pos))
        if dist < 1e-4:
            continue

        mujoco.mjv_connector(
            scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.004,                          # radius
            child_pos,#float(parent_pos[0]), float(parent_pos[1]), float(parent_pos[2]),
            parent_pos,#float(child_pos[0]),  float(child_pos[1]),  float(child_pos[2]),
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
    right_lines = [
        f"Clip  {clip_id} ({clip_label})",
        f"Frame {frame}/{n_frames}",
        f"Speed {speed:.2f}x",
        f"Ghost {'ON ' if ghost_on else 'OFF'}",
        "",
        "[ / ]  prev / next clip",
        ",  / .  speed down / up",
        "R  reset    G  ghost",
        "Space  play / pause",
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
        help="Path to a step checkpoint directory, e.g. runs/my_run/step_0000050000",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256,
        help="NerveNet hidden size — must match the checkpoint (default: 256)",
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
    config.naconmax = 512
    config.njmax = 1024
    env = ModularImitation(config)

    n_clips: int = int(env.reference_clips.qpos.shape[0])
    n_frames: int = int(config.clip_length)
    print(f"  {n_clips} reference clips, {n_frames} frames each")

    # -----------------------------------------------------------------------
    # Network
    # -----------------------------------------------------------------------

    obs_sizes = {
        k: int(jp.squeeze(jax.tree.reduce(jp.add, o)))
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
    ckpt_meta = load_network_weights(args.checkpoint, network)
    print(f"  Loaded step {ckpt_meta['step']}")

    # -----------------------------------------------------------------------
    # JIT-compile network + env step
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

    @jax.jit
    def _env_step(state, action):
        return env.step(state, action)

    # -----------------------------------------------------------------------
    # CPU model / data
    # -----------------------------------------------------------------------
    m: mujoco.MjModel = env.mj_model
    d: mujoco.MjData = mujoco.MjData(m)
    ghost_d: mujoco.MjData = mujoco.MjData(m)

    # -----------------------------------------------------------------------
    # Initial environment state
    # -----------------------------------------------------------------------
    rng = jax.random.PRNGKey(0)
    rng, sub = jax.random.split(rng)
    print(f"Resetting to clip {clip_id}…")
    env_state = env.reset(sub, clip_idx=clip_id)
    net_state = network.initialize_state(batch_size=())

    # Eagerly trigger JIT compilation before opening the viewer so the
    # first interactive frame is not delayed by a multi-second compile.
    print("JIT-compiling network and env step (this may take a minute)…")
    net_state_tmp, net_out_tmp = _run_network(network, net_state, env_state.obs)
    env_state_tmp = _env_step(env_state, net_out_tmp.actions)
    # Discard the trial step — reset to a clean state
    rng, sub = jax.random.split(rng)
    env_state = env.reset(sub, clip_idx=clip_id)
    net_state = network.initialize_state(batch_size=())
    print("Compilation done.")

    mjx.get_data_into(d, m, env_state.data)

    # -----------------------------------------------------------------------
    # Viewer
    # -----------------------------------------------------------------------
    viewer = mujoco.viewer.launch_passive(
        m, d,
        key_callback=_key_callback,
        show_left_ui=False,
        show_right_ui=False,
    )

    print("\nViewer opened.")
    print("  Space   play/pause")
    print("  R       reset episode")
    print("  G       toggle ghost")
    print("  [/]     previous/next clip")
    print("  ,/.     slower/faster")
    print()

    with viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            # --- Clip change -------------------------------------------------
            if _S["clip_delta"] != 0:
                clip_id = (clip_id + _S["clip_delta"]) % n_clips
                _S["clip_delta"] = 0
                _S["reset"] = True
                print(f"Clip → {clip_id}")

            # --- Episode reset -----------------------------------------------
            if _S["reset"]:
                _S["reset"] = False
                rng, sub = jax.random.split(rng)
                env_state = env.reset(sub, clip_idx=clip_id)
                net_state = network.initialize_state(batch_size=())
                mjx.get_data_into(d, m, env_state.data)
                if viewer.user_scn is not None:
                    _clear_ghost(viewer.user_scn)
                viewer.sync()
                continue

            # --- Simulation step ---------------------------------------------
            if _S["running"]:
                net_state, net_out = _run_network(network, net_state, env_state.obs)
                env_state = _env_step(env_state, net_out.actions)

                # Sync GPU physics state → CPU viewer data
                mjx.get_data_into(d, m, env_state.data)

                # Auto-reset at episode end
                #if float(env_state.done) > 0.5:
                #    _S["reset"] = True

            # --- Ghost -------------------------------------------------------
            if viewer.user_scn is not None:
                if _S["ghost"]:
                    ref_clip = int(env_state.info.get("reference_clip", clip_id))
                    ref_frame = int(env_state.metrics.get("current_frame", 0))
                    ref_frame = min(ref_frame, n_frames - 1)
                    ref_qpos = np.array(
                        env.reference_clips.qpos[ref_clip, ref_frame]
                    )
                    _update_ghost(viewer.user_scn, m, ghost_d, ref_qpos)
                else:
                    _clear_ghost(viewer.user_scn)

            # --- Text overlay ------------------------------------------------
            clip_name = ""
            if env.reference_clips.clip_names is not None:
                clip_name = str(env.reference_clips.clip_names[clip_id])
            cur_frame = int(env_state.metrics.get("current_frame", 0))
            speed = _SPEEDS[_S["speed_idx"]]
            left_txt, right_txt = _format_overlay(
                clip_id, clip_name, cur_frame, n_frames,
                speed, _S["ghost"], env_state.metrics,
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

            # --- Real-time pacing --------------------------------------------
            elapsed = time.perf_counter() - t0
            target_dt = config.ctrl_dt / speed
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
