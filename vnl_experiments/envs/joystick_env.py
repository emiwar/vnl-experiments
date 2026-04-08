"""Joystick-controlled rodent environment for interactive visualization.

Same physics and proprioception structure as ModularImitation_v4, but root
observations are synthesized from gamepad input rather than reference clips.
Intended for use with gamepad_viewer.py, not for training.

The obs structure matches what networks trained with reveal_targets='root_only'
expect:
  - Body modules: {"proprioception": {...}}   (no target keys)
  - Root module:  {"current_target": {"pos", "xaxis"}, "future_target": {...}}

The network's internal _filter_obs extracts obs[k]["proprioception"] for
non-root modules, so omitting target keys is correct and sufficient.
"""

import collections
from typing import Optional

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.modular_rodent import base as modular_base
from vnl_playground.tasks.modular_rodent import consts
from vnl_playground.tasks.modular_rodent.imitation_v4 import (
    ModularImitation_v4,
    default_config,
)


class JoystickEnv(ModularImitation_v4):
    """Joystick-controlled rodent environment.

    Inherits ModularImitation_v4's proprioception so that obs sizes exactly
    match checkpoints trained on that environment.  Reference clips are not
    loaded; reset starts from the model's default qpos (T-pose).

    Args:
        config: Environment config (physics params).  Defaults to
            default_config() from ModularImitation_v4.  Clip-related fields
            are ignored.
        max_speed: Forward speed at full RS-Y deflection (m/s).
        max_yaw_rate: Yaw rate at full RS-X deflection (rad/s).
        time_ahead: How far ahead the "future" target is placed (s).
        current_frac: current_target = this fraction of future_target.
        max_rearing_z: Target z-offset at RT=1 (m).
    """

    def __init__(
        self,
        config=None,
        *,
        max_speed: float = 0.2,
        max_yaw_rate: float = 3.0,
        time_ahead: float = 0.1,
        current_frac: float = 0.1,
        min_rearing_z: float = 0.05,
        max_rearing_z: float = 0.06,
        xaxis_bias: float = 1.0,
        root_command_style: str = "future_pos_only"
    ) -> None:
        if config is None:
            config = default_config()
        # Skip ModularImitation_v4.__init__ which loads reference clips.
        # Call its parent (ModularRodentEnv) directly instead.
        modular_base.ModularRodentEnv.__init__(self, config)
        self.compile()

        self._max_speed = max_speed
        self._max_yaw_rate = max_yaw_rate
        self._time_ahead = time_ahead
        self._current_frac = current_frac
        self._min_rearing_z = min_rearing_z
        self._max_rearing_z = max_rearing_z
        self._xaxis_bias = xaxis_bias
        self.root_command_style = root_command_style

    # -------------------------------------------------------------------------
    # Reset / step
    # -------------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset to default qpos (T-pose), zero velocity."""
        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            njmax=self._config.njmax,
            naconmax=self._config.naconmax,
        )
        data = data.replace(
            qpos=jp.array(self.mj_model.qpos0),
            qvel=jp.zeros(self.mj_model.nv),
        )
        data = mjx.forward(self.mjx_model, data)
        obs = self._compute_obs(data, jp.zeros(3))
        reward = {m: jp.zeros(()) for m in consts.MODULES + ["root"]}
        return mjx_env.State(
            data, obs, reward,
            jp.zeros(()),   # done
            {},             # metrics
            {"step": jp.zeros((), dtype=jp.int32)},
        )

    def step(
        self,
        state: mjx_env.State,
        action: dict,
        joystick: Optional[jax.Array] = None,
    ) -> mjx_env.State:
        """Physics step with joystick-derived root target.

        Args:
            state: Current environment state.
            action: Per-module action dict (same structure as ModularImitation_v4).
            joystick: Shape (3,) array [stick_x, stick_y, rt].
                stick_x: RS-X yaw rate command (right=+1).
                stick_y: RS-Y forward velocity command (forward=+1).
                rt:      RT rearing command (0=none, 1=max).
        """
        if joystick is None:
            joystick = jp.zeros(3)
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        ctrl = self._action_to_ctrl(action)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, n_steps)

        done = (self.root_pos(data)[2] < self._config.min_torso_z).astype(float)
        obs = self._compute_obs(data, joystick)
        reward = {m: jp.zeros(()) for m in consts.MODULES + ["root"]}
        return state.replace(
            data=data, obs=obs, reward=reward, done=done,
            metrics={},
            info={"step": state.info["step"] + 1},
        )

    # -------------------------------------------------------------------------
    # Observation helpers
    # -------------------------------------------------------------------------

    def _compute_obs(self, data: mjx.Data, joystick: jax.Array) -> dict:
        """Build the observation dict expected by the network.

        Non-root modules receive only proprioception (no target keys). The
        network's _filter_obs will extract obs[k]["proprioception"] for these
        modules, which works correctly whether target keys are present or not.

        Root receives only current_target and future_target (no proprioception),
        mirroring the training environment structure.
        """
        prop = self._get_modular_proprioception(data)
        root_cur, root_fut = self._joystick_root_target(data, joystick)

        obs = collections.OrderedDict()
        for module in consts.MODULES:
            obs[module] = collections.OrderedDict(proprioception=prop[module])
        obs["root"] = collections.OrderedDict(
            current_target=root_cur,
            future_target=root_fut,
        )
        return obs

    def _joystick_root_target(
        self, data: mjx.Data, joystick: jax.Array,
    ) -> tuple[dict, dict]:
        """Convert gamepad input to root targets in the current torso frame.

        The joystick positions an imaginary target point in the horizontal plane:
          - xy: stick deflection in the yaw-aligned frame (torso heading projected
                onto the ground plane, pitch and roll ignored).  The offset is
                rotated to world frame by yaw only, then to torso-local frame by
                the full torso_xmat.T.
          - z:  RT sets an absolute height above ground; converted to a relative
                offset from the current torso height in the same transform.

        xaxis is computed as the direction from a reference point xaxis_bias metres
        behind the rat to the target point.  A larger bias means less aggressive
        turning for the same lateral stick deflection.

        Returns:
            (current_target, future_target), each {"pos": (3,), "xaxis": (3,)}.
        """
        stick_x = joystick[0]
        stick_y = joystick[1]
        rt = joystick[2]

        # Current torso pose
        torso_xpos = data.bind(self.mjx_model, self._spec.body("torso")).xpos
        torso_xmat = data.bind(self.mjx_model, self._spec.body("torso")).xmat

        # Yaw angle from torso x-axis projected onto the horizontal plane.
        torso_x_world = torso_xmat[:, 0]
        yaw = jp.arctan2(torso_x_world[1], torso_x_world[0])
        cos_yaw, sin_yaw = jp.cos(yaw), jp.sin(yaw)

        # Stick commands in the yaw-aligned horizontal frame (ignores pitch/roll).
        fwd = (stick_y + jp.abs(stick_x)) * self._max_speed * self._time_ahead
        lat = -0.5 * stick_x * self._max_speed * self._time_ahead
        target_abs_z = self._min_rearing_z + rt * (self._max_rearing_z - self._min_rearing_z)

        # Rotate horizontal offset to world frame (yaw only), include absolute z.
        delta_world = jp.array([
            cos_yaw * fwd - sin_yaw * lat,
            sin_yaw * fwd + cos_yaw * lat,
            target_abs_z - torso_xpos[2],
        ])

        # Transform to torso-local frame (full rotation, so pitch/roll are
        # applied here but the commanded xy is purely in the horizontal plane).
        future_pos = torso_xmat.T @ delta_world

        if self.root_command_style == "full_root":
            current_pos = self._current_frac * future_pos

            # xaxis: direction from xaxis_bias metres behind the rat to the target.
            # Larger bias → smaller turning angle for same lateral deflection.
            # z is included naturally, so rearing tilts the commanded heading.
            future_xaxis = future_pos + jp.array([self._xaxis_bias, 0.0, 0.0])
            future_xaxis = future_xaxis / jp.linalg.norm(future_xaxis)

            current_xaxis = jp.array([1.0, 0.0, 0.0])

            cur = {"pos": current_pos, "xaxis": current_xaxis}
            fut = {"pos": future_pos, "xaxis": future_xaxis}
            return cur, fut
        elif self.root_command_style == "future_pos_only":
            return {}, {"pos": future_pos}
