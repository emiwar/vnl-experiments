"""Turn a dm_control_suite task's torque motors into joint position servos.

The control-suite tasks ship with plain ``<motor>`` actuators, so the policy's action *is*
the normalised joint torque. ``apply_servo`` rewrites them in place into equilibrium-point
servos, ``tau = kp*(q* - q) - kv*qdot`` with ``q* = center + half*ctrl``, which makes the
actuator the only *undelayed* feedback path in a system whose policy is delayed by the
``nnx-ppo`` delay module. That contrast is the experiment.

**The rationale, the algebra and the per-env caveats live in ``servo_control.md``, next to
this file.** Read it before changing anything here or before analysing a run that used it;
the comments below are deliberately short because that document carries the argument.

Three things are worth repeating here because they are load-bearing for the code:

``servo_kp`` is dimensionless
    It is normalised per joint by ``kp_ref = gear/half``, the stiffness at which full-range
    position error saturates the actuator. Gears differ ~1000x across these tasks and 4x
    within Hopper alone, so a raw N*m/rad stiffness would mean something different on every
    joint. The normalisation is also what collapses the gain to ``gainprm[0] = servo_kp``.

``forcerange`` is ``[-1, +1]``, not ``[-gear, +gear]``
    MuJoCo clamps *actuator-space* force, before ``qfrc_actuator = moment.T @ force``. Since
    the baseline motors have ``gainprm[0] == 1`` and ``ctrl`` in ``[-1, 1]``, bounding force
    at 1 reproduces the baseline's joint-torque authority exactly. A ``[-gear, gear]`` bound
    would cap joint torque at ``gear**2`` -- 40x too strong on Hopper's hip, 20x too weak on
    Reacher -- and because the error flips sign with gear it would read as a finding rather
    than as a bug.

``kp == 0`` returns the env untouched
    No field is written and ``put_model`` is not re-run, so the torque baseline is the
    shipped env bit-for-bit rather than a servo configured to be neutral.

Why this is model surgery and not a wrapper: ``step()`` reads ``self.mjx_model`` on the inner
env, so overriding the property changes no physics; and a servo computed in a wrapper would
run once per ``ctrl_dt`` as a zero-order hold rather than as a continuous spring at
``sim_dt``. Playground sets Go1's ``Kp``/``Kd`` in ``__init__`` for the same reason.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from mujoco import mjx

#: ``servo_center`` values. ``qpos0`` puts ``ctrl=0`` at the model's reference pose;
#: ``range`` puts it at the joint-range midpoint. See servo_control.md section 3.3 -- the
#: choice is a posture prior on the initial policy, not a cosmetic one.
CENTER_MODES = ("qpos0", "range")


def _require(ok, message: str) -> None:
    """Fail with a sentence rather than a shape error three layers down."""
    if not np.all(ok):
        raise ValueError(f"servo_kp > 0 is not valid for this model: {message}")


def _check_actuators(m: mujoco.MjModel) -> None:
    """Refuse any model the gain/bias algebra does not actually describe.

    Every assumption here is one the derivation in servo_control.md section 1.1 relies on. A
    future registry entry with tendon transmission, a stateful actuator or a pre-existing
    bias (Fish, Swimmer, anything muscle-like) would silently get wrong physics otherwise.
    """
    _require(m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT,
             "every actuator must drive a joint directly (trntype=JOINT)")
    _require(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED,
             "every actuator must have gaintype=FIXED")
    _require(m.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE,
             "every actuator must start with biastype=NONE, i.e. be a plain torque motor")
    _require(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE,
             "stateful actuators (dyntype != NONE) are not supported")
    _require(m.actuator_actnum == 0, "actuators with internal state are not supported")
    _require(m.actuator_gainprm[:, 0] == 1.0,
             "every actuator must have gainprm[0] == 1; the force-parity bound assumes it")
    _require(m.actuator_ctrllimited.astype(bool), "every actuator must be ctrllimited")
    _require(m.actuator_ctrlrange[:, 0] == -1.0, "every ctrlrange must be [-1, 1]")
    _require(m.actuator_ctrlrange[:, 1] == 1.0, "every ctrlrange must be [-1, 1]")
    _require(m.actuator_gear[:, 0] > 0.0,
             "every actuator gear must be positive; kp_ref = gear/half assumes it")

    joints = m.actuator_trnid[:, 0]
    _require(np.isin(m.jnt_type[joints],
                     [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]),
             "servoed joints must be HINGE or SLIDE (BALL/FREE have a different length)")


def _setpoint_geometry(m: mujoco.MjModel, i: int, center_mode: str,
                       unlimited_half_range: float) -> tuple[float, float]:
    """``(center, half)`` for actuator ``i``: the setpoint at ``ctrl=0`` and its half-range.

    An *unlimited* joint has no range to read, so it ignores ``center_mode`` and spans
    ``qpos0 +- unlimited_half_range``; the caller must have supplied one.

    For a limited joint under ``qpos0``, the centre is **clamped into the joint range**.
    Hopper's knee has ``qpos0 = 0`` outside its own limit ``[0.087, 2.618]``, and without the
    clamp the servo would push against the limit constraint forever at ``ctrl = 0``. ``half``
    then reaches the further limit, which leaves the near side of the ``ctrl`` range mapping
    outside the joint and therefore dead -- see servo_control.md section 3.3.
    """
    j = m.actuator_trnid[i, 0]
    q0 = float(m.qpos0[m.jnt_qposadr[j]])

    if not m.jnt_limited[j]:
        if unlimited_half_range <= 0.0:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            raise ValueError(
                f"joint {name!r} is unlimited, so the servo has no setpoint range to use. "
                f"Set env.servo_unlimited_half_range explicitly (the reacher groups use pi, "
                f"which is the range reacher.py itself resets the joint over). Leaving it at "
                f"0.0 would bury a modelling choice about the plant in a default."
            )
        return q0, float(unlimited_half_range)

    lo, hi = (float(v) for v in m.jnt_range[j])
    if center_mode == "range":
        return (lo + hi) / 2.0, (hi - lo) / 2.0
    center = min(max(q0, lo), hi)
    return center, max(hi - center, center - lo)


def apply_servo(env, *, kp: float, damping_ratio: float, center: str,
                unlimited_half_range: float):
    """Rewrite ``env``'s torque motors as position servos of normalised stiffness ``kp``.

    Returns ``env``. At ``kp == 0.0`` it is returned untouched -- no field written, no
    ``put_model`` -- so the baseline arm of a sweep is the shipped env bit-for-bit.

    The re-``put_model`` at the end is mandatory, not tidying: ``reset()`` builds its data
    from ``self.mj_model`` (CPU) and then runs ``mjx.forward`` on ``self.mjx_model``
    (device). Mutating one without the other leaves the env silently inconsistent --
    initial states from the new model, physics from the old.
    """
    if kp == 0.0:
        return env
    if kp < 0.0:
        raise ValueError(f"servo_kp must be >= 0, got {kp}")
    if center not in CENTER_MODES:
        raise ValueError(f"servo_center must be one of {CENTER_MODES}, got {center!r}")

    m = env.mj_model  # the property hands back the live MjModel; its arrays are writable
    _check_actuators(m)

    for i in range(m.nu):
        c, half = _setpoint_geometry(m, i, center, unlimited_half_range)
        g = float(m.actuator_gear[i, 0])
        inertia = float(m.dof_M0[m.jnt_dofadr[m.actuator_trnid[i, 0]]])

        # kp_joint is the physical stiffness; kv follows from the damping ratio so that zeta
        # stays fixed across a stiffness sweep (servo_control.md section 2).
        kp_joint = kp * g / half
        kv_joint = 2.0 * damping_ratio * np.sqrt(kp_joint * inertia)

        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[i, 0] = kp
        m.actuator_biasprm[i, 0] = kp * c / half
        m.actuator_biasprm[i, 1] = -kp / (g * half)
        m.actuator_biasprm[i, 2] = -kv_joint / g**2

        # Equal peak joint torque with the torque baseline. The bound is in actuator space,
        # so it is 1 rather than gear -- see the module docstring.
        m.actuator_forcerange[i] = (-1.0, 1.0)
        m.actuator_forcelimited[i] = 1

    env._mjx_model = mjx.put_model(m, impl=env._config.impl)
    return env


def servo_report(env) -> dict[str, Any]:
    """Per-joint servo parameters, for the run record. ``{}`` if the env is torque-driven.

    Derived by inverting what :func:`apply_servo` wrote, rather than from a stashed copy, so
    the report cannot drift from the physics actually being simulated. ``servo_kp`` alone
    does not determine the plant -- the per-joint stiffness depends on the model -- so
    without this a reader would have to re-derive it from the XML.
    """
    m = env.mj_model
    if not np.any(m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE):
        return {}

    out: dict[str, Any] = {}
    for i in range(m.nu):
        g = float(m.actuator_gear[i, 0])
        kp = float(m.actuator_gainprm[i, 0])
        half = -kp / (g * float(m.actuator_biasprm[i, 1]))
        kp_joint = kp * g / half
        kv_joint = -float(m.actuator_biasprm[i, 2]) * g**2
        inertia = float(m.dof_M0[m.jnt_dofadr[m.actuator_trnid[i, 0]]])
        omega_n = float(np.sqrt(kp_joint / inertia))

        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        center = float(m.actuator_biasprm[i, 0]) * half / kp
        lo, hi = _usable_ctrl_band(m, i, center, half)
        out[name] = {
            "kp": kp_joint,
            "kv": kv_joint,
            "center": center,
            "half_range": half,
            "omega_n": omega_n,
            # ~4 time constants to settle, in seconds -- directly comparable with the
            # network delay k*ctrl_dt. This is the axis the sweep should be plotted on.
            "settling_time": 4.0 / omega_n,
            # The part of [-1, 1] that maps to a setpoint inside the joint's own limits.
            # Anything outside it commands a position the joint cannot reach, so that slice
            # of the policy's output is dead. Centring on qpos0 costs ~50% of it on Hopper's
            # hip and knee -- recorded per run because it is a handicap on the servo arm
            # that a reward comparison would otherwise absorb silently.
            "usable_ctrl_lo": lo,
            "usable_ctrl_hi": hi,
            "dead_ctrl_fraction": 1.0 - (hi - lo) / 2.0,
        }
    return out


def _usable_ctrl_band(m: mujoco.MjModel, i: int, center: float,
                      half: float) -> tuple[float, float]:
    """The sub-interval of ``[-1, 1]`` whose setpoint lies inside the joint's limits."""
    j = m.actuator_trnid[i, 0]
    if not m.jnt_limited[j]:
        return -1.0, 1.0
    lo, hi = (float(v) for v in m.jnt_range[j])
    return max(-1.0, (lo - center) / half), min(1.0, (hi - center) / half)
