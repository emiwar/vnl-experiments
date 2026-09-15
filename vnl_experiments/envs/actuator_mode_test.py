"""What `apply_servo` writes into the model, and that torque control stays untouched.

Three of these are guards against specific ways this can be wrong while still *looking*
right, because the whole experiment is a comparison between the servo arm and the torque arm:

`test_kp_zero_leaves_the_model_untouched`
    The baseline arm has to be the env playground ships, bit-for-bit. If `servo_kp = 0`
    quietly rewrote fields "neutrally", every delay curve already measured would stop being
    the thing the new curves are compared against.

`test_force_parity_with_the_torque_baseline`
    MuJoCo clamps *actuator-space* force, before the moment multiply, so the bound is 1 and
    not `gear`. With `[-gear, gear]` the servo would get `gear**2` of joint torque -- 40x
    too much on Hopper's hip and 20x too little on Reacher. Since the error flips sign with
    gear it would read as "stiffness helps on Hopper and hurts on Reacher", i.e. as a
    finding rather than as a bug. This is the test that catches it.

`test_the_device_model_is_rebuilt`
    `reset()` builds data from `mj_model` and integrates with `mjx_model`. Editing the first
    without re-running `put_model` leaves initial states from the new model and physics from
    the old, with nothing raising.

The spring law is re-derived here from envs/servo_control.md rather than imported from the
implementation, deliberately: importing the module's own geometry would make the assertion
circular and it would pass with the `gear**2` factor dropped.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from vnl_experiments.envs import actuator_mode
from vnl_experiments.envs import registry as env_registry

#: Every task the stiffness sweep can run on. Reacher needs a setpoint range for its
#: unlimited shoulder; the others read theirs off the joint limits.
TASKS = [
    ("ReacherEasy", np.pi),
    ("ReacherHard", np.pi),
    ("HopperStand", 0.0),
    ("HopperHop", 0.0),
    ("FingerTurnEasy", 0.0),
    ("FingerTurnHard", 0.0),
]

ACTUATOR_FIELDS = (
    "actuator_gaintype", "actuator_gainprm", "actuator_biastype", "actuator_biasprm",
    "actuator_forcerange", "actuator_forcelimited", "actuator_ctrlrange",
    "actuator_ctrllimited", "actuator_gear",
)


def _build(task: str, *, for_eval: bool = False, **overrides):
    spec = env_registry.get(task)
    cfg = spec.default_config()
    for key, value in overrides.items():
        cfg[key] = value
    return spec.build(cfg, for_eval=for_eval)


def _geometry(m: mujoco.MjModel, i: int, unlimited_half_range: float):
    """``(center, half)`` for actuator ``i``, re-derived from servo_control.md section 1."""
    j = m.actuator_trnid[i, 0]
    q0 = float(m.qpos0[m.jnt_qposadr[j]])
    if not m.jnt_limited[j]:
        return q0, unlimited_half_range
    lo, hi = (float(v) for v in m.jnt_range[j])
    center = min(max(q0, lo), hi)
    return center, max(hi - center, center - lo)


def _expected_gains(m: mujoco.MjModel, i: int, kp: float, zeta: float,
                    unlimited_half_range: float):
    """``(kp_joint, kv_joint, center, half)`` in joint units."""
    center, half = _geometry(m, i, unlimited_half_range)
    gear = float(m.actuator_gear[i, 0])
    inertia = float(m.dof_M0[m.jnt_dofadr[m.actuator_trnid[i, 0]]])
    kp_joint = kp * gear / half
    return kp_joint, 2.0 * zeta * np.sqrt(kp_joint * inertia), center, half


@pytest.mark.parametrize("task", [t for t, _ in TASKS])
def test_kp_zero_leaves_the_model_untouched(task: str) -> None:
    """The torque baseline must be the shipped env, not a servo configured to be neutral."""
    import mujoco_playground

    spec = env_registry.get(task)
    cfg = spec.default_config()
    assert cfg.servo_kp == 0.0

    bare = mujoco_playground.registry.load(task, config=spec.default_config()).mj_model
    built = spec.build(cfg, for_eval=True).mj_model

    for field in ACTUATOR_FIELDS:
        np.testing.assert_array_equal(
            getattr(built, field), getattr(bare, field), err_msg=f"{task}.{field}")


@pytest.mark.parametrize("task,half_range", TASKS)
def test_actuator_force_matches_the_spring_law(task: str, half_range: float) -> None:
    """`qfrc_actuator == kp*(q* - q) - kv*qdot`, with `q* = center + half*ctrl`."""
    kp, zeta = 2.0, 1.0
    env = _build(task, for_eval=True, servo_kp=kp, servo_damping_ratio=zeta,
                 servo_unlimited_half_range=half_range)
    m = env.mj_model
    d = mujoco.MjData(m)

    rng = np.random.default_rng(0)
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.3, 0.3)
        d.qpos[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
        d.qvel[m.jnt_dofadr[j]] = rng.uniform(-0.5, 0.5)
    d.ctrl[:] = rng.uniform(-1.0, 1.0, size=m.nu)
    mujoco.mj_forward(m, d)

    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        q = float(d.qpos[m.jnt_qposadr[j]])
        qdot = float(d.qvel[m.jnt_dofadr[j]])
        kp_joint, kv_joint, center, half = _expected_gains(m, i, kp, zeta, half_range)
        setpoint = center + half * float(d.ctrl[i])
        expected = kp_joint * (setpoint - q) - kv_joint * qdot

        # Only meaningful below saturation; the clamp is checked separately.
        if abs(expected) >= float(m.actuator_gear[i, 0]):
            continue
        assert float(d.qfrc_actuator[m.jnt_dofadr[j]]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("task,half_range", TASKS)
def test_force_parity_with_the_torque_baseline(task: str, half_range: float) -> None:
    """A saturated servo produces exactly the baseline's peak joint torque, `gear`.

    The bound written into the model is on actuator-space force, so it is 1 rather than
    `gear`; this asserts the *joint*-space consequence, which is the quantity the two arms
    have to share.
    """
    env = _build(task, for_eval=True, servo_kp=50.0, servo_unlimited_half_range=half_range)
    m = env.mj_model
    d = mujoco.MjData(m)
    # Full positive command against every joint parked at its own setpoint centre or below,
    # which is far past the saturation error at kp=50.
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        center, half = _geometry(m, i, half_range)
        d.qpos[m.jnt_qposadr[j]] = center - half
    d.ctrl[:] = 1.0
    mujoco.mj_forward(m, d)

    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        gear = float(m.actuator_gear[i, 0])
        assert float(d.qfrc_actuator[m.jnt_dofadr[j]]) == pytest.approx(gear, rel=1e-9)

    baseline = _build(task, for_eval=True).mj_model
    bd = mujoco.MjData(baseline)
    bd.ctrl[:] = 1.0
    mujoco.mj_forward(baseline, bd)
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        assert float(bd.qfrc_actuator[baseline.jnt_dofadr[j]]) == pytest.approx(
            float(baseline.actuator_gear[i, 0]), rel=1e-9)


def test_the_device_model_is_rebuilt() -> None:
    """A model edit without `put_model` would leave the simulated physics unchanged."""
    import jax

    torque = _build("HopperHop", for_eval=True)
    servo = _build("HopperHop", for_eval=True, servo_kp=5.0)

    key = jax.random.key(0)
    action = jax.numpy.full((torque.action_size,), 0.5)
    start_a, start_b = torque.reset(key), servo.reset(key)
    np.testing.assert_array_equal(
        np.asarray(start_a.data.qpos), np.asarray(start_b.data.qpos),
        err_msg="premise: the same key must seed the same initial state in both arms")

    a = torque.step(start_a, action)
    b = servo.step(start_b, action)
    assert not np.allclose(np.asarray(a.data.qvel), np.asarray(b.data.qvel)), (
        "the servo produced identical dynamics to torque control, so the device model "
        "was not rebuilt from the patched mj_model"
    )


def test_train_and_eval_share_the_plant() -> None:
    """Control mode is a property of the plant, so the wrappers must not change it.

    The training stack does not forward `mj_model` (RewardScalingWrapper has no such
    attribute), hence the unwrap -- which is also why `train.py` reads the servo report off
    the eval env.
    """
    train = _build("HopperHop", servo_kp=3.0)
    while not hasattr(train, "mj_model"):
        train = train.env
    evaluation = _build("HopperHop", for_eval=True, servo_kp=3.0)
    for field in ACTUATOR_FIELDS:
        np.testing.assert_array_equal(
            getattr(train.mj_model, field), getattr(evaluation.mj_model, field), field)


@pytest.mark.parametrize("task", ["ReacherEasy", "ReacherHard"])
def test_unlimited_joint_without_a_range_raises(task: str) -> None:
    """Reacher's shoulder is unlimited; guessing a range for it would be a silent choice."""
    with pytest.raises(ValueError, match="unlimited"):
        _build(task, for_eval=True, servo_kp=1.0)


def test_the_reacher_groups_supply_the_range() -> None:
    """...which is why the group files set it, so the default path still builds."""
    from vnl_experiments.config.dmc_equivalence_test import composed

    cfg = composed("env=dmc/reacher_hard", "net=delayed_mlp", "train=dmc")
    assert cfg.env.servo_unlimited_half_range == pytest.approx(np.pi)


def test_hopper_knee_centre_is_clamped_into_range() -> None:
    """Hopper's knee has `qpos0 = 0` outside its own limit `[0.087, 2.618]`.

    Unclamped, the servo would push against the limit constraint forever at `ctrl = 0`.
    """
    env = _build("HopperHop", for_eval=True, servo_kp=1.0)
    m = env.mj_model
    knee = next(i for i in range(m.nu)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == "knee")
    j = m.actuator_trnid[knee, 0]
    lo, hi = m.jnt_range[j]
    assert float(m.qpos0[m.jnt_qposadr[j]]) < lo, "premise: qpos0 is below the limit"

    # `servo_report` recovers the centre by inverting the written fields, so it can land a
    # ULP outside the range it was clamped into; the clamp itself is what is being asserted.
    center = actuator_mode.servo_report(env)["knee"]["center"]
    assert center == pytest.approx(float(lo))
    assert float(lo) == pytest.approx(center) and center < float(hi)


def test_servo_report_is_empty_for_torque_control() -> None:
    assert actuator_mode.servo_report(_build("HopperHop", for_eval=True)) == {}
