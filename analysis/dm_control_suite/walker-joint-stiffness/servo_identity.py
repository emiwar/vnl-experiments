"""Did every run in this cohort simulate the same actuator?

``repos.vnl_experiments.dirty`` is ``True`` on all 18 runs, so ``git_commit`` does not
identify the code that ran (README §4) -- and the servo was being actively developed on the
days these launched. The commit hash therefore cannot be the comparability argument.

This checks the thing that actually matters instead. ``train.py`` records
``actuator_mode.servo_report`` into the run config, which is the *derived* per-joint physics
-- stiffness, damping, setpoint centre and half-range -- read back off the built model.
If the servo code had changed between launches, the same ``servo_kp`` would have produced
different per-joint values. So:

1. every servo run's ``kp/servo_kp`` ratio must equal the documented ``kp_ref = gear/half``
   for that joint, and be identical across the cohort;
2. under ``servo_center = range`` every joint's usable ctrl band must be the full
   ``[-1, 1]`` (``dead_ctrl_fraction == 0``), which is why this pilot used ``range``;
3. torque runs must carry no servo parameters at all, which is the check that
   ``servo_kp = 0`` really did leave the model unpatched rather than writing a "neutral"
   servo.

This does not prove the *training* code was unchanged -- nothing available can, with a dirty
tree. It pins the one subsystem this analysis is about.

    ../.venv/bin/python analysis/dm_control_suite/walker-joint-stiffness/servo_identity.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

#: gear / half, with half = (hi - lo)/2 under `servo_center = range`. From the WalkerWalk
#: XML: gears 100/50/20 and joint ranges [-20,100] / [-150,0] / [-45,45] degrees.
EXPECTED_KP_REF = {
    "right_hip": 95.492966, "left_hip": 95.492966,
    "right_knee": 38.197186, "left_knee": 38.197186,
    "right_ankle": 25.464791, "left_ankle": 25.464791,
}
TOL = 1e-4


def main() -> None:
    df = pd.read_csv(HERE / "data.csv")
    servo = df[df.condition == "servo"]
    torque = df[df.condition == "torque"]
    failures, lines = [], []

    lines.append(f"servo runs: {len(servo)}   torque runs: {len(torque)}\n")

    lines.append("1. per-joint kp_ref = kp / servo_kp, over every servo run")
    for joint, expected in EXPECTED_KP_REF.items():
        ratios = (servo[f"kp_{joint}"] / servo["servo_kp"]).round(6).unique()
        ok = len(ratios) == 1 and abs(ratios[0] - expected) < TOL
        lines.append(f"   {joint:12s} {ratios}  expected {expected}  "
                     f"{'OK' if ok else '*** MISMATCH ***'}")
        if not ok:
            failures.append(f"{joint}: {ratios} != {expected}")

    lines.append("\n2. dead_ctrl_fraction under servo_center = range")
    for joint in EXPECTED_KP_REF:
        dead = servo[f"dead_{joint}"].unique()
        ok = len(dead) == 1 and abs(dead[0]) < TOL
        lines.append(f"   {joint:12s} {dead}  {'OK' if ok else '*** NONZERO ***'}")
        if not ok:
            failures.append(f"{joint}: dead_ctrl_fraction {dead} != 0")

    lines.append("\n3. torque runs carry no servo parameters")
    for joint in EXPECTED_KP_REF:
        ok = bool(torque[f"kp_{joint}"].isna().all())
        lines.append(f"   {joint:12s} all NaN: {ok}  {'OK' if ok else '*** PRESENT ***'}")
        if not ok:
            failures.append(f"{joint}: torque run has a servo parameter")

    lines.append("\n4. settling time (hip, the slowest joint) against the delay it faces")
    for kp, g in servo.groupby("servo_kp"):
        t_ms = g["settling_right_hip"].iloc[0] * 1000
        for delay in sorted(g.delay.unique()):
            delay_ms = delay * 25
            verdict = ("servo faster than the loop" if t_ms < delay_ms
                       else "servo SLOWER than the loop")
            lines.append(f"   servo_kp={kp:5g}  delay {delay:2d} ({delay_ms:3d} ms)   "
                         f"t_settle = {t_ms:6.1f} ms   {verdict}")

    lines.append("\n" + ("PASS: the actuator was identical in every run of this cohort."
                         if not failures else
                         "FAIL:\n  " + "\n  ".join(failures)))
    text = "\n".join(lines) + "\n"
    (HERE / "servo_identity.txt").write_text(text)
    print(text)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
