"""What ``torque_actuators=False`` actually makes the action mean.

The mechanism this folder's conclusions rest on is a claim about the model, not about the
runs: under position control the policy's action *is* a target joint configuration, and the
reference joint angles it is asked to imitate are the same quantity under a per-joint
affine map. If that is true, a large part of the imitation task is a static map from an
input the policy already has, and needs no feedback -- which is what Q2 observes.

This script checks it against the XML the runs used. For each actuator it compares the
force law MuJoCo will apply,

    force = gainprm0 * act + biasprm0 + biasprm1 * qpos + biasprm2 * qvel

(``gaintype=FIXED``, ``biastype=AFFINE``) with the servo form

    force = kp * (offset + scale * ctrl - qpos),   kp = -biasprm1

and reports ``scale`` / ``offset`` against the joint's own half-range and midpoint. Where
they agree, ``ctrl = -1 .. +1`` sweeps exactly that joint's full range. It then applies the
same conversion ``tasks/rodent/base.py`` does for ``torque_actuators=True`` and shows what
that leaves.

    ../.venv/bin/python analysis/position-control-open-loop/actuator_map.py
    ../.venv/bin/python analysis/position-control-open-loop/actuator_map.py --check

Writes ``actuator_map.txt``; ``--check`` diffs against the committed copy. No GPU, no run
data -- it reads the XML, which is in the vnl-playground working copy.
"""

import argparse
import difflib
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "actuator_map.txt"

#: The body every run in this folder used, as `env_params.walker_xml_path` records it. The
#: local path is resolved from the installed vnl_playground package rather than the
#: cluster path stored in the config -- same file, and the mismatch between the two is the
#: 2026-08-18 trap in analysis/README.md.
XML_NAME = "rodent_no_tail_collisions.xml"

#: Tolerance on the servo/joint-range match, in radians. Exact equality is what the
#: agreeing actuators actually show; this only guards float representation.
TOL = 1e-6


def walker_xml() -> Path:
    import vnl_playground

    return (Path(vnl_playground.__file__).parent
            / "tasks" / "rodent" / "xmls" / XML_NAME)


def model(*, torque: bool) -> mujoco.MjModel:
    """The compiled walker, with `add_rodent`'s torque conversion applied or not.

    The conversion is reproduced here rather than imported because `add_rodent` also
    attaches the arena and a freejoint, which would renumber every actuator in the report
    for no benefit. It is four lines copied verbatim from `tasks/rodent/base.py`, and the
    copy was checked against the real thing: building through `RodentEnv.add_rodent`
    gives the same `actuator_gainprm` (20 on the six spine/neck actuators, the joint's own
    forcerange max elsewhere) and `biasprm = 0` throughout.

    Note also that `rodent.xml` and `rodent_no_tail_collisions.xml` have *identical*
    actuator, joint-range and damping tables -- they differ in the collision flags of 53
    geoms and in body masses -- so everything below holds for the previous-setup cohort
    too.
    """
    spec = mujoco.MjSpec.from_file(str(walker_xml()))
    if torque:
        for actuator in spec.actuators:
            if actuator.forcerange.size >= 2:
                actuator.gainprm[0] = actuator.forcerange[1]
            actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
            actuator.biasprm = np.zeros((10, 1))
    return spec.compile()


def rows(m: mujoco.MjModel) -> list[dict]:
    joint = m.actuator_trnid[:, 0]
    ranges = m.jnt_range[joint]
    out = []
    for i in range(m.nu):
        kp = -m.actuator_biasprm[i, 1]
        gain = m.actuator_gainprm[i, 0]
        half = (ranges[i, 1] - ranges[i, 0]) / 2
        mid = ranges[i].mean()
        out.append({
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "kp": float(kp),
            "kv": float(-m.actuator_biasprm[i, 2]),
            "scale": float(gain / kp) if kp else float("nan"),
            "offset": float(m.actuator_biasprm[i, 0] / kp) if kp else float("nan"),
            "half_range": float(half),
            "midpoint": float(mid),
            "gain": float(gain),
            "forcerange_max": float(m.actuator_forcerange[i, 1]),
            "full_range": float(2 * half),
            "dyn_tau": float(m.actuator_dynprm[i, 0]),
            "ctrlrange": tuple(float(v) for v in m.actuator_ctrlrange[i]),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    position, torque = model(torque=False), model(torque=True)
    pos_rows, torque_rows = rows(position), rows(torque)

    lines = [f"Actuator semantics of {XML_NAME}", ""]
    lines.append(f"nu = {position.nu} actuators, ctrlrange = "
                 f"{pos_rows[0]['ctrlrange']} on all of them, first-order actuator filter "
                 f"tau = {pos_rows[0]['dyn_tau']:.3f} s")
    lines.append("(the filter is dyntype=FILTER in the XML and the torque conversion does "
                 "not touch it, so both")
    lines.append(" control modes carry the same ~40 ms actuation lag)")
    lines.append("")

    lines.append("torque_actuators=False (position control): "
                 "force = kp * (offset + scale*ctrl - qpos)")
    lines.append(f"{'actuator':<20s} {'kp':>7s} {'scale':>8s} {'half_rng':>9s} "
                 f"{'offset':>8s} {'midpoint':>9s}  match")
    exact = 0
    for row in pos_rows:
        ok = (abs(row["scale"] - row["half_range"]) < TOL
              and abs(row["offset"] - row["midpoint"]) < TOL)
        exact += ok
        lines.append(f"{row['name']:<20s} {row['kp']:>7.2f} {row['scale']:>8.3f} "
                     f"{row['half_range']:>9.3f} {row['offset']:>8.3f} "
                     f"{row['midpoint']:>9.3f}  {'exact' if ok else 'approx'}")
    lines.append("")
    lines.append(f"{exact} of {position.nu} actuators map ctrl = -1..+1 onto exactly that "
                 f"joint's own range.")

    approx = [r for r in pos_rows
              if not (abs(r["scale"] - r["half_range"]) < TOL
                      and abs(r["offset"] - r["midpoint"]) < TOL)]
    if approx:
        worst = max(abs(r["scale"] - r["half_range"]) for r in approx)
        lines.append(f"The other {len(approx)} ({', '.join(r['name'] for r in approx)}) "
                     f"still map ctrl affinely onto")
        lines.append(f"an angle, just not onto the full range: max |scale - half_range| = "
                     f"{worst:.3f} rad.")
    lines.append("")
    lines.append(f"kv (velocity feedback) = "
                 f"{sorted({round(r['kv'], 6) for r in pos_rows})} -- position feedback "
                 f"only; damping comes from the joints.")
    lines.append(f"kp values used: {sorted({round(r['kp'], 4) for r in pos_rows})}. "
                 f"The 6 spine/neck joints get kp = 20;")
    lines.append("every limb joint gets kp <= 0.6, i.e. a soft spring rather than a stiff "
                 "servo.")
    lines.append("")

    lines.append("torque_actuators=True: force = gain * ctrl, no state feedback")
    lines.append("(add_rodent sets gainprm0 = forcerange max and biastype = NONE)")
    biases = sorted({round(r["kp"], 4) for r in torque_rows})
    lines.append(f"  bias terms after conversion: {biases}  "
                 f"(0 = biastype NONE, so no qpos/qvel term)")
    lines.append("")

    lines.append("Force authority is NOT matched between the modes. The XML's forcerange "
                 "-- which is what")
    lines.append("the torque conversion turns into the gain -- was not chosen to match "
                 "kp * range, so each")
    lines.append("joint's peak commandable torque changes when the mode changes:")
    lines.append(f"{'actuator':<20s} {'kp*range':>9s} {'torque gain':>12s} {'ratio':>7s}")
    ratios = []
    for row in pos_rows:
        span = row["kp"] * row["full_range"]
        ratio = row["forcerange_max"] / span if span else float("nan")
        ratios.append(ratio)
        lines.append(f"{row['name']:<20s} {span:>9.3f} {row['forcerange_max']:>12.3f} "
                     f"{ratio:>7.2f}")
    finite = np.array([r for r in ratios if np.isfinite(r)])
    lines.append("")
    lines.append(f"ratio (torque gain) / (kp * range): min {finite.min():.2f}, "
                 f"median {np.median(finite):.2f}, max {finite.max():.2f}.")
    lines.append("Position control is the *stiffer* mode at large error on most joints "
                 "(median ratio < 1),")
    lines.append("but its torque is proportional to the tracking error, so it commands "
                 "little near the target,")
    lines.append("while torque mode can command its full gain in any state. The two are "
                 "therefore not a clean")
    lines.append("feedback-loop-only contrast, and report.md carries this as a caveat.")
    lines.append("")
    lines.append("Consequence for this analysis: under position control the action is a "
                 "target joint")
    lines.append("configuration in normalised joint coordinates, and task_obs carries the "
                 "reference joint")
    lines.append("angles (absolute, under body_target_frame=reference_root). The map from "
                 "that part of the")
    lines.append("input to a good action is static and needs no feedback. Under torque "
                 "control no such map")
    lines.append("exists: the torque that realises a target pose depends on the current "
                 "state.")

    text = "\n".join(lines) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("\n".join(difflib.unified_diff(
                (OUT.read_text() if OUT.exists() else "").splitlines(),
                text.splitlines(), fromfile=f"committed/{OUT.name}",
                tofile=f"rebuilt/{OUT.name}", lineterm="")))
            raise SystemExit(1)
        print(f"CHECK: {OUT.name} unchanged")
    else:
        OUT.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
