"""Does ``body_target_frame="reference_root"`` make ``task_obs`` state-independent?

This folder's Q2 claims a "no proprioception" run is not literally open loop, because
``task_obs`` -- which is never delayed and never ablated -- still carries an undelayed
signal about the walker's own root pose. That is a strong claim to rest on reading the
source, and the natural objection is that ``reference_root`` was introduced precisely to
remove current-state dependence from the imitation target.

It does, for the part it names. ``AbsoluteImitation._get_imitation_target`` computes four
sub-keys, and ``body_target_frame`` gates exactly one of them::

    root_targets = rotate(ref_root_pos - root_pos, root_quat)        # always current-root
    quat_targets = relative_quat(ref_root_quat, root_quat)           # always current-root
    joint_targets = reference.joints                                 # absolute
    body_targets  = current_root frame  |  reference_root frame      # <- the switch

So ``reference_root`` buys independence for ``body`` (270 of the 640 task_obs numbers) and
leaves ``root`` (15) and ``quat`` (20) exactly as they were. The class docstring says so --
"The root position/quaternion targets remain relative to the current root frame, which is
unavoidable for an egocentric representation" -- but the *config* docstring describes
``reference_root`` as "the pure target pose shape, independent of all current state", which
is true of the body targets it is talking about and easy to read as a claim about the whole
target.

Reading the source is not verification (analysis/README.md §6). This script measures it: it
builds the real env at both frame settings, perturbs one part of ``qpos`` at a time, and
reports which ``task_obs`` sub-keys move. A leak shows up as a non-zero change; independence
shows up as an exact zero.

    ../.venv/bin/python analysis/position-control-open-loop/frame_leak.py
    ../.venv/bin/python analysis/position-control-open-loop/frame_leak.py --check

Writes ``frame_leak.txt``. Needs a GPU and the reference clips (~3 min, mostly MuJoCo-warp
compilation); it builds two envs and takes no gradient, so it fits on a laptop.
"""

import argparse
import difflib
import pathlib

import jax
import numpy as np
from mujoco import mjx

import vnl_playground
from vnl_experiments.envs.absolute_imitation import AbsoluteImitation, default_config
from vnl_playground.tasks.reference_clips import ReferenceClips

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "frame_leak.txt"

XML_NAME = "rodent_no_tail_collisions.xml"

#: Two clips is enough -- the question is about the observation function, not the data. The
#: full file is 505 MB and loading all of it would be the slowest part of the script.
N_CLIPS = 2

#: qpos layout of the walker: a free joint (3 position + 4 quaternion) then the joint
#: angles. Perturbing these three slices separately is what separates "depends on where the
#: body is" from "depends on how the body is posed".
SLICES = {
    "root position (qpos[0:3])": slice(0, 3),
    "root orientation (qpos[3:7])": slice(3, 7),
    "joint angles (qpos[7:])": slice(7, None),
}

#: Metres / radians. Large enough to dwarf float noise, small enough to stay a plausible
#: tracking error -- the env terminates at 0.1 m of root error, so 0.01 m is a tenth of the
#: way to failure.
PERTURBATION = 0.01

#: The env resets the walker *onto* the reference pose, so at step 0 the root tracking error
#: is ~0 and the `root` target is ~0 with it -- which makes its sensitivity to a rotation
#: look negligible for a reason that has nothing to do with the question. The second base
#: state is displaced by this much first, so the numbers are read at a realistic drift.
DRIFT = 0.05

#: The termination thresholds these two signals feed, from the env config, quoted so the
#: report's "it sees exactly what kills it" claim has its numbers in the same file.
ROOT_TOO_FAR = 0.1
ROOT_TOO_ROTATED = 60


def build(frame: str):
    xmls = pathlib.Path(vnl_playground.__file__).parent / "tasks" / "rodent" / "xmls"
    config = default_config()
    # ConfigDict is type-strict and this field holds an epath object, not a str.
    config.walker_xml_path = type(config.walker_xml_path)(str(xmls / XML_NAME))
    config.body_target_frame = frame
    clips = ReferenceClips(config.reference_data_path, config.clip_length,
                           np.arange(N_CLIPS))
    return AbsoluteImitation(config, clips=clips)


def target(env, data, info):
    """``task_obs`` as the network would receive it, as flat numpy arrays per sub-key."""
    return {key: np.asarray(value).ravel()
            for key, value in env._get_imitation_target(data, info).items()}


def perturb(env, data, where: slice, amount: float):
    """``data`` with one slice of ``qpos`` shifted, re-run through forward kinematics.

    ``mjx.forward`` is not optional: ``xpos`` / ``xquat`` are outputs of the kinematics,
    so a ``data`` whose ``qpos`` was replaced without it still reports the *old* body
    poses and every difference below would come out as zero -- a false pass.
    """
    qpos = data.qpos.at[where].add(amount)
    return mjx.forward(env.mjx_model, data.replace(qpos=qpos))


def root_error(env, data, info):
    """The walker's actual root-position error, and the `root` target's first frame.

    An identity check rather than a sensitivity one: if ``root`` really is the root tracking
    error expressed in the current root frame, then its first reference frame must equal
    ``rotate(ref_root_pos[0] - root_pos, root_quat)``, whose norm is the very quantity the
    ``root_too_far`` termination thresholds. Showing the two agree is what turns "it changes
    when I move the walker" into "it *is* the feedback signal".
    """
    import brax.math

    reference = env._get_imitation_reference(data, info)
    root_pos = env.root_body(data).xpos
    root_quat = env.root_body(data).xquat
    expected = brax.math.rotate(reference.root_position[0] - root_pos, root_quat)
    got = np.asarray(env._get_imitation_target(data, info)["root"])[0]
    return np.asarray(expected), got


def _report(env, info, frame, base_label, base_data, lines, failures, widths_done):
    base = target(env, base_data, info)

    if not widths_done:
        total = sum(v.size for v in base.values())
        lines.append("task_obs sub-key widths (5 reference frames): "
                     + ", ".join(f"{k} {v.size}" for k, v in base.items())
                     + f"  -> {total} total")
        share = base["root"].size + base["quat"].size
        lines.append(f"root + quat are {share} of {total} numbers "
                     f"({100 * share / total:.1f} %); body is {base['body'].size}.")
        lines.append("")

    lines.append(f"body_target_frame = {frame!r}, base state: {base_label}")
    lines.append(f"  {'perturbed':<30s}" + "".join(f"{k:>12s}" for k in base))
    for label, where in SLICES.items():
        moved = target(env, perturb(env, base_data, where, PERTURBATION), info)
        deltas = {k: float(np.abs(moved[k] - base[k]).max()) for k in base}
        lines.append(f"  {label:<30s}" + "".join(f"{deltas[k]:>12.3e}" for k in base))

        # Written as expectations, so that a future change to the env that fixes -- or
        # breaks -- the leak shows up here as a FAILURE rather than as a silently
        # different table.
        if label.startswith("root position"):
            if deltas["root"] == 0.0:
                failures.append(f"{frame}: root target ignored a root-position change")
            if deltas["joint"] != 0.0:
                failures.append(f"{frame}: joint target moved with root position")
            if frame == "reference_root" and deltas["body"] != 0.0:
                failures.append("reference_root: body target still moved with root "
                                "position")
            if frame == "current_root" and deltas["body"] == 0.0:
                failures.append("current_root: body target did not move with root "
                                "position, so the two frames are not distinguished")
        if label.startswith("root orientation") and deltas["quat"] == 0.0:
            failures.append(f"{frame}: quat target ignored a root rotation")
        if label.startswith("joint angles") and any(v != 0.0 for v in deltas.values()):
            failures.append(f"{frame}: a task_obs sub-key moved with joint angles, which "
                            f"absolute targets should not")

    expected, got = root_error(env, base_data, info)
    lines.append(f"  root[frame 0] = {np.round(got, 5).tolist()};  "
                 f"rotate(ref_root - root, root_quat) = {np.round(expected, 5).tolist()};  "
                 f"|error| = {np.linalg.norm(expected):.4f} m")
    if not np.allclose(expected, got, atol=1e-6):
        failures.append(f"{frame}/{base_label}: root target is not the root-position error")
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    lines = ["Does task_obs depend on the walker's own state under each "
             "body_target_frame?", ""]
    lines.append(f"Perturbation: +{PERTURBATION} added to one qpos slice at a time, then "
                 f"mjx.forward.")
    lines.append("Reported: max |change| in each task_obs sub-key. Exactly 0 means that "
                 "sub-key does not")
    lines.append("see the perturbed part of the walker's state. The last line of each block "
                 "checks that")
    lines.append(f"`root` *is* the root-position error, against the {ROOT_TOO_FAR} m "
                 f"root_too_far threshold.")
    lines.append("")

    failures = []
    widths_done = False
    for frame in ("current_root", "reference_root"):
        env = build(frame)
        state = env.reset(jax.random.key(0))
        # Two base states. The env resets the walker onto the reference pose, so at step 0
        # the tracking error -- and hence the `root` target -- is ~0, which makes its
        # sensitivity to a rotation look negligible for a reason that has nothing to do with
        # the question. The drifted state is where a real policy lives.
        for base_label, base_data in (
                ("at reset (root error ~ 0)", state.data),
                (f"drifted +{DRIFT} m in x", perturb(env, state.data, slice(0, 1), DRIFT))):
            _report(env, state.info, frame, base_label, base_data, lines, failures,
                    widths_done)
            widths_done = True

    lines.append("Verdict")
    lines.append("  `reference_root` does what its name says, for the *body* targets: they "
                 "stop depending")
    lines.append("  on the walker's root pose (270 of the 640 task_obs numbers). It changes "
                 "nothing about")
    lines.append("  `root` and `quat` (35 numbers), which are the reference root pose "
                 "expressed relative to")
    lines.append("  the CURRENT root in both settings -- an undelayed root position and "
                 "orientation error, as")
    lines.append("  the identity check on each block's last line confirms. Neither setting "
                 "lets the joint")
    lines.append("  configuration into task_obs, which is what `absolute` buys and what "
                 "makes the delay")
    lines.append("  experiment meaningful at all.")
    lines.append("")
    lines.append("  So a `dec_use_proprioception=False` run has no joint-level feedback and "
                 "no delayed input")
    lines.append("  of any kind, but it does see, fresh every step, how far and how "
                 "wrongly-oriented its root")
    lines.append(f"  is relative to the reference -- exactly the two quantities that "
                 f"terminate the episode")
    lines.append(f"  (`root_too_far` at {ROOT_TOO_FAR} m, `root_too_rotated` at "
                 f"{ROOT_TOO_ROTATED} deg).")
    lines.append("")
    lines.append("FAILURES: " + (", ".join(failures) if failures else "none"))

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

    if failures:
        raise SystemExit("the frame-leak expectations do not hold; see above")


if __name__ == "__main__":
    main()
