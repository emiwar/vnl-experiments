"""What each ``body_target_frame`` lets the walker's own state into.

The whole point of ``AbsoluteImitation`` is to control which parts of the current state
reach ``task_obs``, because the delay study delays and ablates the *proprioception* stream
and then reasons about what the policy can still see. ``task_obs`` is neither delayed nor
ablated by those knobs, so a leak there is silent: nothing crashes, the numbers just mean
something other than what the report says. That already happened once --
``reference_root`` was read as making the target state-independent when it only ever
governed the ``body`` sub-key -- which is why these are tests and not a docstring.

``_get_imitation_target`` is called unbound on a stand-in that supplies exactly the six
things it reads. That keeps the tests free of MuJoCo, a GPU and the 505 MB reference-clip
file while still exercising the real method body, which is where a regression would live.
``analysis/position-control-open-loop/frame_leak.py`` is the same property measured on the
real env; if these tests and that script ever disagree, the stand-in has drifted.
"""

from __future__ import annotations

import brax.math
import jax.numpy as jp
import numpy as np
import pytest

from vnl_experiments.envs.absolute_imitation import (
    OPEN_LOOP_FRAME,
    AbsoluteImitation,
    default_config,
    validate_body_target_frame,
)

FRAMES = ("current_root", "reference_root", OPEN_LOOP_FRAME)

N_REFERENCE_FRAMES = 3
N_JOINTS = 4
BODY_NAMES = ("torso", "foot_L", "skull")


def _quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return jp.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


class _Reference:
    """The slice of clip data ``_get_imitation_target`` reads."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.root_position = jp.array(rng.normal(size=(N_REFERENCE_FRAMES, 3)))
        self.root_quaternion = jp.array(
            np.stack([np.asarray(_quat([0, 0, 1], a))
                      for a in rng.uniform(-1, 1, N_REFERENCE_FRAMES)]))
        self.joints = jp.array(rng.normal(size=(N_REFERENCE_FRAMES, N_JOINTS)))
        self._bodies = {
            name: jp.array(rng.normal(size=(N_REFERENCE_FRAMES, 3)))
            for name in BODY_NAMES
        }

    def body_xpos(self, name):
        return self._bodies[name]


class _CurrentReference:
    """The single reference frame ``_get_current_target`` returns."""

    def __init__(self, position, quaternion):
        self.root_position = position
        self.root_quaternion = quaternion


class _Env:
    """Everything ``AbsoluteImitation._get_imitation_target`` touches, and nothing else.

    ``walker`` is the walker's own root pose and body positions -- the state the tests
    perturb. ``reference`` and ``current_reference`` are the clip, which the tests hold
    fixed: a frame that "does not leak" must produce a target that is unchanged when only
    ``walker`` moves.
    """

    def __init__(self, frame: str, *, root_pos, root_quat, reference=None,
                 current_reference=None, body_pos=None):
        config = default_config()
        config.body_target_frame = frame
        self._config = config
        self._root_pos = root_pos
        self._root_quat = root_quat
        self._reference = reference or _Reference()
        self._current = current_reference or _CurrentReference(
            jp.array([0.3, -0.2, 0.05]), _quat([0, 0, 1], 0.4))
        self._body_pos = body_pos or {
            name: jp.array([0.1 * i, 0.0, 0.0]) for i, name in enumerate(BODY_NAMES)
        }

    # -- the six things the method under test reads ------------------------------------
    def _get_imitation_reference(self, data, info):
        return self._reference

    def _get_current_target(self, data, info):
        return self._current

    def root_body(self, data):
        return type("Body", (), {"xpos": self._root_pos, "xquat": self._root_quat})()

    def _get_bodies_pos(self, data, flatten=False):
        return self._body_pos


def _target(frame, *, root_pos=(0.0, 0.0, 0.0), root_quat=None, **kwargs):
    env = _Env(frame, root_pos=jp.array(root_pos),
               root_quat=_quat([0, 0, 1], 0.0) if root_quat is None else root_quat,
               **kwargs)
    out = AbsoluteImitation._get_imitation_target(env, data=None, info=None)
    return {key: np.asarray(value) for key, value in out.items()}


class TestInterfaceIsUnchanged:
    """Downstream code keys off the shape of this dict, so all three must agree on it."""

    def test_same_keys_and_shapes_under_every_frame(self):
        shapes = [
            {key: value.shape for key, value in _target(frame).items()}
            for frame in FRAMES
        ]
        assert shapes[0] == shapes[1] == shapes[2]
        assert set(shapes[0]) == {"root", "quat", "joint", "body"}

    def test_key_order_is_stable(self):
        """`Concat` and the network's input layout are built from insertion order."""
        for frame in FRAMES:
            env = _Env(frame, root_pos=jp.zeros(3), root_quat=_quat([0, 0, 1], 0.0))
            keys = list(AbsoluteImitation._get_imitation_target(env, None, None))
            assert keys == ["root", "quat", "joint", "body"]

    @pytest.mark.parametrize("frame", FRAMES)
    def test_every_documented_frame_validates(self, frame):
        assert validate_body_target_frame(frame) == frame

    @pytest.mark.parametrize("frame", ["reference", "open_loop", "", None, True])
    def test_an_unknown_frame_is_rejected(self, frame):
        """A mistyped frame must not reach training: nothing downstream would notice."""
        with pytest.raises(ValueError, match="body_target_frame"):
            validate_body_target_frame(frame)

    def test_the_default_is_unchanged(self):
        assert default_config().body_target_frame == "current_root", (
            "the default must stay current_root so an old config.json reloads to the "
            "behaviour it trained with")


class TestWhatEachFrameLeaks:
    """Move the walker, hold the clip fixed, and see which sub-keys move."""

    @staticmethod
    def _moved(frame, **perturbation):
        base = _target(frame)
        other = _target(frame, **perturbation)
        return {key: float(np.abs(other[key] - base[key]).max()) for key in base}

    @pytest.mark.parametrize("frame", FRAMES)
    def test_joint_targets_never_see_the_walker(self, frame):
        """This is what `absolute` buys, and it holds under every frame."""
        for perturbation in ({"root_pos": (0.05, 0.0, 0.0)},
                             {"root_quat": _quat([0, 0, 1], 0.3)},
                             {"body_pos": {name: jp.array([1.0, 2.0, 3.0])
                                           for name in BODY_NAMES}}):
            assert self._moved(frame, **perturbation)["joint"] == 0.0

    @pytest.mark.parametrize("frame", FRAMES)
    def test_body_positions_never_reach_the_target(self, frame):
        """The walker's own body positions are read for their names only."""
        moved = self._moved(frame, body_pos={name: jp.array([1.0, 2.0, 3.0])
                                             for name in BODY_NAMES})
        assert max(moved.values()) == 0.0

    @pytest.mark.parametrize("frame", ("current_root", "reference_root"))
    def test_the_original_frames_leak_the_root_pose(self, frame):
        """The regression guard for the two settings every existing run used.

        If this ever starts passing as "no leak", the runs in
        ``analysis/position-control-open-loop/`` stop meaning what that report says.
        """
        assert self._moved(frame, root_pos=(0.05, 0.0, 0.0))["root"] > 1e-3
        assert self._moved(frame, root_quat=_quat([0, 0, 1], 0.3))["quat"] > 1e-3

    def test_reference_root_frees_only_the_body_targets(self):
        displaced = {"root_pos": (0.05, 0.0, 0.0)}
        assert self._moved("current_root", **displaced)["body"] > 1e-3
        assert self._moved("reference_root", **displaced)["body"] == 0.0

    def test_open_loop_frees_every_sub_key(self):
        for perturbation in ({"root_pos": (0.05, 0.0, 0.0)},
                             {"root_quat": _quat([0, 0, 1], 0.3)}):
            moved = self._moved(OPEN_LOOP_FRAME, **perturbation)
            assert max(moved.values()) == 0.0, moved


class TestOpenLoopValues:
    """Not just state-free -- state-free *and* still the right quantity."""

    def test_root_is_the_references_own_displacement(self):
        reference = _Reference(seed=1)
        current = _CurrentReference(jp.array([0.3, -0.2, 0.05]),
                                    _quat([0, 0, 1], 0.4))
        got = _target(OPEN_LOOP_FRAME, root_pos=(9.0, 9.0, 9.0),
                      root_quat=_quat([0, 1, 0], 1.1),
                      reference=reference, current_reference=current)
        expected = np.stack([
            np.asarray(brax.math.rotate(pos - current.root_position,
                                        current.root_quaternion))
            for pos in reference.root_position
        ])
        np.testing.assert_allclose(got["root"], expected, atol=1e-6)

    def test_quat_is_the_references_own_rotation(self):
        reference = _Reference(seed=2)
        current = _CurrentReference(jp.array([0.0, 1.0, 0.0]),
                                    _quat([0, 0, 1], -0.7))
        got = _target(OPEN_LOOP_FRAME, root_pos=(9.0, 9.0, 9.0),
                      root_quat=_quat([1, 0, 0], 0.9),
                      reference=reference, current_reference=current)
        expected = np.stack([
            np.asarray(brax.math.relative_quat(quat, current.root_quaternion))
            for quat in reference.root_quaternion
        ])
        np.testing.assert_allclose(got["quat"], expected, atol=1e-6)

    def test_body_targets_match_reference_root(self):
        """Open loop changes the anchor for root/quat only; `body` is `reference_root`'s."""
        reference = _Reference(seed=3)
        kwargs = dict(reference=reference, root_pos=(0.4, 0.1, -0.2),
                      root_quat=_quat([0, 0, 1], 0.25))
        np.testing.assert_allclose(_target(OPEN_LOOP_FRAME, **kwargs)["body"],
                                   _target("reference_root", **kwargs)["body"],
                                   atol=1e-7)

    def test_joint_targets_are_the_raw_reference_angles(self):
        reference = _Reference(seed=4)
        for frame in FRAMES:
            np.testing.assert_allclose(
                _target(frame, reference=reference)["joint"],
                np.asarray(reference.joints), atol=1e-7)


class TestExistingFramesAreBitIdentical:
    """Adding the third value must not move a single number in the other two.

    Every run in the project trained under one of them, and several committed analyses
    compare across them, so a change here would silently invalidate stored results and
    any offline reconstruction from a checkpoint.
    """

    @pytest.mark.parametrize("frame", ("current_root", "reference_root"))
    def test_root_and_quat_are_still_taken_against_the_walker(self, frame):
        reference = _Reference(seed=5)
        root_pos = jp.array([0.2, -0.4, 0.11])
        root_quat = _quat([0, 0, 1], 0.6)
        got = _target(frame, root_pos=tuple(np.asarray(root_pos)),
                      root_quat=root_quat, reference=reference)
        expected_root = np.stack([
            np.asarray(brax.math.rotate(pos - root_pos, root_quat))
            for pos in reference.root_position])
        expected_quat = np.stack([
            np.asarray(brax.math.relative_quat(quat, root_quat))
            for quat in reference.root_quaternion])
        np.testing.assert_allclose(got["root"], expected_root, atol=1e-6)
        np.testing.assert_allclose(got["quat"], expected_quat, atol=1e-6)

    def test_current_root_body_targets_are_still_in_the_current_frame(self):
        reference = _Reference(seed=6)
        root_pos = jp.array([-0.3, 0.7, 0.02])
        root_quat = _quat([0, 0, 1], -0.35)
        got = _target("current_root", root_pos=tuple(np.asarray(root_pos)),
                      root_quat=root_quat, reference=reference)
        expected = np.stack([
            np.stack([np.asarray(brax.math.rotate(pos - root_pos, root_quat))
                      for pos in reference.body_xpos(name)])
            for name in BODY_NAMES])
        np.testing.assert_allclose(got["body"], expected, atol=1e-6)

    def test_reference_root_body_targets_are_still_per_frame_reference_relative(self):
        reference = _Reference(seed=7)
        got = _target("reference_root", root_pos=(0.9, 0.9, 0.9),
                      root_quat=_quat([0, 1, 0], 0.2), reference=reference)
        expected = np.stack([
            np.stack([np.asarray(brax.math.rotate(pos - rpos, rquat))
                      for pos, rpos, rquat in zip(reference.body_xpos(name),
                                                  reference.root_position,
                                                  reference.root_quaternion)])
            for name in BODY_NAMES])
        np.testing.assert_allclose(got["body"], expected, atol=1e-6)
