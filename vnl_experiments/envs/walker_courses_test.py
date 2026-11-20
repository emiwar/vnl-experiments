"""Fixed courses: every policy walks the same ground, and training is untouched.

The property that matters is "same ground regardless of trajectory": with a course, the
blocks in the world at any moment are always a contiguous run of course entries, placed
at their absolute course positions -- whichever key reset the walker and however fast it
moved. These tests drive the walker by teleporting it (as the treadmill test does), which
exercises recycling far harder than a policy would.

``impl="jax"`` throughout, so the tests run without a working Warp install.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from vnl_experiments.envs import registry as env_registry
from vnl_experiments.envs import walker_courses as wc
from vnl_experiments.envs.walker_terrain import PARK_X


def _env(**terrain):
    spec = env_registry.get("WalkerWalkTerrain")
    cfg = spec.default_config()
    cfg.impl = "jax"
    for key, value in terrain.items():
        cfg.terrain[key] = value
    return spec.build(cfg, for_eval=True)


@pytest.fixture(scope="module")
def env():
    return _env(height=0.3, irregularity=1.0)


def _course_arrays(course):
    xs, tops = wc.stack_courses([course])
    return jp.asarray(xs[0]), jp.asarray(tops[0])


def _blocks(env, state):
    """(x, top) of every block, sorted by x."""
    pos = np.asarray(state.data.mocap_pos)
    order = np.argsort(pos[:, 0])
    return pos[order, 0], np.asarray(env.block_tops(state.data))[order]


def _walk(env, state, step, hops):
    """Teleport forward ``hops`` times and step once after each; yield every state."""
    for hop in hops:
        state = state.replace(data=state.data.replace(
            qpos=state.data.qpos.at[1].add(hop)))
        state = step(state, jp.zeros(env.action_size))
        yield state


def _assert_on_course(env, state, course):
    x, top = _blocks(env, state)
    live = x > PARK_X / 2
    assert live.all(), "a block was parked: the course ran out"
    # The blocks are a contiguous run of course entries, at the course positions.
    i0 = int(np.searchsorted(course.x, x[0]))
    np.testing.assert_array_equal(x, course.x[i0:i0 + len(x)])
    expect_top = np.where(course.top[i0:i0 + len(x)] > 1e-6,
                          course.top[i0:i0 + len(x)], -0.01)
    np.testing.assert_allclose(top, expect_top, atol=1e-6)


class TestCourses:
    def test_committed_courses_regenerate_identically(self, tmp_path):
        for made in wc.regenerate(out_dir=tmp_path):
            assert made.sha256 == wc.load_course(made.name).sha256, made.name

    def test_course_matches_the_training_distribution(self):
        c = wc.load_course("h0.3_irr1_s0")
        assert c.top.min() >= 0 and c.top.max() <= 0.6 + 1e-6
        gaps = np.diff(c.x) - 2 * c.meta["block_length"]
        assert gaps.min() >= -1e-5 and gaps.max() <= 2.0 + 1e-5
        assert c.x[-1] >= 399.0
        assert c.x[0] - c.meta["block_length"] == pytest.approx(c.meta["start_gap"])

    def test_sets(self):
        eps = wc.courses_for_episodes("h0.3_irr1", 3)
        assert [c.name for c in eps] == ["h0.3_irr1_s0", "h0.3_irr1_s1", "h0.3_irr1_s2"]
        assert [c.name for c in wc.courses_for_episodes("flat", 3)] == ["flat"] * 3
        assert len({c.sha256 for c in eps}) == 3


class TestResetOnCourse:
    def test_initial_layout_is_the_course_start(self, env):
        course = wc.load_course("h0.3_irr1_s0")
        state = jax.jit(env.reset_on_course)(jax.random.PRNGKey(0), *_course_arrays(course))
        _assert_on_course(env, state, course)

    def test_initial_pose_is_unchanged_by_the_course(self, env):
        course = wc.load_course("h0.3_irr1_s0")
        a = jax.jit(env.reset)(jax.random.PRNGKey(4))
        b = jax.jit(env.reset_on_course)(jax.random.PRNGKey(4), *_course_arrays(course))
        np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        np.testing.assert_array_equal(a.info["rng"], b.info["rng"])

    @pytest.mark.parametrize("hops", [[1.0] * 40, [3.7] * 12, [0.2] * 30 + [8.0] * 4])
    def test_blocks_stay_on_course_at_any_speed(self, env, hops):
        course = wc.load_course("h0.3_irr1_s0")
        state = jax.jit(env.reset_on_course)(jax.random.PRNGKey(1), *_course_arrays(course))
        step = jax.jit(env.step)
        for state in _walk(env, state, step, hops):
            _assert_on_course(env, state, course)

    def test_same_ground_for_different_keys(self, env):
        course = wc.load_course("h0.3_irr1_s1")
        arrays = _course_arrays(course)
        step = jax.jit(env.step)
        layouts = []
        for key in (0, 99):
            state = jax.jit(env.reset_on_course)(jax.random.PRNGKey(key), *arrays)
            for state in _walk(env, state, step, [2.5] * 10):
                pass
            # Reset poses differ, so compare at the same torso position.
            layouts.append((_blocks(env, state), float(state.data.qpos[1])))
        (b0, x0), (b1, x1) = layouts
        assert x0 == pytest.approx(x1, abs=0.5)
        # Same window of the course in view (up to a block at the recycle boundary).
        assert len(set(b0[0]) & set(b1[0])) >= env._K - 1

    def test_vmapped_episodes_walk_their_own_courses(self, env):
        courses = wc.courses_for_episodes("h0.3_irr1", 3)
        xs, tops = wc.stack_courses(courses)
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        state = jax.jit(jax.vmap(env.reset_on_course))(keys, jp.asarray(xs), jp.asarray(tops))
        step = jax.jit(jax.vmap(env.step))
        for _ in range(8):
            state = state.replace(data=state.data.replace(
                qpos=state.data.qpos.at[:, 1].add(3.0)))
            state = step(state, jp.zeros((3, env.action_size)))
        for i, course in enumerate(courses):
            one = jax.tree.map(lambda a: a[i], state)
            _assert_on_course(env, one, course)

    def test_running_off_the_end_parks_blocks(self, env):
        course = wc.load_course("h0.3_irr1_s0")
        short = wc.Course("short", course.x[:20], course.top[:20], course.meta)
        state = jax.jit(env.reset_on_course)(jax.random.PRNGKey(0), *_course_arrays(short))
        parked_at = []
        for state in _walk(env, state, jax.jit(env.step), [5.0] * 12):
            x = np.asarray(state.data.mocap_pos[:, 0])
            parked_at.append(int((x == PARK_X).sum()))
            # Once the course has started running out, never wrap back to its start.
            if parked_at[-1]:
                assert not ((x > PARK_X / 2) & (x < short.x[-1] - 15.0)).any()
        assert parked_at[-1] == env._K         # everything parked, and stays parked
        assert parked_at == sorted(parked_at)  # monotonic: nothing un-parks

    def test_bad_courses_are_rejected(self, env):
        course = wc.load_course("h0.3_irr1_s0")
        with pytest.raises(ValueError, match="float"):
            env.check_course(course.x, np.full_like(course.top, 1.5))
        with pytest.raises(ValueError, match="ascending"):
            env.check_course(course.x[::-1], course.top)
        with pytest.raises(ValueError, match="blocks"):
            env.check_course(course.x[:3], course.top[:3])
