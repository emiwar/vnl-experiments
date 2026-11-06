"""WalkerWalkTerrain: flat is plain WalkerWalk, layouts resample, the treadmill keeps up.

`test_flat_terrain_is_plain_walker_walk` is the one the delay comparison rests on. The
terrain cohort is read against the WalkerWalk cohort, so at ``terrain.height = 0`` the env
has to be the shipped walker bit-for-bit: same observation (the blocks must not leak into
the ``xmat`` slice), same reward, same initial pose for the same key, and no extra
contacts from blocks sitting flush with the floor.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground._src.dm_control_suite import walker

from vnl_experiments.envs import registry as env_registry
from vnl_experiments.envs import walker_terrain


def _env(**terrain):
    """The eval (unwrapped) env, with ``terrain.<key>`` overrides."""
    spec = env_registry.get("WalkerWalkTerrain")
    cfg = spec.default_config()
    for key, value in terrain.items():
        cfg.terrain[key] = value
    return spec.build(cfg, for_eval=True)


def _rollout(env, key, n):
    state = jax.jit(env.reset)(key)
    step = jax.jit(env.step)
    actions = jax.random.uniform(jax.random.PRNGKey(7), (n, env.action_size),
                                 minval=-1, maxval=1)
    obs, rewards = [state.obs], [state.reward]
    for a in actions:
        state = step(state, a)
        obs.append(state.obs)
        rewards.append(state.reward)
    return np.asarray(obs), np.asarray(rewards), state


def test_flat_terrain_is_plain_walker_walk():
    flat = _env(height=0.0)
    base = walker.PlanarWalker(walker.WALK_SPEED)
    assert flat.observation_size == base.observation_size
    o1, r1, _ = _rollout(flat, jax.random.PRNGKey(3), 60)
    o0, r0, _ = _rollout(base, jax.random.PRNGKey(3), 60)
    np.testing.assert_array_equal(o1, o0)
    np.testing.assert_array_equal(r1, r0)


def test_blocks_collide_with_the_walker_only():
    m = _env().mj_model
    block = [m.geom(f"block{i}").id for i in range(m.nmocap)]
    walker_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0,)
                    and g not in block]
    floor = m.geom("floor").id

    def collide(a, b):
        return bool((m.geom_contype[a] & m.geom_conaffinity[b])
                    or (m.geom_contype[b] & m.geom_conaffinity[a]))

    assert all(collide(b, w) for b in block for w in walker_geoms)
    assert not any(collide(b, floor) for b in block)
    assert not any(collide(a, b) for a in block for b in block if a != b)
    assert all(collide(w, floor) for w in walker_geoms)


def test_layout_is_resampled_per_key_and_reproducible():
    env = _env(irregularity=0.5)
    reset = jax.jit(env.reset)
    a = reset(jax.random.PRNGKey(0)).data.mocap_pos
    b = reset(jax.random.PRNGKey(1)).data.mocap_pos
    a2 = reset(jax.random.PRNGKey(0)).data.mocap_pos
    np.testing.assert_array_equal(a, a2)
    assert not np.allclose(a, b)


def test_zero_irregularity_is_a_periodic_staircase():
    env = _env(height=0.12, gap=0.8, irregularity=0.0)
    d = jax.jit(env.reset)(jax.random.PRNGKey(0)).data
    np.testing.assert_allclose(env.block_tops(d), 0.12, atol=1e-6)
    period = 2 * env._terrain.block_length + 0.8
    np.testing.assert_allclose(np.diff(np.asarray(d.mocap_pos[:, 0])), period, atol=1e-5)
    assert float(d.mocap_pos[0, 0]) - env._L == pytest.approx(env._terrain.start_gap)


def test_heights_and_gaps_stay_in_range():
    h, g, sigma = 0.1, 1.0, 0.7
    env = _env(height=h, gap=g, irregularity=sigma)
    pos = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 256)).data.mocap_pos
    tops = np.asarray(pos[..., 2] + env._H)
    assert tops.min() >= h * (1 - sigma) - 1e-6 and tops.max() <= h * (1 + sigma) + 1e-6
    gaps = np.diff(np.asarray(pos[..., 0]), axis=-1) - 2 * env._L
    assert gaps.min() >= g * (1 - sigma) - 1e-5 and gaps.max() <= g * (1 + sigma) + 1e-5


def test_treadmill_recycles_blocks_behind_the_torso():
    env = _env(irregularity=1.0)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    # Teleport the walker 10 m downfield: several blocks are now behind it at once.
    state = state.replace(data=state.data.replace(qpos=state.data.qpos.at[1].add(10.0)))
    state = step(state, jp.zeros(env.action_size))
    torso_x = float(state.data.xpos[env._torso_id, 0])
    x = np.sort(np.asarray(state.data.mocap_pos[:, 0]))
    assert x.min() + env._L >= torso_x - env._terrain.back_margin
    assert np.diff(x).min() >= 2 * env._L - 1e-5  # no overlapping blocks
    assert x.max() - torso_x >= 3.0  # the queue still reaches ahead


def test_observed_terrain_reports_the_next_blocks():
    env = _env(observed=True, obs_blocks=2,
               height=0.1, irregularity=0.0)
    base = _env(observed=False)
    assert env.observation_size == base.observation_size + 4
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    dist0, top0, dist1, top1 = np.asarray(s.obs[-4:])
    torso_x = float(s.data.xpos[env._torso_id, 0])
    assert dist0 == pytest.approx(env._terrain.start_gap - torso_x, abs=1e-5)
    assert dist1 > dist0
    assert top0 == pytest.approx(0.1, abs=1e-6) and top1 == pytest.approx(0.1, abs=1e-6)


def test_relative_stand_height_subtracts_the_block_under_the_torso():
    env = _env(relative_stand_height=True, height=0.3, irregularity=0.0,
               start_gap=0.0)
    d = jax.jit(env.reset)(jax.random.PRNGKey(0)).data
    x0 = d.mocap_pos[0, 0]
    assert float(env._ground_height_under(d, x0)) == pytest.approx(0.3, abs=1e-6)
    assert float(env._ground_height_under(d, x0 + env._L + 0.01)) == 0.0


@pytest.mark.parametrize("overrides", [
    dict(height=0.6, irregularity=1.0),   # tallest block would float
    dict(irregularity=1.5),
    dict(num_blocks=2),                           # queue too short
])
def test_bad_configs_are_rejected(overrides):
    with pytest.raises(ValueError):
        _env(**overrides)
