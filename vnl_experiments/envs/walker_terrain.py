"""WalkerWalk over obstacle blocks that are regenerated every episode.

The point is a *parametric* source of unpredictability that lives in the plant rather than
in the controller: block height sets the difficulty, and ``terrain.irregularity`` sets how
unpredictable the layout is at fixed mean difficulty. ``irregularity = 0`` is a perfectly
periodic staircase that can be learned as a rhythm; ``irregularity = 1`` draws every height
and gap uniformly between 0 and twice its mean.

Why mocap blocks and not a heightfield
--------------------------------------
``hfield_data`` lives in the *Model*, and nnx-ppo vmaps a single-instance env over the
*Data* only, so a heightfield is one terrain shared by every env and every episode
(that is how playground's rough-terrain locomotion tasks work). ``mocap_pos`` lives in the
Data, so each world gets its own layout for free, set in ``reset`` and resampled whenever
nnx-ppo re-resets a finished env with a fresh key. No model batching, and it works with
both ``impl="warp"`` and ``impl="jax"``. Playground's fish/finger/swimmer move their
targets the same way.

How the blocks work
-------------------
* ``K = terrain.num_blocks`` box-shaped mocap bodies are appended at the *end* of the
  worldbody. The walker's body indices are therefore unchanged, and ``_get_obs`` slices
  ``xmat`` back to the walker's bodies, so a flat run's observation is the plain walker's.
* Every box has the same half-height ``H = terrain.block_halfheight``. A block of height
  ``h`` is placed with its centre at ``h - H``, i.e. mostly buried, so per-world heights
  need no model change. A zero-height block is buried by a full centimetre, so it never
  doubles up with the floor's contacts. That makes ``terrain.height = 0`` the
  unmodified WalkerWalk (``walker_terrain_test.py`` checks this exactly).
* Collision filtering: the blocks are ``contype=0, conaffinity=1``, so they collide with
  the walker's geoms (``contype=1``) but not with each other. The floor is switched from
  ``contype=1`` to ``0`` so it does not collide with the blocks either. That has no
  physical effect on the walker: walker geoms have ``conaffinity=0``, so the floor's
  ``contype`` never mattered for them.
* Treadmill: once a block is ``terrain.back_margin`` behind the torso, ``step`` moves it to
  the front of the queue with a freshly drawn height and gap. ``K`` small blocks therefore
  cover an episode of any length at any speed, and the broadphase only ever sees
  ``K x 9`` new geom pairs.

Per-instance state is just ``data.mocap_pos`` plus the existing ``info["rng"]``. Nothing
goes in the network carry, and nnx-ppo's ``tree_where(done, reset_states, ...)`` swaps in
a new terrain with the rest of the state.

Things that differ from WalkerWalk at nonzero height
----------------------------------------------------
* The stand reward reads the torso's *absolute* height, so standing on a block reads as
  slightly better posture. ``terrain.relative_stand_height=True`` measures it from the ground
  under the torso instead. It is off by default so the reward keeps the original task's
  definition.
* ``terrain.observed=True`` appends ``(distance, height)`` of the next
  ``terrain.obs_blocks`` blocks ahead of the torso. Blind (the default) makes blocks an
  unforeseen perturbation; observed makes them anticipatable.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import walker

#: How far below the floor a zero-height block's top sits. Anything below the floor plane
#: is only reachable by a foot that is already penetrating it, so 1 cm is plenty.
_BURY_DEPTH = 0.01


def default_terrain_config() -> config_dict.ConfigDict:
    """The ``terrain`` sub-config; ``conf/terrain/*.yaml`` hold presets onto it."""
    return config_dict.create(
        # Difficulty: mean step height in metres (the walker's leg is ~1 m, its torso
        # starts at 1.3 m). 0 is plain WalkerWalk.
        height=0.1,
        # Mean gap between neighbouring block edges, in metres.
        gap=1.0,
        # Unpredictability in [0, 1]: height and gap are drawn as
        # mean * (1 + irregularity * U[-1, 1]).
        irregularity=0.0,
        # Block half-length along x, half-width along y, and the fixed half-height every
        # block has (its top is placed at the sampled height by burying it).
        block_length=0.25,
        block_width=0.5,
        block_halfheight=0.5,
        # Blocks in the treadmill queue, flat run-up before the first block, and how far
        # behind the torso a block goes before it is recycled to the front.
        num_blocks=12,
        start_gap=2.0,
        back_margin=2.0,
        # Observation and reward options; see the module docstring.
        observed=False,
        obs_blocks=2,
        obs_max_distance=5.0,
        relative_stand_height=False,
    )


def default_config() -> config_dict.ConfigDict:
    cfg = walker.default_config()
    cfg.terrain = default_terrain_config()
    return cfg


class WalkerTerrain(walker.PlanarWalker):
    """``PlanarWalker`` with per-episode block terrain. See the module docstring."""

    def __init__(
        self,
        move_speed: float = walker.WALK_SPEED,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict[str, Any] | None = None,
    ):
        super().__init__(move_speed, config, config_overrides)
        c = self._config
        t = self._terrain = c.terrain
        self._check_config()

        self._n_walker_bodies = self._mj_model.nbody - 1  # minus the world body
        self._K = int(t.num_blocks)
        self._L = float(t.block_length)
        self._H = float(t.block_halfheight)

        spec = mujoco.MjSpec.from_file(self._xml_path)
        spec.geom("floor").contype = 0
        for i in range(self._K):
            body = spec.worldbody.add_body(
                name=f"block{i}", mocap=True, pos=[-100.0 - i, 0.0, -2.0 * self._H]
            )
            body.add_geom(
                name=f"block{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self._L, float(t.block_width), self._H],
                contype=0,
                conaffinity=1,
                rgba=[0.55, 0.45, 0.35, 1.0],
            )
        self._mj_model = spec.compile()
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=c.impl)
        self._post_init()

    def _check_config(self) -> None:
        t = self._terrain
        sigma = float(t.irregularity)
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(f"terrain.irregularity must be in [0, 1], got {sigma}")
        if t.height < 0 or t.gap < 0:
            raise ValueError("terrain.height and terrain.gap must be >= 0")
        max_h = t.height * (1 + sigma)
        if max_h > 2 * t.block_halfheight - _BURY_DEPTH:
            raise ValueError(
                f"the tallest block ({max_h:g} m) would float: raise "
                f"terrain.block_halfheight above {max_h / 2:g}"
            )
        # The queue has to reach far enough ahead that the walker never runs off its
        # front: K blocks at their shortest period must cover the back margin plus a few
        # metres of road ahead.
        min_period = 2 * t.block_length + t.gap * (1 - sigma)
        reach = t.num_blocks * min_period - t.back_margin
        if reach < 3.0:
            raise ValueError(
                f"terrain.num_blocks={t.num_blocks} covers only {reach:.2f} m "
                f"ahead of the torso at the shortest block period; need >= 3 m"
            )

    def _post_init(self) -> None:
        super()._post_init()
        # Joint ranges for the walker's own joints; mocap bodies add none.
        self._lowers = self._mj_model.jnt_range[3:, 0]
        self._uppers = self._mj_model.jnt_range[3:, 1]

    # ------------------------------------------------------------------------------
    # Terrain sampling
    # ------------------------------------------------------------------------------

    def _sample_blocks(self, rng: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
        """``n`` (top height, period) pairs, where period = block length + the gap before it."""
        t = self._terrain
        sigma = t.irregularity
        k_h, k_g = jax.random.split(rng)
        u_h = jax.random.uniform(k_h, (n,), minval=-1.0, maxval=1.0)
        u_g = jax.random.uniform(k_g, (n,), minval=-1.0, maxval=1.0)
        heights = t.height * (1 + sigma * u_h)
        gaps = t.gap * (1 + sigma * u_g)
        return heights, 2 * self._L + gaps

    def _block_z(self, heights: jax.Array) -> jax.Array:
        """Centre z for a block whose top is at ``heights`` (buried when ~0)."""
        top = jp.where(heights > 1e-6, heights, -_BURY_DEPTH)
        return top - self._H

    def _initial_mocap_pos(self, rng: jax.Array) -> jax.Array:
        heights, periods = self._sample_blocks(rng, self._K)
        # The first block's near edge is terrain.start_gap ahead of the origin, and every
        # later one is one period further on.
        first_centre = self._terrain.start_gap + self._L
        x = first_centre + jp.cumsum(periods) - periods[0]
        return jp.stack([x, jp.zeros_like(x), self._block_z(heights)], axis=-1)

    def _recycle(self, mocap_pos: jax.Array, torso_x: jax.Array,
                 rng: jax.Array) -> jax.Array:
        """Move every block that is far enough behind the torso to the front of the queue.

        Vectorised for any number of recycled blocks at once: they are queued in index
        order behind the current front-most block. Their relative order does not matter,
        since all of them end up ahead of every block that stays.
        """
        x = mocap_pos[:, 0]
        behind = x + self._L < torso_x - self._terrain.back_margin
        heights, periods = self._sample_blocks(rng, self._K)
        offsets = jp.cumsum(jp.where(behind, periods, 0.0))
        new_x = jp.where(behind, jp.max(x) + offsets, x)
        new_z = jp.where(behind, self._block_z(heights), mocap_pos[:, 2])
        return jp.stack([new_x, mocap_pos[:, 1], new_z], axis=-1)

    def block_tops(self, data: mjx.Data) -> jax.Array:
        """Top height of every block (negative for a buried one)."""
        return data.mocap_pos[:, 2] + self._H

    # ------------------------------------------------------------------------------
    # Env API
    # ------------------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # The first three lines consume rng exactly as PlanarWalker.reset does, so the
        # initial pose matches the plain walker for the same key.
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[2].set(jax.random.uniform(rng1, (), minval=-jp.pi, maxval=jp.pi))
        qpos = qpos.at[3:].set(
            jax.random.uniform(
                rng2, (self.mjx_model.nq - 3,), minval=self._lowers, maxval=self._uppers
            )
        )
        rng, rng_terrain = jax.random.split(rng)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            mocap_pos=self._initial_mocap_pos(rng_terrain),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        metrics = {
            "reward/standing": jp.zeros(()),
            "reward/upright": jp.zeros(()),
            "reward/stand": jp.zeros(()),
            "reward/move": jp.zeros(()),
        }
        info = {"rng": rng}
        reward_, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward_, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        reward_ = self._get_reward(data, action, state.info, state.metrics)

        # Recycle after the physics step and before the observation, so an observed
        # terrain is consistent with the layout the next step will simulate. Moving a
        # mocap body only updates mocap_pos; its xpos follows at the next step's forward
        # pass, which is fine because a recycled block is far from the walker either way.
        rng, rng_recycle = jax.random.split(state.info["rng"])
        torso_x = data.xpos[self._torso_id, 0]
        data = data.replace(mocap_pos=self._recycle(data.mocap_pos, torso_x, rng_recycle))
        info = {**state.info, "rng": rng}

        obs = self._get_obs(data, info)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return mjx_env.State(data, obs, reward_, done.astype(float), state.metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info
        # Walker bodies only: the plain walker reads xmat[1:], which here would include
        # the blocks.
        orientations = data.xmat[1:1 + self._n_walker_bodies, [0, 0], [0, 2]].ravel()
        height = data.xmat[self._torso_id, 2, 2]
        parts = [orientations, height.reshape(1), data.qvel]
        if self._terrain.observed:
            parts.append(self._terrain_obs(data))
        return jp.concatenate(parts)

    def _terrain_obs(self, data: mjx.Data) -> jax.Array:
        """``(distance to near edge, top height)`` of the next blocks ahead of the torso.

        A block counts as "ahead" until its far edge passes the torso, so the block being
        stood on is the first entry, at distance <= 0. Buried blocks report height 0.
        """
        t = self._terrain
        n = int(t.obs_blocks)
        torso_x = data.xpos[self._torso_id, 0]
        x = data.mocap_pos[:, 0]
        near_edge = x - self._L - torso_x
        ahead = x + self._L > torso_x
        key = jp.where(ahead, near_edge, jp.inf)
        idx = jp.argsort(key)[:n]
        dist = jp.clip(near_edge[idx], -2 * self._L, t.obs_max_distance)
        tops = jp.maximum(self.block_tops(data)[idx], 0.0)
        return jp.stack([dist, tops], axis=-1).ravel()

    def _ground_height_under(self, data: mjx.Data, x: jax.Array) -> jax.Array:
        over = jp.abs(data.mocap_pos[:, 0] - x) <= self._L
        return jp.maximum(jp.max(jp.where(over, self.block_tops(data), 0.0)), 0.0)

    def _get_stand_reward(self, data, action, info, metrics) -> jax.Array:
        if not self._terrain.relative_stand_height:
            return super()._get_stand_reward(data, action, info, metrics)
        del action, info
        torso_x, torso_z = data.xpos[self._torso_id, 0], data.xpos[self._torso_id, 2]
        torso_height = torso_z - self._ground_height_under(data, torso_x)
        standing = reward.tolerance(
            torso_height,
            bounds=(walker._STAND_HEIGHT, float("inf")),
            margin=walker._STAND_HEIGHT / 2,
        )
        metrics["reward/standing"] = standing
        upright = (1 + data.xmat[self._torso_id, 2, 2]) / 2
        metrics["reward/upright"] = upright
        stand_reward = (3 * standing + upright) / 4
        metrics["reward/stand"] = stand_reward
        return stand_reward
