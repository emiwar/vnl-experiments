"""Turn a diverged MuJoCo step into a finite, terminal transition.

MJX occasionally diverges: the integrator produces a NaN in ``qpos``/``qvel``
and the dm_control_suite tasks already notice, setting
``done = isnan(qpos).any() | isnan(qvel).any()``. What they do *not* do is keep
the reward finite, and a non-finite reward is fatal to PPO — both the actor and
the critic loss reduce over the batch with ``jp.mean``, so one NaN reward in one
env at one timestep turns every gradient NaN, and Adam's moments make that
permanent. A 1e9-step HumanoidWalk run died this way at ~850M steps.

Whether the reward survives a divergence is pure luck. ``mjx.step`` runs
``forward`` (which fills ``xpos``, ``xmat``, ``sensordata``) and integrates
``qpos``/``qvel`` *last*, so those fields lag one integration behind. The
dm_control rewards read the lagged ones. A divergence in the final substep
therefore leaves a finite reward and only a NaN observation — survivable — while
one in an earlier substep poisons the lagged fields too and kills the run. With
``ctrl_dt/sim_dt`` substeps per control step, most divergences are the fatal
kind.

This wrapper removes the coin flip by making both cases behave like the
survivable one: the step terminates with reward 0.

**Termination, not truncation.** The env already reports a divergence as
``done=1, truncated=0``, and GAE turns that into ``advantage = reward - V(s)``
with the bootstrap dropped — a large negative advantage. That is worth keeping:
if divergences ever correlate with what the policy is doing (extreme torques, a
fragile contact configuration) we want them discouraged, and being consistent
with the path the env already takes matters more than the fact that a diverged
state is not a genuine zero-value terminal.

**The observation is deliberately left alone.** A NaN observation on a
terminated step never reaches a gradient: `nnx_ppo.algorithms.rollout` replaces
the whole state with a reset via ``tree_where(done, ...)``, and GAE drops the
bootstrap through ``jp.where(done, 0.0, next_value)``. Leaving it is what lets
`nnx-ppo`'s ``diagnostics/nonfinite_next_obs`` counter keep working as a
divergence-rate signal — sanitising it here would make divergences invisible
again, which is the thing that made the original failure so expensive to find.
"""

from typing import Any

import jax
import jax.numpy as jp
from jaxtyping import PRNGKeyArray


class NaNGuardWrapper:
    """Terminate with zero reward on any non-finite reward or observation.

    Wrap *innermost*, directly around the task and inside ``EpisodeWrapper``, so
    that ``EpisodeWrapper`` folds this ``done`` into its own with ``truncated``
    still False — i.e. a termination.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self, rng: PRNGKeyArray) -> Any:
        return self.env.reset(rng)

    def step(self, state: Any, action: Any) -> Any:
        next_state = self.env.step(state, action)
        diverged = jp.logical_or(
            jp.any(~jp.isfinite(next_state.reward)),
            _any_nonfinite(next_state.obs),
        )
        return next_state.replace(
            reward=jp.where(diverged, 0.0, next_state.reward),
            done=jp.where(diverged, 1.0, next_state.done).astype(
                jp.asarray(next_state.done).dtype
            ),
        )

    @property
    def observation_size(self) -> Any:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    def __getattr__(self, name: str) -> Any:
        # Forward render(), mj_model, dt, ... to the wrapped task. Only reached
        # for attributes this wrapper does not define itself.
        return getattr(self.env, name)


def _any_nonfinite(tree: Any) -> Any:
    leaves = [
        x for x in jax.tree.leaves(tree)
        if hasattr(x, "dtype") and jp.issubdtype(x.dtype, jp.inexact)
    ]
    if not leaves:
        return jp.array(False)
    return jp.any(jp.stack([jp.any(~jp.isfinite(x)) for x in leaves]))
