"""Fixed Gaussian motor noise, placed before or after the efference copy.

The question this exists for: on dm_control the efference copy barely helps a delayed
policy, plausibly because the tasks are deterministic, so the actions in flight are
already predictable from the delayed observation. Adding noise to the executed action
breaks that predictability, and *where* the noise is added decides whether the
efference queue gets to see it:

``"pre_efference"``
    ``MotorNoise`` sits inside ``EfferenceCopy.inner``, right after the sampler. The queue
    records the noisy action, so the in-flight actions carry information that the delayed
    observation does not.
``"post_efference"``
    ``MotorNoise`` sits after ``EfferenceCopy``. The queue records the intended action and
    the body executes the noisy one: unobserved motor noise. This is the same perturbation
    as the eval-only ``action_noise`` in :mod:`vnl_experiments.delays.evaluation`.

In both places the body receives exactly the same perturbation (in policy units, after
the tanh, clipped back into ``[-1, 1]``), so the placement is the only difference between
the two arms. Adding the noise after the tanh keeps it a fixed size in actuator units:
noise added before the tanh would be squashed wherever the policy saturates, and before
the efference copy it would be indistinguishable from a wider policy distribution.

Interaction with PPO. The sampler's ``log_likelihood`` passes through unchanged: PPO
scores the sampler's stored pre-tanh action, so from the policy gradient's point of view
the noise is part of the environment. The noise sample *is* emitted as ``rollout_extras``
and reused in the loss replay. Without that, the replayed efference queue (rebuilt from
the replayed actions) would differ from the one the rollout acted on.

Evaluation. The noise is part of the world, so it stays on under ``networks.eval()``.
That is why the module has no attribute called ``deterministic``: ``nnx.Module.eval()``
sets every attribute of that name to True.

Temporal correlation. ``tau > 0`` (in control steps) makes the noise an AR(1) process,
``n_t = rho * n_{t-1} + std * sqrt(1 - rho**2) * eps_t`` with ``rho = exp(-1 / tau)``, so
its stationary std is still ``std``. ``tau = inf`` gives ``rho = 1``: one offset per
episode. The rationale: iid noise is largely low-pass filtered away by the body, and noise
much slower than the delay is corrected by the delayed feedback anyway, so the band where
only a copy that sees the noise can help is ``tau`` around the delay. The process state is
carried per env and redrawn from the stationary distribution at the first step of every
episode (``reset_state`` has no key, so it only marks the carry as fresh). ``tau = 0`` is
the iid module, stateless and bit-identical to the original.
"""

from typing import Any

import math

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import (
    ModuleState,
    PPONetworkOutput,
    StatefulModule,
    StatefulModuleOutput,
)

#: Valid values of the ``motor_noise_site`` net-param.
MOTOR_NOISE_SITES = ("pre_efference", "post_efference")


def validate_motor_noise_site(site: Any) -> str:
    if site not in MOTOR_NOISE_SITES:
        raise ValueError(
            f"motor_noise_site must be one of {MOTOR_NOISE_SITES}, got {site!r}"
        )
    return site


def perturb_actions(actions: Any, std: float, key: jax.Array) -> Any:
    """``clip(a + std * N(0, 1), -1, 1)`` on every leaf of ``actions``.

    One key per leaf: ``actions`` is a single array for most networks but a pytree for a
    multi-sampler bank, whose leaves must not share noise.
    """
    leaves, treedef = jax.tree.flatten(actions)
    return jax.tree.unflatten(treedef, [
        jp.clip(a + std * jax.random.normal(k, a.shape), -1.0, 1.0)
        for a, k in zip(leaves, jax.random.split(key, len(leaves)))
    ])


def _is_sampler_dict(x: Any) -> bool:
    return isinstance(x, dict) and "action" in x and "log_likelihood" in x


class MotorNoise(StatefulModule):
    """Add fixed Gaussian noise to the action, leaving the log-likelihood untouched.

    Accepts either a sampler dict ``{"action", "log_likelihood"}`` (or a pytree of them),
    or a :class:`PPONetworkOutput`, whose ``actions`` field is perturbed. The second form
    is what follows an ``EfferenceCopy`` that wraps a whole ``PPOAdapter``
    (``DelayedMLP`` with ``privileged_critic=False``).

    Args:
        std: Noise standard deviation in policy units (actions live in ``[-1, 1]``).
            A plain float rather than an ``nnx.Variable``, so the module adds nothing
            to the saved network state. With ``tau > 0`` it is the AR(1) process's
            stationary std.
        rngs: The network's shared ``nnx.Rngs``.
        tau: Correlation time in control steps. ``0`` (default) is iid noise;
            ``math.inf`` is a constant offset per episode.
        action_size: Width of the (single-array) action. Required when ``tau > 0``,
            to size the carried noise state.
    """

    def __init__(self, std: float, rngs: nnx.Rngs, tau: float = 0.0,
                 action_size: int | None = None):
        if std < 0:
            raise ValueError(f"MotorNoise std must be non-negative, got {std}")
        if not tau >= 0:
            raise ValueError(f"MotorNoise tau must be non-negative, got {tau}")
        if tau > 0 and action_size is None:
            raise ValueError("MotorNoise with tau > 0 needs action_size")
        self.std = float(std)
        self.tau = float(tau)
        self.action_size = action_size
        self.rng = rngs

    @property
    def rho(self) -> float:
        """The AR(1) coefficient ``exp(-1 / tau)``; 0 for iid noise, 1 at ``tau = inf``."""
        return math.exp(-1.0 / self.tau) if self.tau > 0 else 0.0

    def initialize_state(self, batch_size: int) -> ModuleState:
        if self.tau == 0:
            return ()
        return {"n": jp.zeros((batch_size, self.action_size)),
                "fresh": jp.ones((batch_size,), bool)}

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        if self.tau == 0:
            return prev_state
        return {"n": prev_state["n"], "fresh": jp.ones_like(prev_state["fresh"])}

    def __call__(
        self,
        state: ModuleState,
        x: Any,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        # Draw even in the loss replay, as the sampler does, so the RNG stream advances
        # the same way in both phases.
        key = self.rng()

        if isinstance(x, PPONetworkOutput):
            actions = x.actions
        else:
            actions = jax.tree.map(
                lambda d: d["action"], x, is_leaf=_is_sampler_dict
            )

        if self.tau == 0:
            next_state = ()
            if rollout_extras is None:
                noisy = jax.lax.stop_gradient(perturb_actions(actions, self.std, key))
            else:
                noisy = rollout_extras
            extras = noisy
        else:
            if not isinstance(actions, jax.Array):
                raise TypeError("MotorNoise with tau > 0 supports a single action array")
            if rollout_extras is None:
                k_init, k_step = jax.random.split(key)
                stationary = self.std * jax.random.normal(k_init, actions.shape)
                innovation = (self.std * math.sqrt(1.0 - self.rho ** 2)
                              * jax.random.normal(k_step, actions.shape))
                n = jp.where(state["fresh"][:, None], stationary,
                             self.rho * state["n"] + innovation)
            else:
                n = rollout_extras
            n = jax.lax.stop_gradient(n)
            noisy = jp.clip(actions + n, -1.0, 1.0)
            next_state = {"n": n, "fresh": jp.zeros_like(state["fresh"])}
            # The process value, not the clipped action: the next step's AR update needs it.
            extras = n

        if isinstance(x, PPONetworkOutput):
            output = x.replace(actions=noisy)
        else:
            output = jax.tree.map(
                lambda d, a: {**d, "action": a}, x, noisy, is_leaf=_is_sampler_dict
            )
        return StatefulModuleOutput(
            next_state=next_state,
            output=output,
            regularization_loss=jp.array(0.0),
            metrics={},
            rollout_extras=extras,
        )
