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
"""

from typing import Any

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import (
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
            to the saved network state.
        rngs: The network's shared ``nnx.Rngs``.
    """

    def __init__(self, std: float, rngs: nnx.Rngs):
        if std < 0:
            raise ValueError(f"MotorNoise std must be non-negative, got {std}")
        self.std = float(std)
        self.rng = rngs

    def __call__(
        self,
        state: tuple[()],
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

        if rollout_extras is None:
            noisy = jax.lax.stop_gradient(perturb_actions(actions, self.std, key))
        else:
            noisy = rollout_extras

        if isinstance(x, PPONetworkOutput):
            output = x.replace(actions=noisy)
        else:
            output = jax.tree.map(
                lambda d, a: {**d, "action": a}, x, noisy, is_leaf=_is_sampler_dict
            )
        return StatefulModuleOutput(
            next_state=(),
            output=output,
            regularization_loss=jp.array(0.0),
            metrics={},
            rollout_extras=noisy,
        )
