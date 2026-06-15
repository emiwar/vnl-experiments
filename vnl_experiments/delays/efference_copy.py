"""Efference-copy wrapper that feeds an actor its own recent actions.

Wraps an inner ``StatefulModule`` (the actor branch ending in an
``ActionSampler``) so that, on every step, the inner module receives its
own ``queue_length`` most-recently-emitted actions appended to the
observation. This is the Markovian sufficient statistic for a policy that
sees stale observations, and is biologically motivated as the efference
copy of motor commands.

The queue is stored in the wrapper's carry state and reset to zeros on
episode boundaries, mirroring the convention used by ``Delay`` and
``DelayedObsNetwork``.
"""

from typing import Any

import jax
import jax.numpy as jp

from nnx_ppo.networks.types import (
    StatefulModule,
    StatefulModuleOutput,
    ModuleState,
)


def _flatten_queue(queue):
    """Flatten queue leaves [B, L, *leaf] -> [B, L * prod(leaf)] and concat.

    Concatenation order follows the pytree's flatten order. For a single
    leaf this is just a reshape.
    """
    leaves = jax.tree_util.tree_leaves(queue)
    flat = [leaf.reshape(leaf.shape[0], -1) for leaf in leaves]
    return jp.concatenate(flat, axis=-1)


class EfferenceCopy(StatefulModule):
    """Augment an inner actor's input with its own last ``queue_length`` actions.

    The inner module is expected to terminate in an ``ActionSampler`` so that
    its forward output is a sampler dict ``{"action", "log_likelihood"}``.
    ``EfferenceCopy`` extracts the ``"action"`` field and pushes it onto a
    shift-register queue for the next step.

    Carry::

        {
          "inner": <inner module's carry>,
          "queue": <pytree mirroring sample_action, leaves [B, L, *leaf]>,
        }

    The queue is newest-first (index 0 is the most recent action).

    Args:
        inner: The wrapped actor pipeline. Must produce a sampler dict.
        sample_action: A single unbatched example of the action pytree
            (typically ``jp.zeros(action_size)``). Used only to capture
            leaf shapes, dtypes, and tree structure for queue allocation.
        queue_length: Number of recent actions to keep. ``0`` short-circuits
            to a pure pass-through (no queue, no concat) so the same wrapper
            can be used for the no-efference baseline.
        inject_key: Output routing for the queue. ``None`` (default) keeps the
            original behaviour: ``obs`` is a tensor and the flattened queue is
            concatenated onto its last axis. When set to a string, ``obs`` is
            treated as a **dict** and the (unflattened, newest-first
            ``[B, L, *leaf]``) queue is inserted under that key, so the inner
            module can access it by name rather than by index slicing. With
            ``queue_length == 0`` the dict passes through unchanged (no key).
    """

    def __init__(
        self,
        inner: StatefulModule,
        sample_action: Any,
        queue_length: int,
        inject_key: str | None = None,
    ):
        if queue_length < 0:
            raise ValueError(
                f"queue_length must be non-negative, got {queue_length}"
            )
        self.inner = inner
        self.queue_length = queue_length
        self.inject_key = inject_key
        leaves, self._treedef = jax.tree_util.tree_flatten(sample_action)
        self._leaf_shapes = tuple(leaf.shape for leaf in leaves)
        self._leaf_dtypes = tuple(leaf.dtype for leaf in leaves)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def initialize_state(self, batch_size: int) -> ModuleState:
        inner_state = self.inner.initialize_state(batch_size)
        if self.queue_length == 0:
            return {"inner": inner_state}
        queue_leaves = [
            jp.zeros((batch_size, self.queue_length) + shape, dtype)
            for shape, dtype in zip(self._leaf_shapes, self._leaf_dtypes)
        ]
        queue = jax.tree_util.tree_unflatten(self._treedef, queue_leaves)
        return {"inner": inner_state, "queue": queue}

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        inner_reset = self.inner.reset_state(prev_state["inner"])
        if self.queue_length == 0:
            return {"inner": inner_reset}
        return {
            "inner": inner_reset,
            "queue": jax.tree.map(jp.zeros_like, prev_state["queue"]),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        state: ModuleState,
        obs: Any,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        inner_extras = (
            None if rollout_extras is None else rollout_extras["inner"]
        )

        if self.queue_length == 0:
            inner_out = self.inner(state["inner"], obs, inner_extras)
            return StatefulModuleOutput(
                next_state={"inner": inner_out.next_state},
                output=inner_out.output,
                regularization_loss=inner_out.regularization_loss,
                metrics=inner_out.metrics,
                rollout_extras={"inner": inner_out.rollout_extras},
            )

        queue = state["queue"]
        if self.inject_key is None:
            queue_flat = _flatten_queue(queue)
            augmented = jp.concatenate([obs, queue_flat], axis=-1)
        else:
            # Dict mode: hand the queue to the inner module under a named key
            # instead of flattening it into the observation tensor.
            augmented = {**obs, self.inject_key: queue}

        inner_out = self.inner(state["inner"], augmented, inner_extras)

        # Inner output is a sampler dict {"action", "log_likelihood"} (or a
        # pytree of them for multi-sampler banks). For multi-sampler banks
        # the user is expected to provide a matching sample_action pytree.
        new_action = jax.tree.map(
            lambda d: d["action"],
            inner_out.output,
            is_leaf=lambda x: isinstance(x, dict) and "action" in x,
        )

        # Shift register: prepend new action, drop oldest.
        new_queue = jax.tree.map(
            lambda q, a: jp.concatenate([a[:, None], q[:, :-1]], axis=1),
            queue,
            new_action,
        )

        return StatefulModuleOutput(
            next_state={"inner": inner_out.next_state, "queue": new_queue},
            output=inner_out.output,
            regularization_loss=inner_out.regularization_loss,
            metrics=inner_out.metrics,
            rollout_extras={"inner": inner_out.rollout_extras},
        )

    def update_statistics(self, rollout_extras: Any) -> None:
        self.inner.update_statistics(rollout_extras["inner"])
