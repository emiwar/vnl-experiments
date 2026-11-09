"""Silence MuJoCo Warp's per-step "solver iterations limit reached" printf.

From MuJoCo 4.0 the bundled MuJoCo Warp defaults ``opt.warn_overflow`` to
``OverflowType.ALL``. That flag includes ``ITERATIONS`` and ``LS_ITERATIONS``, which
``wp.printf`` one line *per world per physics step* whenever the constraint solver or
line search uses its full iteration budget. Our solver budgets are deliberately small
(e.g. playground's WalkerWalk runs CG with 2 iterations), so hitting the limit is the
normal case, not a fault. Measured: ~413k lines for 256 walker envs x 200 control steps,
which buries the training log.

Only those two bits are cleared. The buffer-overflow warnings (``NEFC``, ``BROADPHASE``,
``NARROWPHASE``, ...) stay on, because they *do* signal a mis-sized ``njmax`` /
``naconmax`` that silently drops constraints.

The flag only controls printing, so physics is unchanged. It is read when the kernel is
compiled, which is why it is set on the ``mjx.Model`` rather than per step.

On MuJoCo < 4.0 (no ``warn_overflow`` field) or the JAX backend (no Warp options) this
is a no-op, so the same code runs on both stacks.
"""

from __future__ import annotations

from typing import Any


def silence_solver_iteration_warnings(env: Any) -> Any:
    """Clear the ITERATIONS / LS_ITERATIONS warning bits on ``env._mjx_model``, in place.

    Call it on the *unwrapped* env, after anything that re-runs ``put_model``
    (``actuator_mode.apply_servo`` does). Returns ``env`` for chaining.
    """
    mjx_model = getattr(env, "_mjx_model", None)
    impl_opt = getattr(getattr(mjx_model, "opt", None), "_impl", None)
    if impl_opt is None or not hasattr(impl_opt, "warn_overflow"):
        return env  # MuJoCo < 4.0, or impl="jax"

    try:
        from mujoco.mjx.third_party.mujoco_warp import OverflowType
    except ImportError:
        return env
    if not hasattr(OverflowType, "ITERATIONS"):
        return env  # older Warp: no per-step iteration warning to silence

    mask = int(impl_opt.warn_overflow) & ~int(
        OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS
    )
    env._mjx_model = mjx_model.replace(
        opt=mjx_model.opt.replace(_impl=impl_opt.replace(warn_overflow=mask))
    )
    return env
