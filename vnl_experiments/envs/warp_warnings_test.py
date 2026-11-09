"""`silence_solver_iteration_warnings` clears exactly the two noisy bits, and survives the build path.

The build-path test matters because ``actuator_mode.apply_servo`` re-runs ``put_model``:
silencing before it would be silently undone. The tests pass on both MuJoCo 3.x (where the
field does not exist and the helper is a no-op) and 4.x.
"""

from absl.testing import absltest

from vnl_experiments.envs import registry as env_registry
from vnl_experiments.envs.warp_warnings import silence_solver_iteration_warnings

TASK = "CartpoleSwingup"  # cheapest dm_control task to build


def _build(impl, **overrides):
    spec = env_registry.get(TASK)
    cfg = spec.default_config()
    cfg.impl = impl
    for k, v in overrides.items():
        cfg[k] = v
    return spec.build(cfg, for_eval=True)  # unwrapped env, so `_mjx_model` is reachable


def _warn_overflow(env):
    impl_opt = getattr(env._mjx_model.opt, "_impl", None)
    return getattr(impl_opt, "warn_overflow", None)


class SilenceSolverIterationWarningsTest(absltest.TestCase):

    def _overflow_type(self):
        try:
            from mujoco.mjx.third_party.mujoco_warp import OverflowType
        except ImportError:
            return None
        return OverflowType if hasattr(OverflowType, "ITERATIONS") else None

    def test_build_path_clears_only_iteration_bits(self):
        OverflowType = self._overflow_type()
        if OverflowType is None:
            self.skipTest("MuJoCo Warp has no OverflowType.ITERATIONS (MuJoCo < 4.0)")
        # servo_kp > 0 makes apply_servo re-run put_model, the case that would undo it.
        for overrides in ({}, {"servo_kp": 1.0}):
            with self.subTest(**overrides):
                flags = int(_warn_overflow(_build("warp", **overrides)))
                self.assertFalse(flags & int(OverflowType.ITERATIONS))
                self.assertFalse(flags & int(OverflowType.LS_ITERATIONS))
                # Buffer overflows still warn: they mean dropped constraints.
                self.assertTrue(flags & int(OverflowType.NEFC))
                self.assertTrue(flags & int(OverflowType.NARROWPHASE))

    def test_noop_without_warp_options(self):
        env = _build("jax")
        before = env._mjx_model
        self.assertIs(silence_solver_iteration_warnings(env)._mjx_model, before)


if __name__ == "__main__":
    absltest.main()
