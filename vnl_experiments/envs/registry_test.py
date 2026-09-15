"""The env registry's contract, and the resume hook in particular.

``EnvSpec.resume_reset`` is what makes a preempted run resumable: a light checkpoint omits
the env states, so they have to be redrawn, and how to spread their episode phases differs
by family. The tests here pin the two things that would otherwise fail silently -- a family
that forgot the hook, and a hook that draws phases over the wrong range.
"""

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp

from vnl_experiments.envs import registry as env_registry

#: Cheapest dm_control task to build: no contacts (`naconmax=0`, `njmax=2`).
CHEAP_TASK = "CartpoleSwingup"

#: The locally installed mujoco_playground defaults dm_control tasks to `impl="warp"`,
#: which raises out of `mujoco/mjx/warp/types.py` under `nnx.vmap` on CPU. Every run in
#: the `nnx-ppo-delays` project was recorded with `impl=jax`, so this both matches the
#: cohort and is the only backend that runs here.
TEST_IMPL = "jax"


def _cheap_env(task=CHEAP_TASK):
    spec = env_registry.get(task)
    cfg = spec.default_config()
    cfg.impl = TEST_IMPL
    return spec, cfg, spec.build(cfg)


class ResumeHookIsDeclaredTest(parameterized.TestCase):
    """Every family must say how it resumes.

    The field has no default precisely so this test can exist: a family without a hook
    would otherwise resume with every env at the same point in its episode, which looks
    like a training artefact rather than a bug.
    """

    @parameterized.named_parameters(*((n, n) for n in sorted(env_registry.ENVS)))
    def test_every_registered_spec_has_a_resume_reset(self, task):
        self.assertIsNotNone(env_registry.ENVS[task].resume_reset,
                             f"{task} cannot be resumed from a light checkpoint")

    def test_the_two_families_use_their_own_hook(self):
        """And not each other's -- the imitation one would crash on a dm_control env."""
        self.assertIs(env_registry.ENVS["AbsoluteImitation"].resume_reset,
                      env_registry._imitation_resume_reset)
        self.assertIs(env_registry.ENVS["Imitation"].resume_reset,
                      env_registry._imitation_resume_reset)
        for task, spec in env_registry.ENVS.items():
            if spec.obs_layout == "flat":
                self.assertIs(spec.resume_reset, env_registry._dmc_resume_reset, task)


class DmcResumeResetTest(absltest.TestCase):
    """The dm_control hook must spread phase over the *whole* episode.

    ``EpisodeWrapper.reset`` already spreads, but only over ``[0, max_len // 2)``, so a
    plainly-reset population has nothing in the second half of its episode. Resuming that
    way leaves every env unable to truncate for ~500 steps and then truncating together.
    """

    N_ENVS = 64

    @classmethod
    def setUpClass(cls):
        cls.spec, cls.cfg, cls.env = _cheap_env()
        cls.max_len = int(cls.cfg.episode_length)
        cls.states, cls.note = cls.spec.resume_reset(
            cls.env, cls.cfg, cls.N_ENVS, jax.random.key(0))
        cls.counter = cls.states.info["step_counter"]

    def test_phases_span_both_halves_of_the_episode(self):
        half = self.max_len // 2
        self.assertLess(int(self.counter.min()), half)
        self.assertGreater(int(self.counter.max()), half)
        # The point of the hook: a plain reset puts *nothing* above the half-way mark.
        above = float((self.counter >= half).mean())
        self.assertGreater(above, 0.25, f"only {above:.0%} of envs past the half-way mark")

    def test_a_plain_reset_would_not(self):
        """The baseline this hook exists to fix, asserted rather than asserted-about."""
        from flax import nnx

        plain = nnx.vmap(self.env.reset)(
            jax.random.split(jax.random.key(1), self.N_ENVS))
        self.assertEqual(
            float((plain.info["step_counter"] >= self.max_len // 2).mean()), 0.0)

    def test_no_env_starts_already_truncated(self):
        """The draw excludes max_len itself, so `truncated` needs no fixing up."""
        self.assertLess(int(self.counter.max()), self.max_len)
        self.assertEqual(float(jp.asarray(self.states.info["truncated"]).mean()), 0.0)

    def test_shapes_and_dtype_match_the_wrapper(self):
        self.assertEqual(self.counter.shape, (self.N_ENVS,))
        self.assertEqual(self.counter.dtype, jp.int32)
        self.assertEqual(jp.asarray(self.states.info["truncated"]).shape,
                         (self.N_ENVS,))
        self.assertEqual(self.states.obs.shape[0], self.N_ENVS)

    def test_the_redrawn_states_can_be_stepped(self):
        """A state whose info dict was rebuilt must still be a valid env state."""
        from flax import nnx

        out = jax.jit(nnx.vmap(self.env.step))(
            self.states, jp.zeros((self.N_ENVS, self.env.action_size)))
        self.assertTrue(
            bool(jp.all(out.info["step_counter"] == self.counter + 1)))
        self.assertTrue(bool(jp.all(jp.isfinite(out.reward))))

    def test_the_note_names_the_range(self):
        """It goes into the resume log, which is where a wrong range would be spotted."""
        self.assertIn(str(self.max_len), self.note)
        self.assertIn(str(self.N_ENVS), self.note)

    def test_info_keys_are_preserved(self):
        """Rebuilding the dict must not drop a key the task put there."""
        from flax import nnx

        plain = nnx.vmap(self.env.reset)(
            jax.random.split(jax.random.key(2), self.N_ENVS))
        self.assertEqual(set(self.states.info), set(plain.info))


class ValidateResumableTest(absltest.TestCase):
    """The startup guard, tested without building an env."""

    def test_a_spec_without_a_hook_is_refused(self):
        from vnl_experiments.config import OverrideError
        from vnl_experiments.train import validate_resumable

        spec = env_registry.EnvSpec(default_config=dict, build=lambda *a, **k: None)
        self.assertIsNone(spec.resume_reset)
        with self.assertRaises(OverrideError) as cm:
            validate_resumable(spec, "MadeUpTask")
        message = str(cm.exception)
        self.assertIn("MadeUpTask", message)
        self.assertIn("resume_reset", message)
        # It must also name the escape hatch, or the reader is stuck.
        self.assertIn("full_checkpoints", message)

    def test_a_registered_spec_passes(self):
        from vnl_experiments.train import validate_resumable

        for task in ("AbsoluteImitation", CHEAP_TASK):
            validate_resumable(env_registry.ENVS[task], task)


if __name__ == "__main__":
    absltest.main()
