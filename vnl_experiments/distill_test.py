"""Tests for the distillation path: the teacher -> student target mapping and teacher loading.

The mapping tests run on the stub env from ``network_builders_test`` (CPU, seconds). The
end-to-end tests use mujoco_playground's CartpoleBalance on the JAX backend, as nnx-ppo's
own distillation tests do.
"""

import json
import tempfile
from pathlib import Path

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
import mujoco_playground
from flax import nnx
from ml_collections import config_dict

from nnx_ppo.algorithms import distillation, ppo
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn
from nnx_ppo.algorithms.config import TrainConfig
from nnx_ppo.algorithms.types import LoggingLevel

from vnl_experiments import distill
from vnl_experiments.config import OverrideError
from vnl_experiments.delays import network_builders as nb
from vnl_experiments.delays.network_builders_test import (
    ACTION_SIZE,
    BATCH,
    FlatStubEnv,
    flat_stub_obs,
)

TEACHER = {"network_class": "DelayedMLP", "actor_hidden_sizes": [16],
           "critic_hidden_sizes": [16], "delay_k": 0, "efference_length": 0}

STUDENTS = {
    "DelayedMLP": {"network_class": "DelayedMLP", "actor_hidden_sizes": [16]},
    "DelayedMLP_delayedcritic": {"network_class": "DelayedMLP",
                                 "actor_hidden_sizes": [16], "privileged_critic": False},
    "FlatRecurrent": {"network_class": "FlatRecurrent", "actor_pre_hidden_sizes": [16],
                      "rnn_hidden_sizes": [8], "actor_post_hidden_sizes": [16]},
    "FlatForwardModel": {"network_class": "FlatForwardModel", "actor_hidden_sizes": [16],
                         "predictor_hidden_sizes": [16]},
}


def _student(name, env=None, delay=3):
    params = {**STUDENTS[name], "critic_hidden_sizes": [16],
              "delay_k": delay, "efference_length": delay}
    return nb.build_network(params, env or FlatStubEnv(), nnx.Rngs(1))


def _teacher(env=None):
    return nb.build_network(TEACHER, env or FlatStubEnv(), nnx.Rngs(0))


class SamplerTargetTest(parameterized.TestCase):

    def _extras(self, net):
        return net(net.initialize_state(BATCH), flat_stub_obs()).rollout_extras

    @parameterized.named_parameters(*((n, n) for n in STUDENTS))
    def test_student_shaped_with_the_teacher_action(self, student_name):
        teacher, student = _teacher(), _student(student_name)
        teacher.eval()
        t_extras, s_extras = self._extras(teacher), self._extras(student)
        # The premise: the trees differ, which is why the mapping exists.
        self.assertNotEqual(jax.tree.structure(t_extras), jax.tree.structure(s_extras))

        target = distill.sampler_target(t_extras, s_extras)
        self.assertEqual(jax.tree.structure(target), jax.tree.structure(s_extras))
        _, t_action = distill._action_leaf(t_extras, "teacher")
        _, got = distill._action_leaf(target, "target")
        self.assertTrue(jp.array_equal(got, t_action))
        self.assertEqual(got.shape, (BATCH, ACTION_SIZE))

        # And the student can replay against it -- the loss-replay contract.
        replay = student(student.initialize_state(BATCH), flat_stub_obs(), target)
        self.assertTrue(jp.all(jp.isfinite(replay.output.loglikelihoods)))

    def test_normalizer_leaf_is_the_students(self):
        teacher, student = _teacher(), _student("DelayedMLP")
        s_extras = self._extras(student)
        target = distill.sampler_target(self._extras(teacher), s_extras)
        self.assertIs(target[0], s_extras[0])

    def test_refuses_mismatched_action_shapes(self):
        t = [None, {"action": [jp.zeros((BATCH, 2))], "value": None}]
        s = [None, {"action": [jp.zeros((BATCH, 3))], "value": None}]
        with self.assertRaisesRegex(ValueError, "same space"):
            distill.sampler_target(t, s)

    def test_refuses_several_samplers(self):
        t = {"action": {"a": jp.zeros((BATCH, 2)), "b": jp.zeros((BATCH, 2))}}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            distill.sampler_target(t, t)


def _cartpole():
    return mujoco_playground.registry.load(
        "CartpoleBalance", config_overrides={"impl": "jax"})


class EndToEndTest(absltest.TestCase):

    def test_distillation_step_with_a_delayed_student(self):
        env = _cartpole()
        teacher, student = _teacher(env), _student("FlatRecurrent", env)
        teacher.eval()
        state = distillation.new_distillation_state(env, teacher, student, 4, seed=0)
        state, metrics = distillation.distillation_step(
            env, teacher, state, 4, 4, 2, 2, LoggingLevel.LOSSES, None,
            distill.sampler_target)
        self.assertEqual(int(state.steps_taken), 16)
        nll = [v for k, v in metrics.items() if k.startswith("losses/distillation_nll")]
        self.assertNotEmpty(nll)
        self.assertTrue(all(jp.all(jp.isfinite(v)) for v in nll))

    def test_load_teacher_round_trip(self):
        env = _cartpole()
        env_config = config_dict.create(task="CartpoleBalance", episode_length=1000)
        teacher = _teacher(env)
        # Saved with its TrainConfig, as every real run is: load_network rebuilds the
        # optimizer template from it, and the clipping setting changes its tree.
        train_config = TrainConfig()
        ts = ppo.new_training_state(
            env, teacher, n_envs=1, seed=0,
            gradient_clipping=train_config.ppo.gradient_clipping)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "CartpoleBalance_DelayedMLP_delay0_eff0-x"
            run_dir.mkdir()
            make_checkpoint_fn(str(run_dir), train_config, include_env_state=False)(ts, 0)
            (run_dir / "config.json").write_text(json.dumps(
                {"env_params": env_config.to_dict(), "net_params": TEACHER}, default=str))

            # Resolved under checkpoint_root when not found as given.
            loaded = distill.load_teacher(run_dir.name, tmp, env=env, env_config=env_config,
                                          obs_layout="flat", seed=7)

            with self.assertRaises(OverrideError):
                distill.load_teacher(run_dir.name, tmp, env=env, env_config=env_config,
                                     obs_layout="dict", seed=7)
            with self.assertRaises(OverrideError):
                distill.load_teacher("missing", tmp, env=env, env_config=env_config,
                                     obs_layout="flat", seed=7)

        self.assertEqual(loaded.step, 0)
        self.assertEqual(loaded.info()["teacher_delay_k"], 0)
        want = jax.tree.leaves(nnx.state(teacher, nnx.Param))
        got = jax.tree.leaves(nnx.state(loaded.network, nnx.Param))
        for a, b in zip(want, got):
            self.assertTrue(jp.array_equal(a, b))


if __name__ == "__main__":
    absltest.main()
