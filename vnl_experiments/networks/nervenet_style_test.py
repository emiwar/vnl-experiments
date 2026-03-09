"""Tests for NerveNetNetwork: forward pass and checkpoint round-trip."""

import os
import shutil
import sys
import tempfile

from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx
import optax

# Allow imports from sibling repos when running from vnl-experiments/
_HERE = os.path.dirname(__file__)
_VNL = os.path.dirname(os.path.dirname(_HERE))
for _p in ["nnx-ppo"]:
    _d = os.path.join(_VNL, _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

from networks import nervenet_style
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn, load_checkpoint
from nnx_ppo.algorithms.types import TrainingState

# ---------------------------------------------------------------------------
# Module names that NerveNetNetwork.__call__ references by name
# ---------------------------------------------------------------------------
_ALL_OBS_MODULES = [
    "hand_L", "arm_L", "hand_R", "arm_R",
    "foot_L", "leg_L", "foot_R", "leg_R",
    "torso", "head", "root",
]
_ACTION_MODULES = [
    "hand_L", "arm_L", "hand_R", "arm_R",
    "foot_L", "leg_L", "foot_R", "leg_R",
    "torso", "head",
]

_OBS_DIM = 6      # small synthetic obs dim per module
_HIDDEN = 8       # small hidden size for fast tests
_ACTION_DIM = 2   # actions per module


def _make_network(seed: int = 0) -> nervenet_style.NerveNetNetwork:
    obs_sizes = {mod: _OBS_DIM for mod in _ALL_OBS_MODULES}
    action_sizes = {mod: _ACTION_DIM for mod in _ACTION_MODULES}
    return nervenet_style.NerveNetNetwork(
        obs_sizes, action_sizes, _HIDDEN, rngs=nnx.Rngs(seed)
    )


def _make_training_state(network, seed: int = 0) -> TrainingState:
    optimizer = nnx.Optimizer(network, optax.adam(1e-4), wrt=nnx.Param)
    return TrainingState(
        networks=network,
        network_states=network.initialize_state(batch_size=()),
        env_states=None,
        optimizer=optimizer,
        rng_key=jax.random.key(seed),
        steps_taken=jp.array(0.0),
    )


def _synthetic_obs(batch: int = 1) -> dict:
    """Return a dict of synthetic obs arrays, one per module."""
    return {mod: jp.ones((batch, _OBS_DIM)) for mod in _ALL_OBS_MODULES}


# ---------------------------------------------------------------------------
# Forward-pass smoke test
# ---------------------------------------------------------------------------

class NerveNetForwardTest(absltest.TestCase):

    def setUp(self):
        self.network = _make_network(seed=1)
        self.obs = _synthetic_obs(batch=2)

    def test_output_shapes(self):
        _, out = self.network((), self.obs)
        for mod in _ACTION_MODULES:
            self.assertIn(mod, out.actions)
            self.assertEqual(out.actions[mod].shape, (2, _ACTION_DIM))
        for mod in _ALL_OBS_MODULES:
            self.assertIn(mod, out.value_estimates)
            self.assertEqual(out.value_estimates[mod].shape, (2,))

    def test_loglikelihoods_has_batch_shape(self):
        _, out = self.network((), self.obs)
        self.assertEqual(out.loglikelihoods.shape, (2,))

    def test_initialize_state_returns_empty_tuple(self):
        state = self.network.initialize_state(batch_size=4)
        self.assertEqual(state, ())


# ---------------------------------------------------------------------------
# Checkpoint round-trip tests
# ---------------------------------------------------------------------------

class NerveNetCheckpointTest(absltest.TestCase):

    def setUp(self):
        self.network = _make_network(seed=17)
        self.state = _make_training_state(self.network, seed=42)

    def _fresh_state(self, seed: int = 99) -> TrainingState:
        net = _make_network(seed=seed)
        return _make_training_state(net, seed=seed)

    def test_round_trip_weights_preserved(self):
        """Checkpoint save + load_checkpoint produces identical forward-pass output."""
        tmpdir = tempfile.mkdtemp()
        try:
            fn = make_checkpoint_fn(tmpdir)
            fn(self.state, step=500)

            template = self._fresh_state()
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000000500"),
                template.networks,
                template.optimizer,
            )
            loaded = ckpt["training_state"]

            obs = _synthetic_obs(batch=2)
            _, out_orig = self.state.networks(self.state.network_states, obs)
            _, out_loaded = loaded.networks(loaded.network_states, obs)

            for mod in _ACTION_MODULES:
                self.assertTrue(
                    jp.allclose(out_orig.actions[mod], out_loaded.actions[mod]),
                    f"Actions for {mod} differ after checkpoint round-trip.",
                )
            for mod in _ALL_OBS_MODULES:
                self.assertTrue(
                    jp.allclose(
                        out_orig.value_estimates[mod],
                        out_loaded.value_estimates[mod],
                    ),
                    f"Value estimates for {mod} differ after checkpoint round-trip.",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_normalizer_statistics_preserved(self):
        """Normalizer counter (NormalizerStatistics variable) survives a checkpoint round-trip."""
        tmpdir = tempfile.mkdtemp()
        try:
            self.state.networks.normalizer.counter.set_value(jp.array(1234.0))

            fn = make_checkpoint_fn(tmpdir)
            fn(self.state, step=100)

            template = self._fresh_state()
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000000100"),
                template.networks,
                template.optimizer,
            )
            loaded = ckpt["training_state"]

            self.assertAlmostEqual(
                float(loaded.networks.normalizer.counter.get_value()),
                1234.0,
                msg="Normalizer counter not preserved after checkpoint.",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_network_weights_viewer_helper(self):
        """load_network_weights() from policy_viewer restores the same weights."""
        # Import here so path setup has already happened
        sys.path.insert(0, os.path.join(_VNL, "vnl-experiments"))
        from tools.policy_viewer import load_network_weights  # noqa: PLC0415

        tmpdir = tempfile.mkdtemp()
        try:
            fn = make_checkpoint_fn(tmpdir)
            fn(self.state, step=0)

            template_net = _make_network(seed=99)
            load_network_weights(
                os.path.join(tmpdir, "step_0000000000"), template_net
            )

            obs = _synthetic_obs(batch=2)
            _, out_orig = self.state.networks(self.state.network_states, obs)
            _, out_loaded = template_net(template_net.initialize_state(batch_size=()), obs)

            for mod in _ALL_OBS_MODULES:
                self.assertTrue(
                    jp.allclose(
                        out_orig.value_estimates[mod],
                        out_loaded.value_estimates[mod],
                    ),
                    f"Value estimates for {mod} differ after load_network_weights.",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    absltest.main()