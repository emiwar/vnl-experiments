"""Tests for ForwardModel and EfferenceCopy's dict-injection mode."""

from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.containers import Sequential
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput
from nnx_ppo.networks.utils import Flattener, Map

from vnl_experiments.delays.efference_copy import EfferenceCopy
from vnl_experiments.delays.forward_model import ForwardModel


LATENT = 3
PROPRIO = 4
ACTION = 2
L = 3  # efference queue length


def _make_decoder(rngs, latent=LATENT, proprio=PROPRIO, action=ACTION):
    return Sequential(
        [
            *make_mlp_layers(
                [latent + proprio, 8, action * 2],
                rngs,
                nnx.swish,
                activation_last_layer=False,
            ),
            NormalTanhSampler(rngs, entropy_weight=1e-2, min_std=1e-1),
        ]
    )


def _make_predictor(rngs, efference_length=L, proprio=PROPRIO, action=ACTION):
    return make_mlp(
        [proprio + efference_length * action, 16, proprio],
        rngs,
        nnx.swish,
        activation_last_layer=False,
    )


def _make_fm(rngs, delay_steps=2, efference_length=L, loss_weight=1.0):
    return ForwardModel(
        decoder=_make_decoder(rngs),
        predictor=_make_predictor(rngs, efference_length=efference_length),
        proprio_size=PROPRIO,
        delay_steps=delay_steps,
        loss_weight=loss_weight,
    )


def _make_input(batch, with_efference=True, key=jax.random.key(0)):
    k1, k2, k3 = jax.random.split(key, 3)
    x = {
        "task_obs": jax.random.normal(k1, (batch, LATENT)),
        "proprioception": jax.random.normal(k2, (batch, PROPRIO)),
    }
    if with_efference:
        x["efference"] = jax.random.normal(k3, (batch, L, ACTION))
    return x


class _ProbeSampler(StatefulModule):
    """Records the (possibly dict) input it received and echoes an action."""

    def __init__(self, action_size: int):
        self.action_size = action_size

    def __call__(self, state, x, rollout_extras=None):
        # In dict mode the probe pulls its action from "proprioception".
        ref = x["proprioception"] if isinstance(x, dict) else x
        action = ref[..., : self.action_size]
        return StatefulModuleOutput(
            next_state=(),
            output={
                "action": action,
                "log_likelihood": jp.zeros(ref.shape[0]),
                "received": x,
            },
            regularization_loss=jp.zeros(ref.shape[0]),
            metrics={},
            rollout_extras=None,
        )

    def initialize_state(self, batch_size: int):
        return ()


class ForwardModelTest(absltest.TestCase):

    def test_forward_shapes(self):
        rngs = nnx.Rngs(0)
        fm = _make_fm(rngs)
        batch = 5
        state = fm.initialize_state(batch)
        out = fm(state, _make_input(batch))

        self.assertEqual(out.output["action"].shape, (batch, ACTION))
        self.assertEqual(out.regularization_loss.shape, (batch,))
        self.assertIn("fm_pred_mse", out.metrics)
        self.assertEqual(out.metrics["fm_pred_mse"].shape, (batch,))
        self.assertEqual(set(out.next_state), {"delay", "pred", "dec"})

    def test_rejects_negative_delay(self):
        with self.assertRaises(ValueError):
            _make_fm(nnx.Rngs(0), delay_steps=-1)

    def test_fm_loss_trains_only_predictor(self):
        """The self-supervised L2 must not touch decoder params."""
        rngs = nnx.Rngs(0)
        fm = _make_fm(rngs)
        batch = 6
        state = fm.initialize_state(batch)
        x = _make_input(batch)

        def fm_loss(model):
            return jp.mean(model(state, x).metrics["fm_pred_mse"])

        grads = nnx.grad(fm_loss)(fm)
        decoder_grads = jax.tree.leaves(grads["decoder"])
        predictor_grads = jax.tree.leaves(grads["predictor"])
        self.assertTrue(all(jp.allclose(g, 0.0) for g in decoder_grads))
        self.assertTrue(any(jp.any(g != 0.0) for g in predictor_grads))

    def test_fm_loss_independent_of_latent(self):
        """The L2 error does not depend on the task latent, so no gradient
        flows back into the (upstream) encoder."""
        rngs = nnx.Rngs(0)
        fm = _make_fm(rngs)
        batch = 6
        x1 = _make_input(batch, key=jax.random.key(3))
        x2 = {**x1, "task_obs": x1["task_obs"] + 100.0}
        mse1 = fm(fm.initialize_state(batch), x1).metrics["fm_pred_mse"]
        mse2 = fm(fm.initialize_state(batch), x2).metrics["fm_pred_mse"]
        self.assertTrue(jp.allclose(mse1, mse2))

    def test_policy_loss_does_not_touch_predictor(self):
        """The actor objective (via log-likelihood) must leave the predictor
        untouched because the prediction is stop-gradient'd into the decoder."""
        rngs = nnx.Rngs(0)
        fm = _make_fm(rngs)
        batch = 6
        state = fm.initialize_state(batch)
        x = _make_input(batch)

        def policy_loss(model):
            return jp.sum(model(state, x).output["log_likelihood"])

        grads = nnx.grad(policy_loss)(fm)
        decoder_grads = jax.tree.leaves(grads["decoder"])
        predictor_grads = jax.tree.leaves(grads["predictor"])
        self.assertTrue(all(jp.allclose(g, 0.0) for g in predictor_grads))
        self.assertTrue(any(jp.any(g != 0.0) for g in decoder_grads))

    def test_delay_zero_learns_identity(self):
        """delay=0 with no efference => predictor learns proprio_t -> proprio_t.
        A few SGD steps should reduce the reconstruction error."""
        rngs = nnx.Rngs(0)
        fm = _make_fm(rngs, delay_steps=0, efference_length=0)
        batch = 32
        state = fm.initialize_state(batch)
        self.assertEqual(state["delay"], ())
        x = _make_input(batch, with_efference=False, key=jax.random.key(1))

        def fm_loss(model):
            return jp.mean(model(state, x).metrics["fm_pred_mse"])

        l0 = fm_loss(fm)
        lr = 0.05
        for _ in range(200):
            grads = nnx.grad(fm_loss)(fm)
            params = nnx.state(fm, nnx.Param)
            params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
            nnx.update(fm, params)
        l1 = fm_loss(fm)
        self.assertLess(float(l1), 0.5 * float(l0))

    def test_init_reset_roundtrip_and_vmap_shapes(self):
        rngs = nnx.Rngs(0)
        fm = _make_fm(rngs)
        batch = 7
        state = fm.initialize_state(batch)
        # Delay buffer carries the leading batch axis.
        self.assertEqual(state["delay"]["buffer"].shape, (batch, 2, PROPRIO))
        self.assertEqual(state["delay"]["idx"].shape, (batch,))
        reset = fm.reset_state(state)
        self.assertTrue(jp.allclose(reset["delay"]["buffer"], 0.0))
        self.assertEqual(set(reset), {"delay", "pred", "dec"})

    def test_jit_compatible(self):
        rngs = nnx.Rngs(0)
        fm = _make_fm(rngs)
        batch = 4
        state = fm.initialize_state(batch)
        graph, params = nnx.split(fm)

        @jax.jit
        def step(graph, params, state, x):
            model = nnx.merge(graph, params)
            out = model(state, x)
            return out.next_state, out.output["action"]

        _, action = step(graph, params, state, _make_input(batch))
        self.assertEqual(action.shape, (batch, ACTION))


class EfferenceCopyDictModeTest(absltest.TestCase):

    def test_dict_injection_passes_queue_by_key(self):
        probe = _ProbeSampler(action_size=ACTION)
        ec = EfferenceCopy(
            probe, jp.zeros((ACTION,)), queue_length=L, inject_key="efference"
        )
        batch = 2
        state = ec.initialize_state(batch)
        obs = {
            "task_obs": jp.ones((batch, LATENT)),
            "proprioception": jp.ones((batch, PROPRIO)),
        }
        out = ec(state, obs)
        received = out.output["received"]
        self.assertIsInstance(received, dict)
        self.assertEqual(received["efference"].shape, (batch, L, ACTION))
        # Original keys are preserved untouched.
        self.assertTrue(jp.allclose(received["task_obs"], obs["task_obs"]))
        # First step queue is all zeros.
        self.assertTrue(jp.allclose(received["efference"], 0.0))

    def test_dict_injection_queue_updates_newest_first(self):
        probe = _ProbeSampler(action_size=ACTION)
        ec = EfferenceCopy(
            probe, jp.zeros((ACTION,)), queue_length=L, inject_key="efference"
        )
        batch = 1
        state = ec.initialize_state(batch)
        actions = []
        for t in range(3):
            obs = {
                "task_obs": jp.zeros((batch, LATENT)),
                "proprioception": jp.full((batch, PROPRIO), float(t + 1)),
            }
            out = ec(state, obs)
            actions.append(out.output["action"])
            state = out.next_state
        # After 3 steps the queue's newest slot holds the last action.
        self.assertTrue(jp.allclose(state["queue"][:, 0], actions[-1]))

    def test_l0_dict_passthrough(self):
        probe = _ProbeSampler(action_size=ACTION)
        ec = EfferenceCopy(
            probe, jp.zeros((ACTION,)), queue_length=0, inject_key="efference"
        )
        state = ec.initialize_state(batch_size=2)
        obs = {
            "task_obs": jp.ones((2, LATENT)),
            "proprioception": jp.ones((2, PROPRIO)),
        }
        out = ec(state, obs)
        # No efference key injected when there is no queue.
        self.assertNotIn("efference", out.output["received"])


class FullActorIntegrationTest(absltest.TestCase):

    def test_actor_pipeline_runs_and_exposes_metric(self):
        rngs = nnx.Rngs(0)
        task_obs_size = LATENT  # encoder consumes the task obs directly here
        encoder = Sequential(
            make_mlp_layers(
                [task_obs_size, 8, LATENT],
                rngs,
                nnx.swish,
                activation_last_layer=False,
            )
        )
        fm = ForwardModel(
            decoder=_make_decoder(rngs),
            predictor=_make_predictor(rngs),
            proprio_size=PROPRIO,
            delay_steps=2,
        )
        actor = Sequential(
            [
                Map(task_obs=encoder, proprioception=Flattener()),
                EfferenceCopy(
                    inner=fm,
                    sample_action=jp.zeros(ACTION),
                    queue_length=L,
                    inject_key="efference",
                ),
            ]
        )
        batch = 5
        state = actor.initialize_state(batch)
        obs = {
            "task_obs": jax.random.normal(jax.random.key(0), (batch, task_obs_size)),
            "proprioception": jax.random.normal(
                jax.random.key(1), (batch, PROPRIO)
            ),
        }
        out = actor(state, obs)
        self.assertEqual(out.output["action"].shape, (batch, ACTION))
        # Sequential keys metrics by layer index; EfferenceCopy forwards the
        # ForwardModel metrics, which carry fm_pred_mse.
        self.assertIn("fm_pred_mse", out.metrics[1])
        self.assertEqual(out.metrics[1]["fm_pred_mse"].shape, (batch,))


if __name__ == "__main__":
    absltest.main()
