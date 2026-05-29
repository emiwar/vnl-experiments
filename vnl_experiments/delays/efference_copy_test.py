"""Tests for EfferenceCopy."""

from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.containers import Sequential
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput

from vnl_experiments.delays.efference_copy import EfferenceCopy


class _ProbeSampler(StatefulModule):
    """Tiny sampler-like module that returns a deterministic action and
    records the augmented input it received for inspection.

    Used to verify that EfferenceCopy concatenates the queue correctly.
    """

    def __init__(self, action_size: int):
        self.action_size = action_size

    def __call__(self, state, x, rollout_extras=None):
        # action == first `action_size` features of the input, so the test
        # can also feed deterministic actions back through the queue.
        action = x[..., : self.action_size]
        return StatefulModuleOutput(
            next_state=(),
            output={
                "action": action,
                "log_likelihood": jp.zeros(x.shape[0]),
                "augmented_input": x,  # diagnostic — see test below
            },
            regularization_loss=jp.zeros(x.shape[0]),
            metrics={},
            rollout_extras=None,
        )

    def initialize_state(self, batch_size: int):
        return ()


class EfferenceCopyTest(absltest.TestCase):

    def test_rejects_negative_queue_length(self):
        with self.assertRaises(ValueError):
            EfferenceCopy(_ProbeSampler(2), jp.zeros((2,)), queue_length=-1)

    def test_l0_is_passthrough(self):
        sampler = _ProbeSampler(action_size=3)
        ec = EfferenceCopy(sampler, jp.zeros((3,)), queue_length=0)
        state = ec.initialize_state(batch_size=4)
        self.assertNotIn("queue", state)
        x = jp.arange(4 * 5, dtype=jp.float32).reshape(4, 5)
        out = ec(state, x)
        self.assertTrue(jp.allclose(out.output["action"], x[:, :3]))
        self.assertTrue(jp.allclose(out.output["augmented_input"], x))

    def test_initial_state_shapes(self):
        sampler = _ProbeSampler(action_size=3)
        ec = EfferenceCopy(sampler, jp.zeros((3,)), queue_length=4)
        state = ec.initialize_state(batch_size=2)
        self.assertEqual(state["queue"].shape, (2, 4, 3))

    def test_queue_starts_zero_then_fills_newest_first(self):
        action_size = 2
        obs_size = 3
        L = 3
        batch = 1
        sampler = _ProbeSampler(action_size=action_size)
        ec = EfferenceCopy(sampler, jp.zeros((action_size,)), queue_length=L)
        state = ec.initialize_state(batch)

        # Drive distinguishable actions by setting the first action_size
        # entries of each obs to t. The sampler echoes those as the action.
        recorded_inputs = []
        recorded_actions = []
        for t in range(6):
            obs = jp.full((batch, obs_size), float(t + 1))
            out = ec(state, obs)
            recorded_inputs.append(out.output["augmented_input"])
            recorded_actions.append(out.output["action"])
            state = out.next_state

        # At step 0 the queue is all zeros, so augmented_input[..., obs_size:]
        # should be zero.
        self.assertTrue(
            jp.allclose(recorded_inputs[0][:, obs_size:], 0.0),
            msg="step 0 should see an all-zero queue",
        )

        # At step t > 0, the augmented input's queue slot should be
        # [a_{t-1}, a_{t-2}, ..., a_{t-L}] in that order (newest first),
        # with zeros for steps before the episode started.
        for t in range(1, 6):
            queue_part = recorded_inputs[t][0, obs_size:]
            queue_reshaped = queue_part.reshape(L, action_size)
            for slot in range(L):
                prev_step = t - 1 - slot
                if prev_step < 0:
                    expected = jp.zeros(action_size)
                else:
                    expected = recorded_actions[prev_step][0]
                self.assertTrue(
                    jp.allclose(queue_reshaped[slot], expected),
                    msg=(
                        f"step {t}, queue slot {slot}: "
                        f"expected action from step {prev_step}, "
                        f"got {queue_reshaped[slot]}, expected {expected}"
                    ),
                )

    def test_reset_state_zeros_queue(self):
        sampler = _ProbeSampler(action_size=2)
        ec = EfferenceCopy(sampler, jp.zeros((2,)), queue_length=3)
        state = ec.initialize_state(batch_size=2)
        # Dirty the queue
        for t in range(5):
            obs = jp.full((2, 4), float(t + 1))
            state = ec(state, obs).next_state
        self.assertFalse(jp.allclose(state["queue"], 0.0))
        reset = ec.reset_state(state)
        self.assertTrue(jp.allclose(reset["queue"], 0.0))

    def test_jit_compatible(self):
        sampler = _ProbeSampler(action_size=2)
        ec = EfferenceCopy(sampler, jp.zeros((2,)), queue_length=3)
        state = ec.initialize_state(batch_size=2)
        graph, params = nnx.split(ec)

        @jax.jit
        def step(graph, params, state, obs):
            ec_local = nnx.merge(graph, params)
            out = ec_local(state, obs)
            return out.next_state, out.output["action"]

        obs = jp.ones((2, 4))
        new_state, action = step(graph, params, state, obs)
        self.assertEqual(action.shape, (2, 2))
        self.assertEqual(new_state["queue"].shape, (2, 3, 2))

    def test_minibatch_slicing(self):
        """The carry leaves all have a leading batch axis, so we should be
        able to slice the state along it and run a subset independently."""
        sampler = _ProbeSampler(action_size=2)
        ec = EfferenceCopy(sampler, jp.zeros((2,)), queue_length=3)
        state = ec.initialize_state(batch_size=8)
        # Run a couple of steps to fill the queue with distinguishable values
        for t in range(2):
            obs = jp.arange(8.0).reshape(8, 1) * jp.ones((8, 5)) + t
            state = ec(state, obs).next_state

        obs_next = jp.ones((8, 5)) * 99
        out_full = ec(state, obs_next)
        # Slice halves
        state_mb1 = jax.tree.map(lambda v: v[:4], state)
        state_mb2 = jax.tree.map(lambda v: v[4:], state)
        out_mb1 = ec(state_mb1, obs_next[:4])
        out_mb2 = ec(state_mb2, obs_next[4:])

        self.assertTrue(
            jp.allclose(out_full.output["action"][:4], out_mb1.output["action"])
        )
        self.assertTrue(
            jp.allclose(out_full.output["action"][4:], out_mb2.output["action"])
        )

    def test_inside_sequential(self):
        """A real-ish pipeline: Dense -> EfferenceCopy(probe). Shapes must
        line up and the network must run without error."""
        rngs = nnx.Rngs(0)
        obs_size = 4
        action_size = 2
        L = 2
        encoder = Dense(obs_size, obs_size, rngs)
        sampler = _ProbeSampler(action_size=action_size)
        ec = EfferenceCopy(sampler, jp.zeros((action_size,)), queue_length=L)
        # EfferenceCopy concatenates the queue (L * action_size) to the obs,
        # so it expects obs_size + L * action_size input dim. Put a second
        # Dense after Sequential's encoder to match the dim.
        net = Sequential([encoder, ec])
        state = net.initialize_state(batch_size=3)
        # encoder output: [3, obs_size] = [3, 4]; ec concats L * action_size = 4
        # -> probe sees [3, 8]; probe.action = first 2 features.
        obs = jp.ones((3, obs_size))
        out = net(state, obs)
        self.assertEqual(out.output["action"].shape, (3, action_size))


if __name__ == "__main__":
    absltest.main()
