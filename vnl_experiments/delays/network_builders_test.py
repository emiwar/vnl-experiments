"""Tests for the delays architecture registry.

Two jobs:

1. **Contract** -- every registered architecture builds, and its carry survives
   an ``initialize_state`` -> forward -> ``reset_state`` round trip with an
   unchanged pytree structure. This is what the offline rollouts in
   ``evaluation.py`` / ``record_activations.py`` / ``eval_videos.py`` rely on, and
   it is all a new architecture needs to satisfy to work with them.

2. **Regression** -- the two original feedforward architectures must keep
   building the exact same parameter tree. Their weights are restored from
   checkpoints by shape, and their ``param_counts`` land in stored eval
   artifacts, so a structural change silently breaks ~370 existing runs.

A stub env stands in for the rodent so these run on CPU in seconds.
"""

import json

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Concat, Sequential
from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.utils import Filter, Flattener, Map

from vnl_experiments.delays import network_builders as nb
from vnl_experiments.delays.efference_copy import EfferenceCopy

TASK_OBS = {"a": 10, "b": 7}
PROPRIO = {"c": 20}
ACTION_SIZE = 5
BATCH = 4


class StubEnv:
    """Only what the builders actually query."""

    non_flattened_observation_size = {
        "state": {"task_obs": TASK_OBS, "proprioception": PROPRIO}
    }
    action_size = ACTION_SIZE


def stub_obs(batch=BATCH):
    return {
        "state": {
            "task_obs": {k: jp.ones((batch, n)) for k, n in TASK_OBS.items()},
            "proprioception": {k: jp.ones((batch, n)) for k, n in PROPRIO.items()},
        }
    }


#: Small-but-complete net_params per architecture, plus the exact parameter count
#: the current code produces. The count pins every layer width at once, so any
#: accidental change to a builder's pipeline fails here.
SMALL = {
    "enc_hidden_sizes": [32],
    "critic_hidden_sizes": [32],
    "latent_size": "8",
    "delay_k": "5",
    "efference_length": "5",
}

CASES = {
    "RodentEncDecDelays": ({**SMALL, "dec_hidden_sizes": [24]}, 3899),
    "RodentForwardModel": (
        {**SMALL, "dec_hidden_sizes": [24], "fm_hidden_sizes": [24]},
        4903,
    ),
    "RodentEncDecRecurrent": (
        {
            **SMALL,
            "rnn_hidden_sizes": [16],
            "dec_pre_hidden_sizes": [24],
            "dec_post_hidden_sizes": [24],
        },
        6931,
    ),
}


def build(network_class, extra=None):
    net_params = {**CASES[network_class][0], "network_class": network_class,
                  **(extra or {})}
    return nb.build_network(net_params, StubEnv(), nnx.Rngs(0)), net_params


def param_count(module):
    return sum(int(leaf.size) for leaf in jax.tree.leaves(nnx.state(module, nnx.Param)))


def param_shapes(module):
    flat, _ = jax.tree_util.tree_flatten_with_path(nnx.state(module, nnx.Param))
    return {
        "/".join(str(getattr(k, "key", k)) for k in path): tuple(leaf.shape)
        for path, leaf in flat
    }


ALL_ARCHITECTURES = tuple((name, name) for name in sorted(nb.ARCHITECTURES))
RNN_CELL_NAMES = tuple((name, name) for name in sorted(nb.RNN_CELLS))


class RegistryTest(absltest.TestCase):

    def test_every_architecture_is_self_consistent(self):
        """``name`` is the registry key, and defaults() is callable."""
        for key, arch in nb.ARCHITECTURES.items():
            self.assertEqual(key, arch.name)
            self.assertIn("activation", arch.defaults())

    def test_get_architecture_exact_and_substring(self):
        self.assertEqual(
            nb.get_architecture("RodentEncDecDelays").name, "RodentEncDecDelays"
        )
        # The distillation scripts record str(type(student)), not a bare name.
        self.assertEqual(
            nb.get_architecture(
                "<class 'vnl_experiments.nets.RodentEncDecDelays'>"
            ).name,
            "RodentEncDecDelays",
        )
        self.assertIsNone(nb.get_architecture("NoSuchNetwork"))

    def test_build_network_warns_and_returns_none_for_unknown(self):
        with self.assertWarns(UserWarning):
            self.assertIsNone(
                nb.build_network({"network_class": "Nope"}, StubEnv(), nnx.Rngs(0))
            )

    def test_original_architectures_have_no_param_groups_hook(self):
        """Byte-stability guard.

        ``param_counts`` output for these two lands in ~370 stored eval artifacts.
        A ``param_groups`` hook would change their JSON, re-minting every eval
        ``spec_id``. Adding one is a producer VERSION bump, not a tidy-up.
        """
        for name in ("RodentEncDecDelays", "RodentForwardModel"):
            self.assertIsNone(nb.ARCHITECTURES[name].param_groups, name)


class ArchitectureContractTest(parameterized.TestCase):
    """What every architecture must satisfy to work with the offline rollouts."""

    @parameterized.named_parameters(*ALL_ARCHITECTURES)
    def test_builds_and_runs(self, network_class):
        nets, _ = build(network_class)
        out = nets(nets.initialize_state(BATCH), stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))
        self.assertEqual(out.output.value_estimates.shape, (BATCH,))
        self.assertFalse(jp.any(jp.isnan(out.output.actions)))

    @parameterized.named_parameters(*ALL_ARCHITECTURES)
    def test_carry_round_trip_preserves_structure(self, network_class):
        nets, _ = build(network_class)
        state = nets.initialize_state(BATCH)
        out = nets(state, stub_obs())
        reset = nets.reset_state(out.next_state)
        self.assertEqual(jax.tree.structure(out.next_state), jax.tree.structure(state))
        self.assertEqual(jax.tree.structure(reset), jax.tree.structure(state))
        for a, b in zip(jax.tree.leaves(reset), jax.tree.leaves(state)):
            self.assertEqual(a.shape, b.shape)

    @parameterized.named_parameters(*ALL_ARCHITECTURES)
    def test_param_count_is_pinned(self, network_class):
        nets, _ = build(network_class)
        self.assertEqual(param_count(nets), CASES[network_class][1])

    @parameterized.named_parameters(*ALL_ARCHITECTURES)
    def test_survives_config_json_round_trip(self, network_class):
        """``net_params`` must rebuild identically after config.json serialisation.

        The train script writes ``json.dump(..., default=str)``, so every value
        comes back as a string and ``_parse_net_params`` has to recover its type.
        A new key whose type does not survive that trip would silently rebuild a
        differently-shaped network -- and only fail later, at checkpoint restore.
        """
        nets, net_params = build(network_class)
        round_tripped = json.loads(json.dumps(net_params, default=str))
        rebuilt = nb.build_network(round_tripped, StubEnv(), nnx.Rngs(0))
        self.assertEqual(param_shapes(rebuilt), param_shapes(nets))

    @parameterized.named_parameters(*ALL_ARCHITECTURES)
    def test_defaults_build_at_full_size(self, network_class):
        """The registered defaults -- not just the small test config -- must build."""
        arch = nb.ARCHITECTURES[network_class]
        net_params = {
            **arch.defaults().to_dict(),
            "delay_k": 5,
            "efference_length": 5,
            "network_class": network_class,
        }
        nets = nb.build_network(net_params, StubEnv(), nnx.Rngs(0))
        out = nets(nets.initialize_state(2), stub_obs(2))
        self.assertEqual(out.output.actions.shape, (2, ACTION_SIZE))


class FeedforwardRegressionTest(absltest.TestCase):
    """Pin the pipeline of the two pre-existing architectures."""

    def test_delay_network_pipeline(self):
        nets, _ = build("RodentEncDecDelays")
        self.assertEqual(
            [type(l) for l in nets.layers],
            [Flattener, Normalizer, Filter, PPOAdapter],
        )
        adapter = nets.layers[-1]
        self.assertEqual([type(l) for l in adapter.action.layers],
                         [Concat, EfferenceCopy])
        self.assertIsInstance(adapter.action.layers[1].inner, Sequential)

    def test_forward_model_pipeline(self):
        nets, _ = build("RodentForwardModel")
        adapter = nets.layers[-1]
        self.assertEqual([type(l) for l in adapter.action.layers],
                         [Map, EfferenceCopy])

    def test_no_normalizer_when_disabled(self):
        nets, _ = build("RodentEncDecDelays", {"normalize_obs": "False"})
        self.assertEqual([type(l) for l in nets.layers],
                         [Flattener, Filter, PPOAdapter])

    def test_delay_layer_presence_follows_delay_k(self):
        for delay_k, expect_delay in (("5", True), ("0", False)):
            nets, _ = build("RodentEncDecDelays",
                            {"delay_k": delay_k, "efference_length": "5"})
            proprio = nets.layers[-1].action.layers[0].components["proprioception"]
            has_delay = any(isinstance(l, Delay) for l in proprio.layers)
            self.assertEqual(has_delay, expect_delay, f"delay_k={delay_k}")

    def test_default_initializer_scale_uses_nnx_defaults(self):
        """At scale 1.0 the builder must not override nnx's kernel_init.

        The training script has always used the nnx defaults, so overriding here
        would mean a fresh run and its offline rebuild start from different
        distributions -- and every future feedforward run would differ from the
        historical ones.
        """
        # critic = Sequential([Flattener(), *dense_layers]), so layers[1] is Dense.
        nets, _ = build("RodentEncDecDelays")
        default_kernel = nets.layers[-1].value.layers[1].linear.kernel[...]
        scaled, _ = build("RodentEncDecDelays", {"initializer_scale": "3.0"})
        scaled_kernel = scaled.layers[-1].value.layers[1].linear.kernel[...]
        self.assertEqual(default_kernel.shape, scaled_kernel.shape)
        self.assertFalse(jp.allclose(default_kernel, scaled_kernel))

    def test_forward_model_honours_detach_prediction(self):
        """Only affects gradients, but the train script sets it -- so it must apply."""
        for value, expected in (("True", True), ("False", False)):
            nets, _ = build("RodentForwardModel", {"detach_prediction": value})
            fm = nets.layers[-1].action.layers[1].inner
            self.assertEqual(fm.detach_prediction, expected)


class RecurrentArchitectureTest(parameterized.TestCase):

    @parameterized.named_parameters(*RNN_CELL_NAMES)
    def test_every_cell_builds_and_carries(self, cell):
        nets, _ = build("RodentEncDecRecurrent", {"rnn_cell": cell})
        state = nets.initialize_state(BATCH)
        out = nets(state, stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))
        # A recurrent decoder must actually carry something forward.
        decoder = nets.layers[-1].action.layers[1].inner
        rnn = [l for l in decoder.layers
               if isinstance(l, tuple(nb.RNN_CELLS.values()))]
        self.assertLen(rnn, 1)
        self.assertIsInstance(rnn[0], nb.RNN_CELLS[cell])

    @parameterized.named_parameters(*RNN_CELL_NAMES)
    def test_state_actually_advances(self, cell):
        """Same input twice from different carries must give different outputs."""
        nets, _ = build("RodentEncDecRecurrent", {"rnn_cell": cell})
        obs = stub_obs()
        first = nets(nets.initialize_state(BATCH), obs)
        second = nets(first.next_state, obs)
        self.assertFalse(jp.allclose(first.output.actions, second.output.actions))

    def test_depth_is_the_length_of_rnn_hidden_sizes(self):
        for sizes in ([16], [16, 16], [16, 16, 16]):
            nets, _ = build("RodentEncDecRecurrent", {"rnn_hidden_sizes": sizes})
            self.assertLen(self._rnn_layers(nets), len(sizes))

    def test_non_uniform_widths(self):
        """Cells are independent modules, so the stack may taper."""
        nets, _ = build("RodentEncDecRecurrent", {"rnn_hidden_sizes": [32, 16, 8]})
        rnn = self._rnn_layers(nets)
        self.assertEqual([l.in_features for l in rnn], [24, 32, 16])  # 24 = pre-MLP out
        self.assertEqual([l.hidden_features for l in rnn], [32, 16, 8])
        out = nets(nets.initialize_state(BATCH), stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))

    @staticmethod
    def _rnn_layers(nets):
        decoder = nets.layers[-1].action.layers[1].inner
        return [l for l in decoder.layers
                if isinstance(l, tuple(nb.RNN_CELLS.values()))]

    def test_efference_zero_is_passthrough(self):
        """`--efference 0` is the 'recurrence instead of efference copy' condition."""
        nets, _ = build("RodentEncDecRecurrent", {"efference_length": "0"})
        out = nets(nets.initialize_state(BATCH), stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))

    def test_empty_pre_and_post_mlps(self):
        nets, _ = build("RodentEncDecRecurrent",
                        {"dec_pre_hidden_sizes": [], "dec_post_hidden_sizes": []})
        out = nets(nets.initialize_state(BATCH), stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))

    def test_unknown_cell_is_rejected(self):
        with self.assertRaises(ValueError):
            build("RodentEncDecRecurrent", {"rnn_cell": "transformer"})

    def test_empty_rnn_reduces_to_the_feedforward_decoder(self):
        """The reduction test that makes `rnn_hidden_sizes=[]` worth allowing.

        With no recurrent layers the decoder is pre-MLP -> post-MLP -> sampler.
        The pre-MLP activates its last layer and the post-MLP does not, so the
        activation pattern is exactly the feedforward decoder's, and splitting
        `dec_hidden_sizes` across pre/post must give a structurally identical
        network. If this ever fails, the recurrent decoder is mis-wired
        independently of the cells.
        """
        recurrent, _ = build("RodentEncDecRecurrent", {
            "rnn_hidden_sizes": [],
            "dec_pre_hidden_sizes": [24, 18],
            "dec_post_hidden_sizes": [12],
        })
        feedforward, _ = build("RodentEncDecDelays",
                               {"dec_hidden_sizes": [24, 18, 12]})
        self.assertEqual(param_shapes(recurrent), param_shapes(feedforward))
        self.assertEqual(param_count(recurrent), param_count(feedforward))
        self.assertEmpty(self._rnn_layers(recurrent))
        # And it still runs, with a carry containing no recurrent state.
        out = recurrent(recurrent.initialize_state(BATCH), stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))

    def test_empty_rnn_with_empty_pre_mlp(self):
        """Degenerate corner: nothing before the post-MLP at all."""
        nets, _ = build("RodentEncDecRecurrent", {
            "rnn_hidden_sizes": [], "dec_pre_hidden_sizes": [],
            "dec_post_hidden_sizes": [12],
        })
        out = nets(nets.initialize_state(BATCH), stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))

    def test_param_groups_hook(self):
        nets, _ = build("RodentEncDecRecurrent")
        groups = nb.recurrent_param_groups(nets)
        self.assertEqual(set(groups), {"encoder", "decoder", "rnn"})
        self.assertGreater(groups["rnn"], 0)
        self.assertLess(groups["rnn"], groups["decoder"])

    def test_gated_cells_have_more_params_than_vanilla(self):
        counts = {
            cell: nb.recurrent_param_groups(
                build("RodentEncDecRecurrent", {"rnn_cell": cell})[0]
            )["rnn"]
            for cell in ("rnn", "gru", "lstm")
        }
        self.assertLess(counts["rnn"], counts["gru"])
        self.assertLess(counts["gru"], counts["lstm"])


if __name__ == "__main__":
    absltest.main()
