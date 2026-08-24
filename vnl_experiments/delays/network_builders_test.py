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
import warnings

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Concat, Sequential
from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.utils import Filter, Flattener, Map
from nnx_ppo.networks.variational import VariationalBottleneck

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


class ParseNetParamsTest(absltest.TestCase):
    """Regression guard for the 2026-08-24 float-truncation bug.

    ``_parse_net_params`` sits on the *training* path as well as the eval path, so a
    value it corrupts is a hyperparameter the run never had. The original code ran
    ``int(v)`` on every value; ``int(0.01) == 0``, so every sub-1.0 float silently
    became zero -- removing the entropy bonus, the KL penalty and the policy-std
    floor from every run trained after the registry refactor.
    """

    #: The four that were zeroed, with the values the defaults actually specify.
    SUB_ONE_FLOATS = {"entropy_weight": 0.01, "kl_weight": 0.001,
                      "min_std": 0.1, "latent_min_std": 0.01}

    def test_sub_one_floats_survive(self):
        parsed = nb._parse_net_params(dict(self.SUB_ONE_FLOATS))
        self.assertEqual(parsed, self.SUB_ONE_FLOATS)

    def test_every_architecture_default_survives_parsing(self):
        """The whole live net_config, for every architecture, must round-trip."""
        for name, arch in nb.ARCHITECTURES.items():
            defaults = arch.defaults().to_dict()
            parsed = nb._parse_net_params(defaults)
            for key, value in defaults.items():
                if isinstance(value, float):
                    self.assertEqual(parsed[key], value, f"{name}.{key}")

    def test_config_json_round_trip_preserves_floats(self):
        """As written by train_rodent.py and read back by the eval scripts."""
        defaults = nb.delay_defaults().to_dict()
        parsed = nb._parse_net_params(json.loads(json.dumps(defaults, default=str)))
        for key, value in self.SUB_ONE_FLOATS.items():
            self.assertEqual(parsed[key], value, key)

    def test_legacy_string_values_still_decode(self):
        """Old config.json files were stringified by json.dump(default=str)."""
        parsed = nb._parse_net_params({
            "latent_size": "32", "min_std": "0.1", "normalize_obs": "True",
            "detach_prediction": "False", "latent_ar1_weight": "None",
            "enc_hidden_sizes": ["512", "512"], "activation": "swish",
        })
        self.assertEqual(parsed, {
            "latent_size": 32, "min_std": 0.1, "normalize_obs": True,
            "detach_prediction": False, "latent_ar1_weight": None,
            "enc_hidden_sizes": [512, 512], "activation": "swish",
        })

    def test_builder_receives_the_configured_std_floor(self):
        """End-to-end: the floor must reach the sampler, not just the parser."""
        nets, _ = build("RodentEncDecDelays", {"min_std": 0.1, "entropy_weight": 0.01,
                                               "kl_weight": 0.001, "latent_min_std": 0.01})
        sampler = nets.layers[-1].action.layers[1].inner.layers[-1]
        self.assertEqual(sampler.min_std, 0.1)
        self.assertEqual(sampler.entropy_weight, 0.01)
        vb = nets.layers[-1].action.layers[0].components["task_obs"].layers[-1]
        self.assertEqual(vb.kl_weight, 0.001)
        self.assertEqual(vb.min_std, 0.01)


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


class DecoderInputAblationTest(absltest.TestCase):
    """``dec_use_intention`` / ``dec_use_proprioception``.

    The completeness ablations for the enc-dec decoder's three input streams (the
    third, the efference copy, is ablated by ``efference_length=0``). The invariant
    that matters most here is the first test: every ``net_params`` dict written
    before these flags existed must rebuild byte-for-byte as it did before.
    """

    #: task_obs 17 -> latent 8, proprio 20, efference 5 x 5 = 25.
    FULL_WIDTH = 8 + 20 + 25

    @staticmethod
    def _decoder_in(nets):
        decoder = nets.layers[-1].action.layers[1].inner
        return decoder.layers[0].linear.in_features

    @staticmethod
    def _streams(nets):
        return set(nets.layers[-1].action.layers[0].components)

    def test_absent_flags_rebuild_the_historical_network(self):
        """No keys == both keys True: the back-compat contract for old config.json."""
        old, _ = build("RodentEncDecDelays")
        new, _ = build("RodentEncDecDelays", {"dec_use_intention": True,
                                              "dec_use_proprioception": True})
        self.assertEqual(param_shapes(new), param_shapes(old))
        self.assertEqual(self._decoder_in(old), self.FULL_WIDTH)
        self.assertEqual(self._streams(old), {"task_obs", "proprioception"})

    def test_no_intention_drops_the_latent_and_the_encoder(self):
        nets, _ = build("RodentEncDecDelays", {"dec_use_intention": False})
        self.assertEqual(self._decoder_in(nets), self.FULL_WIDTH - 8)
        self.assertEqual(self._streams(nets), {"proprioception"})
        out = nets(nets.initialize_state(BATCH), stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))
        self.assertTrue(jp.all(jp.isfinite(out.output.actions)))

    def test_no_intention_has_no_variational_bottleneck(self):
        """The encoder is not built at all, so the run has no KL term to weight.

        Worth pinning separately: a zero-width latent would still carry a bottleneck
        (and would crash the metrics logger on its (B, 0) mu/sigma) -- this ablation
        removes the branch instead.
        """
        full, _ = build("RodentEncDecDelays")
        self.assertIsInstance(
            full.layers[-1].action.layers[0].components["task_obs"].layers[-1],
            VariationalBottleneck)
        nets, _ = build("RodentEncDecDelays", {"dec_use_intention": False})
        actor_modules = [m for _, m in nnx.iter_graph(nets.layers[-1].action)]
        self.assertFalse(any(isinstance(m, VariationalBottleneck)
                             for m in actor_modules))

    def test_no_proprioception_drops_that_branch_only(self):
        nets, _ = build("RodentEncDecDelays", {"dec_use_proprioception": False})
        self.assertEqual(self._decoder_in(nets), self.FULL_WIDTH - 20)
        self.assertEqual(self._streams(nets), {"task_obs"})
        out = nets(nets.initialize_state(BATCH), stub_obs())
        self.assertEqual(out.output.actions.shape, (BATCH, ACTION_SIZE))
        self.assertTrue(jp.all(jp.isfinite(out.output.actions)))

    def test_critic_is_never_ablated(self):
        """The ablations are actor-only -- the critic keeps its full, undelayed input."""
        full, _ = build("RodentEncDecDelays")
        expected = param_shapes(full.layers[-1].value)
        for flag in ("dec_use_intention", "dec_use_proprioception"):
            nets, _ = build("RodentEncDecDelays", {flag: False})
            self.assertEqual(param_shapes(nets.layers[-1].value), expected, flag)

    def test_ablating_both_is_rejected(self):
        with self.assertRaises(ValueError):
            build("RodentEncDecDelays", {"dec_use_intention": False,
                                         "dec_use_proprioception": False})

    def test_flags_survive_the_config_json_round_trip(self):
        """config.json stringifies everything, so "False" must still ablate."""
        for flag, dropped in (("dec_use_intention", 8),
                              ("dec_use_proprioception", 20)):
            nets, net_params = build("RodentEncDecDelays", {flag: False})
            rebuilt = nb.build_network(
                json.loads(json.dumps(net_params, default=str)),
                StubEnv(), nnx.Rngs(0))
            self.assertEqual(param_shapes(rebuilt), param_shapes(nets), flag)
            self.assertEqual(self._decoder_in(rebuilt), self.FULL_WIDTH - dropped)

    def test_defaults_carry_the_flags_on(self):
        defaults = nb.delay_defaults()
        self.assertTrue(defaults["dec_use_intention"])
        self.assertTrue(defaults["dec_use_proprioception"])
        # And they are delay-net only: an inert key on another architecture is the
        # `latent_ar1_weight` mistake.
        for other in ("RodentForwardModel", "RodentEncDecRecurrent"):
            self.assertNotIn("dec_use_intention",
                             nb.ARCHITECTURES[other].defaults())

    def test_param_counts_survives_a_missing_encoder(self):
        """The offline eval's semantic param groups must degrade, not raise."""
        from vnl_experiments.delays import evaluation

        nets, _ = build("RodentEncDecDelays", {"dec_use_intention": False})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            counts = evaluation.param_counts(nets, "RodentEncDecDelays")
        self.assertEqual(counts["total"], param_count(nets))
        self.assertNotIn("encoder", counts)
        self.assertIn("decoder", counts)


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
