"""Tests for the ``weights`` artifact kind.

The checkpoints here are written by the real ``nnx_ppo`` ``make_checkpoint_fn`` on a
small CartpoleBalance MLP, so the export reads exactly the on-disk layout training
produces. What is pinned:

* the export round-trips every non-``RngKey`` variable bit-exactly, and
  :func:`restore_into` puts them back into a freshly built network;
* the spec_id depends only on the step (that is what lets ``pull`` compute file names);
* step selection, including that half-written ``.tmp-step_*`` dirs never count;
* a step rewritten after export is detected as stale and re-exported by ``ensure``;
* manifest rows for weights are compact while the sidecar keeps the full stamp.
"""

from __future__ import annotations

import json
import shutil

import jax
import mujoco_playground
import numpy as np
import pytest
from flax import nnx
from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn
from nnx_ppo.networks import factories

from vnl_experiments.artifacts import weights as W
from vnl_experiments.artifacts.producers import get_producer
from vnl_experiments.artifacts.store import Store, manifest_row, read_manifest


@pytest.fixture(scope="module")
def env():
    return mujoco_playground.registry.load("CartpoleBalance",
                                           config_overrides={"impl": "jax"})


def make_nets(env, seed):
    return factories.make_mlp_actor_critic(env.observation_size, env.action_size,
                                           [8, 8], [8], nnx.Rngs(seed))


@pytest.fixture
def run_dir(tmp_path, env):
    """A run directory with checkpoints at three steps (weights differ per step)."""
    run = tmp_path / "ckpt" / "runA"
    nets = make_nets(env, seed=1)
    state = ppo.new_training_state(env, nets, n_envs=2, seed=0)
    fn = make_checkpoint_fn(str(run), include_env_state=False)
    for i, step in enumerate((0, 1000, 2000)):
        _perturb(nets, i)
        fn(state, step)
    (run / "config.json").write_text(json.dumps(
        {"env_params": {"x": 1}, "net_params": {"network_class": "MLP"}}))
    # A half-written checkpoint, which must never be selected.
    (run / ".tmp-step_0000003000" / "networks").mkdir(parents=True)
    return run


def _perturb(nets, i):
    params = nnx.state(nets, nnx.Param)
    nnx.update(nets, jax.tree_util.tree_map(lambda x: x + 0.1 * i, params))


def _non_key_flat(nets):
    _, _, state = nnx.split(nets, nnx.RngKey, ...)
    return {W.flat_key(p): np.asarray(v.get_value()) for p, v in nnx.to_flat_state(state)}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


def ensure(store, run_dir, steps, wid="runA"):
    producer = get_producer("weights")
    ctx = {"checkpoint_dir": run_dir}
    return [producer.ensure(store, wid, spec, ctx=ctx)
            for spec in producer.expand_specs(producer.spec(), ctx, steps)]


# --------------------------------------------------------------------------------------


class TestIdentity:
    def test_spec_id_depends_only_on_step(self):
        producer = get_producer("weights")
        assert producer.spec_id(producer.spec(step=1000)) == W.weights_spec_id(1000)
        assert W.weights_spec_id(1000) != W.weights_spec_id(2000)
        assert W.weights_spec_id(1000).startswith("w-s0000001000-")

    def test_step_round_trips_through_spec_id(self):
        assert W.step_from_spec_id(W.weights_spec_id(450007040)) == 450007040
        assert W.step_from_spec_id("eval3ds-382e9e69") is None

    def test_version_changes_the_id(self):
        assert W.weights_spec_id(5, version=1) != W.weights_spec_id(5, version=2)

    def test_spec_without_step_cannot_be_identified(self):
        producer = get_producer("weights")
        with pytest.raises(ValueError):
            producer.spec_id(producer.spec())


class TestStepSelector:
    AVAIL = [0, 10, 20, 30, 40, 50, 60]

    @pytest.mark.parametrize("text,expected", [
        ("last", [60]),
        (None, [60]),
        ("all", AVAIL),
        ("every:3", [0, 30, 60]),
        ("every:4", [0, 40, 60]),        # last appended when the stride misses it
        ("20,0", [0, 20]),
        ("20,999", [20]),                # absent explicit steps are dropped by select()
    ])
    def test_select(self, text, expected):
        assert W.StepSelector.parse(text).select(self.AVAIL) == expected

    def test_empty_run_selects_nothing(self):
        assert W.StepSelector.parse("last").select([]) == []

    @pytest.mark.parametrize("bad", ["every:0", "latest", "1,x"])
    def test_bad_selectors_raise(self, bad):
        with pytest.raises(ValueError):
            W.StepSelector.parse(bad)

    def test_checkpoint_steps_skips_tmp(self, run_dir):
        assert W.checkpoint_steps(run_dir) == [0, 1000, 2000]


class TestExport:
    def test_round_trip_is_exact(self, store, run_dir, env):
        ensure(store, run_dir, "all")
        assert W.available_steps("runA", store) == [0, 1000, 2000]

        w = W.load_weights("runA", 1000, store)
        assert w.step == 1000
        assert w.net_params == {"network_class": "MLP"}
        assert w.entry.resolved["n_nonfinite"] == 0
        # Reference: the same checkpoint restored by the training code's own loader.
        from nnx_ppo.algorithms.checkpointing import load_checkpoint
        ref = make_nets(env, seed=5)
        ts = ppo.new_training_state(env, ref, n_envs=2, seed=0)
        load_checkpoint(str(W.step_dir(run_dir, 1000)), ts.networks, ts.optimizer)
        expected = _non_key_flat(ts.networks)
        assert set(w.arrays) == set(expected)
        for k, v in expected.items():
            np.testing.assert_array_equal(w.arrays[k], v, err_msg=k)
        # Normaliser statistics are part of the weights.
        assert {"layers/0/mean", "layers/0/M2", "layers/0/counter"} <= set(w.arrays)

    def test_last_is_highest_step(self, store, run_dir):
        ensure(store, run_dir, "all")
        assert W.load_weights("runA", store=store).step == 2000

    def test_restore_into_fresh_network(self, store, run_dir, env):
        ensure(store, run_dir, "last")
        w = W.load_weights("runA", store=store)
        fresh = make_nets(env, seed=99)
        assert not all(np.array_equal(w.arrays[k], v)
                       for k, v in _non_key_flat(fresh).items())
        W.restore_into(fresh, w)
        for k, v in _non_key_flat(fresh).items():
            np.testing.assert_array_equal(v, w.arrays[k], err_msg=k)

    def test_restore_into_wrong_architecture_names_paths(self, store, run_dir, env):
        ensure(store, run_dir, "last")
        w = W.load_weights("runA", store=store)
        other = factories.make_mlp_actor_critic(env.observation_size, env.action_size,
                                                [8], [8], nnx.Rngs(0))
        with pytest.raises(ValueError, match="missing from weights|not in network"):
            W.restore_into(other, w)

    def test_iter_weights_in_order(self, store, run_dir):
        ensure(store, run_dir, "all")
        assert [w.step for w in W.iter_weights("runA", "every:2", store)] == [0, 2000]

    def test_explicit_missing_step_raises(self, store, run_dir):
        with pytest.raises(FileNotFoundError, match="999"):
            ensure(store, run_dir, "0,999")


class TestStaleness:
    def test_rewritten_step_is_reexported(self, store, run_dir, env):
        (entry,) = ensure(store, run_dir, "2000")
        before = W.load_weights("runA", 2000, store).arrays

        # A requeue rewind re-checkpoints the same step with different weights.
        nets = make_nets(env, seed=42)
        state = ppo.new_training_state(env, nets, n_envs=2, seed=0)
        make_checkpoint_fn(str(run_dir), include_env_state=False)(state, 2000)
        assert W.is_stale(entry, run_dir)

        (entry2,) = ensure(store, run_dir, "2000")
        assert entry2.resolved["source_sha256"] != entry.resolved["source_sha256"]
        after = W.load_weights("runA", 2000, store).arrays
        assert any(not np.array_equal(before[k], after[k]) for k in before)
        assert not W.is_stale(entry2, run_dir)

    def test_absent_source_is_not_stale(self, store, run_dir):
        (entry,) = ensure(store, run_dir, "last")
        shutil.rmtree(run_dir)
        assert not W.is_stale(entry, run_dir)


class TestManifest:
    def test_weights_rows_are_compact(self, store, run_dir, tmp_path):
        (entry,) = ensure(store, run_dir, "last")
        assert "packages" in entry.producer          # sidecar keeps the full stamp
        row = manifest_row(entry)
        assert set(row["producer"]) <= {"module", "version", "git_commit", "host"}
        assert row["producer"]["version"] == W.VERSION

        manifest = tmp_path / "manifest.jsonl"
        store.reindex(manifest)
        (read,) = read_manifest(manifest)
        assert read.spec == {"step": 2000}
        assert read.resolved["checkpoint_step"] == 2000
