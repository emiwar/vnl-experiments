"""The ``activations_dmc`` producer, end to end on tiny real checkpoints.

Each test checkpoint is written by nnx-ppo's own ``make_checkpoint_fn`` from a small
DelayedMLP or FlatForwardModel built by ``network_builders.build_network`` -- the training
path -- next to a ``config.json`` shaped like training's. The producer then rebuilds
the env and network from that directory, exactly as ``ensure`` does on the cluster.
``impl="jax"`` and a 40-step episode keep it fast and independent of Warp.
"""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest
from flax import nnx
from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn

from vnl_experiments.artifacts import weights as W
from vnl_experiments.artifacts.producers import get_producer
from vnl_experiments.artifacts.store import Store
from vnl_experiments.delays.network_builders import build_network
from vnl_experiments.dmc import record_activations as R
from vnl_experiments.envs import registry as env_registry
from vnl_experiments.envs import walker_courses
from vnl_experiments.probes.linear_decoding import Recording

STEP = 1000
N_STEPS = 40


def _env_params(**terrain):
    spec = env_registry.get("WalkerWalkTerrain")
    cfg = spec.default_config()
    cfg.impl = "jax"
    cfg.episode_length = N_STEPS
    for key, value in terrain.items():
        cfg.terrain[key] = value
    return cfg.to_dict()


# Four actor layers, as every walker-terrain run has: `Recording.action_leaf_name()`
# finds the sampler by the suffix "5/action", which only holds at that depth.
NETS = {
    "DelayedMLP": dict(network_class="DelayedMLP", actor_hidden_sizes=[8] * 4,
                       critic_hidden_sizes=[16], delay_k=2, efference_length=2,
                       privileged_critic=False),
    "FlatForwardModel": dict(network_class="FlatForwardModel", actor_hidden_sizes=[8] * 4,
                             critic_hidden_sizes=[16], predictor_hidden_sizes=[8] * 4,
                             delay_k=2, efference_length=2),
}


def _write_checkpoint(run_dir, net_params, env_params):
    env = R.build_env("WalkerWalkTerrain", env_params)
    nets = build_network(net_params, env, nnx.Rngs(3))
    state = ppo.new_training_state(env, nets, n_envs=2, seed=0)
    make_checkpoint_fn(str(run_dir), include_env_state=False)(state, STEP)
    (run_dir / "config.json").write_text(json.dumps(
        {"env_params": env_params, "net_params": net_params}))
    return nets


@pytest.fixture(scope="module", params=sorted(NETS))
def recorded(request, tmp_path_factory):
    """One recording per network class, on the training-distribution course set."""
    tmp = tmp_path_factory.mktemp(request.param)
    run_dir = tmp / "ckpt" / "run"
    _write_checkpoint(run_dir, NETS[request.param], _env_params(height=0.3, irregularity=1.0))
    store = Store(tmp / "store")
    producer = get_producer("activations_dmc")
    spec = producer.spec(course="h0.3_irr1", step=STEP, n_episodes=2)
    entry = producer.ensure(store, "run", spec, ctx={
        "checkpoint_dir": run_dir, "env_class": "WalkerWalkTerrain", "wandb_name": "run"})
    return request.param, entry, store.root / entry.path, run_dir


class TestRecording:
    def test_readable_by_the_probe_reader(self, recorded):
        name, entry, path, _ = recorded
        rec = Recording(path)
        assert rec.action_leaf_name().endswith("5/action")
        assert entry.resolved["network_class"] == name
        assert entry.resolved["n_episodes"] == 2 and entry.resolved["n_steps"] == N_STEPS

    def test_shapes_and_contents(self, recorded):
        _, entry, path, _ = recorded
        import h5py

        with h5py.File(path) as f:
            t, n = N_STEPS, 2
            assert f["target_proprio"].shape[:2] == (t, n)
            assert f["qpos"].shape == (t, n, 9) and f["qvel"].shape == (t, n, 9)
            assert f["actions"].shape == (t, n, 6) and f["reward"].shape == (t, n)
            assert f["reward_terms"].shape == (t, n, 4)
            assert f["foot_pos"].shape == (t, n, 2, 2)
            assert f["foot_ground_height"].shape == (t, n, 2)
            assert f["block_sensor"].shape == (t, n, 4)
            # Episodes walk their own courses, as named.
            assert list(f.attrs["course_names"]) == ["h0.3_irr1_s0", "h0.3_irr1_s1"]
            for i, cname in enumerate(f.attrs["course_names"]):
                c = walker_courses.load_course(cname)
                np.testing.assert_array_equal(f["course_x"][i, : c.n_blocks], c.x)
            # The block sensor is recorded although this run is blind: the first block
            # is start_gap ahead of the origin at t=0.
            assert not f.attrs["terrain_observed"]
            assert np.all(f["block_sensor"][0, :, 0] > 1.0)
            assert np.all(f["block_sensor"][0, :, 1] >= 0)
            # The recorded action is the sampler's (deterministic) action leaf.
            leaf = Recording(path).action_leaf_name()
            np.testing.assert_allclose(f["activations"][leaf][:], f["actions"][:],
                                       atol=2e-3)

    def test_provenance(self, recorded):
        _, entry, _, run_dir = recorded
        r = entry.resolved
        assert r["checkpoint_step"] == STEP
        assert r["weights_spec_id"] == W.weights_spec_id(STEP)
        assert r["source_sha256"] == W.directory_sha256(W.step_dir(run_dir, STEP) / "networks")
        assert r["terrain"]["height"] == 0.3 and r["impl"] == "jax"
        assert entry.spec["course_sha256"] == walker_courses.set_sha256("h0.3_irr1")


class TestCourseSets:
    def test_flat_has_no_ground_under_the_feet(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_checkpoint(run_dir, NETS["DelayedMLP"], _env_params(height=0.0))
        config, arrays, _ = R.load_checkpoint_step(run_dir, STEP)
        out = tmp_path / "flat.h5"
        R.record(out, task="WalkerWalkTerrain", config=config, arrays=arrays, step=STEP,
                 course_set="flat", n_episodes=2)
        import h5py

        with h5py.File(out) as f:
            assert np.all(f["foot_ground_height"][:] == 0)
            assert list(f.attrs["course_names"]) == ["flat", "flat"]

    def test_running_off_the_course_raises(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        _write_checkpoint(run_dir, NETS["DelayedMLP"], _env_params(height=0.3))
        config, arrays, _ = R.load_checkpoint_step(run_dir, STEP)
        # Twelve blocks, all already behind the walker: the first recycle exhausts it.
        meta = walker_courses.load_course("stairs_h0.3").meta
        behind = walker_courses.Course("behind", np.arange(12, dtype=np.float32) * 1.5 - 40,
                                       np.full(12, 0.3, np.float32), meta)
        monkeypatch.setattr(R.walker_courses, "courses_for_episodes",
                            lambda set_name, n: [behind] * n)
        with pytest.raises(RuntimeError, match="ran off the end"):
            R.record(tmp_path / "x.h5", task="WalkerWalkTerrain", config=config,
                     arrays=arrays, step=STEP, course_set="stairs_h0.3", n_episodes=1)
        assert not (tmp_path / "x.h5").exists()

    def test_missing_step_lists_what_exists(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_checkpoint(run_dir, NETS["DelayedMLP"], _env_params())
        with pytest.raises(FileNotFoundError, match=f"last {STEP}"):
            R.load_checkpoint_step(run_dir, 480215040)


class TestSpec:
    def test_spec_needs_course_and_step(self):
        producer = get_producer("activations_dmc")
        with pytest.raises(ValueError, match="course"):
            producer.spec_id(producer.spec(step=1))

    def test_stale_course_hash_is_refused(self):
        producer = get_producer("activations_dmc")
        with pytest.raises(ValueError, match="changed"):
            producer.spec(course="flat", step=1, course_sha256="0" * 64)

    def test_string_values_hash_like_ints(self):
        producer = get_producer("activations_dmc")
        a = producer.spec_id(producer.spec(course="flat", step=5, n_episodes=3))
        b = producer.spec_id(producer.spec(course="flat", step="5", n_episodes="3"))
        assert a == b and a.startswith("dact-flat-s0000000005-")
