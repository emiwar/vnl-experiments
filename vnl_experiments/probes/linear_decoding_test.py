"""Pins the decoder's arithmetic, so a refactor cannot quietly move published numbers.

Two levels:

* synthetic fixtures with known answers, which run anywhere;
* a reproduction of rows from ``analysis/implicit-forward-model/data.csv``, which runs only
  where that analysis's activation recordings are still on disk. That analysis is written up
  and frozen; if this module stops reproducing its numbers, its report is no longer
  reproducible and the change needs a `DECODE_VERSION` bump and a re-run, not a shrug.

    ../.venv/bin/python -m pytest vnl_experiments/probes/linear_decoding_test.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vnl_experiments.probes import linear_decoding as ld
from vnl_experiments.probes import pathways

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ACTIVATIONS = REPO_ROOT / "eval_results" / "activations"
LEGACY_DATA = REPO_ROOT / "analysis" / "implicit-forward-model" / "data.csv"


# ---------------------------------------------------------------------------
# Targets, masks, efference
# ---------------------------------------------------------------------------

def test_make_targets_shifts_within_clip():
    target = np.arange(5 * 2 * 1, dtype=np.float32).reshape(5, 2, 1)
    tg = ld.make_targets(target, delay_k=2)
    assert np.array_equal(tg["delayed"][:2], np.zeros((2, 2, 1), np.float32))
    assert np.array_equal(tg["delayed"][2:], target[:-2])
    assert np.array_equal(tg["delta"], target - tg["delayed"])


def test_delay_zero_makes_delta_degenerate():
    target = np.random.default_rng(0).normal(size=(6, 3, 4)).astype(np.float32)
    tg = ld.make_targets(target, delay_k=0)
    assert np.array_equal(tg["delayed"], target)
    assert not tg["delta"].any()
    assert ld.degenerate_targets(0) == {"proprio": False, "delta": True}
    assert ld.degenerate_targets(10)["delta"] is False


def test_valid_mask_requires_alive_and_warm():
    dones = np.zeros((5, 2), bool)
    dones[4, 1] = True
    mask = ld.valid_mask(dones, delay_k=2)
    assert not mask[:2].any()          # buffer not yet filled
    assert mask[2:, 0].all()
    assert not mask[4, 1]              # dead


def test_efference_queue_is_the_last_eff_actions():
    actions = np.arange(4 * 1 * 2, dtype=np.float32).reshape(4, 1, 2)
    q = ld.efference_queue(actions, eff=2)
    assert q.shape == (4, 1, 4)
    assert not q[0].any()                                   # nothing before t=0
    assert np.array_equal(q[1, 0, :2], actions[0, 0])       # a(t-1)
    assert np.array_equal(q[2, 0, 2:], actions[0, 0])       # a(t-2)
    assert ld.efference_queue(actions, eff=0).shape == (4, 1, 0)


# ---------------------------------------------------------------------------
# The decoder itself
# ---------------------------------------------------------------------------

def _synthetic(n_clips=20, T=60, d=4, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, n_clips, d)).astype(np.float32)
    W = rng.normal(size=(d, 2)).astype(np.float32)
    Y = X @ W + noise * rng.normal(size=(T, n_clips, 2)).astype(np.float32)
    return X, Y.astype(np.float32), np.ones((T, n_clips), bool), n_clips


def test_noiseless_linear_target_is_recovered():
    X, Y, mask, n = _synthetic(noise=0.0)
    out = ld.decode(X, Y, mask, n, seed=0)
    assert out["test_r2"] > 0.99
    assert out["n_features"] == 4


def test_unpredictable_target_scores_about_zero():
    X, _, mask, n = _synthetic()
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(*mask.shape, 2)).astype(np.float32)
    assert ld.decode(X, Y, mask, n, seed=0)["test_r2"] < 0.05


def test_too_few_rows_gives_nan_not_a_wrong_number():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 4, 50)).astype(np.float32)      # 50 features, ~12 rows
    Y = rng.normal(size=(3, 4, 2)).astype(np.float32)
    out = ld.decode(X, Y, np.ones((3, 4), bool), 4, seed=0)
    assert np.isnan(out["test_r2"])


def test_decode_is_deterministic():
    X, Y, mask, n = _synthetic(noise=0.5)
    a = ld.decode(X, Y, mask, n, seed=0)
    b = ld.decode(X, Y, mask, n, seed=0)
    assert a["test_r2"] == b["test_r2"] and a["lambda"] == b["lambda"]


def test_split_is_by_clip_and_disjoint():
    train, val, test = ld._split_clips(20, 0.3, 0.2, seed=0)
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train | val | test) == 20


# ---------------------------------------------------------------------------
# Pathways
# ---------------------------------------------------------------------------

def test_actor_and_encoder_stages_resolve():
    assert pathways.resolve_stage("input::delayed_plus_efference") == \
        ("actor", 0, "input\n(delayed+eff)")
    assert pathways.resolve_stage("layer::3/action/1/predictor/4")[:2] == ("actor", 5)
    assert pathways.resolve_stage("layer::3/action/1/decoder/0")[:2] == ("actor", 6)
    assert pathways.resolve_stage("layer::3/action/1/decoder/5/action")[:2] == ("actor", 11)
    # The un-nested (efference-only) architecture shares the decoder stages.
    assert pathways.resolve_stage("layer::3/action/1/0")[:2] == ("actor", 6)
    assert pathways.resolve_stage("layer::3/action/0/task_obs/6")[:2] == ("encoder", 6)
    assert pathways.resolve_stage("layer::3/value/3")[:2] == ("critic", 3)


def test_off_pathway_probes_are_classified_not_dropped():
    for probe, pathway in [
        ("input::current_proprio", "reference"),
        ("layer::3/action/1/delay", "reference"),
        ("layer::3/action/0/proprioception", "reference"),
        ("layer::1/state/proprioception", "preprocessing"),
        ("layer::3/action/1/decoder/5/log_likelihood", "diagnostic"),
    ]:
        assert pathways.resolve_stage(probe)[0] == pathway, probe
    assert pathways.resolve_stage("layer::something/new")[0] == "other"


def test_predictor_is_not_swallowed_by_the_decoder_pattern():
    """The optional `(decoder/)?` group must not also match `predictor/N`."""
    for i in range(5):
        pathway, index, _ = pathways.resolve_stage(f"layer::3/action/1/predictor/{i}")
        assert (pathway, index) == ("actor", i + 1)


# ---------------------------------------------------------------------------
# Regression against the frozen analysis
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LEGACY_DATA.exists(), reason="frozen analysis not present")
def test_reproduces_frozen_implicit_forward_model_rows():
    """Re-decode one legacy recording and match its committed R² to 1e-9."""
    import pandas as pd

    committed = pd.read_csv(LEGACY_DATA)
    for run_name, group in committed.groupby("run_name"):
        h5 = LEGACY_ACTIVATIONS / f"{run_name}__old_eval.h5"
        if h5.exists():
            break
    else:
        pytest.skip("no legacy activation recording on disk")

    rows = pd.DataFrame(ld.decode_file(h5, seed=0))
    merged = group.merge(rows, on=["probe", "target"], suffixes=("_old", "_new"))
    assert len(merged) == len(group), "probe/target set changed"

    old, new = merged["test_r2_old"], merged["test_r2_new"]
    both_nan = old.isna() & new.isna()
    assert (both_nan | (np.abs(old - new) < 1e-9)).all(), \
        merged.loc[~(both_nan | (np.abs(old - new) < 1e-9)),
                   ["probe", "target", "test_r2_old", "test_r2_new"]]
