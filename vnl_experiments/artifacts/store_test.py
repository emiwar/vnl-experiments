"""Tests for the artifact store's identity and bookkeeping.

The store had no tests. Most of it is ordinary filesystem work, but two properties are
load-bearing in a way that fails *silently* when broken, and those are what this covers.

**`spec_id` stability.** The id is a hash of the normalised spec plus the producer
version, and it is the primary key for 1900+ stored artifacts and for the `REQUIRES`
pins in a dozen committed analyses. If the hash changes, every pinned analysis reports
0/N coverage and every artifact looks missing; if it *fails* to change when the producer
version does, results made by different code get pooled into one figure. Neither raises.

**The `None`-dropping rule.** `normalise_spec` drops `None` values specifically so that a
producer can gain a new optional spec key without invalidating everything it made before
(`EvalProducer.action_noise` is the precedent). That is a deliberate design decision with
no other expression in the code, so it is asserted here.

There is a real instance of getting this wrong on record: `--set action_noise=0` hashed
differently from `0.0` and quietly minted a parallel `spec_id`, splitting one sweep across
two ids. Hence `test_int_and_float_hash_differently`, which pins the behaviour rather than
the bug -- the fix belongs in the producer's `spec()`, which coerces the type.
"""

from __future__ import annotations

import json

import pytest

from vnl_experiments.artifacts.store import (
    Entry,
    Store,
    normalise_spec,
    spec_id,
)


class TestNormaliseSpec:
    def test_keys_are_sorted(self):
        assert list(normalise_spec({"b": 1, "a": 2})) == ["a", "b"]

    def test_none_values_are_dropped(self):
        """So a new optional spec key does not invalidate existing artifacts."""
        assert normalise_spec({"a": 1, "b": None}) == {"a": 1}

    def test_adding_a_none_key_does_not_change_the_id(self):
        """The whole point of the rule, stated as the property it protects."""
        before = spec_id("eval", {"checkpoint": "last"}, 3)
        after = spec_id("eval", {"checkpoint": "last", "action_noise": None}, 3)
        assert before == after

    def test_tuples_and_sets_become_lists(self):
        """JSON has no tuple or set, so they must canonicalise before hashing."""
        assert normalise_spec({"a": (1, 2)}) == {"a": [1, 2]}
        assert normalise_spec({"a": {2, 1}}) == {"a": [1, 2]}

    def test_a_set_hashes_regardless_of_iteration_order(self):
        assert spec_id("x", {"a": {1, 2, 3}}, 1) == spec_id("x", {"a": {3, 1, 2}}, 1)


class TestSpecId:
    def test_shape_is_prefix_and_eight_hex(self):
        sid = spec_id("eval3ds", {"checkpoint": "last"}, 3)
        prefix, _, digest = sid.rpartition("-")
        assert prefix == "eval3ds"
        assert len(digest) == 8
        assert all(c in "0123456789abcdef" for c in digest)

    def test_stable_across_calls_and_key_order(self):
        """Committed analyses pin these strings, so they cannot drift between runs."""
        a = spec_id("eval", {"checkpoint": "last", "dataset": "old_eval"}, 3)
        b = spec_id("eval", {"dataset": "old_eval", "checkpoint": "last"}, 3)
        assert a == b == spec_id("eval", {"checkpoint": "last", "dataset": "old_eval"}, 3)

    def test_a_version_bump_changes_the_id(self):
        """What keeps results made by different producer code from being pooled."""
        assert spec_id("eval", {"checkpoint": "last"}, 3) != spec_id(
            "eval", {"checkpoint": "last"}, 4)

    def test_a_changed_value_changes_the_id(self):
        assert spec_id("eval", {"dataset": "old_eval"}, 3) != spec_id(
            "eval", {"dataset": "new_eval"}, 3)

    def test_int_and_float_hash_differently(self):
        """`0` and `0.0` are different specs.

        This is JSON's behaviour and not something the store should paper over -- but it
        is why a producer's `spec()` coerces numeric fields to a fixed type. Getting it
        wrong once split an action-noise sweep across two parallel ids.
        """
        assert spec_id("e", {"action_noise": 0}, 1) != spec_id("e", {"action_noise": 0.0}, 1)


class TestStore:
    @pytest.fixture
    def store(self, tmp_path):
        return Store(root=tmp_path)

    def _write(self, store, kind="eval", wandb_id="abc123", sid="eval-deadbeef",
               payload=b'{"reward": 1}'):
        path = store.path_for(kind, wandb_id, sid, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_unknown_kind_is_refused(self, store):
        with pytest.raises(ValueError) as e:
            store.dir_for("nonsense", "abc123")
        assert "nonsense" in str(e.value)

    def test_record_then_lookup_round_trips(self, store):
        path = self._write(store)
        store.record("eval", "abc123", "eval-deadbeef", path,
                     spec={"checkpoint": "last"}, producer={"module": "m", "version": 3},
                     resolved={"checkpoint_step": 600_000_000})
        entry = store.lookup("eval", "abc123", "eval-deadbeef")
        assert isinstance(entry, Entry)
        assert entry.spec == {"checkpoint": "last"}
        assert entry.resolved["checkpoint_step"] == 600_000_000
        assert entry.producer["version"] == 3
        assert entry.path == "eval/abc123/eval-deadbeef.json"

    def test_the_sidecar_is_the_record(self, store):
        """No central database: the sidecar next to the bytes is the source of truth.

        This is what makes `rsync`-ing a directory between cluster and laptop
        self-describing, so it is worth asserting the file is really there and complete.
        """
        path = self._write(store)
        store.record("eval", "abc123", "eval-deadbeef", path,
                     spec={"checkpoint": "last"}, producer={"module": "m", "version": 3})
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        assert meta["spec_id"] == "eval-deadbeef"
        assert meta["wandb_id"] == "abc123"
        assert meta["sha256"]
        assert meta["bytes"] == len(b'{"reward": 1}')

    def test_lookup_needs_both_sidecar_and_data(self, store):
        """A sidecar whose data file has gone is not a hit -- it would be re-produced."""
        path = self._write(store)
        store.record("eval", "abc123", "eval-deadbeef", path,
                     spec={}, producer={"module": "m", "version": 1})
        assert store.have("eval", "abc123", "eval-deadbeef")
        path.unlink()
        assert not store.have("eval", "abc123", "eval-deadbeef")

    def test_record_refuses_an_absent_data_file(self, store):
        with pytest.raises(FileNotFoundError):
            store.record("eval", "abc123", "eval-deadbeef",
                         store.path_for("eval", "abc123", "eval-deadbeef", ".json"),
                         spec={}, producer={})

    def test_missing_lists_only_the_runs_without_the_artifact(self, store):
        path = self._write(store, wandb_id="has")
        store.record("eval", "has", "eval-deadbeef", path,
                     spec={}, producer={"module": "m", "version": 1})
        assert store.missing("eval", "eval-deadbeef", ["has", "hasnt"]) == ["hasnt"]

    def test_other_specs_distinguishes_wrong_version_from_no_data(self, store):
        """"Made by a different eval version" must not read as "no data"."""
        path = self._write(store, sid="eval-oldoldold")
        store.record("eval", "abc123", "eval-oldoldold", path,
                     spec={}, producer={"module": "m", "version": 2})
        others = store.other_specs("eval", ["abc123"], exclude="eval-deadbeef")
        assert "eval-oldoldold" in others

    def test_reindex_rebuilds_the_manifest_from_sidecars(self, store, tmp_path):
        for wandb_id in ("aaa", "bbb"):
            path = self._write(store, wandb_id=wandb_id)
            store.record("eval", wandb_id, "eval-deadbeef", path,
                         spec={}, producer={"module": "m", "version": 1})
        manifest = tmp_path / "manifest.jsonl"
        entries = store.reindex(manifest_path=manifest)
        assert len(entries) == 2
        lines = manifest.read_text().strip().split("\n")
        assert len(lines) == 2
        assert {json.loads(l)["wandb_id"] for l in lines} == {"aaa", "bbb"}
