"""Tests for the YAML -> ``TrainConfig`` boundary.

Most of this is about the two fields OmegaConf cannot represent natively. They are worth
testing hard because neither failure is loud: an unhashable ``logging_percentiles`` raises
somewhere inside jit with a message that does not mention config, and a wrong
``logging_level`` just quietly logs the wrong metrics for a 12-hour run.
"""

from __future__ import annotations

import dataclasses

import pytest
from omegaconf import OmegaConf

from nnx_ppo.algorithms.config import PPOConfig, TrainConfig
from nnx_ppo.algorithms.types import LoggingLevel
from vnl_experiments.config.train_builder import (
    PPOSchema,
    build_train_config,
    resolve_logging_level,
    validate_train_config,
)


class TestLoggingLevel:
    def test_single_name(self):
        assert resolve_logging_level("THROUGHPUT") is LoggingLevel.THROUGHPUT

    def test_list_of_names_is_reduced_with_or(self):
        got = resolve_logging_level(["LOSSES", "THROUGHPUT", "ENV_METRICS"])
        assert got == (LoggingLevel.LOSSES | LoggingLevel.THROUGHPUT | LoggingLevel.ENV_METRICS)

    def test_empty_list_is_none(self):
        assert resolve_logging_level([]) is LoggingLevel.NONE

    def test_an_existing_flag_passes_through(self):
        flag = LoggingLevel.LOSSES | LoggingLevel.WEIGHTS
        assert resolve_logging_level(flag) is flag

    def test_unknown_name_names_itself_and_the_alternatives(self):
        with pytest.raises(ValueError) as e:
            resolve_logging_level(["LOSSES", "THRUPUT"])
        assert "THRUPUT" in str(e.value)
        assert "THROUGHPUT" in str(e.value)


class TestBuildTrainConfig:
    def test_returns_the_real_dataclasses(self):
        """Not a DictConfig: train_ppo, pickling and asdict all need the real type."""
        cfg = build_train_config(OmegaConf.create({}))
        assert type(cfg) is TrainConfig
        assert type(cfg.ppo) is PPOConfig

    def test_defaults_come_from_nnx_ppo(self):
        """An empty group reproduces nnx-ppo's own defaults, so nothing is restated."""
        assert dataclasses.asdict(build_train_config(OmegaConf.create({}))) == (
            dataclasses.asdict(TrainConfig())
        )

    def test_percentiles_are_a_hashable_tuple(self):
        """A jit static argument (nnx_ppo/algorithms/ppo.py:115). A list would raise."""
        cfg = build_train_config(
            OmegaConf.create({"ppo": {"logging_percentiles": [0, 50, 100]}})
        )
        assert cfg.ppo.logging_percentiles == (0, 50, 100)
        assert isinstance(cfg.ppo.logging_percentiles, tuple)
        hash(cfg.ppo.logging_percentiles)

    def test_percentiles_may_be_null(self):
        cfg = build_train_config(OmegaConf.create({"eval": {"logging_percentiles": None}}))
        assert cfg.eval.logging_percentiles is None

    def test_nested_override_reaches_a_ppo_field(self):
        cfg = build_train_config(OmegaConf.create({"ppo": {"clip_range": 0.3}}))
        assert cfg.ppo.clip_range == 0.3

    def test_every_ppo_field_is_reachable(self):
        """The point of the migration: not just the six that used to have flags.

        Inheriting the schema from PPOConfig is what guarantees this, so a field added
        upstream becomes overridable with no change here.
        """
        schema = OmegaConf.to_container(OmegaConf.structured(PPOSchema))
        assert {f.name for f in dataclasses.fields(PPOConfig)} <= set(schema)

    def test_unknown_key_is_refused(self):
        """Struct mode: a typo must not be silently accepted as a new field."""
        with pytest.raises(Exception):
            build_train_config(OmegaConf.create({"ppo": {"clip_rnage": 0.3}}))


class TestValidate:
    def test_indivisible_minibatches_is_refused(self):
        cfg = build_train_config(
            OmegaConf.create({"ppo": {"n_envs": 100, "n_minibatches": 8}})
        )
        with pytest.raises(ValueError) as e:
            validate_train_config(cfg)
        assert "n_minibatches" in str(e.value)

    def test_divisible_passes_through(self):
        cfg = build_train_config(
            OmegaConf.create({"ppo": {"n_envs": 4096, "n_minibatches": 8}})
        )
        assert validate_train_config(cfg) is cfg
