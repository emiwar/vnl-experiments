"""Tests for composing a YAML group onto a real ``default_config()``.

The asset-path handling gets the most attention here. Committed YAML cannot hold an
absolute path -- vnl-playground lives in a different place on the laptop and on the
cluster -- but getting the resolution wrong is not a crash, it is a run that trains on a
different body. That is the 2026-08-18 failure, worth up to 42 % of the reward, and it is
the same hazard ``envs.config_io`` guards on the reload side.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from etils import epath
from ml_collections import config_dict
from omegaconf import OmegaConf

from vnl_experiments.config.env_builder import build_env_config, build_net_config
from vnl_experiments.config.overrides import OverrideError


def _default():
    cfg = config_dict.ConfigDict()
    cfg.ctrl_dt = 0.01
    cfg.njmax = 1200
    cfg.solver = "cg"
    cfg.torque_actuators = True
    cfg.walker_xml_path = epath.Path("/opt/playground/xmls/rodent.xml")
    cfg.reward_terms = config_dict.ConfigDict()
    cfg.reward_terms.joints = config_dict.ConfigDict()
    cfg.reward_terms.joints.weight = 1.0
    return cfg


class TestDeltas:
    def test_unspecified_fields_keep_the_default(self):
        """The group is a delta, not a replacement: silence means 'use the default'."""
        cfg = build_env_config(_default, OmegaConf.create({"njmax": 256}))
        assert cfg.njmax == 256
        assert cfg.ctrl_dt == 0.01
        assert cfg.solver == "cg"

    def test_nested_delta(self):
        cfg = build_env_config(_default, OmegaConf.create(
            {"reward_terms": {"joints": {"weight": 0.0}}}))
        assert cfg.reward_terms.joints.weight == 0.0

    def test_no_deltas_is_the_plain_default(self):
        assert build_env_config(_default, None).to_dict() == _default().to_dict()

    def test_extra_is_applied_after_the_group(self):
        """Runtime-computed values win over the YAML, which is what `extra` is for."""
        cfg = build_env_config(_default, OmegaConf.create({"njmax": 256}),
                               extra={"njmax": 512})
        assert cfg.njmax == 512

    def test_unknown_key_is_refused(self):
        with pytest.raises(OverrideError) as e:
            build_env_config(_default, OmegaConf.create({"njmaxx": 256}))
        assert "njmaxx" in str(e.value)


class TestAssetPaths:
    def test_bare_basename_resolves_against_the_default_s_directory(self):
        cfg = build_env_config(
            _default, OmegaConf.create({"walker_xml_path": "rodent_no_tail.xml"}))
        assert str(cfg.walker_xml_path) == "/opt/playground/xmls/rodent_no_tail.xml"

    def test_the_epath_type_is_preserved(self):
        """ml_collections hard-TypeErrors on a plain str here, so this must not regress."""
        cfg = build_env_config(
            _default, OmegaConf.create({"walker_xml_path": "rodent_no_tail.xml"}))
        assert isinstance(cfg.walker_xml_path, type(epath.Path("/a")))

    def test_an_explicit_directory_is_left_alone(self):
        """An out-of-tree asset can still be pointed at, so this is not a trap."""
        cfg = build_env_config(
            _default, OmegaConf.create({"walker_xml_path": "/elsewhere/custom.xml"}))
        assert str(cfg.walker_xml_path) == "/elsewhere/custom.xml"

    def test_the_real_group_names_the_no_tail_body(self):
        """Against the real config group and the real default config, not stand-ins.

        Which body a run trains on is the single field most worth an explicit assertion:
        it is invisible in the run name and was silently wrong for 395 artifacts before.
        Composed through Hydra rather than read from the file, because the group's
        `defaults:` list is part of how it resolves.
        """
        from hydra import compose, initialize_config_dir

        from vnl_experiments.conf_schema import register as register_schemas
        from vnl_experiments.envs import registry as env_registry

        register_schemas()
        conf_dir = str((Path(__file__).parent.parent / "conf").resolve())
        with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
            cfg = compose(config_name="train")

        built = build_env_config(
            env_registry.get(cfg.env_spec.task).default_config, cfg.env)
        assert Path(str(built.walker_xml_path)).name == "rodent_no_tail_collisions.xml"
        assert Path(str(built.walker_xml_path)).exists()


class TestNetConfig:
    def test_defaults_plus_deltas(self):
        from vnl_experiments.delays.network_builders import recurrent_defaults

        cfg = build_net_config(recurrent_defaults, OmegaConf.create({"rnn_cell": "gru"}))
        assert cfg.rnn_cell == "gru"
        assert cfg.latent_size == 32  # untouched registry default

    def test_unknown_net_key_lists_the_available_ones(self):
        from vnl_experiments.delays.network_builders import delay_defaults

        with pytest.raises(OverrideError) as e:
            build_net_config(delay_defaults, OmegaConf.create({"latnet_size": 64}))
        assert "latnet_size" in str(e.value)
        assert "latent_size" in str(e.value)
