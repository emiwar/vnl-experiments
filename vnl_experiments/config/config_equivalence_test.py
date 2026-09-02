"""The Hydra config groups reproduce the configs the pre-Hydra entry point produced.

This is the migration's correctness anchor. `conf/env/rodent_imitation.yaml` and
`conf/train/rodent.yaml` are transcriptions of what `train_rodent.make_env_config()` and
`make_train_config()` used to build, and a transcription error would not raise -- it would
train a subtly different model and put it in the run index next to runs it is no longer
comparable to.

`testdata/pre_hydra_config.json` is a snapshot of that old output, taken while the old
code was still present and verified equal to the Hydra composition at the time. Comparing
against a frozen snapshot rather than against the live old code is deliberate: it means
the guarantee survives deleting `train_rodent.py`, instead of quietly turning into a
skipped test. Every run in the existing WandB index was trained with those values, so
this is what keeps new runs comparable to them.

Regenerating the snapshot is therefore *not* routine. It requires the pre-Hydra module,
which is gone; if a config value genuinely should change, edit the snapshot in the same
commit as the YAML and say why in the message, so the diff shows what moved.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from vnl_experiments.conf_schema import register as register_schemas
from vnl_experiments.config import build_env_config, build_net_config, build_train_config
from vnl_experiments.delays.network_builders import ARCHITECTURES
from vnl_experiments.envs import registry as env_registry

HERE = Path(__file__).parent
CONF_DIR = str((HERE.parent / "conf").resolve())
SNAPSHOT = json.loads((HERE / "testdata" / "pre_hydra_config.json").read_text())


def composed(*overrides: str):
    """The fully composed config, exactly as the entry point would see it."""
    register_schemas()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        return compose(config_name="train", overrides=list(overrides))


def _jsonish(value):
    """Match the snapshot's encoding: JSON types, asset paths as basenames."""
    return json.loads(json.dumps(value, default=str))


def test_env_group_matches_the_pre_hydra_config() -> None:
    """`conf/env/rodent_imitation.yaml` == the old `make_env_config()`, field for field."""
    cfg = composed()
    built = build_env_config(env_registry.get(cfg.env_spec.task).default_config, cfg.env)
    got = {k: (Path(str(v)).name if k.endswith("_path") else v)
           for k, v in _jsonish(built.to_dict()).items()}
    assert got == SNAPSHOT["env_config"]


def test_train_group_matches_the_pre_hydra_config() -> None:
    """`conf/train/rodent.yaml` == the old `make_train_config()` for default arguments."""
    built = build_train_config(composed().train)
    assert _jsonish(dataclasses.asdict(built)) == SNAPSHOT["train_config"]


def test_net_groups_match_the_registry_defaults() -> None:
    """Each net group composes to exactly its architecture's `defaults()`.

    The groups carry no deltas, so any difference means the generated schema lost
    something in the YAML round trip -- a tuple, a null, a nested value.
    """
    for group, arch_name in (("encdec", "RodentEncDecDelays"),
                             ("forward_model", "RodentForwardModel"),
                             ("recurrent", "RodentEncDecRecurrent")):
        cfg = composed(f"net={group}")
        assert cfg.net_spec.architecture == arch_name
        arch = ARCHITECTURES[arch_name]
        assert build_net_config(arch.defaults, cfg.net).to_dict() == arch.defaults().to_dict(), (
            f"{group} drifted"
        )


def test_train_group_preserves_the_flag_and_tuple_types() -> None:
    """The two fields OmegaConf cannot represent natively come back correct.

    `logging_percentiles` in particular must be a tuple: it is a jit static argument
    (nnx_ppo/algorithms/ppo.py:115) and a list would be unhashable.
    """
    from nnx_ppo.algorithms.types import LoggingLevel

    built = build_train_config(composed().train)
    assert isinstance(built.ppo.logging_percentiles, tuple)
    hash(built.ppo.logging_percentiles)  # must not raise
    assert built.eval.logging_percentiles is None
    assert LoggingLevel.THROUGHPUT in built.ppo.logging_level
    assert LoggingLevel.ENV_METRICS in built.ppo.logging_level
    assert LoggingLevel.THROUGHPUT not in built.eval.logging_level


def test_smoke_group_composes_over_rodent() -> None:
    """`train=smoke` inherits the rodent config and shrinks only what it names."""
    built = build_train_config(composed("train=smoke").train)
    assert built.ppo.n_envs == 64
    assert built.ppo.total_steps == 200_000
    assert built.video.enabled is False
    # Inherited from rodent.yaml, not restated in smoke.yaml.
    assert built.ppo.learning_rate == 1.0e-4
    assert built.ppo.discounting_factor == 0.95
    assert built.ppo.rollout_length == 20


def test_command_line_overrides_reach_the_built_configs() -> None:
    """The end-to-end point of the migration: any key, no `+` prefix."""
    cfg = composed("net=recurrent", "net.rnn_cell=gru", "net.latent_size=64",
                   "env.ctrl_dt=0.02", "env.reward_terms.joints.weight=0.5",
                   "train.ppo.clip_range=0.3")
    env = build_env_config(env_registry.get(cfg.env_spec.task).default_config, cfg.env)
    net = build_net_config(ARCHITECTURES["RodentEncDecRecurrent"].defaults, cfg.net)
    train = build_train_config(cfg.train)

    assert env.ctrl_dt == 0.02
    assert env.reward_terms["joints"]["weight"] == 0.5
    assert net.rnn_cell == "gru"
    assert net.latent_size == 64
    assert train.ppo.clip_range == 0.3


def test_a_typo_is_refused_before_anything_runs() -> None:
    """Struct mode plus the generated schema: an unknown key cannot reach the config."""
    from hydra.errors import ConfigCompositionException

    with pytest.raises(ConfigCompositionException):
        composed("net=recurrent", "net.rnn_cel=gru")
