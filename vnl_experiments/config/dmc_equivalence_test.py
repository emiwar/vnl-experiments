"""`train=dmc` reproduces the config the existing control-suite runs were trained with.

The `nnx-ppo-delays` project holds 90 runs (2026-05-29 to 2026-07-08, delays 0-25) trained
by the pre-Hydra `train_delays.py`. A delay sweep is only readable against those points, so
the optimisation hyperparameters have to stay identical -- and a drifted learning rate or
advantage scale does not look wrong, it just quietly makes the new points incomparable.

The old script derived a few values from `brax_ppo_config(env)` and overrode the rest
inline; both are asserted here. `conf/train/dmc.yaml` pins brax's values as literals
rather than reading them at runtime, so this test is also what catches brax changing a
default underneath us.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from nnx_ppo.algorithms.types import LoggingLevel
from vnl_experiments.conf_schema import register as register_schemas
from vnl_experiments.config import build_train_config
from vnl_experiments.envs import registry as env_registry

CONF_DIR = str((Path(__file__).parent.parent / "conf").resolve())

#: What `train_delays.py` built, with brax's per-task values substituted in. `n_envs`,
#: `total_steps` and `clip_range` were the script's own choices, not brax's.
OLD_PPO = {
    "n_envs": 8192,
    "rollout_length": 30,          # brax unroll_length
    "total_steps": 480_000_000,    # brax num_timesteps x 8
    "gae_lambda": 0.95,
    "discounting_factor": 0.99,
    "clip_range": 0.3,
    "learning_rate": 1e-4,
    "normalize_advantages": True,
    "n_epochs": 4,
    "n_minibatches": 8,
    "critic_loss_weight": 1.0,
    "logging_percentiles": None,
}
OLD_LOGGING_LEVEL = (
    LoggingLevel.LOSSES | LoggingLevel.THROUGHPUT | LoggingLevel.ROLLOUT_STATS
    | LoggingLevel.CRITIC_EXTRA | LoggingLevel.ACTOR_EXTRA
)


def composed(*overrides: str):
    register_schemas()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        return compose(config_name="train", overrides=list(overrides))


@pytest.fixture(scope="module")
def train_config():
    return build_train_config(
        composed("env=dmc/walker_walk", "net=delayed_mlp", "train=dmc").train
    )


@pytest.mark.parametrize("field,expected", sorted(OLD_PPO.items(), key=lambda kv: kv[0]))
def test_ppo_field_matches_the_historical_run(train_config, field, expected) -> None:
    assert getattr(train_config.ppo, field) == expected


def test_logging_level_matches(train_config) -> None:
    """The existing delay analyses read the actor/critic and rollout subtrees."""
    assert train_config.ppo.logging_level == OLD_LOGGING_LEVEL


def test_eval_and_video_cadence_match(train_config) -> None:
    assert train_config.eval.every_steps == 600_000     # brax 60M / num_evals=100
    assert train_config.eval.n_envs == 256
    assert train_config.eval.max_episode_length == 1000
    assert train_config.video.every_steps == 6_000_000  # brax 60M / 10


def test_brax_reference_values_have_not_moved() -> None:
    """The literals above came from brax; fail loudly if brax changes them.

    Skipped rather than failed when the reference config is unavailable -- this asserts
    an upstream fact, not our own code.
    """
    params = pytest.importorskip(
        "mujoco_playground.config.dm_control_suite_params",
        reason="brax reference PPO config unavailable",
    ).brax_ppo_config("WalkerWalk")
    assert params.unroll_length == 30
    assert params.num_timesteps == 60_000_000
    assert params.episode_length == 1000
    assert params.reward_scaling == 10.0


class TestWrapperParameters:
    """`episode_length` and `reward_scale` live on the env config, not `train`.

    They belong to the wrappers rather than the task, so they are injected into the env
    config -- which means they are overridable per run and, more importantly, recorded in
    `config.json`'s `env_params` where an analysis can read them back.
    """

    def test_reward_scale_defaults_to_brax(self) -> None:
        cfg = env_registry.get("WalkerWalk").default_config()
        assert cfg.reward_scale == 10.0
        assert cfg.episode_length == 1000

    def test_reward_scale_is_overridable(self) -> None:
        cfg = composed("env=dmc/walker_walk", "net=delayed_mlp", "env.reward_scale=1.0")
        assert cfg.env.reward_scale == 1.0

    def test_every_registered_dmc_env_has_both_fields(self) -> None:
        """A new registry entry that forgot them would train unscaled and unwrapped."""
        for name, spec in env_registry.ENVS.items():
            if spec.obs_layout != "flat":
                continue
            cfg = spec.default_config()
            assert "reward_scale" in cfg, name
            assert "episode_length" in cfg, name


def test_the_dmc_groups_pin_the_control_suite_project() -> None:
    """So a control-suite run cannot land in the rodent project by forgetting a flag."""
    assert composed("env=dmc/cartpole_swingup", "net=delayed_mlp").wandb.project == (
        "nnx-ppo-delays"
    )
    # ...and the rodent default is untouched.
    assert composed().wandb.project == "nnx-ppo-rodent-delays"
