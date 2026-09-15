"""`train=dmc` reproduces the config the existing control-suite runs were trained with.

The `nnx-ppo-delays` project holds 90 runs (2026-05-29 to 2026-07-08, delays 0-25) trained
by the pre-Hydra `train_delays.py`. A delay sweep is only readable against those points, so
the optimisation hyperparameters have to stay identical -- and a drifted learning rate or
advantage scale does not look wrong, it just quietly makes the new points incomparable.

The old script derived a few values from `brax_ppo_config(env)` and overrode the rest
inline; both are asserted here. `conf/train/dmc.yaml` pins brax's values as literals
rather than reading them at runtime, so this test is also what catches brax changing a
default underneath us.

**Two values deliberately no longer match the old script**: the eval and video cadences,
changed on 2026-09-15. They are measurement cadences, not optimisation: they change how
often a curve is sampled and how much wall clock that costs, and change nothing about the
weights a run arrives at. They are pinned in their own test below rather than dropped, so
the divergence stays a decision with a date on it and cannot drift further unnoticed.
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


#: What the pre-Hydra script used, kept so the divergence below stays legible and so a
#: cohort spanning 2026-09-15 can be identified by these numbers in its logged config.
LEGACY_CADENCE = {"eval_every_steps": 600_000, "video_every_steps": 6_000_000}


def test_eval_shape_matches_the_historical_run(train_config) -> None:
    """How an eval is *measured* must not drift -- only how often it is taken.

    These two set what one eval point means: 256 episodes of at most 1000 steps. Change
    either and the eval curves either side are different quantities, which is the thing
    `every_steps` is explicitly allowed to be and these are not.
    """
    assert train_config.eval.n_envs == 256
    assert train_config.eval.max_episode_length == 1000
    assert train_config.video.episode_length == 1000


def test_measurement_cadences_are_the_2026_09_15_values(train_config) -> None:
    """Deliberately not the old script's. See the module docstring and dmc.yaml.

    600_000 was brax's 60M / `num_evals = 100` and was never rescaled when `total_steps`
    was multiplied by 8, so a 480M run took 800 evals rather than the 100 the number was
    chosen to give -- and on the locomotion tasks that was 58-62 % of the wall clock,
    against 22-27 % for training itself
    (`analysis/dm_control_suite/where-the-wall-clock-goes/`). 6_000_000 for video was
    brax's 60M / 10, unrescaled the same way.

    The values are asserted rather than merely commented because the *count* is what the
    choice was about: 4.8M and 30M give exactly 100 evals and 16 videos over the 480M
    budget, and a later edit to `total_steps` that quietly changed those counts is the
    same class of mistake this whole file exists to catch.
    """
    assert train_config.eval.every_steps == 4_800_000
    assert train_config.video.every_steps == 30_000_000
    assert train_config.eval.every_steps != LEGACY_CADENCE["eval_every_steps"]

    budget = train_config.ppo.total_steps
    assert budget % train_config.eval.every_steps == 0
    assert budget % train_config.video.every_steps == 0
    assert budget // train_config.eval.every_steps == 100
    assert budget // train_config.video.every_steps == 16


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

    def test_every_registered_dmc_env_defaults_to_torque_control(self) -> None:
        """The servo fields are injected the same way, and must default to inert.

        `servo_kp = 0.0` is what makes the model identical to the one playground ships, so
        an entry that lost the field would silently be a different plant from the rest of
        the cohort. See envs/servo_control.md.
        """
        for name, spec in env_registry.ENVS.items():
            if spec.obs_layout != "flat":
                continue
            cfg = spec.default_config()
            assert cfg.servo_kp == 0.0, name
            assert cfg.servo_damping_ratio == 1.0, name
            assert cfg.servo_center == "qpos0", name
            assert cfg.servo_unlimited_half_range == 0.0, name


def test_the_dmc_groups_pin_the_control_suite_project() -> None:
    """So a control-suite run cannot land in the rodent project by forgetting a flag."""
    assert composed("env=dmc/cartpole_swingup", "net=delayed_mlp").wandb.project == (
        "nnx-ppo-delays"
    )
    # ...and the rodent default is untouched.
    assert composed().wandb.project == "nnx-ppo-rodent-delays"


class TestEvalEnvIsNotWrapped:
    """The training wrappers must not reach the measurement env.

    Two of the three distort what the eval reports, and neither failure is loud:

    * ``RewardScalingWrapper`` multiplies reward by 10, so `eval/episode_reward` comes out
      in different units than the historical runs -- comparable in shape, wrong in scale.
    * ``EpisodeWrapper.reset`` seeds ``step_counter`` to a random value in
      ``[0, max_len/2)``. That phase-spread is deliberate and correct for training (envs
      must not truncate in lockstep) but on an eval env it means an episode can start 499
      steps in, so the reported lifespan lands near 750 rather than 1000, with large
      variance.

    The pre-Hydra script passed the bare env as `eval_env` for exactly this reason.

    ``NaNGuardWrapper`` is the third, and is kept off eval for a different reason: it
    would barely distort anything, but a simulator divergence costs one NaN eval point
    rather than the whole run, so there is no reason to move the numbers the cohort is
    compared on. See `_dmc_builder`.
    """

    def _envs(self, task="CartpoleSwingup"):
        spec = env_registry.get(task)
        cfg = spec.default_config()
        return spec.build(cfg), spec.build(cfg, for_eval=True)

    def test_train_env_is_wrapped(self) -> None:
        train, _ = self._envs()
        assert type(train).__name__ == "RewardScalingWrapper"

    def test_eval_env_is_bare(self) -> None:
        _, ev = self._envs()
        name = type(ev).__name__
        assert "Wrapper" not in name, f"eval env is wrapped in {name}"

    def test_eval_reward_is_unscaled(self) -> None:
        """A 10x reward would silently rescale every eval curve.

        The two envs cannot be compared by resetting both with the same key --
        ``EpisodeWrapper.reset`` splits the rng and passes only half to the inner env, so
        they start from different states. Wrap the eval env by hand instead and step both
        from *one* state, which isolates the scaling from everything else.
        """
        import jax

        spec = env_registry.get("CartpoleSwingup")
        cfg = spec.default_config()
        assert cfg.reward_scale == 10.0

        from nnx_ppo.wrappers import reward_scaling_wrapper

        bare = spec.build(cfg, for_eval=True)
        scaled = reward_scaling_wrapper.RewardScalingWrapper(bare, cfg.reward_scale)

        state = bare.reset(jax.random.key(0))
        action = jax.numpy.zeros((bare.action_size,))
        r_bare = float(bare.step(state, action).reward)
        r_scaled = float(scaled.step(state, action).reward)
        assert r_scaled == pytest.approx(r_bare * cfg.reward_scale, rel=1e-5)

    def test_eval_episodes_start_at_step_zero(self) -> None:
        """The randomised start counter is what cut the reported lifespan short."""
        import jax

        spec = env_registry.get("CartpoleSwingup")
        cfg = spec.default_config()
        train, ev = spec.build(cfg), spec.build(cfg, for_eval=True)
        counters = [int(train.reset(jax.random.key(s)).info["step_counter"])
                    for s in range(12)]
        assert max(counters) > 0, "training envs should be phase-spread"
        # The bare env keeps no counter at all, so nothing truncates the eval episode
        # before `eval.max_episode_length` does.
        assert "step_counter" not in ev.reset(jax.random.key(0)).info


class TestNamingAndTagsArePerEnvFamily:
    """Naming and env-family tags are declared by the env group, not hardcoded.

    A control-suite name has to lead with the task -- there are nine of them, and
    `DelayedMLP_delay5_eff5` does not say which problem was solved. The rodent has one
    task, so the architecture leads. `TrainEvalSplit` asserts an in-training eval on
    held-out clips, which is untrue for a task that has no clips.
    """

    def _setup(self, *overrides):
        from vnl_experiments import train as entry

        return entry.build_run(composed(*overrides))

    def test_rodent_name_and_tags_are_unchanged(self) -> None:
        s = self._setup("delay=5")
        assert s.name_stem == "RodentEncDec_delay5_eff5"
        assert {"warp", "TrainEvalSplit"} <= set(s.tags)

    def test_dmc_name_leads_with_the_task(self) -> None:
        s = self._setup("env=dmc/walker_walk", "net=delayed_mlp", "train=dmc", "delay=5")
        assert s.name_stem == "WalkerWalk_DelayedMLP_delay5_eff5"

    def test_dmc_tags_carry_the_task_and_not_the_rodent_ones(self) -> None:
        s = self._setup("env=dmc/walker_walk", "net=delayed_mlp", "train=dmc", "delay=5")
        assert "WalkerWalk" in s.tags
        assert "TrainEvalSplit" not in s.tags
        assert "warp" not in s.tags

    def test_ablation_tokens_still_reach_the_name(self) -> None:
        s = self._setup("delay=0", "efference=3", "net.dec_use_proprioception=false")
        assert s.name_stem == "RodentEncDec_delay0_eff3_noproprio"

    def test_every_dmc_group_declares_both(self) -> None:
        """A new group that forgot them would fall back to the rodent convention."""
        from pathlib import Path

        for f in sorted((Path(__file__).parent.parent / "conf/env/dmc").glob("*.yaml")):
            cfg = composed(f"env=dmc/{f.stem}", "net=delayed_mlp")
            assert cfg.env_spec.name_template.startswith("{task}"), f.stem
            assert cfg.env_spec.task in list(cfg.env_spec.tags), f.stem
            assert "TrainEvalSplit" not in list(cfg.env_spec.tags), f.stem


class TestVideoCameraFollowsThePlant:
    """The locomoting tasks must render from a tracking camera.

    `mjx_env.render_array` falls back to camera ``-1`` -- a *static* free camera aimed
    at the model's origin -- when none is given. A walker that has run eight metres
    downfield is then simply not in the picture, which is what every control-suite
    video did before this was set. Each dm_control XML already ships a `trackcom`
    camera; the group just has to name it, and the name is the XML's, so it is set per
    env rather than once on `train=dmc`.
    """

    #: Camera per group, or None for a task whose plant cannot leave the frame:
    #: cartpole's rail is limited to +-1.8 m and ball_in_cup's cup is spring-anchored,
    #: and neither model defines a `side` camera to name anyway.
    EXPECTED = {
        "walker_walk": "side", "walker_run": "side", "walker_stand": "side",
        "cheetah_run": "side",
        "humanoid_walk": "side", "humanoid_stand": "side",
        "cartpole_balance": None, "cartpole_swingup": None, "ball_in_cup": None,
        # Hopper travels, but its XML has no `side` camera; `cam0` and `back` are both
        # trackcom. Reacher and Finger are anchored to the world and cannot leave the frame
        # -- and neither defines a trackcom camera to name anyway (reacher's `hand` is
        # mode="track", finger's `cam0`/`cam1` are static).
        "hopper_stand": "cam0", "hopper_hop": "cam0",
        "reacher_easy": None, "reacher_hard": None,
        "finger_turn_easy": None, "finger_turn_hard": None,
    }

    @staticmethod
    def _groups():
        return sorted(p.stem for p in
                      (Path(__file__).parent.parent / "conf/env/dmc").glob("*.yaml"))

    def test_the_table_covers_every_group(self) -> None:
        """A new group has to make the choice rather than silently inherit -1."""
        assert set(self._groups()) == set(self.EXPECTED)

    @pytest.mark.parametrize("group", sorted(EXPECTED))
    def test_group_sets_the_expected_camera(self, group: str) -> None:
        cfg = composed(f"env=dmc/{group}", "net=delayed_mlp", "train=dmc")
        camera = cfg.train.video.render_kwargs.get("camera")
        assert camera == self.EXPECTED[group]

    @pytest.mark.parametrize(
        "group", sorted(g for g, c in EXPECTED.items() if c is not None))
    def test_the_camera_exists_and_actually_tracks(self, group: str) -> None:
        """Named cameras are resolved at render time, so a typo fails mid-training.

        Asserting the mode as well as the name is what makes this a guard: a camera
        that exists but is bolted to the world would leave the video exactly as
        broken as camera -1.
        """
        import mujoco

        cfg = composed(f"env=dmc/{group}", "net=delayed_mlp", "train=dmc")
        spec = env_registry.get(cfg.env_spec.task)
        model = spec.build(spec.default_config(), for_eval=True).mj_model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                 for i in range(model.ncam)]
        camera = cfg.train.video.render_kwargs.camera
        assert camera in names, f"{group}: {camera!r} not among {names}"
        assert model.cam(camera).mode == mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM

    def test_the_rodent_camera_is_untouched(self) -> None:
        """The rodent sets its own camera on `train=rodent`; this must not shadow it."""
        cfg = composed()
        assert cfg.train.video.render_kwargs.camera == "close_profile-rodent"
