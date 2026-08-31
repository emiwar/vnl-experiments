"""Tests for ``--env-config`` (and the coercion it shares with ``--net-config``).

The env config is the record of what a run actually simulated, so the thing worth
testing hardest is not that a valid override lands, but that an invalid one is
*refused*: a typo that silently created a key, or silently did nothing, would
produce a run whose `env_params` and whose behaviour disagree.
"""

from absl.testing import absltest
from ml_collections import config_dict

from vnl_experiments.delays.train_rodent import (
    _coerce,
    apply_env_config_overrides,
)


def _config():
    """A stand-in with one field of every shape the real env config has."""
    cfg = config_dict.ConfigDict()
    cfg.ctrl_dt = 0.01
    cfg.njmax = 256
    cfg.solver = "newton"
    cfg.torque_actuators = True
    cfg.keep_clips_idx = None
    cfg.start_frame_range = [0, 44]
    cfg.reward_terms = config_dict.ConfigDict()
    cfg.reward_terms.joints = config_dict.ConfigDict()
    cfg.reward_terms.joints.weight = 1.0
    cfg.reward_terms.joints.exp_scale = 1.4
    cfg.reward_terms.torso_z_range = config_dict.ConfigDict()
    cfg.reward_terms.torso_z_range.healthy_z_range = (0.0325, 0.5)
    cfg.termination_criteria = config_dict.ConfigDict()
    cfg.termination_criteria.pose_error = config_dict.ConfigDict()
    cfg.termination_criteria.pose_error.max_l2_error = 4.5
    return cfg


class CoerceTest(absltest.TestCase):

    def test_bool_before_int(self):
        """bool is a subclass of int; the bool branch has to come first."""
        self.assertIs(_coerce(True, "false"), False)
        self.assertIs(_coerce(False, "yes"), True)
        with self.assertRaises(ValueError):
            _coerce(True, "maybe")

    def test_numbers_keep_their_type(self):
        self.assertEqual(_coerce(256, "512"), 512)
        self.assertIsInstance(_coerce(0.01, "0.02"), float)
        # The int(0.01) == 0 trap: a float field must not be read as an int.
        self.assertEqual(_coerce(0.01, "0.02"), 0.02)

    def test_int_list(self):
        self.assertEqual(_coerce([0, 44], "0,120"), [0, 120])
        self.assertEqual(_coerce([512, 512], "[256,256]"), [256, 256])
        self.assertEqual(_coerce([1, 2], ""), [])

    def test_float_list_stays_float(self):
        """Element type comes from the default; int(x) would raise here."""
        self.assertEqual(_coerce((0.0325, 0.5), "0.04,0.6"), [0.04, 0.6])

    def test_empty_default_list_assumes_int(self):
        self.assertEqual(_coerce([], "1,2"), [1, 2])

    def test_none_default_reads_a_literal(self):
        self.assertEqual(_coerce(None, "[1,2,3]"), [1, 2, 3])
        self.assertEqual(_coerce(None, "42"), 42)
        self.assertEqual(_coerce(None, "some_string"), "some_string")


class ApplyEnvConfigOverridesTest(absltest.TestCase):

    def test_flat_values(self):
        cfg = _config()
        apply_env_config_overrides(cfg, [
            "ctrl_dt=0.02", "njmax=512", "solver=cg", "torque_actuators=false",
            "start_frame_range=0,120",
        ])
        self.assertEqual(cfg.ctrl_dt, 0.02)
        self.assertEqual(cfg.njmax, 512)
        self.assertEqual(cfg.solver, "cg")
        self.assertIs(cfg.torque_actuators, False)
        self.assertEqual(cfg.start_frame_range, [0, 120])

    def test_nested_values(self):
        cfg = _config()
        apply_env_config_overrides(cfg, [
            "reward_terms.joints.weight=0.5",
            "termination_criteria.pose_error.max_l2_error=6.0",
            "reward_terms.torso_z_range.healthy_z_range=0.04,0.6",
        ])
        self.assertEqual(cfg.reward_terms.joints.weight, 0.5)
        self.assertEqual(cfg.termination_criteria.pose_error.max_l2_error, 6.0)
        # A tuple field stays a tuple: the list the parser produced does not fit,
        # so it is rebuilt in the field's own type.
        self.assertEqual(cfg.reward_terms.torso_z_range.healthy_z_range, (0.04, 0.6))
        # Untouched siblings stay put.
        self.assertEqual(cfg.reward_terms.joints.exp_scale, 1.4)

    def test_empty_override_list_is_a_no_op(self):
        cfg, before = _config(), _config().to_dict()
        apply_env_config_overrides(cfg, [])
        self.assertEqual(cfg.to_dict(), before)

    def test_overrides_reach_to_dict(self):
        """to_dict() is what becomes config.json and WandB env_params."""
        cfg = _config()
        apply_env_config_overrides(cfg, ["ctrl_dt=0.02",
                                         "reward_terms.joints.weight=0.5"])
        as_dict = cfg.to_dict()
        self.assertEqual(as_dict["ctrl_dt"], 0.02)
        self.assertEqual(as_dict["reward_terms"]["joints"]["weight"], 0.5)

    def test_tuple_field_stays_a_tuple(self):
        cfg = _config()
        apply_env_config_overrides(
            cfg, ["reward_terms.torso_z_range.healthy_z_range=0.04,0.6"])
        value = cfg.reward_terms.torso_z_range.healthy_z_range
        self.assertIsInstance(value, tuple)
        self.assertEqual(value, (0.04, 0.6))

    def test_type_strict_field_keeps_its_type(self):
        """Some real fields hold epath objects, not str."""
        from etils import epath
        cfg = config_dict.ConfigDict()
        cfg.walker_xml_path = epath.Path("/a/rodent.xml")
        apply_env_config_overrides(cfg, ["walker_xml_path=/b/other.xml"])
        self.assertEqual(str(cfg.walker_xml_path), "/b/other.xml")
        self.assertIsInstance(cfg.walker_xml_path, type(epath.Path("/a")))

    # --- refusals -----------------------------------------------------------

    def test_unknown_top_level_key(self):
        with self.assertRaises(SystemExit) as e:
            apply_env_config_overrides(_config(), ["ctrl_dtt=0.02"])
        self.assertIn("ctrl_dtt", str(e.exception))
        self.assertIn("ctrl_dt", str(e.exception))  # lists what is available

    def test_unknown_nested_group(self):
        with self.assertRaises(SystemExit) as e:
            apply_env_config_overrides(_config(), ["reward_terms.jointz.weight=1"])
        self.assertIn("jointz", str(e.exception))
        self.assertIn("reward_terms", str(e.exception))

    def test_unknown_leaf_under_a_valid_group(self):
        with self.assertRaises(SystemExit) as e:
            apply_env_config_overrides(_config(), ["reward_terms.joints.wieght=1"])
        self.assertIn("wieght", str(e.exception))
        self.assertIn("exp_scale", str(e.exception))

    def test_descending_into_a_scalar(self):
        with self.assertRaises(SystemExit) as e:
            apply_env_config_overrides(_config(), ["ctrl_dt.foo=1"])
        self.assertIn("not a group", str(e.exception))

    def test_unparsable_value(self):
        with self.assertRaises(SystemExit) as e:
            apply_env_config_overrides(_config(), ["ctrl_dt=fast"])
        self.assertIn("ctrl_dt", str(e.exception))

    def test_missing_equals_sign(self):
        with self.assertRaises(SystemExit) as e:
            apply_env_config_overrides(_config(), ["ctrl_dt"])
        self.assertIn("KEY=VALUE", str(e.exception))

    def test_a_refused_override_leaves_earlier_ones_applied_but_stops(self):
        """Fail fast: the run must not start with a half-applied config."""
        cfg = _config()
        with self.assertRaises(SystemExit):
            apply_env_config_overrides(cfg, ["ctrl_dt=0.02", "nonsense=1",
                                             "njmax=512"])
        self.assertEqual(cfg.ctrl_dt, 0.02)
        self.assertEqual(cfg.njmax, 256)  # never reached


if __name__ == "__main__":
    absltest.main()
