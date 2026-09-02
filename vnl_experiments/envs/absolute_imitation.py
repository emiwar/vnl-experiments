"""Imitation environment with *absolute* (non-relative) joint and body targets.

This is a drop-in subclass of
:class:`vnl_playground.tasks.rodent.imitation.Imitation` used for the
proprioceptive-delay study (``vnl_experiments/train.py``).

The baseline ``Imitation`` task builds its imitation target *relative* to the
agent's current state: the joint target subtracts the current joint angles and
the body target subtracts the current body positions. Both of those quantities
are themselves proprioceptive, so when only the proprioception stream is delayed
the (un-delayed) target can leak near-current proprioceptive information back
into the network, confounding the delay experiment.

``AbsoluteImitation`` overrides only ``_get_imitation_target`` to make the joint
and body targets absolute while keeping them in egocentric coordinates. The root
position/quaternion targets remain relative to the current root frame, which is
unavoidable for an egocentric representation. The returned dict has identical
keys and shapes to the parent, so no downstream network code changes.
"""

import collections
from typing import Any, Mapping

import brax.math
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from vnl_playground.tasks.rodent.imitation import (
    Imitation,
    default_config as _imitation_default_config,
)

_BODY_TARGET_FRAMES = ("current_root", "reference_root")


def default_config() -> config_dict.ConfigDict:
    """Imitation default config extended with ``body_target_frame``.

    ``body_target_frame`` selects the egocentric frame for the absolute body
    targets:

    * ``"current_root"`` (default): reference body positions expressed in the
      *current* root's egocentric frame. Removes the dependence on the current
      joint configuration (no proprioception leak) but still depends on the
      current root pose.
    * ``"reference_root"``: reference body positions expressed in the
      *reference* root's egocentric frame. The pure target pose shape,
      independent of all current state.
    """
    config = _imitation_default_config()
    config.body_target_frame = "current_root"
    return config


class AbsoluteImitation(Imitation):
    """Imitation env whose joint/body targets are absolute, not relative."""

    def __init__(self, config: config_dict.ConfigDict = default_config(), *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        if self._config.body_target_frame not in _BODY_TARGET_FRAMES:
            raise ValueError(
                "config.body_target_frame must be one of "
                f"{_BODY_TARGET_FRAMES}, got {self._config.body_target_frame!r}."
            )

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Absolute imitation target in egocentric coordinates.

        Same structure and shapes as :meth:`Imitation._get_imitation_target`,
        but the joint and body targets do not subtract the agent's current
        state.
        """
        reference = self._get_imitation_reference(data, info)

        # Root pos/quat targets stay relative to the current root frame
        # (unavoidable for an egocentric representation), identical to parent.
        root_pos = self.root_body(data).xpos
        root_quat = self.root_body(data).xquat
        root_targets = jax.vmap(
            lambda ref_pos: brax.math.rotate(ref_pos - root_pos, root_quat)
        )(reference.root_position)
        quat_targets = jax.vmap(
            lambda ref_quat: brax.math.relative_quat(ref_quat, root_quat)
        )(reference.root_quaternion)

        # Absolute joint target: the reference joint angles directly. Joint
        # angles are intrinsically in each joint's own frame (already
        # egocentric), so no current-state subtraction is needed.
        joint_targets = reference.joints

        # Absolute body target, expressed in the chosen egocentric frame.
        body_names = list(self._get_bodies_pos(data, flatten=False).keys())
        # (n_bodies, n_reference_frames, 3)
        ref_body_pos = jp.array(
            [reference.body_xpos(name) for name in body_names]
        )
        if self._config.body_target_frame == "current_root":
            # Reference body positions in the *current* root frame.
            to_ego = jax.vmap(
                lambda pos: brax.math.rotate(pos - root_pos, root_quat)
            )
            body_targets = jax.vmap(to_ego)(ref_body_pos)
        else:  # "reference_root"
            # Reference body positions in the *reference* root frame, per frame.
            ref_root_pos = reference.root_position  # (n_reference_frames, 3)
            ref_root_quat = reference.root_quaternion  # (n_reference_frames, 4)

            def per_body(body_pos):  # body_pos: (n_reference_frames, 3)
                return jax.vmap(
                    lambda p, rpos, rquat: brax.math.rotate(p - rpos, rquat)
                )(body_pos, ref_root_pos, ref_root_quat)

            body_targets = jax.vmap(per_body)(ref_body_pos)

        return collections.OrderedDict(
            root=root_targets,
            quat=quat_targets,
            joint=joint_targets,
            body=body_targets,
        )
