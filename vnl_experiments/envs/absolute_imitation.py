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
and body targets absolute while keeping them in egocentric coordinates. The
returned dict has identical keys and shapes to the parent under every setting, so
no downstream network code changes.

``body_target_frame`` selects how much of the current state the target is allowed
to depend on, in three steps -- see :func:`default_config` for the per-value
detail and the history:

============================ ============= ================================
value                        ``body``      ``root`` / ``quat``
============================ ============= ================================
``current_root`` (default)   current root  current root
``reference_root``           reference     current root
``reference_root_open_loop`` reference     reference at the current frame
============================ ============= ================================

Only the third makes the whole target independent of the walker's own state: it is
a function of the clip and the current frame index alone. The first two both hand
the network an undelayed root position and orientation *error*, which is what a
delay or proprioception ablation is usually trying to withhold.
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

#: The value that makes the target fully state-independent. Named rather than inlined
#: because the training script keys a run-name token off it and the tests assert on it.
OPEN_LOOP_FRAME = "reference_root_open_loop"

#: Every valid value, and the single source of truth for "is this an AbsoluteImitation
#: config?". The offline eval paths decide which env class to rebuild by testing
#: ``env_params["body_target_frame"]`` against this, so a value added here is picked up
#: there for free -- and a config recording something else (some early ones say
#: ``"neither"``) still resolves to the base ``Imitation``, as it should.
BODY_TARGET_FRAMES = ("current_root", "reference_root", OPEN_LOOP_FRAME)


def default_config() -> config_dict.ConfigDict:
    """Imitation default config extended with ``body_target_frame``.

    ``body_target_frame`` selects the egocentric frame the imitation target is
    expressed in. Each value removes one more dependence on the walker's own state
    than the one above it:

    * ``"current_root"`` (default): reference body positions expressed in the
      *current* root's egocentric frame. Removes the dependence on the current
      joint configuration (no proprioception leak) but still depends on the
      current root pose.
    * ``"reference_root"``: reference body positions expressed in the
      *reference* root's egocentric frame, so the ``body`` targets are the pure
      target pose shape. ``root`` and ``quat`` are **unchanged** from
      ``current_root`` -- they remain the reference root pose relative to the
      *current* root, i.e. an undelayed root position and orientation error.
    * ``"reference_root_open_loop"``: as ``reference_root``, and ``root`` / ``quat``
      are additionally taken relative to the **reference** root at the current
      frame instead of the walker's actual root. The whole target is then a
      function of ``(clip, current frame)`` alone and carries no information about
      the walker's state.

    Why the third value exists, and why it is a value rather than a separate key.
    ``reference_root`` was added to stop the target leaking current state, and its
    name and original docstring ("the pure target pose shape, independent of all
    current state") were read that way -- but it only ever governed the ``body``
    sub-key, which is 270 of the 640 numbers in the rodent's ``task_obs``. The 35 in
    ``root`` / ``quat`` stayed a live, undelayed root-tracking-error signal in both
    settings, so a run with ``dec_use_proprioception=False`` was never actually open
    loop. ``analysis/position-control-open-loop/`` is where that surfaced, and its
    ``frame_leak.py`` measures which sub-keys move when the walker is displaced.

    It is a third value because every committed analysis selects its cohorts on
    ``env_params.body_target_frame``. A new value is invisible to those selectors, so
    they keep excluding the runs they were written before; a separate key would have
    been invisible *to the wrong side* -- every existing ``== "reference_root"`` filter
    would have silently pooled open-loop runs into cohorts that predate them, which is
    the ``dec_use_intention`` / ``dec_use_proprioception`` trap in
    ``analysis/README.md`` §6 all over again. The cost is that the field name now
    undersells its scope; the compensation is this docstring and the table in the module
    header. The 2x2 a separate key would have offered also has a dead cell:
    ``current_root`` body targets depend on the current root pose, so pairing them with a
    state-free ``root`` / ``quat`` would not be open loop either.

    The default stays ``current_root``, so an old ``config.json`` reloads to exactly the
    behaviour it trained with, and nothing about the two existing values changes. **This is
    the reload default, not the launch default** -- new runs get theirs from
    ``conf/env/rodent_imitation.yaml``, which is where to change the standard. Leave this one
    alone: it is what a stored config that predates the key falls back to when the env class
    is forced to ``AbsoluteImitation`` (the ``f315e336`` runs, via ``eval_videos.ENV_OVERRIDES``);
    a config with no key *and* no forced class resolves to base ``Imitation`` instead and
    never consults it.

    One consequence to plan for when using the open-loop value: it takes the root error away
    from the **critic** too. ``delays.network_builders.build_delay_network`` feeds the critic
    undelayed ``task_obs + proprioception``, and proprioception carries no root-position
    error, so the value function loses its most direct predictor of a ``root_too_far``
    termination. A reward drop under this setting is therefore not purely an actor effect.
    """
    config = _imitation_default_config()
    config.body_target_frame = "current_root"
    return config


def validate_body_target_frame(frame: Any) -> str:
    """Return ``frame`` if it names a known target frame, else raise.

    Separate from the constructor so that a launcher, a config check or a test can reject
    a typo without building an env (which needs MuJoCo, a GPU and the reference clips). A
    silently-unknown value would be the worst outcome: the target is only read by the
    network, so a mistyped frame produces a run that trains happily and means something
    other than its config says.
    """
    if frame not in BODY_TARGET_FRAMES:
        raise ValueError(
            "config.body_target_frame must be one of "
            f"{BODY_TARGET_FRAMES}, got {frame!r}."
        )
    return frame


class AbsoluteImitation(Imitation):
    """Imitation env whose joint/body targets are absolute, not relative."""

    def __init__(self, config: config_dict.ConfigDict = default_config(), *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        validate_body_target_frame(self._config.body_target_frame)

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Absolute imitation target in egocentric coordinates.

        Same structure and shapes as :meth:`Imitation._get_imitation_target` under
        every ``body_target_frame``, but the joint and body targets do not subtract
        the agent's current state.
        """
        reference = self._get_imitation_reference(data, info)
        frame = self._config.body_target_frame
        open_loop = frame == OPEN_LOOP_FRAME

        # The pose that `root` / `quat` are expressed relative to. Under the two original
        # settings it is the walker's own root, which makes those two sub-keys a root
        # tracking *error* -- undelayed feedback, and the reason a proprioception ablation
        # alone does not give an open-loop policy. Under OPEN_LOOP_FRAME it is instead the
        # reference root at the *current* frame, so the same numbers become the reference's
        # own upcoming displacement: a function of the clip and the frame index, with no
        # dependence on the walker at all.
        if open_loop:
            current = self._get_current_target(data, info)
            anchor_pos = current.root_position
            anchor_quat = current.root_quaternion
        else:
            anchor_pos = self.root_body(data).xpos
            anchor_quat = self.root_body(data).xquat

        root_targets = jax.vmap(
            lambda ref_pos: brax.math.rotate(ref_pos - anchor_pos, anchor_quat)
        )(reference.root_position)
        quat_targets = jax.vmap(
            lambda ref_quat: brax.math.relative_quat(ref_quat, anchor_quat)
        )(reference.root_quaternion)

        # Absolute joint target: the reference joint angles directly. Joint
        # angles are intrinsically in each joint's own frame (already
        # egocentric), so no current-state subtraction is needed.
        joint_targets = reference.joints

        # Absolute body target, expressed in the chosen egocentric frame. `data` is read
        # here only for the body *names*, whose order is a property of the model rather
        # than of the state, so this is not a route by which state reaches the target.
        body_names = list(self._get_bodies_pos(data, flatten=False).keys())
        # (n_bodies, n_reference_frames, 3)
        ref_body_pos = jp.array(
            [reference.body_xpos(name) for name in body_names]
        )
        if frame == "current_root":
            # Reference body positions in the *current* root frame. `anchor_*` is the
            # current root pose here, so this is unchanged from before the open-loop
            # option existed.
            to_ego = jax.vmap(
                lambda pos: brax.math.rotate(pos - anchor_pos, anchor_quat)
            )
            body_targets = jax.vmap(to_ego)(ref_body_pos)
        else:  # "reference_root" or OPEN_LOOP_FRAME
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
