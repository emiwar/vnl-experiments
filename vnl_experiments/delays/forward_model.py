"""Explicit forward model for the proprioception-delay experiment.

Where the plain encoder-decoder lets the decoder *implicitly* compensate for a
proprioceptive delay (it is simply fed the delayed proprioception plus an
efference-copy action buffer), this module makes the compensation *explicit*: a
learned predictor maps the delayed proprioception and the recent action buffer
to an estimate of the **current** (undelayed) proprioception, and only that
estimate is handed to the decoder.

A self-supervised L2 loss trains the prediction toward the true current
proprioception (privileged information available only at train time, analogous
to the privileged critic). Two ``stop_gradient`` s keep the two learning signals
disjoint:

* ``stop_gradient`` on the prediction *before* the decoder — the policy / actor
  gradient never reaches the predictor. This one is **conditional** on
  ``detach_prediction`` (default ``True``); set it ``False`` to train the
  predictor with the policy gradient (see the ablation note below).
* ``stop_gradient`` on the predictor's *inputs* (delayed proprioception + the
  flattened action buffer) and on the L2 *target* — during loss replay the
  action queue carries decoder-parameter dependence through BPTT, so detaching
  the inputs and target ensures the L2 updates **only** the predictor. These two
  are always on, independent of ``detach_prediction``.

The ``detach_prediction`` flag, combined with ``loss_weight``, separates the
contribution of the *architecture* from that of the *forward loss*:

* ``detach_prediction=True``, ``loss_weight>0`` — the forward model: the
  predictor is shaped only by the L2 prediction objective.
* ``detach_prediction=False``, ``loss_weight=0`` — architecture-only ablation:
  the predictor is an ordinary policy layer, shaped purely by the policy
  gradient. Comparing the two isolates the value of the forward loss itself.
* ``detach_prediction=True``, ``loss_weight=0`` — degenerate: the predictor
  receives no gradient at all and stays frozen at its initialisation.

The module is designed to sit inside an :class:`EfferenceCopy` running in
``inject_key`` (dict) mode, so it receives a dict
``{"task_obs": latent, "proprioception": proprio_cur, "efference": queue}`` and
reads each entry by name rather than by index slicing.
"""

from typing import Any

import jax
import jax.numpy as jp

from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.types import (
    ModuleState,
    StatefulModule,
    StatefulModuleOutput,
)


class ForwardModel(StatefulModule):
    """Predict the current proprioception, then decode an action from it.

    Carry::

        {
          "delay": <Delay carry, or () when delay_steps == 0>,
          "pred":  <predictor carry>,
          "dec":   <inner decoder carry>,
        }

    Args:
        decoder: The inner actor pipeline ending in an ``ActionSampler`` so its
            forward ``output`` is a sampler dict ``{"action", "log_likelihood"}``.
            It is fed ``[latent, predicted_proprioception]``.
        predictor: An MLP mapping
            ``proprio_size + efference_length * action_size -> proprio_size``.
        proprio_size: Flattened proprioception width (used to build the internal
            delay buffer).
        delay_steps: Proprioception delay. ``>= 1`` embeds a :class:`Delay`;
            ``0`` keeps the predictor active as an identity model
            (``proprio_t -> proprio_t``), a useful capacity diagnostic.
        loss_weight: Scale on the self-supervised L2 loss added to
            ``regularization_loss``.
        detach_prediction: When ``True`` (default) the prediction is
            ``stop_gradient`` 'd before the decoder, so the policy gradient never
            reaches the predictor (the forward model). When ``False`` the policy
            gradient flows into the predictor's params.
        latent_key: Dict key carrying the task latent (decoder context).
            ``None`` means there is no task latent (e.g. a flat observation with
            no task/proprioception split); the decoder then sees only the
            predicted proprioception.
        proprio_key: Dict key carrying the current proprioception.
        efference_key: Dict key carrying the action queue. Absent (or ``None``)
            means no efference — the predictor sees the delayed proprioception
            only.
    """

    def __init__(
        self,
        decoder: StatefulModule,
        predictor: StatefulModule,
        proprio_size: int,
        delay_steps: int,
        loss_weight: float = 1.0,
        detach_prediction: bool = True,
        latent_key: str | None = "task_obs",
        proprio_key: str = "proprioception",
        efference_key: str = "efference",
    ):
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        self.decoder = decoder
        self.predictor = predictor
        self.loss_weight = loss_weight
        self.detach_prediction = detach_prediction
        self.latent_key = latent_key
        self.proprio_key = proprio_key
        self.efference_key = efference_key
        self.delay = (
            Delay(jp.zeros(proprio_size), k_steps=delay_steps)
            if delay_steps >= 1
            else None
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def initialize_state(self, batch_size: int) -> ModuleState:
        return {
            "delay": (
                self.delay.initialize_state(batch_size)
                if self.delay is not None
                else ()
            ),
            "pred": self.predictor.initialize_state(batch_size),
            "dec": self.decoder.initialize_state(batch_size),
        }

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        return {
            "delay": (
                self.delay.reset_state(prev_state["delay"])
                if self.delay is not None
                else ()
            ),
            "pred": self.predictor.reset_state(prev_state["pred"]),
            "dec": self.decoder.reset_state(prev_state["dec"]),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        state: ModuleState,
        x: Any,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        latent = x[self.latent_key] if self.latent_key is not None else None
        proprio_cur = x[self.proprio_key]
        queue = x.get(self.efference_key)

        # Most recent observation actually available to the agent.
        if self.delay is not None:
            delay_out = self.delay(state["delay"], proprio_cur)
            p_delayed = delay_out.output
            delay_state = delay_out.next_state
        else:
            p_delayed = proprio_cur
            delay_state = state["delay"]

        # Predictor inputs are detached so the L2 loss trains only the
        # predictor (the action queue otherwise carries decoder-parameter
        # gradient through BPTT during loss replay).
        pred_terms = [p_delayed]
        if queue is not None:
            pred_terms.append(queue.reshape(queue.shape[0], -1))
        pred_in = jax.lax.stop_gradient(jp.concatenate(pred_terms, axis=-1))
        pred_out = self.predictor(state["pred"], pred_in)
        p_hat = pred_out.output

        # Self-supervised loss toward the true (privileged) current
        # proprioception.
        target = jax.lax.stop_gradient(proprio_cur)
        fm_loss = jp.mean((p_hat - target) ** 2, axis=-1)

        # Decoder sees the task latent (when present) + the prediction. By
        # default the prediction is detached so the policy gradient never flows
        # back into the predictor; with detach_prediction=False the predictor is
        # trained by the policy gradient (architecture-only ablation).
        p_for_decoder = (
            jax.lax.stop_gradient(p_hat) if self.detach_prediction else p_hat
        )
        dec_in = (
            jp.concatenate([latent, p_for_decoder], axis=-1)
            if latent is not None
            else p_for_decoder
        )
        dec_out = self.decoder(state["dec"], dec_in, rollout_extras)

        regularization_loss = (
            dec_out.regularization_loss
            + pred_out.regularization_loss
            + self.loss_weight * fm_loss
        )

        metrics = {
            "fm_pred_mse": fm_loss,
            "decoder": dec_out.metrics,
            "predictor": pred_out.metrics,
        }
        if self.delay is not None:
            metrics["delay"] = delay_out.metrics
        return StatefulModuleOutput(
            next_state={
                "delay": delay_state,
                "pred": pred_out.next_state,
                "dec": dec_out.next_state,
            },
            output=dec_out.output,
            regularization_loss=regularization_loss,
            metrics=metrics,
            rollout_extras=dec_out.rollout_extras,
        )

    def update_statistics(self, rollout_extras: Any) -> None:
        # Only the decoder threads rollout_extras (its sampler); the predictor
        # and delay emit nothing and own no running statistics.
        self.decoder.update_statistics(rollout_extras)
