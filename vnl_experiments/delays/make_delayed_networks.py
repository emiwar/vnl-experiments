"""Factory for an asymmetric, actor-only-delayed PPO network.

The constructed pipeline is::

    Sequential([
        Normalizer(obs_size)?,            # if normalize_obs (sees fresh obs)
        PPOAdapter(
            action=Sequential([
                Delay(obs, delay_k)?,     # if delay_k > 0
                EfferenceCopy(
                    inner=Sequential([*actor_mlp, sampler]),
                    queue_length=efference_length,
                ),
            ]),
            value=critic_mlp,             # un-delayed, privileged
        ),
    ])

Both adapter branches receive the same upstream tensor (the normalised obs);
the Delay sits inside the action branch only, so the critic gets fresh obs
automatically. This realises the "actor delay only / privileged critic"
design discussed in the delays plan.
"""

from collections.abc import Callable
from typing import Union

import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Sequential
from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.types import StatefulModule
from nnx_ppo.networks.utils import Filter

from vnl_experiments.delays.efference_copy import EfferenceCopy
from vnl_experiments.delays.forward_model import ForwardModel


def make_delayed_mlp_actor_critic(
    obs_size: int,
    action_size: int,
    actor_hidden_sizes: list[int],
    critic_hidden_sizes: list[int],
    delay_k: int,
    efference_length: int,
    rngs: nnx.Rngs,
    *,
    activation: Union[Callable, str] = nnx.swish,
    normalize_obs: bool = True,
    initializer_scale: float = 1.0,
    entropy_weight: float = 1e-2,
    min_std: float = 1e-3,
    std_scale: float = 1.0,
) -> StatefulModule:
    """Build an actor-only-delayed, privileged-critic PPO network.

    Args:
        obs_size: Flat observation size.
        action_size: Flat action size.
        actor_hidden_sizes: Hidden layer widths for the actor MLP.
        critic_hidden_sizes: Hidden layer widths for the critic MLP.
        delay_k: Number of steps to delay the observation seen by the
            actor. ``0`` disables the delay (baseline). The critic always
            sees the un-delayed observation.
        efference_length: Number of past actor actions to append to the
            (possibly delayed) observation before it enters the actor MLP.
            ``0`` disables the efference copy.
        rngs: NNX random number generators.
        activation: Activation function (callable or one of
            ``{"swish", "tanh", "relu"}``).
        normalize_obs: If True, prepend a :class:`Normalizer` that updates
            running statistics from the un-delayed obs stream.
        initializer_scale: Variance-scaling scale for Dense layers.
        entropy_weight: Entropy bonus weight for the sampler.
        min_std: Minimum policy std.
        std_scale: Multiplicative scale on the policy std.

    Returns:
        A :class:`StatefulModule` whose forward output is a
        :class:`PPONetworkOutput`. Pass it straight to ``ppo.train_ppo``.
    """
    if delay_k < 0:
        raise ValueError(f"delay_k must be non-negative, got {delay_k}")
    if efference_length < 0:
        raise ValueError(
            f"efference_length must be non-negative, got {efference_length}"
        )

    if isinstance(activation, str):
        activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
            activation
        ]

    kernel_init = nnx.initializers.variance_scaling(
        initializer_scale, "fan_in", "uniform"
    )

    actor_in = obs_size + efference_length * action_size
    actor_mlp = make_mlp_layers(
        [actor_in] + actor_hidden_sizes + [action_size * 2],
        rngs,
        activation,  # type: ignore[arg-type]
        activation_last_layer=False,
        kernel_init=kernel_init,
    )
    sampler = NormalTanhSampler(
        rngs,
        entropy_weight=entropy_weight,
        min_std=min_std,
        std_scale=std_scale,
    )
    actor_with_efference = EfferenceCopy(
        inner=Sequential([*actor_mlp, sampler]),
        sample_action=jp.zeros(action_size),
        queue_length=efference_length,
    )

    action_branch_layers: list[StatefulModule] = []
    if delay_k > 0:
        action_branch_layers.append(
            Delay(jp.zeros(obs_size), k_steps=delay_k)
        )
    action_branch_layers.append(actor_with_efference)

    critic = make_mlp(
        [obs_size] + critic_hidden_sizes + [1],
        rngs,
        activation,  # type: ignore[arg-type]
        activation_last_layer=False,
        kernel_init=kernel_init,
    )

    adapter = PPOAdapter(
        action=Sequential(action_branch_layers),
        value=critic,
    )
    if normalize_obs:
        return Sequential([Normalizer(obs_size), adapter])
    return adapter


def make_forward_model_actor_critic(
    obs_size: int,
    action_size: int,
    actor_hidden_sizes: list[int],
    critic_hidden_sizes: list[int],
    delay_k: int,
    efference_length: int,
    rngs: nnx.Rngs,
    *,
    predictor_hidden_sizes: list[int] | None = None,
    fm_loss_weight: float = 1.0,
    detach_prediction: bool = True,
    activation: Union[Callable, str] = nnx.swish,
    normalize_obs: bool = True,
    initializer_scale: float = 1.0,
    entropy_weight: float = 1e-2,
    min_std: float = 1e-3,
    std_scale: float = 1.0,
) -> StatefulModule:
    """Build an explicit-forward-model, privileged-critic PPO network for flat obs.

    The flat-observation analogue of the rodent :class:`ForwardModel` path: the
    whole observation plays the role of "proprioception" (there is no
    task/proprioception split and hence no encoder/latent). A learned predictor
    maps ``[delayed obs, action buffer] -> current obs``; the actor MLP consumes
    only the predicted current obs. A self-supervised L2 loss (logged as
    ``fm_pred_mse``) trains the prediction toward the true current obs. The
    critic sees the full, un-delayed observation (privileged).

    Args:
        obs_size: Flat observation size (also the predicted proprioception width).
        action_size: Flat action size.
        actor_hidden_sizes: Hidden layer widths for the decoder MLP.
        critic_hidden_sizes: Hidden layer widths for the critic MLP.
        delay_k: Observation delay in steps. ``0`` keeps the predictor active as
            an identity model (capacity diagnostic / baseline).
        efference_length: Number of past actor actions fed to the predictor.
            ``0`` disables the efference copy (predictor sees delayed obs only).
        rngs: NNX random number generators.
        predictor_hidden_sizes: Hidden layer widths for the predictor MLP.
            Defaults to ``actor_hidden_sizes``.
        fm_loss_weight: Weight on the self-supervised forward-model L2 loss.
        detach_prediction: When ``True`` (default) the prediction is detached
            before the decoder (true forward model — the policy gradient never
            reaches the predictor). ``False`` trains the predictor with the
            policy gradient too (architecture ablation).
        activation: Activation function (callable or one of
            ``{"swish", "tanh", "relu"}``).
        normalize_obs: If True, prepend a :class:`Normalizer` over the fresh obs.
        initializer_scale: Variance-scaling scale for Dense layers.
        entropy_weight: Entropy bonus weight for the sampler.
        min_std: Minimum policy std.
        std_scale: Multiplicative scale on the policy std.

    Returns:
        A :class:`StatefulModule` whose forward output is a
        :class:`PPONetworkOutput`. Pass it straight to ``ppo.train_ppo``.
    """
    if delay_k < 0:
        raise ValueError(f"delay_k must be non-negative, got {delay_k}")
    if efference_length < 0:
        raise ValueError(
            f"efference_length must be non-negative, got {efference_length}"
        )

    if isinstance(activation, str):
        activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
            activation
        ]

    if predictor_hidden_sizes is None:
        predictor_hidden_sizes = actor_hidden_sizes

    kernel_init = nnx.initializers.variance_scaling(
        initializer_scale, "fan_in", "uniform"
    )

    # Decoder sees only the predicted current observation (no task latent).
    decoder = Sequential(
        [
            *make_mlp_layers(
                [obs_size] + actor_hidden_sizes + [action_size * 2],
                rngs,
                activation,  # type: ignore[arg-type]
                activation_last_layer=False,
                kernel_init=kernel_init,
            ),
            NormalTanhSampler(
                rngs,
                entropy_weight=entropy_weight,
                min_std=min_std,
                std_scale=std_scale,
            ),
        ]
    )
    # Predictor: [delayed obs, action buffer] -> current obs.
    predictor = make_mlp(
        [obs_size + efference_length * action_size]
        + predictor_hidden_sizes
        + [obs_size],
        rngs,
        activation,  # type: ignore[arg-type]
        activation_last_layer=False,
        kernel_init=kernel_init,
    )

    actor = Sequential(
        [
            # Flat obs tensor -> {"proprioception": obs} so EfferenceCopy /
            # ForwardModel can address it by name.
            Filter({"proprioception": lambda x: x}),
            EfferenceCopy(
                inner=ForwardModel(
                    decoder=decoder,
                    predictor=predictor,
                    proprio_size=obs_size,
                    delay_steps=delay_k,
                    loss_weight=fm_loss_weight,
                    detach_prediction=detach_prediction,
                    latent_key=None,
                ),
                sample_action=jp.zeros(action_size),
                queue_length=efference_length,
                inject_key="efference",
            ),
        ]
    )

    critic = make_mlp(
        [obs_size] + critic_hidden_sizes + [1],
        rngs,
        activation,  # type: ignore[arg-type]
        activation_last_layer=False,
        kernel_init=kernel_init,
    )

    adapter = PPOAdapter(action=actor, value=critic)
    if normalize_obs:
        return Sequential([Normalizer(obs_size), adapter])
    return adapter
