"""MLP-based modular network for multi-module rodent imitation.

Shared-encoder MLP with per-module actor heads and per-module value
heads, expressed as a plain :class:`Sequential` ending in a two-port
:class:`PPOAdapter`. Suitable as a distillation teacher or a flat
baseline against modular topologies like ``nervenet_style_v4``.
"""

from collections.abc import Callable, Mapping
from typing import Union

import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Parallel, Sequential
from nnx_ppo.networks.factories import make_mlp
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput
from nnx_ppo.networks.utils import Filter, Flattener


class JointSamplerBank(StatefulModule):
    """Collapse a per-key Parallel of samplers into one joint sampler dict.

    Wraps ``Parallel({pop: sampler})``. The Parallel emits
    ``{pop: {"action": ..., "log_likelihood": ...}}``; this module reshapes
    that into a *single* sampler dict::

        {"action": {pop: action_pop}, "log_likelihood": sum_pop LL_pop}

    The summed log-likelihood is the joint policy log-probability under
    the conditional-independence factoring across pops. ``PPOAdapter``
    then sees one sampler dict (one leaf) and produces a scalar
    ``loglikelihoods`` field on ``PPONetworkOutput`` — restoring the
    centralised-actor / joint-clip PPO semantics required for stable
    training on multi-actuator setups.
    """

    def __init__(self, samplers: dict[str, StatefulModule]):
        self.bank = Parallel(samplers)

    def __call__(self, state, x, rollout_extras=None):
        out = self.bank(state, x, rollout_extras)
        actions = {k: v["action"] for k, v in out.output.items()}
        ll = sum(v["log_likelihood"] for v in out.output.values())
        return StatefulModuleOutput(
            next_state=out.next_state,
            output={"action": actions, "log_likelihood": ll},
            regularization_loss=out.regularization_loss,
            metrics=out.metrics,
            rollout_extras=out.rollout_extras,
        )

    def initialize_state(self, batch_size):
        return self.bank.initialize_state(batch_size)

    def reset_state(self, prev_state):
        return self.bank.reset_state(prev_state)

    def update_statistics(self, rollout_extras):
        self.bank.update_statistics(rollout_extras)


def _make_reveal_filter(
    obs_sizes: Mapping[str, int], reveal_targets: str
) -> Filter | None:
    """Build the up-front :class:`Filter` that implements ``reveal_targets``.

    Returns ``None`` for ``reveal_targets="all"`` (no filtering needed).
    For the other modes the filter keeps only the requested obs streams,
    matching the per-key totals encoded in ``obs_sizes``.
    """
    if reveal_targets == "all":
        return None

    spec: dict[str, object] = {}
    for k in obs_sizes:
        if k == "root":
            continue
        spec[k] = (k, "proprioception")

    if reveal_targets == "none":
        spec["root"] = lambda obs: jp.zeros(obs["root"]["current_target"].shape[:1] + (0,))
    elif reveal_targets == "root_only":
        spec["root"] = "root"
    elif reveal_targets == "joystick_only":
        spec["root"] = ("root", "future_target", "pos")
    elif reveal_targets == "proprio_only":
        # No root info at all.
        pass
    else:
        raise ValueError(f"Unknown reveal_targets={reveal_targets!r}")
    return Filter(spec)


def MLPModularNetwork(
    obs_sizes: Mapping[str, int],
    action_sizes: Mapping[str, int],
    actor_hidden_sizes: list[int],
    critic_hidden_sizes: list[int],
    rngs: nnx.Rngs,
    activation: Union[str, Callable] = nnx.swish,
    normalize_obs: bool = True,
    entropy_weight: float = 1e-2,
    min_std: float = 1e-1,
    std_scale: float = 1.0,
    initializer_scale: float = 1.0,
    reveal_targets: str = "all",
) -> Sequential:
    """Build a shared-encoder MLP modular network.

    Pipeline shape::

        Sequential([
            Filter(reveal_targets)?,            # optional, see _make_reveal_filter
            Flattener(),                        # all obs → single (B, D) tensor
            Normalizer(flat_obs_size)?,         # if normalize_obs
            PPOAdapter(
                action=Sequential([
                    actor_encoder,              # → (B, H_a)
                    Parallel({pop: Sequential([
                        Dense(H_a, 2 * A[pop]), # per-pop action params
                        NormalTanhSampler(...), # per-pop sampler
                    ]) for pop in action_keys}),
                ]),
                value=Sequential([
                    critic_encoder,             # → (B, H_c)
                    Parallel({pop: Dense(H_c, 1) for pop in body_modules}),
                ]),
            ),
        ])

    Returns a :class:`Sequential` whose forward output is a
    :class:`PPONetworkOutput` keyed per-module.
    """
    if isinstance(activation, str):
        activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
            activation
        ]

    flat_obs_size = int(sum(obs_sizes.values()))
    kernel_init = nnx.initializers.variance_scaling(
        initializer_scale, "fan_in", "uniform"
    )

    # Preprocessing pipeline.
    pre: list[StatefulModule] = []
    reveal_filter = _make_reveal_filter(obs_sizes, reveal_targets)
    if reveal_filter is not None:
        pre.append(reveal_filter)
    pre.append(Flattener())
    if normalize_obs:
        pre.append(Normalizer(flat_obs_size))

    # Actor port: shared encoder, then per-module head + sampler.
    actor_encoder = make_mlp(
        [flat_obs_size] + actor_hidden_sizes,
        rngs,
        activation,
        activation_last_layer=True,
        kernel_init=kernel_init,
    )
    actor_feature_size = actor_hidden_sizes[-1]
    action_port: StatefulModule = Sequential([
        actor_encoder,
        JointSamplerBank({
            pop: Sequential([
                Dense(
                    actor_feature_size,
                    2 * int(action_sizes[pop]),
                    rngs,
                    kernel_init=kernel_init,
                ),
                NormalTanhSampler(rngs, entropy_weight, min_std, std_scale),
            ])
            for pop in action_sizes
        }),
    ])

    # Value port: shared encoder, then per-module Dense(H, 1) head.
    critic_encoder = make_mlp(
        [flat_obs_size] + critic_hidden_sizes,
        rngs,
        activation,
        activation_last_layer=True,
        kernel_init=kernel_init,
    )
    critic_feature_size = critic_hidden_sizes[-1]
    value_port: StatefulModule = Sequential([
        critic_encoder,
        Parallel({
            pop: Dense(critic_feature_size, 1, rngs, kernel_init=kernel_init)
            for pop in obs_sizes
        }),
    ])

    return Sequential([
        *pre,
        PPOAdapter(action=action_port, value=value_port),
    ])
