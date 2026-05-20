"""NerveNet-style modular network for the rodent body.

Rewritten on top of :class:`nnx_ppo.networks.graph.PopulationGraph`. The
legacy v3 hand-wired a two-stage afferent/efferent message-passing
scheme inside ``__call__`` with ~100 lines of ``Dense + add`` calls.
Here the same topology is expressed declaratively:

* one input population per body module reads its obs slice;
* one hidden-state population per body module sum-integrates its
  incoming connections and applies the transfer function once;
* afferent connections (end-effector → limb → root, plus torso/head →
  root) at ``delay=0`` — same-step propagation;
* efferent connections (root → limb → end-effector, plus root →
  torso/head) at ``delay=1`` — they appear one step later, breaking the
  otherwise-illegal delay-0 cycles.

This is a behavioural change relative to the legacy implementation: the
legacy network applied a connection-level activation inside each Dense
(double-nonlinearity per limb-step) and unrolled the two passes inside a
single forward call; here populations apply their transfer function once
per step, and the round-trip limb↔root takes two simulation steps. Old
checkpoints do not transfer.
"""

from collections.abc import Mapping, Callable
from typing import Any, Optional, Union

import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.adapter import PPOAdapter
from nnx_ppo.algorithms.distributions import NormalTanhSampler
from nnx_ppo.networks.containers import Flattener, Sequential
from nnx_ppo.networks.factories import make_mlp_layers
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.graph import PopulationGraph
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.types import (
    Context,
    StatefulModule,
    StatefulModuleOutput,
)


NON_END_EFFECTORS = ["arm_L", "arm_R", "leg_L", "leg_R", "torso", "head"]
END_EFFECTORS = ["hand_L", "hand_R", "foot_L", "foot_R"]
LIMB_PAIRS = [
    ("arm_L", "hand_L"),
    ("arm_R", "hand_R"),
    ("leg_L", "foot_L"),
    ("leg_R", "foot_R"),
]


class _Scale(StatefulModule):
    """Multiply input by a fixed scalar. Used to apply motor_scale /
    critic_scale to head outputs without baking the factor into Dense weights.
    """

    def __init__(self, factor: float):
        self.factor = float(factor)

    def __call__(self, state, x, *, context: Context = Context.INFERENCE):
        return StatefulModuleOutput(state, x * self.factor, jp.array(0.0), {})


class _FlattenObsValues(StatefulModule):
    """Per-key Flattener for dict observations.

    Input: ``{k: pytree}``. Output: ``{k: [B, -1] array}`` for each key.
    Playground modular rodent envs hand the network observations whose
    per-body-module values are themselves small pytrees (proprio /
    target / etc.). Downstream consumers (``Normalizer``,
    ``PopulationGraph``) want a flat tensor per key matching the
    declared ``obs_sizes[k]``.

    Idempotent: if a value is already a flat ``[B, D]`` array, the
    reshape-and-concat is a no-op.
    """

    def __call__(self, state, obs, *, context: Context = Context.INFERENCE):
        flattener = Flattener()
        out = {k: flattener((), v).output for k, v in obs.items()}
        return StatefulModuleOutput(state, out, jp.array(0.0), {})


class _MotorPlusDetachedCritic(StatefulModule):
    """Runs the population graph (motor outputs) and a separate MLP critic
    on the same input, splits the critic's ``[B, n_modules]`` output by
    module key, and returns a single flat dict combining both. Used to
    feed ``PPOAdapter`` when ``detached_critic=True``.
    """

    def __init__(
        self,
        graph: StatefulModule,
        critic: StatefulModule,
        value_keys: list[str],
    ):
        self.graph = graph
        self.critic = critic
        self._value_keys = list(value_keys)

    def __call__(self, state, obs, *, context: Context = Context.INFERENCE):
        graph_out = self.graph(state["graph"], obs, context=context)
        critic_out = self.critic(state["critic"], obs, context=context)
        merged = dict(graph_out.output)
        for i, k in enumerate(self._value_keys):
            merged[k] = critic_out.output[..., i]
        return StatefulModuleOutput(
            next_state={
                "graph": graph_out.next_state,
                "critic": critic_out.next_state,
            },
            output=merged,
            regularization_loss=(
                jp.sum(graph_out.regularization_loss)
                + jp.sum(critic_out.regularization_loss)
            ),
            metrics={"graph": graph_out.metrics, "critic": critic_out.metrics},
        )

    def initialize_state(self, batch_size: int):
        return {
            "graph": self.graph.initialize_state(batch_size),
            "critic": self.critic.initialize_state(batch_size),
        }

    def reset_state(self, prev_state):
        return {
            "graph": self.graph.reset_state(prev_state["graph"]),
            "critic": self.critic.reset_state(prev_state["critic"]),
        }


class NerveNetNetwork_v4(PPOAdapter):
    """Modular NerveNet-style PPO network for the rodent body."""

    def __init__(
        self,
        obs_sizes: Mapping[str, int],
        action_sizes: Mapping[str, int],
        hidden_size: int,
        root_size: int,
        rngs: nnx.Rngs,
        entropy_weight: float = 0.01,
        min_std: float = 0.1,
        motor_scale: float = 1.0,
        critic_scale: float = 1.0,
        normalize_obs: bool = True,
        activation: Union[str, Callable] = nnx.swish,
        detached_critic: bool = False,
        detached_critic_hidden_sizes: Optional[list[int]] = None,
    ):
        if isinstance(activation, str):
            activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
                activation
            ]

        body_modules = list(obs_sizes.keys())
        graph = PopulationGraph(rngs)

        # Input populations: Dense(obs_size, hidden_or_root) + activation.
        for k in body_modules:
            out_size = root_size if k == "root" else hidden_size
            graph.add_population(
                f"in_{k}",
                input_size=int(obs_sizes[k]),
                output_size=out_size,
                compute=Dense(int(obs_sizes[k]), out_size, rngs),
                activation=activation,
                input_from=k,
            )

        # Hidden-state populations: sum-integrating junctions with the
        # transfer function applied once.
        for k in body_modules:
            out_size = root_size if k == "root" else hidden_size
            graph.add_population(k, output_size=out_size, activation=activation)
            graph.connect(f"in_{k}", k)

        # Afferents (toward root). delay=0 — same-step propagation.
        for limb, ee in LIMB_PAIRS:
            if ee in body_modules and limb in body_modules:
                graph.connect(ee, limb)
            if limb in body_modules:
                graph.connect(limb, "root")
        for k in ("torso", "head"):
            if k in body_modules:
                graph.connect(k, "root")

        # Efferents (away from root). delay=1 — break the limb↔root cycle.
        for limb, ee in LIMB_PAIRS:
            if limb in body_modules:
                graph.connect("root", limb, delay=1)
            if ee in body_modules and limb in body_modules:
                graph.connect(limb, ee, delay=1)
        for k in ("torso", "head"):
            if k in body_modules:
                graph.connect("root", k, delay=1)

        # Per-module motor heads (only for keys that have actions).
        for k, a in action_sizes.items():
            head_layers: list[StatefulModule] = [
                Dense(
                    hidden_size,
                    2 * int(a),
                    rngs,
                    kernel_init=nnx.initializers.zeros,
                )
            ]
            if motor_scale != 1.0:
                head_layers.append(_Scale(motor_scale))
            head = head_layers[0] if len(head_layers) == 1 else Sequential(head_layers)
            graph.add_output(f"motor_{k}", source=k, head=head)

        # Per-module value heads. In the default (attached) mode each body
        # module gets a per-population Dense → 1 head, fed by that
        # module's hidden state. In detached mode a separate MLP on the
        # full normalised obs produces all module values; the graph
        # itself has no value heads.
        value_keys = [f"value_{k}" for k in body_modules]
        if not detached_critic:
            for k in body_modules:
                out_size = root_size if k == "root" else hidden_size
                head_layers = [
                    Dense(out_size, 1, rngs, kernel_init=nnx.initializers.zeros)
                ]
                if critic_scale != 1.0:
                    head_layers.append(_Scale(critic_scale))
                head = (
                    head_layers[0]
                    if len(head_layers) == 1
                    else Sequential(head_layers)
                )
                graph.add_output(f"value_{k}", source=k, head=head)

        graph.finalize()

        # Build the trunk that sits behind the (optional) Normalizer.
        # Attached: the graph alone already emits motor_* and value_*.
        # Detached: pair the graph (motor_* only) with a separate MLP
        # whose output is split into one value per body module.
        trunk: StatefulModule
        if detached_critic:
            hidden_layers = detached_critic_hidden_sizes or [512, 512]
            total_obs_size = sum(int(s) for s in obs_sizes.values())
            critic_sizes = [total_obs_size, *hidden_layers, len(body_modules)]
            critic_layers: list[StatefulModule] = [Flattener()]
            critic_layers.extend(
                make_mlp_layers(
                    critic_sizes, rngs, activation, activation_last_layer=False
                )
            )
            if critic_scale != 1.0:
                critic_layers.append(_Scale(critic_scale))
            critic = Sequential(critic_layers)
            trunk = _MotorPlusDetachedCritic(graph, critic, value_keys)
        else:
            trunk = graph

        # Flatten per-key obs sub-pytrees to flat [B, D] tensors before
        # anything downstream sees them. Required because Playground
        # modular envs hand the network nested obs values (proprio /
        # target / ...) per body module, while Normalizer and the graph
        # expect flat arrays matching obs_sizes[k].
        pre_trunk: list[StatefulModule] = [_FlattenObsValues()]
        if normalize_obs:
            pre_trunk.append(Normalizer(dict(obs_sizes)))
        inner: StatefulModule = Sequential([*pre_trunk, trunk])

        action_specs = {
            f"motor_{k}": NormalTanhSampler(rngs, entropy_weight, min_std)
            for k in action_sizes
        }
        value_specs = value_keys

        # Expose underlying graph + (optional) normalizer for introspection
        # by checkpoint utilities / logging code.
        self.graph = graph
        # The Normalizer, if present, is the second entry of `inner.layers`
        # (after the _FlattenObsValues prepass).
        self.normalizer = (
            inner.layers[1] if normalize_obs else None  # type: ignore[index]
        )

        super().__init__(
            inner=inner, action_specs=action_specs, value_specs=value_specs
        )
