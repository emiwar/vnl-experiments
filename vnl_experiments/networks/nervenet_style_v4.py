"""NerveNet-style modular network for the rodent body.

Rewritten on the final Phase-B pseudocode shape: the network is a plain
``Sequential`` ending in a two-port :class:`PPOAdapter`. The
:class:`PopulationGraph` does the heavy lifting in the middle; the
samplers and value extraction live inside the adapter's ports.

Topology
~~~~~~~~

* One input population per body module (reads ``obs[k]``).
* One hidden-state population per body module (sum-integrates incoming
  connections, applies the activation once).
* Reciprocal ``delay=1`` connections between adjacent body modules
  (end-effector ↔ limb, limb ↔ root, torso/head ↔ root).
* For each module with an action: an output population
  ``"action_{k}"`` of size ``2 * action_sizes[k]`` (mean + log_std for a
  Gaussian-tanh head). No activation.
* Optionally (``output_value=True``): a per-module value-head output
  population ``"value_{k}"`` of size 1.

This is the same compositional shape as the user-authored pseudocode
in ``pseudocode_notes_refactor.txt``.
"""

from collections.abc import Mapping, Callable
from typing import Optional, Union

from flax import nnx

from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Sequential, Splitter
from nnx_ppo.networks.factories import make_mlp_layers
from nnx_ppo.networks.graph import PopulationGraph
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.types import StatefulModule
from nnx_ppo.networks.utils import Filter, Flattener, Map


#NON_END_EFFECTORS = ["arm_L", "arm_R", "leg_L", "leg_R", "torso", "head"]
#END_EFFECTORS = ["hand_L", "hand_R", "foot_L", "foot_R"]
LIMB_PAIRS = [
    ("arm_L", "hand_L"),
    ("arm_R", "hand_R"),
    ("leg_L", "foot_L"),
    ("leg_R", "foot_R"),
]


def _build_nervenet_graph(
    obs_sizes: Mapping[str, int],
    action_sizes: Mapping[str, int],
    activation: Callable,
    hidden_size: int,
    root_size: int,
    output_value: bool,
    rngs: nnx.Rngs,
) -> PopulationGraph:
    body_modules = list(obs_sizes.keys())
    pg = PopulationGraph(rngs)

    # Hidden-state populations + their input-from-obs feeders.
    for pop in body_modules:
        size = root_size if pop == "root" else hidden_size
        pg.add_population(pop, size=size, activation=activation)
        pg.add_input(f"in_{pop}", size=int(obs_sizes[pop]), input_from=pop)
        pg.connect(f"in_{pop}", pop)

    # Per-module action-params outputs (size = 2 * action_size, linear).
    for pop, a in action_sizes.items():
        pg.add_output(f"action_{pop}", size=2 * int(a))
        pg.connect(pop, f"action_{pop}")

    # Per-module value-head outputs.
    if output_value:
        for pop in body_modules:
            pg.add_output(f"value_{pop}", size=1)
            pg.connect(pop, f"value_{pop}")

    # Reciprocal delay=1 connections between body modules. The delay
    # breaks the otherwise-illegal cycle limb ↔ root in the topo sort.
    for limb, ee in LIMB_PAIRS:
        pg.connect(ee, limb, delay=1, reciprocal=True)
        pg.connect(limb, "root", delay=1, reciprocal=True)
    for k in ("torso", "head"):
        pg.connect(k, "root", delay=1, reciprocal=True)

    pg.finalize()
    return pg


def NerveNetNetwork_v4(
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
) -> Sequential:
    """Build the NerveNet-v4 :class:`Sequential` network.

    Returns a :class:`Sequential` whose forward output is a
    :class:`PPONetworkOutput` keyed by body-module name (matching the
    env's per-key actions and rewards).

    Note: ``motor_scale`` and ``critic_scale`` are kept as constructor
    arguments for parity with earlier configs but are currently
    ignored (the graph's per-population output heads use the default
    Dense initializer; scaling can be added via a :class:`Scale` layer
    on the adapter's action / value port if needed).
    """
    del motor_scale, critic_scale  # currently unused; see docstring.

    if isinstance(activation, str):
        activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
            activation
        ]
    preprocessing: list[StatefulModule] = [Flattener(preserve_levels=1)]
    if normalize_obs:
        preprocessing.append(Normalizer(dict(obs_sizes)))

    body_modules = list(obs_sizes.keys())
    action_keys = list(action_sizes.keys())

    # Graph emits per-module action-params (and per-module values if attached).
    graph = _build_nervenet_graph(
        obs_sizes=obs_sizes,
        action_sizes=action_sizes,
        activation=activation,
        hidden_size=hidden_size,
        root_size=root_size,
        output_value=not detached_critic,
        rngs=rngs,
    )



    # Action port: per-module Gaussian-tanh sampler on the graph's
    # ``action_{pop}`` outputs. ``Map`` is per-key dispatch and drops
    # any extra keys (the ``value_*`` entries in the attached-critic
    # case) for free.
    action_port: StatefulModule = Sequential([
        Filter({pop: f"action_{pop}" for pop in action_keys}),
        Map({
            pop: NormalTanhSampler(rngs, entropy_weight, min_std)
            for pop in action_keys
        }),
    ])

    # Value port. Attached: pick the per-module ``value_{pop}``
    # outputs that the graph emits. Detached: run a flat-MLP critic on
    # the (normalised) observation and split its output per module.
    value_port: StatefulModule
    if detached_critic:
        hidden_layers = detached_critic_hidden_sizes or [512, 512]
        total_obs_size = sum(int(s) for s in obs_sizes.values())
        critic_sizes = [total_obs_size, *hidden_layers, len(body_modules)]
        critic = Sequential([
            Flattener(),
            *make_mlp_layers(
                critic_sizes, rngs, activation, activation_last_layer=False
            ),
            Splitter(**{pop: 1 for pop in body_modules}),
        ])
        actor = Sequential([graph, action_port])
        return Sequential([
            *preprocessing,
            PPOAdapter(action=actor, value=critic)
        ])
    else:
        value_port = Filter({pop: f"value_{pop}" for pop in body_modules})
        return Sequential([
            *preprocessing,
            graph,
            PPOAdapter(action=action_port, value=value_port)
        ])
