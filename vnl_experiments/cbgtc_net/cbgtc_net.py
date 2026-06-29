"""Cortico-basal ganglia-thalamo-cortical (CBGTC) network.

A brain-region-structured policy built on :class:`PopulationGraph`. Each
population is a brain region; connections carry a one-step delay, making the
whole graph a recurrent discrete-time dynamical system (one synaptic hop per
step). The basal-ganglia loop follows the canonical circuit: cortico-striatal
projections, the direct (D1->GPi) and indirect (D2->GPe->STN->GPi) pathways,
the hyperdirect (cortex->STN) pathway, GPi inhibition of thalamus, and the
GPi->superior colliculus->brainstem descending route.

Two external inputs:

* ``proprioception`` enters at the spinal cord (bottom-up body state).
* the imitation reference target (``task_obs``) enters at the thalamus as a
  top-down goal; the cortico-BG-thalamic loop selects actions that move the
  body toward it.

Built for the :class:`AbsoluteImitation` env, whose observation tree is
``{state: {task_obs, proprioception}}``. The preprocessing flattens each leaf,
normalises, and lifts to a flat ``{task_obs, proprioception}`` dict that the
graph's ``add_input`` reads directly.
"""

from typing import Callable

import jax
from flax import nnx

from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Sequential
from nnx_ppo.networks.factories import make_mlp_layers
from nnx_ppo.networks.graph import PopulationGraph
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.utils import Filter, Flattener


def build_cbgtc_net(
    obs_size,
    action_size: int,
    base_size: int,
    critic_hidden_sizes: list[int],
    activation: Callable,
    entropy_weight: float,
    rngs: nnx.Rngs,
    *,
    min_std: float = 0.1,
    std_scale: float = 1.0,
) -> Sequential:
    """Build the CBGTC network as a :class:`Sequential` ending in a
    :class:`PPOAdapter`.

    Args:
        obs_size: The env's ``non_flattened_observation_size`` tree, shaped
            ``{state: {task_obs, proprioception}}``.
        action_size: Number of action dimensions.
        base_size: Base population width; individual regions scale from this.
        critic_hidden_sizes: Hidden widths for the flat-MLP critic.
        activation: Population transfer function.
        entropy_weight: Entropy bonus weight for the action sampler.
        rngs: NNX rng container (pass explicitly for reproducible seeding).
        min_std: Minimum action std.
        std_scale: Action std scaling.
    """
    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(
        sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0])
    )

    p = PopulationGraph(rngs)
    p.add_population("spinal_cord", size=base_size, activation=activation)
    p.add_population("brainstem", size=base_size, activation=activation)
    p.add_population("thalamus", size=base_size, activation=activation)
    p.add_population("motor_cortex", size=2 * base_size, activation=activation)
    p.add_population("striatum_d1", size=base_size, activation=activation)
    p.add_population("striatum_d2", size=base_size, activation=activation)
    p.add_population("GPe", size=base_size // 8, activation=activation)
    p.add_population("GPi", size=base_size // 8, activation=activation)
    p.add_population("STN", size=base_size // 8, activation=activation)
    p.add_population(
        "superior_colliculus", size=base_size // 2, activation=activation
    )

    # Bottom-up body state into the spinal cord; top-down reference goal into
    # the thalamus. Output is mean+std for the Gaussian-tanh head (2*action).
    p.add_input("in_proprioception", size=proprio_size, input_from="proprioception")
    p.add_input("in_reference", size=task_obs_size, input_from="task_obs")
    p.add_output("out", 2 * action_size)

    p.connect("in_proprioception", "spinal_cord", delay=1)
    p.connect("in_reference", "thalamus", delay=1)
    p.connect("spinal_cord", "out", delay=1)

    p.connect("spinal_cord", "brainstem", delay=1, reciprocal=True)
    p.connect("brainstem", "brainstem", delay=1)
    p.connect("brainstem", "thalamus", delay=1, reciprocal=True)
    p.connect("thalamus", "motor_cortex", delay=1, reciprocal=True)
    p.connect("motor_cortex", "motor_cortex", delay=1)
    p.connect("motor_cortex", "striatum_d1", delay=1)
    p.connect("motor_cortex", "striatum_d2", delay=1)
    p.connect("thalamus", "striatum_d1", delay=1)
    p.connect("thalamus", "striatum_d2", delay=1)
    p.connect("striatum_d1", "GPi", delay=1)
    p.connect("striatum_d2", "GPe", delay=1)
    p.connect("GPe", "STN", delay=1)
    p.connect("GPe", "GPi", delay=1)
    p.connect("STN", "GPi", delay=1)
    p.connect("motor_cortex", "STN", delay=1)
    p.connect("GPi", "thalamus", delay=1)
    p.connect("GPi", "superior_colliculus", delay=1)
    p.connect("superior_colliculus", "brainstem", delay=1)

    p.finalize()

    sampler = NormalTanhSampler(
        rngs, entropy_weight=entropy_weight, min_std=min_std, std_scale=std_scale
    )
    # Graph emits {"out": [B, 2*action]}; Flattener unwraps the single leaf to
    # the bare [B, 2*action] array the sampler expects.
    action_port = Sequential([p, Flattener(), sampler])

    # Privileged critic: flat MLP over the full (normalised) obs.
    critic = Sequential(
        [
            Flattener(),
            *make_mlp_layers(
                [task_obs_size + proprio_size] + list(critic_hidden_sizes) + [1],
                rngs,
                activation,
                activation_last_layer=False,
            ),
        ]
    )

    adapter = PPOAdapter(action=action_port, value=critic)

    # Env wraps obs as {state: {task_obs, proprioception}}. Flatten each inner
    # leaf to 1D (preserve_levels=2 keeps state.<key>), normalise per inner key,
    # then lift to a flat {task_obs, proprioception} dict for the graph/critic.
    pre = Flattener(preserve_levels=2)
    normalizer_shape = {
        "state": {"task_obs": task_obs_size, "proprioception": proprio_size}
    }
    lift = Filter(
        {
            "task_obs": ("state", "task_obs"),
            "proprioception": ("state", "proprioception"),
        }
    )
    return Sequential([pre, Normalizer(normalizer_shape), lift, adapter])
