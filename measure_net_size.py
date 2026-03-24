"""Rodent imitation learning with modular architecture."""

from datetime import datetime
import json

import dataclasses
import jax
import jax.numpy as jp
from flax import nnx
from ml_collections import config_dict

from vnl_playground.tasks.modular_rodent.imitation_v4 import ModularImitation_v4, default_config

from vnl_experiments.networks.nervenet_style_v3 import NerveNetNetwork_v3

net_config = config_dict.create(
    hidden_size=32,
    root_size=32,
    critic_scale=20.0,
    entropy_weight=1e-2,
    min_std=1e-1,
    motor_scale=1.0,
    normalize_obs=True,
    combine_likelihoods=True,
    detached_critic=True,
    detached_critic_hidden_sizes=[512, 512],
    activation="tanh",
)

env = ModularImitation_v4()
# Setup network
rngs = nnx.Rngs(0)

obs_sizes = {k: jp.squeeze(jax.tree.reduce(jp.add, o)) for k,o in env.non_flattened_observation_size.items()}
nets = NerveNetNetwork_v3(
    obs_sizes, env.action_size, rngs=rngs, **net_config
)
tot = sum(jax.tree.leaves(jax.tree.map(lambda x: x.size, nnx.state(nets, nnx.Param))))
critic = sum(jax.tree.leaves(jax.tree.map(lambda x: x.size, nnx.state(nets.critic, nnx.Param))))
print(f"NerveNet #params: {tot-critic}")

mlp_hidden_sizes = [64, 64]
obs_size = env.observation_size
action_size = jax.tree.reduce(jp.add, env.action_size)
mlp_size = (obs_size+1) * mlp_hidden_sizes[0]
for i in range(len(mlp_hidden_sizes)-1):
    mlp_size += (mlp_hidden_sizes[i]+1) * mlp_hidden_sizes[i+1] 
mlp_size += (mlp_hidden_sizes[-1]+1) * action_size*2
print(f"MLP #params: {mlp_size}")
