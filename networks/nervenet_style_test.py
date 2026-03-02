import jax
import jax.numpy as jp
from flax import nnx


from networks import nervenet_style
from vnl_playground.tasks.modular_rodent.imitation import ModularImitation, default_config

SEED = 1

base_env = ModularImitation()

rngs = nnx.Rngs(SEED)
obs_sizes = {k: jp.squeeze(jax.tree.reduce(jp.add, o)) for k,o in base_env.non_flattened_observation_size.items()}
net = nervenet_style.NerveNetNetwork(obs_sizes,
                                     base_env.action_size,
                                     32, rngs, 0.01, 0.1)

env_state = jax.vmap(base_env.reset)(rngs().reshape(1))

_, net_output = net((), env_state.obs)

env_state2 = base_env.step(env_state, net_output.actions)
