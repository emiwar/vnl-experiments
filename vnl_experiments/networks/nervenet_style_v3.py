from collections.abc import Mapping, Callable
from typing import Optional, Union

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.containers import Sequential, Flattener
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.factories import make_mlp_layers

NON_END_EFFECTORS = ["arm_L", "arm_R", "leg_L", "leg_R", "torso", "head"]
END_EFFECTORS = ["hand_L", "hand_R", "foot_L", "foot_R"]

class NerveNetNetwork_v3(PPONetwork):

    def __init__(self,
                 obs_sizes: Mapping[str, int],
                 action_sizes: Mapping[str, int],
                 hidden_size: int,
                 root_size: int,
                 rngs: nnx.Rngs,
                 entropy_weight: float = 0.01,
                 min_std: float = 0.1,
                 motor_scale: float = 1.0,
                 normalize_obs: bool = True,
                 activation: Union[str, Callable] = nnx.swish,
                 critic_scale: float = 1.0,
                 combine_likelihoods: bool = True,
                 detached_critic: bool = False,
                 detached_critic_hidden_sizes: list[int] = [512, 512]):
        if isinstance(activation, str):
            activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[activation]
        all_modules = obs_sizes.keys()

        self.input_layers = nnx.Dict({k: Dense(os, hidden_size, rngs, activation) for k,os in obs_sizes.items()})
        self.input_layers["root"] = Dense(obs_sizes["root"], root_size, rngs, activation)
        self.afferents = nnx.Dict()
        self.efferents = nnx.Dict()
        for k in NON_END_EFFECTORS:
            self.afferents[k] = Dense(hidden_size, root_size, rngs, activation)
            self.efferents[k] = Dense(root_size, hidden_size, rngs, activation) 
        for k in END_EFFECTORS:
            self.afferents[k] = Dense(hidden_size, hidden_size, rngs, activation)
            self.efferents[k] = Dense(hidden_size, hidden_size, rngs, activation)
        self.motor_layers = nnx.Dict({k: Dense(hidden_size, 2*a, rngs, kernel_init=nnx.initializers.zeros) for k,a in action_sizes.items()})
        self.detached_critic = detached_critic
        if detached_critic:
            total_obs_size = jax.tree.reduce(jp.add, obs_sizes)
            critic_sizes = [total_obs_size, *detached_critic_hidden_sizes, len(all_modules)]
            self.critic = Sequential(Flattener(), *make_mlp_layers(critic_sizes, rngs, activation, activation_last_layer=False))
        else:
            self.critics = nnx.Dict({k: Dense(hidden_size, 1, rngs, kernel_init=nnx.initializers.zeros) for k in all_modules})
            self.critics["root"] = Dense(root_size, 1, rngs, kernel_init=nnx.initializers.zeros)
        self.action_samplers = nnx.Dict({k: NormalTanhSampler(rngs, entropy_weight, min_std) for k in action_sizes.keys()})
        self.motor_scale = motor_scale
        self.critic_scale = critic_scale
        self.normalizer = Normalizer(obs_sizes) if normalize_obs else None
        self.combine_likelihoods = combine_likelihoods

    def __call__(self,
                 network_state: tuple[()],
                 obs: Mapping[str, jax.Array],
                 raw_action: Optional[Mapping[str, jax.Array]] = None) -> tuple[tuple[()], PPONetworkOutput]:
        assert obs.keys() == self.input_layers.keys()

        #Flatten
        flattener = Flattener()
        obs_flat = {k: flattener((), xx).output for k,xx in obs.items()}

        if self.normalizer is not None:
            obs_norm = self.normalizer((), obs_flat).output
        
        #Input pass
        layer_1 = {k: self.input_layers[k]((), xx).output for k,xx in obs_norm.items()}

        x = dict()

        #First pass limb communication
        x["arm_L"]  = layer_1["arm_L"]  + self.afferents["hand_L"]((), layer_1["hand_L"]).output
        x["arm_R"]  = layer_1["arm_R"]  + self.afferents["hand_R"]((), layer_1["hand_R"]).output
        x["hand_L"] = layer_1["hand_L"] + self.efferents["hand_L"]((), layer_1["arm_L"]).output
        x["hand_R"] = layer_1["hand_R"] + self.efferents["hand_R"]((), layer_1["arm_R"]).output

        x["leg_L"]  = layer_1["leg_L"]  + self.afferents["foot_L"]((), layer_1["foot_L"]).output
        x["leg_R"]  = layer_1["leg_R"]  + self.afferents["foot_R"]((), layer_1["foot_R"]).output
        x["foot_L"] = layer_1["foot_L"] + self.efferents["foot_L"]((), layer_1["leg_L"]).output
        x["foot_R"] = layer_1["foot_R"] + self.efferents["foot_R"]((), layer_1["leg_R"]).output

        x["torso"] = layer_1["torso"]
        x["head"] = layer_1["head"]

        avg_afferent = (self.afferents["arm_L"]((), x["arm_L"]).output\
            + self.afferents["arm_R"]((), x["arm_R"]).output\
            + self.afferents["leg_L"]((), x["leg_L"]).output\
            + self.afferents["leg_R"]((), x["leg_R"]).output\
            + self.afferents["torso"]((), x["torso"]).output\
            + self.afferents["head"]((), x["head"]).output)/6.0
        x["root"] = layer_1["root"] + avg_afferent
        
        # Efferents
        # Re-using the end effector efferents from first pass
        x["head"] += self.efferents["head"]((), x["root"]).output
        x["torso"] += self.efferents["torso"]((), x["root"]).output

        x["arm_L"] += self.efferents["arm_L"]((), x["root"]).output
        x["hand_L"] += self.efferents["hand_L"]((), x["arm_L"]).output
        x["arm_R"] += self.efferents["arm_R"]((), x["root"]).output
        x["hand_R"] += self.efferents["hand_R"]((), x["arm_R"]).output

        x["leg_L"] += self.efferents["leg_L"]((), x["root"]).output
        x["foot_L"] += self.efferents["foot_L"]((), x["leg_L"]).output
        x["leg_R"] += self.efferents["leg_R"]((), x["root"]).output
        x["foot_R"] += self.efferents["foot_R"]((), x["leg_R"]).output

        #Motor
        motor = {k: ml((), x[k]).output * self.motor_scale for k,ml in self.motor_layers.items()}
        actions = {}
        new_raw_actions = {}
        loglikelihoods = {}
        regularization_loss = jp.array(0.0)
        metrics = {}
        for k in self.action_samplers.keys():
            output = self.action_samplers[k]((), motor[k], raw_action[k] if raw_action is not None else None)
            actions[k], new_raw_actions[k], loglikelihoods[k] = output.output
            regularization_loss += output.regularization_loss
            metrics[k] = output.metrics

        if self.combine_likelihoods:
            # Sum loglikelihoods across all action modules → joint scalar loglikelihood, root excluded
            loglikelihoods = sum(loglikelihoods.values())

        #Critic
        if self.detached_critic:
            critic_output = self.critic(network_state["critic"], obs_norm)
            network_state["critic"] = critic_output.next_state
            value_estimates = {k: critic_output.output[..., i] for i,k in enumerate(obs_norm.keys())}
        else:
            value_estimates = {k: jp.squeeze(self.critics[k]((), xx).output*self.critic_scale, axis=-1) for k,xx in x.items()}

        return network_state, PPONetworkOutput(
            actions=actions,
            raw_actions=new_raw_actions,
            loglikelihoods=loglikelihoods,
            regularization_loss=regularization_loss,
            value_estimates=value_estimates,
            metrics=metrics,
        )

    def initialize_state(self, batch_size) -> tuple[()]:
        if self.detached_critic:
            return {"critic": self.critic.initialize_state(batch_size)}
        else:
            return ()
    
    def update_statistics(self, last_rollout, total_steps) -> None:
        flattener = Flattener()
        last_rollout = last_rollout.replace(
            obs = {k: jax.vmap(lambda o: flattener((), o).output)(o) for k,o in last_rollout.obs.items()}
        )
        if self.normalizer is not None:
            self.normalizer.update_statistics(last_rollout, total_steps)
