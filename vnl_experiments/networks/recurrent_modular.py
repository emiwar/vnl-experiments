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
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers

NON_END_EFFECTORS = ["arm_L", "arm_R", "leg_L", "leg_R", "torso", "head"]
END_EFFECTORS = ["hand_L", "hand_R", "foot_L", "foot_R"]

class RecurrentModularNetwork(PPONetwork):

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
                 min_psi: float = 0.5,
                 max_psi: float = 0.95,
                 detached_critic: bool = True,
                 detached_critic_hidden_sizes: list[int] = [512, 512],
                 reveal_targets: str = "all"):
        if isinstance(activation, str):
            activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[activation]
        module_size = {k: hidden_size for k in obs_sizes.keys()}
        module_size["root"] = root_size
        self.raw_psi = nnx.Dict({k: nnx.Param(rngs.normal(s)) for k,s in module_size.items()})
        self.min_psi = min_psi
        self.max_psi = max_psi
        self.input_layers = nnx.Dict({k: make_mlp([os, module_size[k], module_size[k]], rngs, activation, False) for k,os in obs_sizes.items()})
        self.afferents = nnx.Dict()
        self.efferents = nnx.Dict()
        for k in NON_END_EFFECTORS:
            self.afferents[k] = nnx.Linear(hidden_size, root_size, rngs=rngs)
            self.efferents[k] = nnx.Linear(root_size, hidden_size, rngs=rngs) 
        for k in END_EFFECTORS:
            self.afferents[k] = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
            self.efferents[k] = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.motor_layers = nnx.Dict({k: Dense(hidden_size, 2*a, rngs, kernel_init=nnx.initializers.zeros) for k,a in action_sizes.items()})
        self.detached_critic = detached_critic
        if detached_critic:
            total_obs_size = jax.tree.reduce(jp.add, obs_sizes)
            critic_sizes = [total_obs_size, *detached_critic_hidden_sizes, len(module_size)]
            self.critic = Sequential([Flattener(), *make_mlp_layers(critic_sizes, rngs, activation, activation_last_layer=False)])
        else:
            self.critics = nnx.Dict({k: Dense(s, 1, rngs, kernel_init=nnx.initializers.zeros) for k,s in module_size.items()})
        self.action_samplers = nnx.Dict({k: NormalTanhSampler(rngs, entropy_weight, min_std) for k in action_sizes.keys()})
        self.motor_scale = motor_scale
        self.critic_scale = critic_scale
        self.normalizer = Normalizer(obs_sizes) if normalize_obs else None
        self.combine_likelihoods = combine_likelihoods
        self.activation = activation
        self.reveal_targets = reveal_targets

    def __call__(self,
                 network_state: tuple[()],
                 obs: Mapping[str, jax.Array],
                 raw_action: Optional[Mapping[str, jax.Array]] = None) -> tuple[tuple[()], PPONetworkOutput]:
        obs = self._filter_obs(obs)
        assert obs.keys() == self.input_layers.keys()

        #Flatten
        flattener = Flattener()
        obs_flat = {k: flattener((), xx).output for k,xx in obs.items()}

        if self.normalizer is not None:
            obs_norm = self.normalizer((), obs_flat).output
        
        #Input pass
        layer_1 = {k: self.input_layers[k]([(), ()], xx).output for k,xx in obs_norm.items()}

        module_input = dict()
        h = {k: self.activation(network_state[k]) for k in layer_1.keys()}
        module_input["hand_L"] = layer_1["hand_L"] + self.efferents["hand_L"](h["arm_L"])
        module_input["hand_R"] = layer_1["hand_R"] + self.efferents["hand_R"](h["arm_R"])
        module_input["arm_L"] = layer_1["arm_L"] + self.efferents["arm_L"](h["root"]) + self.afferents["hand_L"](h["hand_L"])
        module_input["arm_R"] = layer_1["arm_R"] + self.efferents["arm_R"](h["root"]) + self.afferents["hand_R"](h["hand_R"])

        module_input["foot_L"] = layer_1["foot_L"] + self.efferents["foot_L"](h["leg_L"])
        module_input["foot_R"] = layer_1["foot_R"] + self.efferents["foot_R"](h["leg_R"])
        module_input["leg_L"] = layer_1["leg_L"] + self.efferents["leg_L"](h["root"]) + self.afferents["foot_L"](h["foot_L"])
        module_input["leg_R"] = layer_1["leg_R"] + self.efferents["leg_R"](h["root"]) + self.afferents["foot_R"](h["foot_R"])
        
        module_input["head"] = layer_1["head"] + self.efferents["head"](h["root"])
        module_input["torso"] = layer_1["torso"] + self.efferents["torso"](h["root"])
        module_input["root"] = layer_1["root"]
        module_input["root"] += sum(self.afferents[k](h[k]) for k in NON_END_EFFECTORS)/len(NON_END_EFFECTORS)

        psi = {k: self._psi(k) for k in module_input.keys()}
        new_state = {k: psi[k] * network_state[k] + (1-psi[k])*mi for k,mi in module_input.items()}

        #Motor
        motor = {k: ml((), self.activation(new_state[k])).output*self.motor_scale for k,ml in self.motor_layers.items()}
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
            new_state["critic"] = critic_output.next_state
            value_estimates = {k: critic_output.output[..., i] for i,k in enumerate(obs_norm.keys())}
        else:
            value_estimates = {k: jp.squeeze(self.critics[k]((), self.activation(xx)).output*self.critic_scale, axis=-1) for k,xx in new_state.items()}

        return new_state, PPONetworkOutput(
            actions=actions,
            raw_actions=new_raw_actions,
            loglikelihoods=loglikelihoods,
            regularization_loss=regularization_loss,
            value_estimates=value_estimates,
            metrics=metrics,
        )

    def initialize_state(self, batch_size: int) -> tuple[()]:
        state = {k: jp.zeros((batch_size, *rp.shape)) for k,rp in self.raw_psi.items()}
        if self.detached_critic:
            state["critic"] = self.critic.initialize_state(batch_size)
        return state
    
    def reset_state(self, prev_state):
        new_state = dict()
        for k in self.raw_psi.keys():
            new_state[k] = jp.zeros_like(prev_state[k])
        if self.detached_critic:
            new_state["critic"] = self.critic.reset_state(prev_state["critic"])
        return new_state
 
    def update_statistics(self, last_rollout, total_steps) -> None:
        obs = self._filter_obs(last_rollout.obs)
        flattener = Flattener()
        last_rollout = last_rollout.replace(
            obs = {k: jax.vmap(lambda o: flattener((), o).output)(o) for k,o in obs.items()}
        )
        if self.normalizer is not None:
            self.normalizer.update_statistics(last_rollout, total_steps)

    def _psi(self, k):
        raw_psi = self.raw_psi[k].value
        lam = 0.5*(1.0 + nnx.tanh(raw_psi))
        psi = (1 - lam) * self.min_psi + lam * self.max_psi
        return psi

    def _filter_obs(self, obs):
        if self.reveal_targets == "all":
            return obs
        obs = obs.copy()
        for k,o in obs.items():
            if k != "root":
                obs[k] = o["proprioception"]
        if self.reveal_targets == "none":
            batch_dim = obs["root"]["current_target"].shape
            obs["root"] = jp.zeros((batch_dim, 0))
            return obs
        elif self.reveal_targets == "root_only":
            return obs
        elif self.reveal_targets == "joystick_only":
            obs["root"] = obs["root"]["future_target"]["pos"]
            return obs
