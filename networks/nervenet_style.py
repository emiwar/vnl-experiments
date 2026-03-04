from collections.abc import Mapping
from typing import Optional

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.containers import Flattener
from nnx_ppo.networks.normalizer import Normalizer

class NerveNetNetwork(PPONetwork):

    def __init__(self, obs_sizes: Mapping[str, int], action_sizes: Mapping[str, int],
                 hidden_size: int, rngs: nnx.Rngs, entropy_weight: float,
                 min_std: float, normalize_obs: bool = True):
        all_modules = obs_sizes.keys()
        self.input_layers = nnx.Dict({k: Dense(os, hidden_size, rngs, nnx.swish) for k,os in obs_sizes.items()})
        self.afferents = nnx.Dict({k: Dense(hidden_size, hidden_size, rngs, nnx.swish) for k in all_modules})
        self.efferents = nnx.Dict({k: Dense(hidden_size, hidden_size, rngs, nnx.swish) for k in all_modules})
        self.motor_layers = nnx.Dict({k: Dense(hidden_size, 2*a, rngs) for k,a in action_sizes.items()})
        self.critics = nnx.Dict({k: Dense(hidden_size, 1, rngs) for k in all_modules})
        self.action_samplers = nnx.Dict({k: NormalTanhSampler(rngs, entropy_weight, min_std) for k in action_sizes.keys()})
        self.normalizer = Normalizer(obs_sizes) if normalize_obs else None

    def __call__(self,
                 network_state: tuple[()],
                 obs: Mapping[str, jax.Array],
                 raw_action: Optional[Mapping[str, jax.Array]] = None) -> tuple[tuple[()], PPONetworkOutput]:
        assert obs.keys() == self.input_layers.keys()

        x = obs

        #Flatten
        flattener = Flattener()
        x = {k: flattener((), xx).output for k,xx in x.items()}

        if self.normalizer is not None:
            x = self.normalizer((), x).output
        
        #Input pass
        x = {k: self.input_layers[k]((), xx).output for k,xx in x.items()}

        #Afferents
        x["arm_L"] += self.afferents["hand_L"]((), x["hand_L"]).output
        x["arm_R"] += self.afferents["hand_R"]((), x["hand_R"]).output
        x["leg_L"] += self.afferents["foot_L"]((), x["foot_L"]).output
        x["leg_R"] += self.afferents["foot_R"]((), x["foot_R"]).output

        x["root"] += self.afferents["arm_L"]((), x["arm_L"]).output\
            + self.afferents["arm_R"]((), x["arm_R"]).output\
            + self.afferents["leg_L"]((), x["leg_L"]).output\
            + self.afferents["leg_R"]((), x["leg_R"]).output\
            + self.afferents["torso"]((), x["torso"]).output\
            + self.afferents["head"]((), x["head"]).output
        
        #Efferents
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
        motor = {k: ml((), x[k]).output for k,ml in self.motor_layers.items()}
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
        # Sum loglikelihoods across all action modules → joint scalar loglikelihood, root excluded
        loglikelihoods = sum(loglikelihoods.values())

        #Critic
        value_estimates = {k: jp.squeeze(self.critics[k]((), xx).output, axis=-1) for k,xx in x.items()}

        return (), PPONetworkOutput(
            actions=actions,
            raw_actions=new_raw_actions,
            loglikelihoods=loglikelihoods,
            regularization_loss=regularization_loss,
            value_estimates=value_estimates,
            metrics=metrics,
        )

    def initialize_state(self, batch_size) -> tuple[()]:
        return ()
    
    def update_statistics(self, last_rollout, total_steps) -> None:
        flattener = Flattener()
        last_rollout = last_rollout.replace(
            obs = {k: jax.vmap(lambda o: flattener((), o).output)(o) for k,o in last_rollout.obs.items()}
        )
        self.normalizer.update_statistics(last_rollout, total_steps)
