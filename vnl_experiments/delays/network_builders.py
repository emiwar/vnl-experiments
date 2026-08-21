"""The architecture registry for the delays experiments: build + restore + defaults.

This module is the **single source of truth** for every network architecture in the
delays family. Both the training scripts (``train_rodent.py``) and the offline eval
scripts (``eval_runs.py``, ``eval_videos.py``, ``record_activations.py``) construct
their networks by calling ``build_network`` here, so an architecture is defined
exactly once and a train/eval drift is structurally impossible.

Adding an architecture means adding one ``Architecture`` entry to
``ARCHITECTURES``: a ``defaults()`` returning its train-time ``net_config``, and a
``build(net_params, env, rngs)`` reconstructing it. Nothing else needs to change —
the rollout loops in ``evaluation.py`` / ``record_activations.py`` /
``eval_videos.py`` thread network carry state generically, so recurrent
architectures work without touching them.

Public API:
    * ``ARCHITECTURES`` / ``get_architecture(network_class)`` — the registry.
    * ``build_network(net_params, env, rngs)`` — dispatch on ``network_class``.
    * ``load_network(ckpt_dir, net_params, env, seed)`` — build + restore latest
      step, returning ``(networks, step)`` or ``None`` when unavailable.
    * ``_parse_net_params(raw)`` — JSON-string → typed net-param dict.
    * ``build_delay_network`` / ``build_forward_model_network`` /
      ``build_recurrent_network`` — the individual builders.

The builders are env-agnostic: they only query ``env.non_flattened_observation_size``
and ``env.action_size``, so the same builders serve both the ``Imitation`` and
``AbsoluteImitation`` environments.

**Back-compat rule.** The ``p.get(key, <default>)`` fallbacks inside each builder
exist for old ``config.json`` files written before a key existed, and their values
are *not* always the same as the corresponding ``defaults()`` entry (e.g.
``latent_size`` 16 vs 32 for the delay net). Never "unify" the two: ``defaults()``
governs new runs, the ``p.get`` fallbacks govern reconstruction of old ones.
Changing a fallback silently re-interprets an existing checkpoint.
"""

import dataclasses
import pickle
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from flax import nnx
from ml_collections import config_dict

from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state
from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Concat, Sequential
from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.recurrent import GRU, LSTM, SimpleRNN
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.utils import Filter, Flattener, Map
from nnx_ppo.networks.variational import VariationalBottleneck

from vnl_experiments.delays.efference_copy import EfferenceCopy
from vnl_experiments.delays.forward_model import ForwardModel

#: Recurrent cell classes selectable via the ``rnn_cell`` net-config key.
RNN_CELLS = {"lstm": LSTM, "gru": GRU, "rnn": SimpleRNN}

_ACTIVATIONS = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}


# ---------------------------------------------------------------------------
# Net-param parsing
# ---------------------------------------------------------------------------

def _parse_net_params(raw: dict) -> dict:
    """Convert JSON string values to proper Python types (matching checkpoint_utils)."""
    result = {}
    for k, v in raw.items():
        if isinstance(v, list):
            result[k] = [int(x) for x in v]
        elif v == "True":
            result[k] = True
        elif v == "False":
            result[k] = False
        elif v == "None":
            result[k] = None
        else:
            try:
                result[k] = int(v)
            except (ValueError, TypeError):
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    result[k] = v
    return result


# ---------------------------------------------------------------------------
# Train-time defaults
#
# One function per architecture, returning the ``net_config`` a fresh run starts
# from. ``train_rodent.py`` takes these and applies any ``--net-config`` overrides.
# These are *not* the same thing as the ``p.get`` fallbacks in the builders below;
# see the back-compat rule in the module docstring.
# ---------------------------------------------------------------------------

def _shared_defaults(**extra):
    """The net-config keys every delays architecture has, plus ``extra``."""
    return config_dict.create(
        enc_hidden_sizes=[512] * 4,
        critic_hidden_sizes=[1024] * 2,
        activation="swish",
        entropy_weight=1e-2,
        min_std=1e-1,
        std_scale=1.0,
        normalize_obs=True,
        initializer_scale=1.0,
        kl_weight=0.001,
        latent_min_std=0.01,
        latent_size=32,
        latent_ar1_weight=None,
        **extra,
    )


def delay_defaults():
    """Defaults for ``RodentEncDecDelays`` (feedforward decoder).

    ``dec_use_intention`` / ``dec_use_proprioception`` are the decoder-input ablation
    switches: set one to False to drop that stream from the decoder's input entirely
    (see ``build_delay_network``). The third stream, the efference copy, is switched
    off by ``efference_length=0`` instead -- that path predates these flags and stays
    as it is so the existing no-efference runs remain comparable.
    """
    return _shared_defaults(
        dec_hidden_sizes=[512] * 4,
        dec_use_intention=True,        # encoder latent -> decoder
        dec_use_proprioception=True,   # (delayed) proprioception -> decoder
    )


def forward_model_defaults():
    """Defaults for ``RodentForwardModel`` (explicit predictor + decoder)."""
    return _shared_defaults(
        dec_hidden_sizes=[512] * 4,
        fm_hidden_sizes=[512] * 4,
        fm_loss_weight=1.0,
        detach_prediction=True,
    )


def recurrent_defaults():
    """Defaults for ``RodentEncDecRecurrent`` (recurrent decoder).

    The decoder is ``pre-MLP -> recurrent stack -> post-MLP -> sampler``.
    ``rnn_hidden_sizes`` gives one width per recurrent layer, so its length is the
    depth; the cells are independent modules with their own carries, so the widths
    need not be uniform (``[512, 256]`` is a valid taper).

    Any of the three lists may be empty. ``dec_pre_hidden_sizes=[]`` feeds the
    concatenated latent/proprioception/efference vector straight into the
    recurrent stack. ``rnn_hidden_sizes=[]`` removes the recurrence altogether and
    is the *reduction* configuration -- see ``build_recurrent_network``.
    """
    return _shared_defaults(
        dec_pre_hidden_sizes=[512] * 2,
        rnn_cell="lstm",                     # one of RNN_CELLS
        rnn_hidden_sizes=[512],              # one width per layer; len = depth
        rnn_trainable_initial_state=False,
        dec_post_hidden_sizes=[512] * 2,
    )


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_delay_network(net_params: dict, env, rngs: nnx.Rngs):
    """Reconstruct the enc-dec delay network from saved net_params.

    The decoder's input is the concatenation of up to three streams: the encoder's
    latent (the "intention"), the delayed proprioception, and the efference copy of
    the in-flight actions. Each can be ablated away: ``dec_use_intention=False`` and
    ``dec_use_proprioception=False`` drop their branch from the ``Concat`` (and the
    corresponding term from the decoder's input width), ``efference_length=0`` makes
    the ``EfferenceCopy`` a pass-through. Both flags default to True, so a
    ``net_params`` dict written before they existed rebuilds exactly as before.

    The critic is untouched by the flags: it keeps seeing undelayed ``task_obs`` and
    ``proprioception`` in every arm, which is what makes the ablations comparable.
    """
    p = _parse_net_params({k: v for k, v in net_params.items()
                           if k != "network_class"})

    obs_size = env.non_flattened_observation_size
    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0]))
    action_size = env.action_size

    delay_k = int(p.get("delay_k", 0))
    efference_length = int(p.get("efference_length", delay_k))

    enc_hidden = list(p.get("enc_hidden_sizes", [512] * 4))
    dec_hidden = list(p.get("dec_hidden_sizes", [512] * 4))
    critic_hidden = list(p.get("critic_hidden_sizes", [1024] * 2))
    latent_size = int(p.get("latent_size", 16))
    kl_weight = float(p.get("kl_weight", 0.01))
    latent_min_std = float(p.get("latent_min_std", 0.01))
    entropy_weight = float(p.get("entropy_weight", 1e-2))
    min_std = float(p.get("min_std", 1e-1))
    std_scale = float(p.get("std_scale", 1.0))
    normalize_obs = bool(p.get("normalize_obs", True))
    initializer_scale = float(p.get("initializer_scale", 1.0))

    use_intention = bool(p.get("dec_use_intention", True))
    use_proprio = bool(p.get("dec_use_proprioception", True))
    if not (use_intention or use_proprio):
        raise ValueError(
            "build_delay_network: the decoder needs at least one of intention / "
            "proprioception (Concat requires at least one component)"
        )

    activation = _ACTIVATIONS[str(p.get("activation", "swish"))]

    # Only override nnx's default initialiser when a non-default scale was asked
    # for. The training script has always used the nnx defaults, so at
    # initializer_scale == 1.0 this reproduces the init a fresh run actually got;
    # passing an explicit variance_scaling unconditionally would silently change
    # the init of every future run (and never matched the trained one).
    kernel_kwargs = {}
    if initializer_scale != 1.0:
        kernel_kwargs["kernel_init"] = nnx.initializers.variance_scaling(
            initializer_scale, "fan_in", "uniform"
        )

    enc_sizes = [task_obs_size] + enc_hidden + [latent_size * 2]
    decoder_in = (
        (latent_size if use_intention else 0)
        + (proprio_size if use_proprio else 0)
        + efference_length * action_size
    )
    dec_sizes = [decoder_in] + dec_hidden + [action_size * 2]
    critic_sizes = [task_obs_size + proprio_size] + critic_hidden + [1]

    # Insertion order fixes the decoder's input layout, so build the dict in the
    # historical order (intention, then proprioception, then the efference queue
    # appended by EfferenceCopy). An ablated branch is not constructed at all, so a
    # no-intention run has no encoder and hence no KL term in its loss.
    components = {}
    if use_intention:
        components["task_obs"] = Sequential([
            Flattener(),
            *make_mlp_layers(enc_sizes, rngs, activation,
                             activation_last_layer=False, **kernel_kwargs),
            VariationalBottleneck(latent_size, rngs, kl_weight, latent_min_std),
        ])

    if use_proprio:
        proprio_branch_layers = [Flattener()]
        if delay_k > 0:
            proprio_branch_layers.append(Delay(jp.zeros(proprio_size), k_steps=delay_k))
        components["proprioception"] = Sequential(proprio_branch_layers)

    decoder = Sequential([
        *make_mlp_layers(dec_sizes, rngs, activation,
                         activation_last_layer=False, **kernel_kwargs),
        NormalTanhSampler(rngs, entropy_weight=entropy_weight,
                          min_std=min_std, std_scale=std_scale),
    ])

    actor = Sequential([
        Concat(components),
        EfferenceCopy(inner=decoder, sample_action=jp.zeros(action_size),
                      queue_length=efference_length),
    ])

    critic = Sequential([
        Flattener(),
        *make_mlp_layers(critic_sizes, rngs, activation,
                         activation_last_layer=False, **kernel_kwargs),
    ])

    adapter = PPOAdapter(action=actor, value=critic)

    pre = Flattener(preserve_levels=2)
    lift = Filter({
        "task_obs": ("state", "task_obs"),
        "proprioception": ("state", "proprioception"),
    })

    if normalize_obs:
        normalizer_shape = {
            "state": {
                "task_obs": task_obs_size,
                "proprioception": proprio_size,
            }
        }
        return Sequential([pre, Normalizer(normalizer_shape), lift, adapter])
    return Sequential([pre, lift, adapter])


def build_forward_model_network(net_params: dict, env, rngs: nnx.Rngs):
    """Reconstruct the explicit-forward-model network (mirrors the train script)."""
    p = _parse_net_params({k: v for k, v in net_params.items() if k != "network_class"})

    obs_size = env.non_flattened_observation_size
    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0]))
    action_size = env.action_size

    delay_k = int(p.get("delay_k", 0))
    efference_length = int(p.get("efference_length", delay_k))
    fm_loss_weight = float(p.get("fm_loss_weight", 1.0))

    enc_hidden = list(p.get("enc_hidden_sizes", [512] * 4))
    dec_hidden = list(p.get("dec_hidden_sizes", [512] * 4))
    fm_hidden = list(p.get("fm_hidden_sizes", [512] * 4))
    critic_hidden = list(p.get("critic_hidden_sizes", [1024] * 2))
    latent_size = int(p.get("latent_size", 32))
    kl_weight = float(p.get("kl_weight", 0.001))
    latent_min_std = float(p.get("latent_min_std", 0.01))
    entropy_weight = float(p.get("entropy_weight", 1e-2))
    min_std = float(p.get("min_std", 1e-1))
    std_scale = float(p.get("std_scale", 1.0))
    normalize_obs = bool(p.get("normalize_obs", True))
    # Only affects gradient flow, so eval output is identical either way -- but the
    # train script sets it from --detach-prediction, so the builder must honour it
    # or a --no-detach-prediction run would train the wrong architecture.
    detach_prediction = bool(p.get("detach_prediction", True))
    activation = _ACTIVATIONS[str(p.get("activation", "swish"))]

    enc_sizes = [task_obs_size] + enc_hidden + [latent_size * 2]
    decoder_in = latent_size + proprio_size
    dec_sizes = [decoder_in] + dec_hidden + [action_size * 2]
    predictor_sizes = (
        [proprio_size + efference_length * action_size] + fm_hidden + [proprio_size]
    )
    critic_sizes = [task_obs_size + proprio_size] + critic_hidden + [1]

    encoder_branch = Sequential([
        Flattener(),
        *make_mlp_layers(enc_sizes, rngs, activation, activation_last_layer=False),
        VariationalBottleneck(latent_size, rngs, kl_weight, latent_min_std),
    ])
    decoder = Sequential([
        *make_mlp_layers(dec_sizes, rngs, activation, activation_last_layer=False),
        NormalTanhSampler(rngs, entropy_weight=entropy_weight,
                          min_std=min_std, std_scale=std_scale),
    ])
    predictor = make_mlp(predictor_sizes, rngs, activation, activation_last_layer=False)

    actor = Sequential([
        Map(task_obs=encoder_branch, proprioception=Flattener()),
        EfferenceCopy(
            inner=ForwardModel(
                decoder=decoder, predictor=predictor, proprio_size=proprio_size,
                delay_steps=delay_k, loss_weight=fm_loss_weight,
                detach_prediction=detach_prediction,
            ),
            sample_action=jp.zeros(action_size),
            queue_length=efference_length,
            inject_key="efference",
        ),
    ])
    critic = Sequential([
        Flattener(),
        *make_mlp_layers(critic_sizes, rngs, activation, activation_last_layer=False),
    ])
    adapter = PPOAdapter(action=actor, value=critic)

    pre = Flattener(preserve_levels=2)
    lift = Filter({
        "task_obs": ("state", "task_obs"),
        "proprioception": ("state", "proprioception"),
    })
    if normalize_obs:
        normalizer_shape = {"state": {
            "task_obs": task_obs_size, "proprioception": proprio_size}}
        return Sequential([pre, Normalizer(normalizer_shape), lift, adapter])
    return Sequential([pre, lift, adapter])


def build_recurrent_network(net_params: dict, env, rngs: nnx.Rngs):
    """Reconstruct the enc-dec network with a *recurrent* decoder.

    Identical to ``build_delay_network`` except for the decoder, which is

        pre-MLP -> [<cell> per entry of rnn_hidden_sizes] -> post-MLP -> sampler

    where ``<cell>`` is chosen by the ``rnn_cell`` net-param (see ``RNN_CELLS``).
    The encoder, the ``Delay``-ed proprioception branch, the ``EfferenceCopy``
    queue and the feedforward privileged critic are all unchanged, so a recurrent
    run differs from its feedforward twin in exactly one factor. Setting
    ``efference_length`` to 0 turns the efference copy into a pass-through, which
    is the "can recurrence replace an explicit efference copy" condition.

    ``rnn_hidden_sizes=[]`` is deliberately legal: it strips the recurrence and
    leaves ``pre-MLP -> post-MLP -> sampler``. Because the pre-MLP activates its
    last layer and the post-MLP does not, that collapses to exactly the
    ``RodentEncDecDelays`` decoder when
    ``dec_pre_hidden_sizes + dec_post_hidden_sizes == dec_hidden_sizes``, which
    makes it a reduction test for this whole code path (asserted in
    ``network_builders_test.py``). It is a debugging configuration, not a
    condition to train: a run with no recurrent layers still records
    ``network_class = "RodentEncDecRecurrent"``, so filter analyses on
    ``net_params.rnn_hidden_sizes`` rather than trusting the class name.
    """
    p = _parse_net_params({k: v for k, v in net_params.items()
                           if k != "network_class"})

    obs_size = env.non_flattened_observation_size
    task_obs_size = int(sum(jax.tree.flatten(obs_size["state"]["task_obs"])[0]))
    proprio_size = int(sum(jax.tree.flatten(obs_size["state"]["proprioception"])[0]))
    action_size = env.action_size

    delay_k = int(p.get("delay_k", 0))
    efference_length = int(p.get("efference_length", delay_k))

    enc_hidden = list(p.get("enc_hidden_sizes", [512] * 4))
    critic_hidden = list(p.get("critic_hidden_sizes", [1024] * 2))
    pre_hidden = list(p.get("dec_pre_hidden_sizes", [512] * 2))
    post_hidden = list(p.get("dec_post_hidden_sizes", [512] * 2))
    rnn_hidden_sizes = list(p.get("rnn_hidden_sizes", [512]))
    rnn_trainable_initial_state = bool(p.get("rnn_trainable_initial_state", False))
    latent_size = int(p.get("latent_size", 32))
    kl_weight = float(p.get("kl_weight", 0.001))
    latent_min_std = float(p.get("latent_min_std", 0.01))
    entropy_weight = float(p.get("entropy_weight", 1e-2))
    min_std = float(p.get("min_std", 1e-1))
    std_scale = float(p.get("std_scale", 1.0))
    normalize_obs = bool(p.get("normalize_obs", True))
    activation = _ACTIVATIONS[str(p.get("activation", "swish"))]

    cell_name = str(p.get("rnn_cell", "lstm")).lower()
    if cell_name not in RNN_CELLS:
        raise ValueError(
            f"Unknown rnn_cell {cell_name!r}; expected one of {sorted(RNN_CELLS)}."
        )
    cell_cls = RNN_CELLS[cell_name]

    enc_sizes = [task_obs_size] + enc_hidden + [latent_size * 2]
    decoder_in = latent_size + proprio_size + efference_length * action_size
    critic_sizes = [task_obs_size + proprio_size] + critic_hidden + [1]

    encoder_branch = Sequential([
        Flattener(),
        *make_mlp_layers(enc_sizes, rngs, activation, activation_last_layer=False),
        VariationalBottleneck(latent_size, rngs, kl_weight, latent_min_std),
    ])

    proprio_branch_layers = [Flattener()]
    if delay_k > 0:
        proprio_branch_layers.append(Delay(jp.zeros(proprio_size), k_steps=delay_k))
    proprio_branch = Sequential(proprio_branch_layers)

    # Pre-MLP projects into the recurrent stack; activation_last_layer=True so the
    # cell sees a nonlinear embedding rather than a bare affine map.
    decoder_layers = list(
        make_mlp_layers([decoder_in] + pre_hidden, rngs, activation,
                        activation_last_layer=True)
    )
    # Each cell is an independent module with its own carry, so consecutive
    # widths need not match: layer i maps its predecessor's width to its own.
    width = pre_hidden[-1] if pre_hidden else decoder_in
    for hidden in rnn_hidden_sizes:
        decoder_layers.append(cell_cls(
            in_features=width,
            hidden_features=hidden,
            rngs=rngs,
            trainable_initial_state=rnn_trainable_initial_state,
        ))
        width = hidden
    # `width` is the pre-MLP output when rnn_hidden_sizes is empty, which is what
    # makes the no-recurrence reduction line up with the feedforward decoder.
    decoder_layers += make_mlp_layers(
        [width] + post_hidden + [action_size * 2], rngs, activation,
        activation_last_layer=False,
    )
    decoder_layers.append(NormalTanhSampler(
        rngs, entropy_weight=entropy_weight, min_std=min_std, std_scale=std_scale))
    decoder = Sequential(decoder_layers)

    actor = Sequential([
        Concat(task_obs=encoder_branch, proprioception=proprio_branch),
        EfferenceCopy(inner=decoder, sample_action=jp.zeros(action_size),
                      queue_length=efference_length),
    ])

    critic = Sequential([
        Flattener(),
        *make_mlp_layers(critic_sizes, rngs, activation, activation_last_layer=False),
    ])

    adapter = PPOAdapter(action=actor, value=critic)

    pre = Flattener(preserve_levels=2)
    lift = Filter({
        "task_obs": ("state", "task_obs"),
        "proprioception": ("state", "proprioception"),
    })

    if normalize_obs:
        normalizer_shape = {"state": {
            "task_obs": task_obs_size, "proprioception": proprio_size}}
        return Sequential([pre, Normalizer(normalizer_shape), lift, adapter])
    return Sequential([pre, lift, adapter])


# ---------------------------------------------------------------------------
# Semantic parameter groups
#
# Optional per-architecture hook consulted by ``evaluation.param_counts``. Only
# define one for a *new* architecture: the two original classes deliberately fall
# through to the generic positional introspection in ``evaluation.py`` so their
# eval-record bytes -- and hence every stored artifact spec_id -- stay identical.
# ---------------------------------------------------------------------------

def recurrent_param_groups(nets) -> dict:
    """``encoder`` / ``decoder`` / ``rnn`` counts for the recurrent architecture."""
    from vnl_experiments.delays.evaluation import _count_params

    adapter = next((l for l in nets.layers if isinstance(l, PPOAdapter)), None)
    if adapter is None:
        return {}
    head, efference = adapter.action.layers[0], adapter.action.layers[1]
    decoder = efference.inner
    rnn_layers = [l for l in decoder.layers if isinstance(l, tuple(RNN_CELLS.values()))]
    return {
        "encoder": _count_params(head.components["task_obs"]),
        "decoder": _count_params(decoder),
        "rnn": sum(_count_params(l) for l in rnn_layers),
    }


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Architecture:
    """One network architecture: how to default it, build it and name it.

    Attributes:
        name: The ``network_class`` tag written into ``config.json`` and used to
            dispatch reconstruction. Never reuse a name for a changed
            architecture -- old checkpoints are rebuilt by this string.
        defaults: Returns the train-time ``net_config`` for a fresh run.
        build: ``(net_params, env, rngs) -> StatefulModule``.
        short_name: Prefix for the experiment / checkpoint directory name.
        tags: Extra WandB tags.
        label: Optional ``(net_params) -> str`` overriding ``short_name`` in the
            run name, for architectures whose variants deserve distinct names.
        param_groups: Optional ``(nets) -> dict`` of semantic parameter counts.
            Leave ``None`` to use the generic introspection in ``evaluation.py``.
    """

    name: str
    defaults: Callable[[], Any]
    build: Callable[[dict, Any, nnx.Rngs], Any]
    short_name: str
    tags: tuple[str, ...] = ()
    label: Callable[[dict], str] | None = None
    param_groups: Callable[[Any], dict] | None = None

    def run_label(self, net_params: dict) -> str:
        return self.label(net_params) if self.label is not None else self.short_name


def _recurrent_label(net_params: dict) -> str:
    """``RodentEncDecLSTM`` / ``...GRU`` / ``...RNN`` -- the cell is the headline."""
    return f"RodentEncDec{str(net_params.get('rnn_cell', 'lstm')).upper()}"


ARCHITECTURES: dict[str, Architecture] = {
    arch.name: arch
    for arch in (
        Architecture(
            name="RodentEncDecDelays",
            defaults=delay_defaults,
            build=build_delay_network,
            short_name="RodentEncDec",
            tags=("MLP", "EncDec"),
        ),
        Architecture(
            name="RodentForwardModel",
            defaults=forward_model_defaults,
            build=build_forward_model_network,
            short_name="RodentForwardModel",
            tags=("MLP", "ForwardModel"),
        ),
        Architecture(
            name="RodentEncDecRecurrent",
            defaults=recurrent_defaults,
            build=build_recurrent_network,
            short_name="RodentEncDecRec",
            tags=("Recurrent", "EncDec"),
            label=_recurrent_label,
            param_groups=recurrent_param_groups,
        ),
    )
}


def get_architecture(network_class: str) -> Architecture | None:
    """Resolve a ``network_class`` string to its ``Architecture``, or None.

    Exact match first, then substring -- the distillation scripts record
    ``str(type(student))`` rather than a bare name, so the stored value can be
    something like ``"<class '...RodentEncDecDelays'>"``.
    """
    network_class = str(network_class)
    if network_class in ARCHITECTURES:
        return ARCHITECTURES[network_class]
    for name, arch in ARCHITECTURES.items():
        if name in network_class:
            return arch
    return None


def build_network(net_params: dict, env, rngs: nnx.Rngs):
    """Dispatch on ``network_class``. Returns None for unknown classes."""
    network_class = str(net_params.get("network_class", ""))
    arch = get_architecture(network_class)
    if arch is None:
        warnings.warn(
            f"Unknown network_class {network_class!r}; add an Architecture entry "
            f"for it in network_builders.ARCHITECTURES. Skipping."
        )
        return None
    return arch.build(net_params, env, rngs)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_network(ckpt_dir: Path, net_params: dict, env, seed: int):
    """Build the network, restore the latest step, return (nets, step) or None."""
    ckpt_dir = Path(ckpt_dir)
    step_dirs = sorted(ckpt_dir.glob("step_*/"))
    if not step_dirs:
        warnings.warn(f"No step_* directory in {ckpt_dir}; skipping.")
        return None
    step_dir = step_dirs[-1]

    nets = build_network(net_params, env, nnx.Rngs(seed))
    if nets is None:
        return None

    with open(step_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    ppo_cfg = meta.get("config")
    ppo_cfg = ppo_cfg.ppo if ppo_cfg is not None else None
    ts = new_training_state(
        env, nets, n_envs=1, seed=seed,
        learning_rate=ppo_cfg.learning_rate if ppo_cfg else 1e-4,
        gradient_clipping=ppo_cfg.gradient_clipping if ppo_cfg else 1.0,
        weight_decay=ppo_cfg.weight_decay if ppo_cfg else None,
    )
    ckpt = load_checkpoint(str(step_dir), ts.networks, ts.optimizer)
    return ts.networks, int(ckpt["step"])
