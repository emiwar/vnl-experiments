"""Shared PPO training entry point for every delays architecture.

The architecture is selected by ``--network`` and built by the registry in
``network_builders.py`` -- the same builder the offline eval scripts use, so a
trained network and its offline reconstruction cannot drift apart. Everything
else here (env config, PPO config, WandB logging, checkpointing, final eval) is
architecture-independent.

Run as::

    # feedforward decoder (the original delays baseline)
    python -m vnl_experiments.delays.train_rodent --delay 5

    # recurrent decoder
    python -m vnl_experiments.delays.train_rodent --delay 5 \
        --network RodentEncDecRecurrent --net-config rnn_cell=gru

    # decoder-input ablations (the three streams the decoder receives; the third,
    # the efference copy, is ablated with --efference 0)
    python -m vnl_experiments.delays.train_rodent --delay 20 \
        --net-config dec_use_intention=false
    python -m vnl_experiments.delays.train_rodent --delay 20 \
        --net-config dec_use_proprioception=false

Any net-config key of the chosen architecture can be overridden on the command
line, e.g. ``--net-config rnn_hidden_sizes=[256,256] --net-config latent_size=64``.
``python -m vnl_experiments.delays.train_rodent --list-networks`` prints the
available architectures with their defaults.

``train_rodent_delays.py`` and ``train_rodent_forward_model.py`` are thin
wrappers over this module and keep their original command lines.
"""

import os


os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import dataclasses
import gc
import json
from datetime import datetime

import jax
import wandb
from flax import nnx

from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.reference_clips import ReferenceClips

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.callbacks import wandb_video_fn
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn
from nnx_ppo.algorithms.config import (
    EvalConfig,
    PPOConfig,
    TrainConfig,
    VideoConfig,
)
from nnx_ppo.algorithms.types import LoggingLevel

from vnl_experiments.provenance import repo_versions
from vnl_experiments.delays import evaluation
from vnl_experiments.delays.network_builders import (
    ARCHITECTURES,
    build_network,
    get_architecture,
)
from vnl_experiments.envs.absolute_imitation import AbsoluteImitation, default_config

DEFAULT_NETWORK = "RodentEncDecDelays"

#: Default WandB note. Deliberately says only what is true of every run here and
#: names no config: the previous default ("New XML + reference_root.") encoded two
#: settings, which is exactly the kind of claim that goes stale and gets trusted --
#: `env_params` / `net_params` are the record for what a run actually used. Pass
#: ``--notes`` to say something run-specific; the manual half of the comparability
#: protocol (analysis/README.md §4) reads it.
DEFAULT_NOTES = "Rodent imitation, proprioception-delay study."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags every delays training script shares.

    The wrapper scripts call this so their command lines stay identical to what
    the slurm scripts already pass.
    """
    p.add_argument("--delay", type=int, default=0,
                   help="Proprioception delay in steps. 0 = no delay.")
    p.add_argument("--efference", type=int, default=None,
                   help="Efference-copy queue length. Defaults to --delay. "
                        "0 disables the efference copy entirely.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="nnx-ppo-rodent-delays")
    p.add_argument("--exp-name-suffix", default="")
    p.add_argument("--notes", default=DEFAULT_NOTES,
                   help="Free-text notes attached to the WandB run. "
                        f"Default: {DEFAULT_NOTES!r}")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Override the configured total_steps (smoke testing).")
    p.add_argument("--n-envs", type=int, default=None,
                   help="Override the configured n_envs (smoke testing).")
    p.add_argument("--rollout-length", type=int, default=None,
                   help="Override the rollout length. This is also the BPTT "
                        "truncation horizon for recurrent architectures.")
    p.add_argument("--n-minibatches", type=int, default=None,
                   help="Override n_minibatches. Must divide n_envs.")
    p.add_argument("--no-video", dest="video", action="store_false",
                   help="Disable video recording (smoke testing).")
    p.add_argument("--no-final-eval", dest="final_eval", action="store_false",
                   help="Skip the end-of-training evaluation.")
    p.add_argument("--checkpoint-every-steps", type=int, default=None,
                   help="Override checkpoint_every_steps (smoke testing).")
    p.add_argument("--eval-every-steps", type=int, default=None,
                   help="Override the in-training eval interval (smoke testing).")
    p.add_argument("--eval-limit-clips", type=int, default=None,
                   help="Cap clips per split in the final eval (memory knob).")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", default=DEFAULT_NETWORK, choices=sorted(ARCHITECTURES),
                   help=f"Architecture to train (default: {DEFAULT_NETWORK}).")
    p.add_argument("--net-config", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a net-config key of the chosen architecture. "
                        "Repeatable. Lists accept JSON or comma-separated values.")
    p.add_argument("--list-networks", action="store_true",
                   help="Print the registered architectures and their defaults, then exit.")
    add_common_args(p)
    return p.parse_args(argv)


def _coerce(default, raw: str):
    """Coerce a ``--net-config`` string to the type of the existing default."""
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(default, (list, tuple)):
        text = raw.strip()
        if text.startswith("["):
            return [int(x) for x in json.loads(text)]
        if text == "":
            return []
        return [int(x) for x in text.split(",")]
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if default is None:
        # An unset optional (e.g. latent_ar1_weight): infer from the literal.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def apply_net_config_overrides(net_config, overrides: list[str]):
    """Apply ``KEY=VALUE`` strings onto an architecture's default net_config."""
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--net-config expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in net_config:
            raise SystemExit(
                f"Unknown net-config key {key!r}. Available: {sorted(net_config.keys())}"
            )
        try:
            net_config[key] = _coerce(net_config[key], raw)
        except (ValueError, json.JSONDecodeError) as e:
            raise SystemExit(f"Bad value for --net-config {key}: {raw!r} ({e})")
    return net_config


def print_networks() -> None:
    for name, arch in sorted(ARCHITECTURES.items()):
        print(f"{name}  (short_name={arch.short_name}, tags={list(arch.tags)})")
        for key, value in sorted(arch.defaults().to_dict().items()):
            print(f"    {key} = {value!r}")
        print()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def make_env_config():
    """The rodent imitation env config shared by all delays runs."""
    env_config = default_config()
    env_config.solver = "newton"
    env_config.reward_terms["bodies_pos"]["weight"] = 0.0
    env_config.reward_terms["joints_vel"]["weight"] = 0.0
    env_config.mujoco_impl = "warp"
    env_config.naconmax = 32 * 4096
    env_config.njmax = 256
    env_config.ctrl_dt = 0.01
    # NOTE: body_target_frame is consumed by the *environment* (AbsoluteImitation),
    # not the network. It must live on env_config to take effect; setting it on
    # net_config is inert (only logged to WandB net_params, never read). See the
    # "reference-representation bug" note in analysis/README.md.
    env_config.body_target_frame = "reference_root"
    env_config.torque_actuators = True
    env_config.walker_xml_path = consts.RODENT_NO_TAIL_COLLISION_XML
    return env_config


def make_train_config(args, *, log_net_metrics: bool = True) -> TrainConfig:
    """The PPO/eval/video config, with the CLI overrides applied.

    ``log_net_metrics`` adds the ``env/*`` and ``net/*`` metric subtrees. It
    defaults on, which is what the forward-model script has always done and what
    you want for a recurrent policy; ``train_rodent_delays.py`` passes False to
    reproduce the exact config its historical runs were trained with.
    """
    extra = (
        LoggingLevel.ENV_METRICS | LoggingLevel.NETWORK_METRICS
        if log_net_metrics
        else LoggingLevel.NONE
    )
    config = TrainConfig(
        ppo=PPOConfig(
            n_envs=4096,
            # Also the BPTT truncation horizon for recurrent architectures: the
            # PPO update minibatches over the env axis only and replays the whole
            # sequence, so gradients flow back exactly this many steps.
            rollout_length=20,
            total_steps=600_000_000,
            discounting_factor=0.95,
            normalize_advantages=True,
            learning_rate=1e-4,
            n_epochs=4,
            n_minibatches=8,
            gradient_clipping=1.0,
            weight_decay=None,
            logging_level=LoggingLevel.BASIC | LoggingLevel.THROUGHPUT | extra,
            logging_percentiles=(0, 25, 50, 75, 100),
        ),
        eval=EvalConfig(
            enabled=True,
            every_steps=10_000_000,
            n_envs=1024,
            max_episode_length=1000,
            logging_level=LoggingLevel.BASIC | extra,
            logging_percentiles=None,
        ),
        video=VideoConfig(
            enabled=True,
            every_steps=50_000_000,
            episode_length=1000,
            render_kwargs={
                "height": 480,
                "width": 640,
                "camera": "close_profile-rodent",
                "add_labels": True,
            },
        ),
        seed=args.seed,
        checkpoint_every_steps=50_000_000,
    )
    if args.n_envs is not None:
        config = dataclasses.replace(
            config, ppo=dataclasses.replace(config.ppo, n_envs=args.n_envs))
    if getattr(args, "rollout_length", None) is not None:
        config = dataclasses.replace(
            config, ppo=dataclasses.replace(
                config.ppo, rollout_length=args.rollout_length))
    if getattr(args, "n_minibatches", None) is not None:
        config = dataclasses.replace(
            config, ppo=dataclasses.replace(
                config.ppo, n_minibatches=args.n_minibatches))
    if not args.video:
        config = dataclasses.replace(
            config, video=dataclasses.replace(config.video, enabled=False))
    if getattr(args, "checkpoint_every_steps", None) is not None:
        config = dataclasses.replace(
            config, checkpoint_every_steps=args.checkpoint_every_steps)
    if getattr(args, "eval_every_steps", None) is not None:
        config = dataclasses.replace(
            config, eval=dataclasses.replace(
                config.eval, every_steps=args.eval_every_steps))

    if config.ppo.n_envs % config.ppo.n_minibatches:
        raise SystemExit(
            f"n_envs ({config.ppo.n_envs}) must be divisible by n_minibatches "
            f"({config.ppo.n_minibatches})."
        )
    return config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(
    args: argparse.Namespace,
    *,
    network: str | None = None,
    net_config_overrides: dict | None = None,
    extra_wandb_config: dict | None = None,
    extra_tags: tuple[str, ...] = (),
    name_token: str = "",
    log_net_metrics: bool = True,
) -> None:
    """Train one run. The wrapper scripts call this with their own defaults.

    Args:
        args: Parsed CLI namespace (see ``add_common_args``).
        network: ``network_class`` to train. Defaults to ``args.network``.
        net_config_overrides: Applied on top of the architecture defaults, before
            the ``--net-config`` flags. Wrappers use this to route their own
            architecture-specific flags (e.g. ``--fm-loss-weight``).
        extra_wandb_config: Extra top-level WandB config keys. Wrappers use this
            to keep historical column names available to existing analyses.
        extra_tags: Extra WandB tags.
        name_token: Inserted into the experiment name before the timestamp.
        log_net_metrics: See ``make_train_config``.
    """
    network = network or getattr(args, "network", DEFAULT_NETWORK)
    arch = get_architecture(network)
    if arch is None:
        raise SystemExit(
            f"Unknown --network {network!r}. Available: {sorted(ARCHITECTURES)}"
        )

    efference_length = args.efference if args.efference is not None else args.delay
    seed = args.seed

    env_config = make_env_config()

    net_config = arch.defaults()
    for key, value in (net_config_overrides or {}).items():
        net_config[key] = value
    apply_net_config_overrides(net_config, getattr(args, "net_config", []))

    config = make_train_config(args, log_net_metrics=log_net_metrics)

    clips = ReferenceClips(env_config.reference_data_path,
                           env_config.clip_length,
                           env_config.keep_clips_idx)
    train_clips, test_clips = clips.split()
    train_env = AbsoluteImitation(env_config, clips=train_clips)
    # In-training eval runs on the held-out clips. Until 2026-08-20 this line was
    # followed by `eval_env = train_env`, so every delays run's WandB `eval/*`
    # series actually measured train-split performance -- see the trap note in
    # analysis/README.md. Do not "simplify" this back.
    eval_env = AbsoluteImitation(env_config, clips=test_clips)

    rngs = nnx.Rngs(seed)

    # One source of truth: config.json (for offline reconstruction by the eval
    # scripts), the network built here, and the end-of-training eval's metadata
    # all read the same dict.
    net_params = {
        **net_config.to_dict(),
        "delay_k": args.delay,
        "efference_length": efference_length,
        "network_class": arch.name,
    }

    # The registry builder -- the same call the offline eval path makes, so the
    # trained architecture and its reconstruction are the same code.
    nets = build_network(net_params, train_env, rngs)
    if nets is None:
        raise SystemExit(f"build_network returned None for {arch.name!r}")

    # Decoder-input ablations get their own name token and tag. Without this a
    # no-intention run at eff == delay is indistinguishable from a standard-arch
    # efference baseline to the analyses that select on delay_k / efference_length /
    # hidden sizes alone. Empty when both flags are at their default, so existing
    # run names and tag sets are unchanged.
    ablations = tuple(
        token for key, token in (("dec_use_intention", "nointent"),
                                 ("dec_use_proprioception", "noproprio"))
        if not net_params.get(key, True)
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{args.exp_name_suffix}" if args.exp_name_suffix else ""
    exp_name = (
        f"{arch.run_label(net_params)}_delay{args.delay}_eff{efference_length}"
        f"{''.join(f'_{t}' for t in ablations)}{name_token}-{timestamp}{suffix}"
    )

    ckpt_dir = f"checkpoints/{exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as _f:
        json.dump({
            "env_params": env_config.to_dict(),
            "net_params": net_params,
        }, _f, indent=2, default=str)

    wandb.init(
        project=args.wandb_project,
        config={
            "env": "AbsoluteImitation",
            "delay_k": args.delay,
            "efference_length": efference_length,
            # Logged so the run index can group by architecture without having to
            # parse the run name. `net_params` is the full dict written to
            # config.json (not the bare net_config), so every architecture-specific
            # knob becomes a `net_params.*` column.
            "network_class": arch.name,
            "seed": seed,
            # Git state of all three repos. WandB's own `git_commit` covers only
            # vnl-experiments; nnx-ppo (the algorithm) and vnl-playground (the task)
            # were previously unrecorded, and `dirty` flags a working copy that has
            # drifted from its commit -- which has happened on the cluster before.
            "repos": repo_versions(),
            "config": dataclasses.asdict(config),
            "net_params": net_params,
            "env_params": env_config.to_dict(),
            **(extra_wandb_config or {}),
        },
        name=exp_name,
        tags=(*arch.tags, "warp", "TrainEvalSplit", arch.name,
              f"delay{args.delay}", f"eff{efference_length}",
              *ablations, *extra_tags),
        notes=getattr(args, "notes", DEFAULT_NOTES),
    )
    result = ppo.train_ppo(
        train_env,
        nets,
        config,
        log_fn=wandb.log,
        video_fn=wandb_video_fn(fps=50),
        checkpoint_fn=make_checkpoint_fn(ckpt_dir, config),
        eval_env=eval_env,
        **({} if args.total_steps is None else {"total_steps": args.total_steps}),
    )

    print(
        f"Training complete: {result.total_steps} steps, "
        f"{result.total_iterations} iterations"
    )
    if result.eval_history:
        print(
            "Final eval reward: "
            f"{result.eval_history[-1].get('eval/episode_reward/mean', 'N/A')}"
        )

    total_steps = result.total_steps

    if args.final_eval:
        # Release the training state (n_envs env states + optimizer moments)
        # before the eval allocates. `nets` is the same object as
        # result.training_state.networks, so the trained weights survive.
        del result
        jax.clear_caches()
        gc.collect()

        evaluation.run_final_eval(
            nets, AbsoluteImitation, env_config,
            ckpt_dir=ckpt_dir,
            wandb_id=wandb.run.id, wandb_name=exp_name, step=total_steps,
            net_params=net_params,
            train_env=train_env, train_clips=train_clips, test_clips=test_clips,
            seed=seed, limit_clips=args.eval_limit_clips,
            summary_fn=wandb.run.summary.update,
        )

    wandb.finish()


def main() -> None:
    args = parse_args()
    if args.list_networks:
        print_networks()
        return
    run(args)


if __name__ == "__main__":
    main()
