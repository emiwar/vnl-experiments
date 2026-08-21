"""Train PPO on the rodent imitation task with an explicit forward model.

Run as::

    python -m vnl_experiments.delays.train_rodent_forward_model --delay 5

The actor encodes the imitation target into a latent, and a learned forward
model predicts *current* proprioception from delayed proprioception plus the
efference-copy action queue. The decoder then acts on [latent, predicted
proprioception]. ``--fm-loss-weight`` weights the self-supervised L2 prediction
loss; ``--no-detach-prediction`` lets policy gradients reach the predictor
(architecture ablation). The critic sees the full un-delayed dict obs.

This is a thin wrapper: the architecture lives in
``network_builders.ARCHITECTURES["RodentForwardModel"]`` and the training
scaffolding in ``train_rodent.py``, which is also what the offline eval scripts
build from. The command line is unchanged, so the slurm scripts keep working.
"""

import argparse

from vnl_experiments.delays import train_rodent

NETWORK = "RodentForwardModel"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    train_rodent.add_common_args(p)
    p.add_argument("--fm-loss-weight", type=float, default=1.0,
                   help="Weight on the self-supervised forward-model L2 loss.")
    p.add_argument("--detach-prediction", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Detach the prediction before the decoder (forward-model "
                        "behavior). Use --no-detach-prediction to train the "
                        "predictor with policy gradients (architecture ablation).")
    p.add_argument("--net-config", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a net-config key. Repeatable.")
    # This script's historical default; the shared entry point defaults to 0.
    p.set_defaults(delay=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rodent.run(
        args,
        network=NETWORK,
        # These are net-config keys of the architecture; the CLI flags are kept
        # for compatibility with the existing slurm scripts.
        net_config_overrides={
            "fm_loss_weight": args.fm_loss_weight,
            "detach_prediction": args.detach_prediction,
        },
        # Also logged top-level, so analyses filtering on the historical
        # `fm_loss_weight` / `detach_prediction` columns keep resolving.
        extra_wandb_config={
            "fm_loss_weight": args.fm_loss_weight,
            "detach_prediction": args.detach_prediction,
        },
        extra_tags=() if args.detach_prediction else ("nodetach",),
        name_token="" if args.detach_prediction else "_nodetach",
    )


if __name__ == "__main__":
    main()
