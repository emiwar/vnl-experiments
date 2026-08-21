"""Train PPO on the rodent imitation task with actor-side proprioception
delay + efference copy.

Run as::

    python -m vnl_experiments.delays.train_rodent_delays --delay 5

Network shape mirrors ``vnl_experiments/non-modular/enc_dec.py`` (encoder
with variational bottleneck on ``imitation_target``, decoder on the latent
+ proprioception), but only the **proprioception** stream is delayed; the
``imitation_target`` is treated as an external reference and stays
un-delayed. The critic sees the full un-delayed dict obs (privileged).
Efference copy feeds the most recent ``efference_length`` actions into the
decoder input. ``--delay 0`` and ``--efference 0`` reproduce the baseline.

This is a thin wrapper: the architecture lives in
``network_builders.ARCHITECTURES["RodentEncDecDelays"]`` and the training
scaffolding in ``train_rodent.py``, which is also what the offline eval scripts
build from. The command line is unchanged, so the slurm scripts keep working.
For a recurrent decoder, use::

    python -m vnl_experiments.delays.train_rodent --network RodentEncDecRecurrent
"""

import argparse

from vnl_experiments.delays import train_rodent

NETWORK = "RodentEncDecDelays"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    train_rodent.add_common_args(p)
    p.add_argument("--net-config", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a net-config key. Repeatable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rodent.run(
        args,
        network=NETWORK,
        # False reproduces the exact TrainConfig this script's historical runs
        # used, so `config.*` stays comparable across the refactor. Pass through
        # train_rodent.py directly if you want the env/net metric subtrees.
        log_net_metrics=False,
    )


if __name__ == "__main__":
    main()
