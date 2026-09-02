"""Expand a Hydra-style sweep into one ``sbatch`` per cell.

    python -m vnl_experiments.sweep delay=0,5,10,20 net=recurrent net.rnn_cell=lstm,gru

prints -- or, with ``--submit``, runs -- eight ``sbatch slurm_rodent_requeue.sh ...``
commands, one per point in the product. Any override the training entry point accepts can
be swept, using Hydra's own syntax (``a,b,c``, ``range(1,4)``, ``choice(...)``), because
the parsing *is* Hydra's: the same ``OverridesParser`` the trainer composes with, so a
sweep cannot mean something different from the run it launches.

Why not ``--multirun`` with the submitit launcher: these jobs run on ``gpu_requeue`` and
own their own preemption protocol -- checkpoint on SIGTERM, exit 42, and let
``slurm_rodent_requeue.sh`` call ``scontrol requeue``. Submitit wants to manage
requeueing itself through its own checkpoint mechanism, and having two systems each think
they are responsible for resuming the job is worse than launching with ``sbatch``.

Dry-run is the default. The intended workflow is to look at the list, then re-run with
``--submit``, and to paste the command into the analysis folder so the cohort is
reproducible from the commit.
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path

from hydra.core.override_parser.overrides_parser import OverridesParser

#: Default batch script. It is the requeue one because a sweep large enough to be worth
#: scripting is a sweep worth running on the cheap preemptible partition.
DEFAULT_SCRIPT = "slurm_rodent_requeue.sh"


def expand(overrides: list[str]) -> list[list[str]]:
    """Every combination of the swept overrides, as lists of ``key=value`` strings.

    Non-swept overrides are passed through unchanged to every cell. Order is preserved so
    the emitted command lines read like the one that was typed.
    """
    parsed = OverridesParser.create().parse_overrides(overrides)

    axes: list[list[str]] = []
    for override in parsed:
        key = override.key_or_group
        if override.is_sweep_override():
            axes.append([f"{key}={v}" for v in override.sweep_string_iterator()])
        else:
            axes.append([f"{key}={override.get_value_element_as_str()}"])
    return [list(cell) for cell in itertools.product(*axes)]


def commands(cells: list[list[str]], script: str, sbatch_args: list[str]) -> list[list[str]]:
    return [["sbatch", *sbatch_args, script, *cell] for cell in cells]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                   help="Hydra overrides. A comma-separated value or range(...) makes "
                        "that key a sweep axis; everything else is passed to every job.")
    p.add_argument("--script", default=DEFAULT_SCRIPT,
                   help="Batch script to submit (default: %(default)s).")
    p.add_argument("--submit", action="store_true",
                   help="Actually submit. Without it the commands are only printed, "
                        "which is the default because a sweep is easy to typo and "
                        "expensive to typo.")
    p.add_argument("--sbatch-arg", action="append", default=[], metavar="ARG",
                   help="Extra argument for sbatch itself, e.g. --sbatch-arg=-t=0-04:00. "
                        "Repeatable.")
    p.add_argument("--max-jobs", type=int, default=64,
                   help="Refuse to submit more than this many jobs (default: "
                        "%(default)s). A guard against a sweep that is larger than "
                        "intended; raise it deliberately.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.overrides:
        print("No overrides given; nothing to sweep.", file=sys.stderr)
        return 2

    try:
        cells = expand(args.overrides)
    except Exception as e:  # noqa: BLE001 - the parser raises a variety of types
        print(f"error: could not parse the overrides: {e}", file=sys.stderr)
        return 2

    cmds = commands(cells, args.script, args.sbatch_arg)
    for cmd in cmds:
        print(shlex.join(cmd))
    print(f"\n{len(cmds)} job(s).", file=sys.stderr)

    if not args.submit:
        print("Dry run; re-run with --submit to launch.", file=sys.stderr)
        return 0

    if len(cmds) > args.max_jobs:
        print(f"error: {len(cmds)} jobs exceeds --max-jobs {args.max_jobs}.",
              file=sys.stderr)
        return 1
    if not Path(args.script).exists():
        print(f"error: {args.script} not found. Submit from the repository root.",
              file=sys.stderr)
        return 1

    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"error: {shlex.join(cmd)}\n{result.stderr.strip()}", file=sys.stderr)
            return 1
        print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
