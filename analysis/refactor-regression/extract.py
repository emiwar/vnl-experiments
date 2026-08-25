"""Did the 2026-08-21 registry refactor change what the enc-dec network learns?

The refactor moved training onto the same ``network_builders.build_network`` path the
offline eval already used. Two things then happened, and this folder separates them:

1. ``_parse_net_params`` ran ``int(v)`` on every value, and ``int(0.01) == 0``, so the
   sub-1.0 floats were truncated to zero. Training therefore ran with no entropy bonus,
   no KL penalty and no policy-std floor (``entropy_weight`` / ``kl_weight`` /
   ``min_std`` / ``latent_min_std``). Fixed in ``a3450a9``.
2. Everything else about the refactor -- the shared builder, the registry, the
   ``eval_env = train_env`` fix -- which was intended to be behaviour-neutral.

So the question has two halves: *how much did the bug cost*, and *once fixed, does the
refactored path reproduce the pre-refactor baseline*. Three code epochs, one architecture.

**Preliminary.** Two liberties are taken and both are recorded per row:

* **Mixed metric source.** The 2026-08-11 baseline is ``state = failed`` (training
  finished; the end-of-training eval crashed), so it has offline ``eval`` artifacts but no
  inline ``final_eval`` summary. The new runs are the reverse -- no artifacts produced
  yet. ``metric_source`` says which was used per run. The two paths were calibrated on 10
  runs holding both: ``old_eval`` agrees to +0.35 % +/- 1.53 %, well inside the effects
  here. A formal version pins one eval spec and produces it for every run.
* **``n_envs`` varies** (1024 and 4096). It is a column, not a condition, and the
  headline comparison uses the 4096 subset where the baseline lives.

Both halves are read on ``old_eval`` -- the held-out 20 % split -- because it is the one
metric measured identically across all three epochs. The in-training ``eval/*`` series is
**not** usable here: before the refactor it scored the *train* split (see the trap note in
analysis/README.md), so it means different things on either side of the boundary.

Run it
------
    ../.venv/bin/python analysis/refactor-regression/extract.py            # frozen rebuild
    ../.venv/bin/python analysis/refactor-regression/extract.py --sync --refresh
    ../.venv/bin/python analysis/refactor-regression/extract.py --check
"""

import json
from pathlib import Path

import pandas as pd

from vnl_experiments.artifacts import store as artifact_store
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: Only the index is required. The eval artifacts are read opportunistically (see the
#: docstring): declaring them would report a GAP for every new run, when the honest
#: statement is "this run's number came from the inline eval instead".
REQUIRES = ["index"]

XML_ROOT = ("/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/vnl-playground/"
            "vnl_playground/tasks/rodent/xmls")
NEW_XML = f"{XML_ROOT}/rodent_no_tail_collisions.xml"

#: Noise-free eval specs to look for, newest producer version first. All are the same
#: measurement; the duplicates exist because ``action_noise`` None-vs-0 minted two ids.
EVAL_SPECS = ("eval3ds-n00-6a6b8d4e", "eval3ds-n00-21b2d9a8")

FIXED_COMMITS = frozenset({
    "a3450a91809a6b2c86adff7bf35d5c6374530e9d",  # the parser fix
    "4245ae4d4d1e0ba0b3f0f5d5e1f0d0c9b8a7f6e5",  # provenance logging, same behaviour
})

#: Training completed, whatever the run's final state says. The 2026-08-11 cohort is
#: `failed` because the end-of-training eval crashed after 600 M steps of training.
MIN_STEPS = 590_000_000


def _standard_encdec(df: pd.DataFrame) -> pd.Series:
    """The plain feedforward enc-dec, efference-matched, on the current body/frame."""
    name = df["wandb_name"].fillna("")
    return (
        name.str.startswith("RodentEncDec_")               # excludes LSTM_ and ForwardModel_
        & df["env_params.walker_xml_path"].eq(NEW_XML)
        & df["env_params.body_target_frame"].eq("reference_root")
        & df["delay_k"].eq(df["efference_length"])
        & df["summary._step"].ge(MIN_STEPS)
        & pipeline.full_decoder_inputs_mask(df)
    )


def _commits(df: pd.DataFrame) -> pd.Series:
    return df["git_commit"].fillna("")


CONDITIONS = {
    # Pre-dates the refactor entirely: the last cohort trained by the old inline path.
    "pre_refactor": lambda df: (
        _standard_encdec(df)
        & ~_commits(df).isin(pipeline.UNREGULARIZED_COMMITS)
        & ~_commits(df).isin(FIXED_COMMITS)
    ),
    "unregularized": lambda df: (
        _standard_encdec(df) & _commits(df).isin(pipeline.UNREGULARIZED_COMMITS)
    ),
    "fixed": lambda df: (
        _standard_encdec(df) & _commits(df).isin(FIXED_COMMITS)
    ),
}

#: `config.ppo.n_envs` is deliberately absent: it varies by design here and is carried as
#: a data column. `git_commit` likewise -- it *is* the axis.
INVARIANTS = [
    "env",
    "config.seed",
    "config.ppo.total_steps",
    "config.ppo.rollout_length",
    "config.ppo.learning_rate",
    "config.ppo.n_minibatches",
    "net_params.latent_size",
    "net_params.enc_hidden_sizes",
    "net_params.dec_hidden_sizes",
    "env_params.body_target_frame",
    "env_params.ctrl_dt",
    "env_params.walker_xml_path",
]


def _artifact_metrics(wandb_id: str) -> tuple[dict | None, str | None]:
    store = artifact_store.Store()
    for sid in EVAL_SPECS:
        entry = store.lookup("eval", wandb_id, sid)
        if entry is None:
            continue
        path = store.root / entry.path
        if not path.exists():
            continue
        return json.loads(path.read_text()), sid
    return None, None


def build_row(run: pd.Series) -> dict:
    record, sid = _artifact_metrics(run["wandb_id"])
    if record is not None:
        datasets = record["datasets"]
        metrics = {
            f"{ds}_reward": datasets[ds]["episode_reward"]["mean"]
            for ds in ("train", "old_eval", "new_eval") if ds in datasets
        }
        metrics["old_eval_lifespan"] = datasets["old_eval"]["lifespan_steps"]["mean"]
        metrics["survived"] = datasets["old_eval"]["termination_rate"]["survived"]
        source = f"artifact:{sid}"
    else:
        metrics = {
            f"{ds}_reward": run.get(
                f"summary.final_eval/{ds}/episode_reward/mean")
            for ds in ("train", "old_eval", "new_eval")
        }
        metrics["old_eval_lifespan"] = run.get(
            "summary.final_eval/old_eval/lifespan_steps/mean")
        metrics["survived"] = run.get(
            "summary.final_eval/old_eval/termination_rate/survived")
        source = "inline_final_eval"

    return {
        "condition": run["condition"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "git_commit": str(run["git_commit"])[:7],
        "gpu": run["gpu"],
        "n_envs": run.get("config.ppo.n_envs"),
        "delay_k": run.get("delay_k"),
        "efference_length": run.get("efference_length"),
        "seed": run.get("config.seed"),
        "actual_step": run.get("summary._step"),
        "total_params": run.get("summary.final_eval/params/total"),
        "metric_source": source,
        **metrics,
    }


def main() -> None:
    args = pipeline.parse_args(__doc__)
    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    df = pd.DataFrame([build_row(run) for _, run in runs.iterrows()])
    df = df.sort_values(["condition", "delay_k", "n_envs", "wandb_id"],
                        ignore_index=True)

    report = comparability_report(runs, invariant_cols=INVARIANTS,
                                  group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
    print(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
