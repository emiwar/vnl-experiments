"""Recurrent decoders: architecture, delay tolerance, action buffer, BPTT horizon.

A single clean cohort -- **4096 envs, seed 42, new XML, `reference_root`, full decoder
inputs, regularisation intact, >=590 M steps** -- used to answer five questions:

1. **Is a recurrent decoder better than feedforward, with or without an explicit forward
   model, given that it carries more parameters?** `total_params` is on every row.
2. **How much delay can it tolerate?** Matched LSTM and feedforward
   `efference_length = 1` sweeps at `rollout_length = 60`, from delay 0 to 60.
3. **Does a longer BPTT horizon help both architectures?** Every cell measured at both
   `rollout_length` 20 and 60 -- four feedforward, two LSTM, one forward-model.
4. **Can a recurrent net run on a one-step action buffer and compensate internally, and
   up to what delay?** `eff_ratio` and the `efference_length = 1` vs `= delay_k` contrast.
5. **Does the cell type matter?** LSTM vs GRU vs vanilla RNN at delay 10.

**Requeued jobs.** The cluster terminated and requeued many of these, leaving a partial
and a complete twin per config. The cohort filter admits a run only once
`summary._step >= config.ppo.total_steps`, i.e. it trained to completion -- not merely
past a threshold -- so no partial can enter, whatever fraction it reached.

`state` is deliberately *not* filtered on: 23 of the pre-refactor (`ef060b7`) feedforward
runs are `failed` despite having trained the full 600 M steps and written a `final_eval`
summary, so the job exited non-zero after the work was done. Filtering on `state` would
silently delete the backbone of the delay sweep.

Completion being guaranteed, a cell holding two runs is a genuine replicate rather than a
partial in disguise, and `main()` reports them rather than failing. Two kinds occur, and
they measure different things: **cross-epoch** replicates differ in `git_commit` (the
feedforward delay-0 and delay-5 pairs, agreeing to +0.06 % per `refactor-regression`),
while **same-config** replicates share commit, seed and every hyperparameter (the LSTM
delay-40 pair) and so measure pure run-to-run nondeterminism -- GPU kernel scheduling, not
seed. `plot.py` averages both.

**Forward-model arm is the canonical one only** (`fm_loss_weight = 1.0`,
`detach_prediction = True`). Older forward-model runs predate those keys and span
1130-1998 reward at delay 10 across variants; pooling them would make the arm meaningless.

**Preliminary.** `metric_source` records whether a row came from an offline `eval`
artifact or the inline `final_eval` summary (calibrated at +0.35 % +/- 1.53 % on
`old_eval` over 10 runs holding both). Every cell is n = 1; replicate spread for this
family is ~2 % at delay 0 rising to ~8.5 % at delay 10, so differences below that are not
resolved.

Run it
------
    ../.venv/bin/python analysis/recurrent-architectures/extract.py
    ../.venv/bin/python analysis/recurrent-architectures/extract.py --sync --refresh
"""

import json
from pathlib import Path

import pandas as pd

from vnl_experiments.artifacts import store as artifact_store
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRES = ["index"]

XML = "rodent_no_tail_collisions.xml"
EVAL_SPECS = ("eval3ds-n00-6a6b8d4e", "eval3ds-n00-21b2d9a8")


def _cohort(df: pd.DataFrame) -> pd.Series:
    """One environment, one batch size, one seed -- so architecture is the only axis."""
    return (
        df["env_params.walker_xml_path"].astype(str).str.contains(XML)
        & df["env_params.body_target_frame"].eq("reference_root")
        & df["config.ppo.n_envs"].eq(4096)
        & df["config.seed"].eq(42)
        # Trained to completion, not merely past a threshold: a requeued partial cannot
        # satisfy this however far it got. `state` is not filtered -- see the docstring.
        & df["summary._step"].ge(df["config.ppo.total_steps"])
        & pipeline.full_decoder_inputs_mask(df)
        & pipeline.regularized_training_mask(df)
    )


def _is_recurrent(df: pd.DataFrame) -> pd.Series:
    return df["net_params.network_class"].fillna("").eq("RodentEncDecRecurrent")


def _cell_is(df: pd.DataFrame, cell: str) -> pd.Series:
    return _cohort(df) & _is_recurrent(df) & df["net_params.rnn_cell"].fillna("").eq(cell)


CONDITIONS = {
    # `RodentEncDec_` in the name covers the runs predating `network_class`; the LSTM/GRU/
    # RNN names share the `RodentEncDec` stem, hence the explicit trailing underscore.
    "feedforward": lambda df: (
        _cohort(df) & ~_is_recurrent(df)
        & df["wandb_name"].fillna("").str.startswith("RodentEncDec_")
    ),
    "forward_model": lambda df: (
        _cohort(df)
        & df["wandb_name"].fillna("").str.startswith("RodentForwardModel_")
        & df["net_params.fm_loss_weight"].eq(1.0)
        & df["net_params.detach_prediction"].eq(True)
    ),
    "lstm": lambda df: _cell_is(df, "lstm"),
    "gru": lambda df: _cell_is(df, "gru"),
    "rnn": lambda df: _cell_is(df, "rnn"),
}

#: `rollout_length`, `delay_k` and `efference_length` are the axes and are carried as
#: columns. `n_envs` and `seed` are pinned by the cohort filter, so they must not vary.
INVARIANTS = [
    "env",
    "config.ppo.n_envs",
    "config.seed",
    "config.ppo.total_steps",
    "config.ppo.learning_rate",
    "config.ppo.n_minibatches",
    "net_params.latent_size",
    "net_params.enc_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "env_params.body_target_frame",
    "env_params.ctrl_dt",
    "env_params.walker_xml_path",
]

#: Identifies an experimental cell; two runs sharing one would be a silent duplicate.
CELL_KEY = ["condition", "delay_k", "efference_length", "rollout_length"]


def _artifact(wandb_id: str):
    store = artifact_store.Store()
    for sid in EVAL_SPECS:
        entry = store.lookup("eval", wandb_id, sid)
        if entry is not None and (store.root / entry.path).exists():
            return json.loads((store.root / entry.path).read_text()), sid
    return None, None


def build_row(run: pd.Series) -> dict:
    record, sid = _artifact(run["wandb_id"])
    if record is not None:
        ds = record["datasets"]
        metrics = {f"{k}_reward": ds[k]["episode_reward"]["mean"]
                   for k in ("train", "old_eval", "new_eval") if k in ds}
        metrics["old_eval_lifespan"] = ds["old_eval"]["lifespan_steps"]["mean"]
        metrics["survived"] = ds["old_eval"]["termination_rate"]["survived"]
        total_params = record.get("param_counts", {}).get("total")
        source = f"artifact:{sid}"
    else:
        metrics = {f"{k}_reward": run.get(f"summary.final_eval/{k}/episode_reward/mean")
                   for k in ("train", "old_eval", "new_eval")}
        metrics["old_eval_lifespan"] = run.get(
            "summary.final_eval/old_eval/lifespan_steps/mean")
        metrics["survived"] = run.get(
            "summary.final_eval/old_eval/termination_rate/survived")
        total_params = run.get("summary.final_eval/params/total")
        source = "inline_final_eval"

    delay = run.get("delay_k")
    eff = run.get("efference_length")
    return {
        "condition": run["condition"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "git_commit": str(run["git_commit"])[:7],
        "rollout_length": run.get("config.ppo.rollout_length"),
        "delay_k": delay,
        "efference_length": eff,
        "eff_ratio": (eff / delay) if delay else None,
        "rnn_cell": run.get("net_params.rnn_cell"),
        "total_params": total_params,
        "actual_step": run.get("summary._step"),
        "metric_source": source,
        **metrics,
    }


def report_replicates(df: pd.DataFrame) -> None:
    """Print every cell holding more than one run, and what its spread measures.

    Completion is guaranteed by the cohort filter, so a duplicate is a real replicate,
    never a partial. Which *kind* matters: a same-commit pair shares seed and every
    hyperparameter, so its spread is pure run-to-run nondeterminism and is the tightest
    noise floor available here; a cross-commit pair also spans a code epoch.
    """
    for key, group in df.groupby(CELL_KEY, dropna=False):
        if len(group) == 1:
            continue
        kind = ("same-config (nondeterminism only)"
                if group["git_commit"].nunique() == 1 else "cross-epoch")
        lo, hi = group.old_eval_reward.min(), group.old_eval_reward.max()
        cond, delay, eff, roll = key
        print(f"replicate cell [{kind}] {cond} delay {delay:.0f} eff {eff:.0f} "
              f"rollout {roll:.0f}: {list(group.wandb_id)} "
              f"spread {100 * (hi - lo) / lo:.1f}% ({lo:.0f}-{hi:.0f}), averaged in plot.py")


def main() -> None:
    args = pipeline.parse_args(__doc__)
    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    df = pd.DataFrame([build_row(run) for _, run in runs.iterrows()])
    df = df.sort_values(["condition", "delay_k", "efference_length", "rollout_length"],
                        ignore_index=True)

    report_replicates(df)

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
