"""Preliminary: does a recurrent (LSTM) decoder help under proprioception delay?

Three sub-questions, in the order they can actually be answered from the data:

1. **Does recurrence help at all?** Same enc-dec pipeline, the only change being that the
   decoder's MLP is replaced by ``pre-MLP -> LSTM(512) -> post-MLP -> sampler``
   (``RodentEncDecRecurrent``, ``rnn_cell = lstm``).
2. **Does it tolerate a shorter efference copy?** The efference queue is an *explicit*
   memory of recent actions; a recurrent decoder can in principle reconstruct that from
   its hidden state. Runs with ``efference_length = 1`` against ``delay_k`` of 5 or 10
   test this. ``eff_ratio`` is the column.
3. **Is it just extra parameters?** The LSTM decoder carries ~50 % more parameters than
   the feedforward one, so any win has to be weighed against a parameter-matched
   feedforward control. ``total_params`` is carried per run for exactly this.

**Preliminary — three liberties, all recorded per row.**

* **Feedforward references are pooled across code epochs.** The ``feedforward`` condition
  includes both the 2026-08-11 pre-refactor cohort and the post-fix runs. That is
  justified by measurement, not assumption: ``refactor-regression`` shows the two epochs
  agree to +0.06 % at the fully matched point. Runs from the two *unregularized* commits
  are excluded via ``pipeline.regularized_training_mask``. ``git_commit`` is carried so
  the pooling can be audited.
* **``n_envs`` (1024, 4096) and ``seed`` (42, 43, 52) both vary.** Neither is a condition.
  Comparisons are only read within a matched (delay, efference, n_envs, seed) cell, and
  the report says which cells are matched and which are not.
* **Mixed metric source** (``metric_source``): offline ``eval`` artifact where one exists,
  otherwise the inline ``final_eval`` summary. Calibrated at +0.35 % +/- 1.53 % on
  ``old_eval`` across 10 runs holding both.

Old-XML reduced-efference runs (the ``no_efference`` / ``efference_trunc`` families in
``action-buffer-length``) are deliberately **not** pooled in: they were trained on
``rodent.xml``, and spanning two bodies is the walker-XML trap, not a liberty.

Run it
------
    ../.venv/bin/python analysis/recurrent-decoder-preliminary/extract.py
    ../.venv/bin/python analysis/recurrent-decoder-preliminary/extract.py --sync --refresh
"""

import json
from pathlib import Path

import pandas as pd

from vnl_experiments.artifacts import store as artifact_store
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

REQUIRES = ["index"]

XML_ROOT = ("/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/vnl-playground/"
            "vnl_playground/tasks/rodent/xmls")
NEW_XML = f"{XML_ROOT}/rodent_no_tail_collisions.xml"
EVAL_SPECS = ("eval3ds-n00-6a6b8d4e", "eval3ds-n00-21b2d9a8")
MIN_STEPS = 590_000_000


def _usable(df: pd.DataFrame) -> pd.Series:
    """Same env, same body, trained to completion, regularisation intact."""
    return (
        df["env_params.walker_xml_path"].eq(NEW_XML)
        & df["env_params.body_target_frame"].eq("reference_root")
        & df["summary._step"].ge(MIN_STEPS)
        & pipeline.full_decoder_inputs_mask(df)
        & pipeline.regularized_training_mask(df)
    )


CONDITIONS = {
    "feedforward": lambda df: (
        _usable(df) & df["wandb_name"].fillna("").str.startswith("RodentEncDec_")
    ),
    "recurrent": lambda df: (
        _usable(df) & df["wandb_name"].fillna("").str.startswith("RodentEncDecLSTM_")
    ),
}

#: `dec_hidden_sizes` is absent by construction: the recurrent decoder replaces it with
#: `dec_pre_hidden_sizes` / `rnn_hidden_sizes` / `dec_post_hidden_sizes`, so it is the
#: axis, not an invariant. `n_envs` and `seed` vary and are carried as columns.
#: `config.ppo.rollout_length` was an invariant until 2026-08-25, when a 20/40/60 sweep
#: made it an experimental axis. It is the BPTT truncation horizon for the recurrent
#: decoder -- at `ctrl_dt = 0.01` those are 0.2 / 0.4 / 0.6 s of credit assignment -- so
#: it is exactly the knob a memory should be sensitive to. Carried as a column.
INVARIANTS = [
    "env",
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


def _artifact(wandb_id: str):
    store = artifact_store.Store()
    for sid in EVAL_SPECS:
        entry = store.lookup("eval", wandb_id, sid)
        if entry is None:
            continue
        path = store.root / entry.path
        if path.exists():
            return json.loads(path.read_text()), sid
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
        "n_envs": run.get("config.ppo.n_envs"),
        "rollout_length": run.get("config.ppo.rollout_length"),
        "seed": run.get("config.seed"),
        "delay_k": delay,
        "efference_length": eff,
        "eff_ratio": (eff / delay) if delay else None,
        "rnn_cell": run.get("net_params.rnn_cell"),
        "rnn_hidden_sizes": run.get("net_params.rnn_hidden_sizes"),
        "total_params": total_params,
        "metric_source": source,
        **metrics,
    }


def main() -> None:
    args = pipeline.parse_args(__doc__)
    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    df = pd.DataFrame([build_row(run) for _, run in runs.iterrows()])
    df = df.sort_values(["condition", "delay_k", "efference_length", "n_envs"],
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
