"""New (almost-full-collision) walker XML + reference_root vs the old (sparse) baseline.

Primary question
----------------
The configuration going forward is **new XML + `reference_root` + torque actuators**.
How does it compare to the old baseline it replaces, **old XML + `current_root` +
torque**? Both arms are the policy-gradient forward model (``fm_loss_weight = 0``,
``detach_prediction = False``), which is the only network with a torque-actuated cohort
on both sides at the standard 600 M-step / ``min_std = 0.1`` settings.

The primary contrast therefore moves **two** things at once (walker XML and target
frame). Two groups of supplementary conditions separate them:

* **XML alone** — EncDec + efference, torque, `current_root` on both sides
  (``encdec_old_current`` vs ``encdec_new_current``).
* **Frame alone** — old XML, torque, `current_root` vs `reference_root`, measured
  independently in two networks (``encdec_*`` and ``expfm_*``). Both reference-root arms
  come from the deliberate 2026-07-06 A/B at ``909e774d``.

The decomposition necessarily crosses networks: there is no cell where XML and frame vary
independently within one architecture. It bounds the split rather than measuring it.

A third supplementary pair covers **position control** (``expfm_*_position``), which is
an XML-only contrast at fixed `current_root` and is treated as a special case.

Excluded by design
------------------
``net_params.min_std == 0.1`` and ``summary._step == 600_064_000`` drop the
larger-min-std (0.25) and longer-training (2 G-step) new-XML runs, which belong to a
separate question. Also excluded: seeds 43/44, non-standard architectures, and runs where
``efference_length != delay_k``.

Data sources
------------
Configs and summaries from the committed run index; reward curves and throughput from
``history`` artifacts; offline evaluation from ``eval`` artifacts where present (see
``coverage.txt``). Nothing here touches the WandB API.

.. warning::

   The eval spec was repointed on 2026-08-19 from ``eval3ds-66aaff5b`` to the post-fix
   ``eval3ds-347333e3``. Every v1 eval of a new-XML run had been simulated on
   ``rodent.xml`` -- the wrong body -- which depressed the new arm's held-out reward by
   3 % at delay 0 rising to ~50 % at delay 60+. That is the shape of this question's former
   headline result, and it does not survive the fix. **Read the retraction at the top of
   ``report.md`` before using any number from this folder.**

Run it
------
    ../.venv/bin/python analysis/collision-model-xml/extract.py           # frozen
    ../.venv/bin/python analysis/collision-model-xml/extract.py --refresh
    ../.venv/bin/python analysis/collision-model-xml/extract.py --check
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRES = ["index", "history", "eval:eval3ds-347333e3"]

NEW_XML = "rodent_no_tail_collisions.xml"
EXPECTED_STEP = 600_064_000
STD_MIN_STD = 0.1
STD_ARCH = {
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
}

REWARD_MEAN_KEYS = ("summary.eval/episode_reward/mean", "summary.episode_reward/mean")
REWARD_STD_KEYS = ("summary.eval/episode_reward/std", "summary.episode_reward/std")
LIFESPAN_KEYS = ("summary.eval/lifespan/mean", "summary.lifespan_mean")
CURVE_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")

#: The **v2** eval spec (``EvalProducer.VERSION = 2``), post the 2026-08-18 walker-XML fix.
#: The v1 id this replaced, ``eval3ds-66aaff5b``, simulated every new-XML run on
#: ``rodent.xml``; see the retraction at the top of ``report.md``. Pinned rather than
#: resolved from the producer so a future bump fails loudly in ``assert_spec_ids``.
EVAL_SPEC_ID = "eval3ds-347333e3"
EVAL_DATASETS = ("train", "old_eval", "new_eval")


def assert_spec_ids() -> None:
    """Fail loudly if the eval producer's spec_id has drifted from the pinned constant."""
    got = get_producer("eval").spec_id(get_producer("eval").spec())
    if got != EVAL_SPEC_ID:
        raise SystemExit(
            f"eval spec_id has drifted: got {got}, expected {EVAL_SPEC_ID}.\n"
            f"The artifacts this analysis reads were made by a different eval spec or "
            f"producer VERSION. Update EVAL_SPEC_ID deliberately, re-produce, and say so "
            f"in report.md -- do not silently repoint at a different generation of data.")


def assert_artifact_body(entry, run: pd.Series) -> None:
    """The eval must say it simulated the body this run trained on.

    This cohort exists to compare two bodies, so the check is per-run rather than against
    one expected name -- and it is the guard that would have caught the 2026-08-18 bug on
    day one. An artifact with no stamp predates the fix and was simulated on ``rodent.xml``
    whatever its run trained on, so absence is an error, not something to shrug at.
    """
    stamp = (entry.resolved or {}).get("walker_xml_path")
    trained = Path(str(run["env_params.walker_xml_path"])).name
    if stamp is None:
        raise SystemExit(
            f"eval artifact for {run['wandb_id']} has no resolved.walker_xml_path, so it "
            f"predates the 2026-08-18 walker-XML fix and was simulated on the wrong body. "
            f"Re-produce it (see analysis/README.md §6).")
    if stamp != trained:
        raise SystemExit(f"eval artifact for {run['wandb_id']} was produced on {stamp}, "
                         f"but the run trained on {trained}.")


# --------------------------------------------------------------------------------------
# condition selectors
# --------------------------------------------------------------------------------------


def _is_new_xml(df: pd.DataFrame) -> pd.Series:
    return df["env_params.walker_xml_path"].astype(str).str.contains(NEW_XML)


def _base(df: pd.DataFrame) -> pd.Series:
    """What every included run shares. Also what excludes the two out-of-scope cohorts:
    ``min_std`` drops the larger-exploration runs, ``_step`` drops the 2 G-step ones."""
    mask = (
        (df["env"] == "AbsoluteImitation")
        & (df["seed"] == 42)
        & (df["summary._step"] == EXPECTED_STEP)
        & (df["net_params.min_std"] == STD_MIN_STD)
        & (df["delay_k"] == df["efference_length"])
        & pipeline.full_decoder_inputs_mask(df)
    )
    for column, value in STD_ARCH.items():
        mask &= df[column] == value
    return mask


def _network(df: pd.DataFrame, kind: str) -> pd.Series:
    """Network identity from tags plus the forward-model knobs.

    No single config field separates the explicit forward model (``fm_loss_weight = 1``,
    prediction detached) from the policy-gradient one (``fm_loss_weight = 0``, gradient
    flowing through the predictor), so both are needed.
    """
    tags = df["tags"].fillna("").str.split(",")
    is_fm = tags.apply(lambda t: "ForwardModel" in t)
    if kind == "encdec":
        return tags.apply(lambda t: "EncDec" in t) & ~is_fm
    if kind == "explicit_fm":
        return (is_fm & (df["fm_loss_weight"] == 1)
                & df["detach_prediction"].fillna(True).astype(bool))
    if kind == "pg_fm":
        return is_fm & (df["fm_loss_weight"] == 0) & (df["detach_prediction"] == False)  # noqa: E712
    raise ValueError(kind)


#: Terminal states that are acceptable *given* that ``summary._step`` reached
#: ``EXPECTED_STEP``. Completing all 600 M steps is the real inclusion criterion; the
#: WandB state only says how the process exited. The 2026-08-11 cohort (46 runs at
#: ``ef060b7``) is marked ``failed`` because it died in the post-training evaluation --
#: it has every training metric, normal 3.5-7.5 h runtimes, and no ``final_eval/*`` keys.
#: A run that died *during* training cannot reach ``EXPECTED_STEP``, so this pair of
#: conditions admits the former and excludes the latter.
ACCEPTED_STATES = ("finished", "failed")


def _cell(network: str, *, new_xml: bool, torque: bool, frame: str,
          commits: tuple[str, ...] | None = None, dates: tuple[str, ...] | None = None):
    """Selector for one cell. ``commits``/``dates`` pin a specific launch tranche where a
    cell spans several, so a baseline stays a single coherent sweep."""

    def selector(df: pd.DataFrame) -> pd.Series:
        mask = (
            _base(df)
            & _network(df, network)
            & (_is_new_xml(df) == new_xml)
            & (df["env_params.torque_actuators"] == torque)
            & (df["env_params.body_target_frame"] == frame)
            & df["tags"].fillna("").str.split(",").apply(lambda t: "TrainEvalSplit" in t)
            & df["state"].isin(ACCEPTED_STATES)
        )
        if commits is not None:
            mask &= df["git_commit"].str[:8].isin(commits)
        if dates is not None:
            mask &= df["created_at"].str[:10].isin(dates)
        return mask

    return selector


CONDITIONS = {
    # --- the EncDec 2x2: the only network where XML and frame vary independently ------
    # The canonical 1cd5838f delay sweep is the reference baseline used across all of
    # these questions; later torque EncDec tranches are left out to keep it one sweep.
    "encdec_old_current": _cell("encdec", new_xml=False, torque=True,
                                frame="current_root", commits=("1cd5838f",)),
    "encdec_old_reference": _cell("encdec", new_xml=False, torque=True,
                                  frame="reference_root"),
    "encdec_new_current": _cell("encdec", new_xml=True, torque=True,
                                frame="current_root"),
    "encdec_new_reference": _cell("encdec", new_xml=True, torque=True,
                                  frame="reference_root"),

    # --- explicit forward model, torque: three of the four cells ---------------------
    "expfm_old_current": _cell("explicit_fm", new_xml=False, torque=True,
                               frame="current_root", commits=("54643764",)),
    "expfm_old_reference": _cell("explicit_fm", new_xml=False, torque=True,
                                 frame="reference_root"),
    "expfm_new_reference": _cell("explicit_fm", new_xml=True, torque=True,
                                 frame="reference_root"),

    # --- policy-gradient forward model, torque: the two diagonal cells ---------------
    "pgfm_old_current": _cell("pg_fm", new_xml=False, torque=True, frame="current_root"),
    "pgfm_new_reference": _cell("pg_fm", new_xml=True, torque=True,
                                frame="reference_root"),

    # --- position control (XML alone, current_root both sides) -----------------------
    "expfm_old_position": _cell("explicit_fm", new_xml=False, torque=False,
                                frame="current_root"),
    "expfm_new_position": _cell("explicit_fm", new_xml=True, torque=False,
                                frame="current_root"),
}

#: Which pairs the plots draw, and what each one isolates. The three ``primary_*`` pairs
#: are the same contrast (old XML + current_root -> new XML + reference_root, torque)
#: measured in three independent networks.
PAIRS = {
    "primary_encdec": ("encdec_old_current", "encdec_new_reference",
                       "XML + frame (EncDec, torque)"),
    "primary_expfm": ("expfm_old_current", "expfm_new_reference",
                      "XML + frame (explicit FM, torque)"),
    "primary_pgfm": ("pgfm_old_current", "pgfm_new_reference",
                     "XML + frame (policy-gradient FM, torque)"),

    # the EncDec 2x2, each factor held at both levels of the other
    "xml_at_current": ("encdec_old_current", "encdec_new_current",
                       "XML alone, at current_root (EncDec)"),
    "xml_at_reference": ("encdec_old_reference", "encdec_new_reference",
                         "XML alone, at reference_root (EncDec)"),
    "frame_at_old": ("encdec_old_current", "encdec_old_reference",
                     "Frame alone, on old XML (EncDec)"),
    "frame_at_new": ("encdec_new_current", "encdec_new_reference",
                     "Frame alone, on new XML (EncDec)"),
    # the same frame contrast in a second network, old XML only
    "frame_expfm": ("expfm_old_current", "expfm_old_reference",
                    "Frame alone, on old XML (explicit FM)"),

    "position_xml": ("expfm_old_position", "expfm_new_position",
                     "XML alone (explicit FM, position, current_root)"),
}

# The experimental axes (xml, frame, network, control mode, delay) and git_commit vary by
# design; everything else must be single-valued within a condition.
INVARIANTS = [
    "env", "seed", "net_params.latent_size", "net_params.kl_weight",
    "net_params.min_std", "net_params.latent_min_std", "net_params.std_scale",
    "net_params.enc_hidden_sizes", "net_params.dec_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "env_params.clip_length", "env_params.ctrl_dt", "env_params.sim_dt",
    "env_params.solver", "env_params.iterations", "env_params.ls_iterations",
    "env_params.njmax", "env_params.naconmax", "env_params.rescale_factor",
    "env_params.mujoco_impl",
    "config.ppo.n_envs", "config.ppo.total_steps", "summary._step",
    "env_params.walker_xml_path", "env_params.torque_actuators",
    "env_params.body_target_frame", "git_commit",
]


# --------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------


NETWORK_OF = {"pgfm": "pg_forward_model", "encdec": "efference",
              "expfm": "forward_model"}


def history_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def median_sps(hist: pd.DataFrame | None, column: str) -> float | None:
    """Median throughput, dropping the first 10 % of samples (XLA compilation)."""
    if hist is None or column not in hist or hist[column].dropna().empty:
        return None
    values = hist[column].dropna().to_numpy()
    return float(np.median(values[max(1, len(values) // 10):]))


def build_row(run: pd.Series, hist: pd.DataFrame | None) -> dict:
    condition = run["condition"]
    xml = str(run["env_params.walker_xml_path"]).rsplit("/", 1)[-1]
    notes = run["notes"]
    return {
        # provenance
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "wandb_project": PROJECT,
        "state": run["state"],
        "git_commit": (run["git_commit"] or "")[:8],
        "tags": run["tags"],
        "notes": notes.strip() if isinstance(notes, str) else "",
        "created_at": run["created_at"],
        # experimental axes
        "condition": condition,
        "xml": "new" if xml == NEW_XML else "old",
        "walker_xml": xml,
        "network": NETWORK_OF[condition.split("_")[0]],
        "control_mode": "torque" if run["env_params.torque_actuators"] else "position",
        "body_target_frame": run["env_params.body_target_frame"],
        "delay_k": run["delay_k"],
        "efference_length": run["efference_length"],
        # fm knobs (sanity)
        "fm_loss_weight": run.get("fm_loss_weight"),
        "detach_prediction": run.get("detach_prediction"),
        "min_std": run["net_params.min_std"],
        # invariants
        "env": run["env"],
        "seed": run["seed"],
        "latent_size": run["net_params.latent_size"],
        "kl_weight": run["net_params.kl_weight"],
        "clip_length": run["env_params.clip_length"],
        "ctrl_dt": run["env_params.ctrl_dt"],
        "solver": run["env_params.solver"],
        "njmax": run["env_params.njmax"],
        "naconmax": run["env_params.naconmax"],
        "mujoco_impl": run["env_params.mujoco_impl"],
        "n_envs": run["config.ppo.n_envs"],
        "total_steps": run["config.ppo.total_steps"],
        "actual_step": run["summary._step"],
        # hardware -- speed comparisons are only valid within one GPU model
        "gpu": run["gpu"],
        "host": run["host"],
        # training-time metrics (eval_env == train_env, so: eval on training clips)
        "reward_mean": pipeline.first_present(run, *REWARD_MEAN_KEYS),
        "reward_std": pipeline.first_present(run, *REWARD_STD_KEYS),
        "lifespan_mean": pipeline.first_present(run, *LIFESPAN_KEYS),
        "train_sps_median": median_sps(hist, "throughput/train_sps"),
        "eval_sps_median": median_sps(hist, "throughput/eval_sps"),
        "runtime_s": run.get("summary._runtime"),
        # inline end-of-training eval, present only on runs from 2026-08-10 onward.
        # A *different* measurement from the batch eval artifacts below (in-memory
        # weights vs the newest checkpoint on disk) -- never mix them in one figure.
        "inline_eval_old_eval_reward": run.get(
            "summary.final_eval/old_eval/episode_reward/mean"),
        "inline_eval_new_eval_reward": run.get(
            "summary.final_eval/new_eval/episode_reward/mean"),
        "inline_eval_new_eval_survived": run.get(
            "summary.final_eval/new_eval/termination_rate/survived"),
    }


def build_curve(run: pd.Series, hist: pd.DataFrame | None) -> list[dict]:
    """The run's eval-reward series (evaluated every 10 M steps -> ~60 points)."""
    if hist is None:
        return []
    key = next((k for k in CURVE_KEYS if k in hist.columns), None)
    if key is None:
        return []
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "delay_k": run["delay_k"], "step": int(row["_step"]),
             "reward_mean": float(row[key])}
            for _, row in hist.dropna(subset=[key]).iterrows()]


def build_eval_rows(store: Store, run: pd.Series) -> list[dict]:
    """One row per (run, dataset) from the batch offline evaluation, if present."""
    entry = store.lookup("eval", run["wandb_id"], EVAL_SPEC_ID)
    if entry is None:
        return []
    assert_artifact_body(entry, run)
    record = json.loads((store.root / entry.path).read_text())
    rows = []
    for dataset in EVAL_DATASETS:
        block = record.get("datasets", {}).get(dataset)
        if not block:
            continue
        lifespan = block["lifespan_steps"]["mean"]
        rows.append({
            "wandb_id": run["wandb_id"],
            "condition": run["condition"],
            "delay_k": run["delay_k"],
            "dataset": dataset,
            "checkpoint_step": entry.resolved.get("checkpoint_step"),
            # The body the eval actually simulated -- the field whose absence *was* the
            # 2026-08-18 bug. Carried into the CSV so the fix is auditable from the data.
            "walker_xml": entry.resolved.get("walker_xml_path"),
            "n_clips": block["n_clips"],
            "episode_reward": block["episode_reward"]["mean"],
            "lifespan_steps": lifespan,
            # Length-fair metrics: raw reward is comparable within a dataset only.
            "reward_per_step": block["episode_reward"]["mean"] / lifespan,
            "survived": block["termination_rate"]["survived"],
            "hazard_rate": ((1 - block["termination_rate"]["survived"])
                            / (lifespan * float(run["env_params.ctrl_dt"]))),
        })
    return rows


def main() -> None:
    args = pipeline.parse_args(__doc__)
    assert_spec_ids()

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    producer = get_producer("history")
    history_spec_id = producer.spec_id(producer.spec())

    rows, curves, evals = [], [], []
    for _, run in runs.iterrows():
        hist = history_of(store, run["wandb_id"], history_spec_id)
        rows.append(build_row(run, hist))
        curves.extend(build_curve(run, hist))
        evals.extend(build_eval_rows(store, run))

    df = pd.DataFrame(rows).sort_values(["condition", "delay_k", "wandb_id"],
                                        ignore_index=True)
    curves_df = pd.DataFrame(curves).sort_values(["condition", "delay_k", "step"],
                                                 ignore_index=True)
    eval_df = pd.DataFrame(evals).sort_values(["condition", "dataset", "delay_k"],
                                              ignore_index=True)

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report + "\n")

    print("\nCohort:")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:22s} n={len(sub):2d} xml={sub['xml'].iat[0]:3s} "
              f"{sub['control_mode'].iat[0]:8s} {sub['body_target_frame'].iat[0]:14s} "
              f"git={sorted(sub['git_commit'].unique())}")
        print(f"  {'':22s} delays={sorted(sub['delay_k'])}")

    print("\nxml x frame (authoritative env_params values):")
    print(pd.crosstab([df["xml"], df["control_mode"]], df["body_target_frame"]).to_string())

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    ok &= pipeline.write_csv(eval_df, HERE / "data_eval.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
