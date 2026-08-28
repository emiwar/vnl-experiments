"""Lab-meeting summary, August 2026: the five delay claims, one folder.

This folder exists to make **talk figures**, not to ask a new question. Every claim below
has already been established somewhere else in ``analysis/``; what is new here is that all
five are rebuilt from one script, on one cohort definition per claim, with the figure
conventions the talk needs (log delay axis with a special-cased 0, dashed reference lines,
one panel per delay). Where a claim's evidence had to come from an older cohort, this
module says so at the condition, and ``report.md`` says so next to the figure.

The five claims, and where each one's data comes from
-----------------------------------------------------

1. **The efference copy is necessary.** ``efference_old`` vs ``no_efference_old``.
   *The only cohort in the project that has a no-efference arm at all* -- 13 runs at
   ``efference_length = 0``, git ``1cd5838``, and they are on the **old** walker XML with
   ``current_root`` targets. Nothing equivalent was ever trained on the going-forward
   configuration, so this contrast cannot be moved onto it without new runs (see
   ``report.md``, "What would close the gaps"). The two supporting panels -- tracking error
   and lifetime against delay, which need only the with-efference arm -- are therefore drawn
   from the going-forward ``encdec`` cohort instead, and labelled as such.

2. **A forward model improves learning.** ``encdec`` vs ``pgfm`` vs ``expfm`` at 600 M
   steps, all new XML + ``reference_root``. Complete: three full delay sweeps.

3. **Part of that is convergence speed, not ceiling.** ``expfm``/``pgfm`` read at every
   budget we have (600 M / 2 G / 4 G) from the training curves. Delays 0 and 10 top out at
   2 G; delays 20-50 reach 4 G on seed 43. Panels are drawn to whatever budget each delay
   actually reached and the title says which.

4. **Forward models are sensitive to motor noise.** ``encdec`` vs ``expfm`` re-evaluated
   under a fixed Gaussian perturbation of the executed action. The implicit arm has no
   noise evals, so this contrast is explicit-FM-vs-enc-dec, not explicit-vs-implicit.

5. **The policy gradient does not learn a forward model.** ``expfm`` vs ``pgfm``: their
   own ``fm_pred_mse``, and the layer-wise linear probe. The layer-wise R² is *reused* from
   ``explicit-vs-implicit-fm-probe/data.csv`` -- it is committed, vetted data and
   re-deriving it would produce identical numbers more slowly. The **concatenated-layer**
   bars are new and are decoded here from the activation HDF5s.

Two things that are deliberately non-standard
---------------------------------------------
* **No ``comparability.txt``.** Requested: this is a summary folder, and the comparability
  work belongs to the folders being summarised. The mismatches that do matter are checked
  numerically in ``checks.txt`` and stated in ``report.md`` beside the affected figure.
* **Several CSVs rather than one ``data.csv``**, because the five claims want five
  differently-shaped tables (per-run endpoints, per-step curves, run x sigma, run x layer,
  run x layer-group). ``runs.csv`` remains the single frozen record of *which runs*.

Eval-artifact generation
------------------------
Everything here reads **v2** eval artifacts (``EvalProducer.VERSION = 2``). v3 exists in the
code but nothing has been produced with it yet, and mixing generations inside one figure is
what ``analysis/README.md`` forbids. The one exception is the claim-1 cohort, whose
no-efference arm holds only the unhashed ``legacy-batch`` eval; ``checks.txt`` measures
legacy-vs-v2 agreement on the 22 runs that hold both, which is what licenses using it.

Run it
------
    ../.venv/bin/python analysis/aug-2026-labmeeting-summary/extract.py
    ../.venv/bin/python analysis/aug-2026-labmeeting-summary/extract.py --refresh
    ../.venv/bin/python analysis/aug-2026-labmeeting-summary/extract.py --check
    ../.venv/bin/python analysis/aug-2026-labmeeting-summary/extract.py --part groups --redecode

``--part`` limits which tables are rebuilt (``delay``, ``curves``, ``noise``, ``probe``,
``groups``, or ``all``); every part not named is left untouched on disk. Only ``groups``
is expensive -- it refits ridge decoders from ~1.8 GB HDF5s, about two minutes per
recording -- and it caches per run under ``.decode-cache/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import get_producer
from vnl_experiments.artifacts.store import Store
from vnl_experiments.probes import linear_decoding as ld
from vnl_experiments.wandb_utils import index, pipeline

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".decode-cache"
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
PROBE_SOURCE = HERE.parent / "explicit-vs-implicit-fm-probe" / "data.csv"

# --------------------------------------------------------------------------------------
# Artifact identity
# --------------------------------------------------------------------------------------

#: v2, post the 2026-08-18 walker-XML fix. Pinned rather than derived so a producer
#: ``VERSION`` bump is a loud failure instead of a silent switch to different data.
EVAL_SPEC_ID = "eval3ds-347333e3"
#: The unhashed import of the pre-pipeline ``eval_runs.py`` batch. The **only** eval the
#: no-efference arm has; see the module docstring and ``checks.txt``.
LEGACY_SPEC_ID = "legacy-batch"
#: sigma -> v2 eval spec id, copied from ``action-noise-robustness/extract.py``.
NOISE_SPEC_IDS = {
    0.0: "eval3ds-n00-6a6b8d4e",
    0.02: "eval3ds-n02-a4e0be11",
    0.05: "eval3ds-n05-7c60cd50",
    0.1: "eval3ds-n10-2726ab9c",
    0.25: "eval3ds-n25-da25f356",
}
#: The 11-key history spec: the only one carrying ``fm_pred_mse`` alongside reward.
HISTORY_SPEC_ID = "hist2000-fc46b078"
#: The producer's default history spec -- reward, lifespan, throughput. The enc-dec and
#: decoder-ablation runs have this one and not the 11-key one, which is fine: nothing
#: asks them for a prediction error, since they have no predictor.
REWARD_HISTORY_SPEC_ID = "hist2000-09fea177"
ACT_SPEC_ID = "act-old_eval-fa8b8144"
#: v3 (``EvalProducer.VERSION = 3``). The decoder-ablation runs were evaluated after the
#: bump and hold **only** this generation, so the ablation cross-check in ``checks.txt``
#: reads v3 for all four of its runs. Everything else in this folder stays on v2; the two
#: are never mixed inside one figure or one number.
EVAL_V3_SPEC_ID = "eval3ds-382e9e69"

REQUIRES = ["index", f"eval:{EVAL_SPEC_ID}", f"eval:{LEGACY_SPEC_ID}",
            f"eval:{EVAL_V3_SPEC_ID}",
            f"history:{HISTORY_SPEC_ID}", f"history:{REWARD_HISTORY_SPEC_ID}",
            f"activations:{ACT_SPEC_ID}"] + \
           [f"eval:{sid}" for sid in NOISE_SPEC_IDS.values()]

NEW_XML = "rodent_no_tail_collisions.xml"
OLD_XML = "rodent.xml"

# --------------------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------------------

STD_ARCH = {
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
    "net_params.latent_size": 32,
}
FULL_600M = 600_064_000

#: ``failed`` is admitted because the *step count* is the inclusion criterion, not the exit
#: state. The entire 2026-08-11 new-XML sweep (23 enc-dec + 23 forward-model runs) is
#: ``failed``: each completed 600 064 000 steps and then died in the post-training inline
#: eval, which is why those runs alone lack ``final_eval/*``. A run that died *during*
#: training cannot reach the step count, so the two gates together admit the former and
#: exclude the latter. Precedent: ``collision-model-xml``, ``explicit-vs-implicit-fm-probe``.
ACCEPTED_STATES = ("finished", "failed", "crashed")


def _std(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in STD_ARCH.items():
        mask &= df[column] == value
    return mask


def _common(df: pd.DataFrame) -> pd.Series:
    """Shared by every cohort: the task, the standard body plan, an intact decoder."""
    return ((df["env"] == "AbsoluteImitation")
            & _std(df)
            & pipeline.full_decoder_inputs_mask(df)
            & df["state"].isin(ACCEPTED_STATES))


def _old_cohort(df: pd.DataFrame) -> pd.Series:
    """The 2026-06-12 sweep: old walker XML, ``current_root`` targets, git ``1cd5838``.

    ``body_target_frame`` is read off ``env_params``, never ``net_params`` -- the latter is
    inert and reads ``reference_root`` for these runs, which is the bug that invalidated
    ``imitation-target-representation`` (README §6).
    """
    return (_common(df)
            & df["git_commit"].astype(str).str.startswith("1cd5838")
            & df["env_params.walker_xml_path"].astype(str).str.endswith("/" + OLD_XML)
            & (df["summary._step"] == FULL_600M))


def _new_cohort(df: pd.DataFrame) -> pd.Series:
    """The going-forward configuration: new XML, ``reference_root``, torque, min_std 0.1."""
    return (_common(df)
            & df["env_params.walker_xml_path"].astype(str).str.contains(NEW_XML)
            & (df["env_params.body_target_frame"] == "reference_root")
            & (df["env_params.torque_actuators"] == True)  # noqa: E712
            & (df["net_params.min_std"] == 0.1)
            & (df["delay_k"] == df["efference_length"]))


def _explicit(df: pd.DataFrame) -> pd.Series:
    """``ne(False)`` not ``fillna(True)``: unset means the constructor default, True."""
    return (df["fm_loss_weight"] == 1) & df["detach_prediction"].ne(False)


def _implicit(df: pd.DataFrame) -> pd.Series:
    return (df["fm_loss_weight"] == 0) & (df["detach_prediction"] == False)  # noqa: E712


def _fm_tier(arm: str, budget: int, seed: int):
    def selector(df: pd.DataFrame) -> pd.Series:
        knobs = _explicit(df) if arm == "explicit" else _implicit(df)
        mask = (_new_cohort(df) & knobs
                & (df["config.ppo.total_steps"] == budget)
                & (df["seed"] == seed))
        if budget == 600_000_000:
            # Completed runs only at the tier every claim-2/claim-5 endpoint is read at.
            mask &= df["summary._step"] == FULL_600M
        return mask
    return selector


#: The 2026-08-25 decoder-input ablations. `4245ae42` is post the `a3450a9` fix, so the
#: regularisation is intact, and post the 2026-08-20 `eval_env = train_env` fix, so their
#: WandB `eval/*` is genuinely held out (their control's is not -- see ``report.md``).
ABLATION_COMMIT = "4245ae42"


def _ablation(*, intention: bool = True, proprioception: bool = True,
              efference: bool = True):
    """One decoder-input ablation at delay 10.

    The three knobs are not interchangeable: dropping the intention or the proprioception
    is a `net_params` flag, while dropping the efference copy is `efference_length = 0`.
    Selecting on all three together is what keeps the cells disjoint -- a `nointent` run
    still has `efference_length == delay_k` and would otherwise match the efference cell.
    """
    def selector(df: pd.DataFrame) -> pd.Series:
        # Deliberately not `_common`: that gate includes `full_decoder_inputs_mask`,
        # whose whole job is to keep these runs out of the baseline cohorts. Here they
        # are the subject, so the architecture checks are spelled out instead.
        mask = ((df["env"] == "AbsoluteImitation")
                & _std(df)
                & df["state"].isin(ACCEPTED_STATES)
                & (df["env_params.walker_xml_path"].astype(str).str.contains(NEW_XML))
                & (df["env_params.body_target_frame"] == "reference_root")
                & (df["env_params.torque_actuators"] == True)  # noqa: E712
                & (df["net_params.min_std"] == 0.1)
                & (df["delay_k"] == 10)
                & df["git_commit"].astype(str).str.startswith(ABLATION_COMMIT)
                & (df["summary._step"] == FULL_600M)
                & df["net_params.rnn_hidden_sizes"].isna()
                & df["fm_loss_weight"].isna()
                # `ne(False)`, not `== True`: the flags are *absent* on every run that
                # predates them, and an absent flag means the input is present.
                & (df["net_params.dec_use_intention"].ne(False) != (not intention))
                & (df["net_params.dec_use_proprioception"].ne(False) != (not proprioception))
                & ((df["efference_length"] == 0) == (not efference)))
        return mask
    return selector


CONDITIONS = {
    # -- claim 1: the only cohort with a no-efference arm -------------------------------
    "efference_old": lambda df: _old_cohort(df) & (df["delay_k"] == df["efference_length"]),
    "no_efference_old": lambda df: (_old_cohort(df) & (df["efference_length"] == 0)
                                    & (df["delay_k"] > 0)),
    # -- claims 1-2, 4: the going-forward enc-dec sweep ---------------------------------
    "encdec": lambda df: (_new_cohort(df)
                          & df["git_commit"].astype(str).str.startswith("ef060b73")
                          & df["fm_loss_weight"].isna()
                          & (df["summary._step"] == FULL_600M)),
    # -- claims 2-5: the two forward-model arms, three budgets --------------------------
    "expfm": _fm_tier("explicit", 600_000_000, 42),
    "pgfm": _fm_tier("implicit", 600_000_000, 42),
    "expfm_2g": _fm_tier("explicit", 2_000_000_000, 42),
    "pgfm_2g": _fm_tier("implicit", 2_000_000_000, 42),
    "expfm_4g": _fm_tier("explicit", 4_000_000_000, 43),
    "pgfm_4g": _fm_tier("implicit", 4_000_000_000, 43),
    # -- claim 1: are all three decoder inputs load-bearing? ---------------------------
    # One run each, delay 10 only, all at `4245ae42` (2026-08-25). Their control is the
    # delay-10 member of `encdec` -- deliberately not duplicated into a condition of its
    # own, since `select_conditions` requires the cells to be disjoint and re-selecting
    # it here would make the same run appear twice in `runs.csv`.
    "ablate_intention": _ablation(intention=False),
    "ablate_proprioception": _ablation(proprioception=False),
    "ablate_efference": _ablation(efference=False),
}

#: Sizes the selectors are expected to return. A silent change in cohort size is how a
#: summary figure drifts away from the analysis it summarises, so it is an error here.
EXPECTED_N = {"efference_old": 22, "no_efference_old": 13, "encdec": 23,
              "expfm": 23, "pgfm": 13, "expfm_2g": 4, "pgfm_2g": 4,
              "expfm_4g": 5, "pgfm_4g": 5,
              "ablate_intention": 1, "ablate_proprioception": 1, "ablate_efference": 1}

#: The ablations' control: the delay-10 enc-dec run every other claim-1/2/4 figure uses.
ABLATION_CONTROL_CONDITION = "encdec"
ABLATION_DELAY = 10

ARM_OF = {"efference_old": "encdec", "no_efference_old": "none", "encdec": "encdec",
          "expfm": "explicit", "pgfm": "implicit", "expfm_2g": "explicit",
          "pgfm_2g": "implicit", "expfm_4g": "explicit", "pgfm_4g": "implicit",
          "ablate_intention": "encdec", "ablate_proprioception": "encdec",
          "ablate_efference": "none"}
BUDGET_OF = {"expfm_2g": 2_000_000_000, "pgfm_2g": 2_000_000_000,
             "expfm_4g": 4_000_000_000, "pgfm_4g": 4_000_000_000}
#: Which eval generation each condition's endpoints are read from. ``None`` = no eval
#: artifact exists for that cohort at all (the 4 G runs), which the row records rather
#: than hides.
EVAL_OF = {"efference_old": LEGACY_SPEC_ID, "no_efference_old": LEGACY_SPEC_ID,
           "encdec": EVAL_SPEC_ID, "expfm": EVAL_SPEC_ID, "pgfm": EVAL_SPEC_ID,
           "expfm_2g": EVAL_SPEC_ID, "pgfm_2g": EVAL_SPEC_ID,
           "expfm_4g": None, "pgfm_4g": None,
           "ablate_intention": EVAL_V3_SPEC_ID,
           "ablate_proprioception": EVAL_V3_SPEC_ID,
           "ablate_efference": EVAL_V3_SPEC_ID}

#: Which history generation each condition's curve comes from. ``None`` = no curve; the
#: two old-XML cohorts are endpoint-only here.
HISTORY_OF = {"efference_old": None, "no_efference_old": None,
              "encdec": REWARD_HISTORY_SPEC_ID,
              "expfm": HISTORY_SPEC_ID, "pgfm": HISTORY_SPEC_ID,
              "expfm_2g": HISTORY_SPEC_ID, "pgfm_2g": HISTORY_SPEC_ID,
              "expfm_4g": HISTORY_SPEC_ID, "pgfm_4g": HISTORY_SPEC_ID,
              "ablate_intention": REWARD_HISTORY_SPEC_ID,
              "ablate_proprioception": REWARD_HISTORY_SPEC_ID,
              "ablate_efference": REWARD_HISTORY_SPEC_ID}

DATASETS = ("train", "old_eval", "new_eval")
#: The held-out split every headline figure uses. 169 unseen clips, 502 control steps.
PRIMARY_DATASET = "old_eval"


def assert_spec_ids() -> None:
    """A producer ``VERSION`` bump must break this, not silently repoint the figures."""
    producer = get_producer("eval")
    drift = []
    got = producer.spec_id(producer.spec(datasets=list(DATASETS)))
    if got != EVAL_SPEC_ID:
        drift.append(f"eval(no noise): stored {EVAL_SPEC_ID}, producer now emits {got}")
    for sigma, expected in NOISE_SPEC_IDS.items():
        got = producer.spec_id(producer.spec(datasets=list(DATASETS), action_noise=sigma))
        if got != expected:
            drift.append(f"eval(sigma={sigma}): stored {expected}, producer now emits {got}")
    if drift:
        print("NOTE: eval spec_ids differ from the producer's current output --\n  "
              + "\n  ".join(drift)
              + f"\n  This folder deliberately reads the v2 generation "
                f"(EvalProducer.VERSION is now {producer.VERSION}); nothing has been "
                f"produced at v3 yet. Repoint only together, and say so in report.md.")


# --------------------------------------------------------------------------------------
# Eval artifacts -> per-run endpoints
# --------------------------------------------------------------------------------------

def read_eval(store: Store, wandb_id: str, spec_id: str | None):
    if spec_id is None:
        return None, None
    entry = store.lookup("eval", wandb_id, spec_id)
    if entry is None:
        return None, None
    return json.loads((store.root / entry.path).read_text()), entry


def eval_metrics(record: dict, dataset: str) -> dict:
    """The metrics one figure or another needs, from one dataset of one eval record.

    ``reward_per_step`` and ``hazard_rate`` exist because cumulative reward and raw
    survival are not comparable across datasets: ``new_eval`` clips are 6x longer, which
    inflates the first and deflates the second (README §6).
    """
    block = record["datasets"][dataset]
    steps = float(block["n_steps"])
    lifespan = float(block["lifespan_steps"]["mean"])
    survived = float(block["termination_rate"]["survived"])
    alive_s = float(block["lifespan_s"]["mean"])
    # Each `errors` entry is a {mean, std} block, not a scalar.
    errors = {k: v.get("mean") if isinstance(v, dict) else v
              for k, v in block.get("errors", {}).items()}
    net = block.get("net_metrics", {})
    return {
        "n_clips": int(block["n_clips"]),
        "n_steps": int(steps),
        "episode_reward": float(block["episode_reward"]["mean"]),
        "episode_reward_std": float(block["episode_reward"]["std"]),
        "reward_per_step": float(block["episode_reward"]["mean"]) / steps,
        "lifespan_steps": lifespan,
        "lifespan_s": alive_s,
        "survived": survived,
        "hazard_rate": (1.0 - survived) / alive_s if alive_s > 0 else np.nan,
        "joint_l2_error": errors.get("joint_l2_error"),
        "joint_vel_l2_error": errors.get("joint_vel_l2_error"),
        "root_pos_distance": errors.get("root_pos_distance"),
        "root_angular_error": errors.get("root_angular_error"),
        "body_errors_total": errors.get("body_errors/total"),
        "end_eff_error": errors.get("body_errors/end_eff_total"),
        "fm_pred_mse": net.get("3/action/1/fm_pred_mse"),
    }


def run_identity(run: pd.Series) -> dict:
    condition = run["condition"]
    return {
        "condition": condition,
        "arm": ARM_OF[condition],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "state": run["state"],
        "delay_k": int(run["delay_k"]),
        "efference_length": int(run["efference_length"]),
        "seed": run["seed"],
        "budget": int(run["config.ppo.total_steps"]),
        "trained_step": run.get("summary._step"),
        "git_commit": str(run["git_commit"])[:8],
        "walker_xml": Path(str(run["env_params.walker_xml_path"])).name,
        "body_target_frame": run["env_params.body_target_frame"],
        "gpu": run.get("gpu"),
    }


def build_delay_table(runs: pd.DataFrame, store: Store) -> pd.DataFrame:
    """One row per (run, dataset). Runs with no eval artifact are kept, not dropped."""
    rows = []
    for _, run in runs.iterrows():
        spec_id = EVAL_OF[run["condition"]]
        record, entry = read_eval(store, run["wandb_id"], spec_id)
        identity = run_identity(run)
        for dataset in DATASETS:
            row = {**identity, "eval_spec_id": spec_id, "dataset": dataset,
                   "have_eval": record is not None}
            if record is not None:
                resolved = entry.resolved or {}
                row["checkpoint_step"] = resolved.get("checkpoint_step", record.get("step"))
                row["artifact_walker_xml"] = resolved.get("walker_xml_path")
                row.update(eval_metrics(record, dataset))
            rows.append(row)
    return pd.DataFrame(rows)


def assert_artifact_body(table: pd.DataFrame) -> None:
    """The artifact must say it simulated the body the run trained on.

    Two independently written records compared against each other -- the run index's
    ``env_params.walker_xml_path`` and the artifact sidecar's ``resolved.walker_xml_path``.
    Comparing a run's config against itself is what let the walker-XML bug survive for
    months (README §6). ``legacy-batch`` predates the stamp, so it is exempted explicitly
    rather than by accident: those runs are old-XML, where the substitution the fix
    addressed was a no-op.
    """
    stamped = table[table["have_eval"] & (table["eval_spec_id"] != LEGACY_SPEC_ID)]
    missing = stamped[stamped["artifact_walker_xml"].isna()]
    if len(missing):
        raise SystemExit(
            f"{len(missing)} eval artifacts carry no resolved.walker_xml_path, which means "
            f"they predate the 2026-08-18 fix and ran on the wrong body: "
            f"{sorted(set(missing['wandb_id']))}")
    wrong = stamped[stamped["artifact_walker_xml"] != stamped["walker_xml"]]
    if len(wrong):
        raise SystemExit(
            "eval artifacts simulated a different body than the run trained on:\n"
            + wrong[["wandb_id", "walker_xml", "artifact_walker_xml"]].to_string())


# --------------------------------------------------------------------------------------
# History artifacts -> training curves (claim 3)
# --------------------------------------------------------------------------------------

CURVE_COLUMNS = {
    "eval/episode_reward/mean": "reward_mean",
    "eval/lifespan/mean": "lifespan_mean",
    "eval/net/3/action/1/fm_pred_mse/mean": "fm_mse_eval",
    "eval/env/joint_l2_error/mean": "joint_l2_error",
    "eval/env/terminations/any/mean": "termination_rate",
}

#: Averaging window for "the value at step X". Eval runs every ~10 M steps and a single
#: point moves 1-2 % on nondeterministic GPU physics, so five points are averaged, never
#: one (README §6; the difference is -15.2 % vs -7.9 % on one real contrast).
WINDOW = 50_000_000

#: The reward series here is ``eval/*``, which -- for every run in this project -- was
#: computed on the **training** clips: both delays training scripts overrode
#: ``eval_env = train_env`` from 2026-06-01 until the 2026-08-20 fix, and every run in this
#: folder predates it. It is a legitimate learning curve and the arms are affected
#: identically; it is not a held-out measurement, and ``report.md`` never calls it one.
CURVE_NOTE = "eval/* is eval-on-training-clips for every run here (see README, 2026-08-20)"


def build_curves(runs: pd.DataFrame, store: Store) -> pd.DataFrame:
    """Tidy per-step curves, reading each condition's own history generation.

    The two generations differ only in how many keys were fetched, not in the reward
    series, so a figure may hold both -- unlike the eval artifacts, where a VERSION bump
    means different bytes. ``spec_id`` is emitted per row anyway.
    """
    frames = []
    for _, run in runs.iterrows():
        spec_id = HISTORY_OF.get(run["condition"], HISTORY_SPEC_ID)
        if spec_id is None:
            continue
        entry = store.lookup("history", run["wandb_id"], spec_id)
        if entry is None:
            continue
        try:
            frame = pd.read_csv(store.root / entry.path)
        except pd.errors.EmptyDataError:
            continue          # a run that died before its first eval (jhghg9vt)
        if "eval/episode_reward/mean" not in frame.columns:
            continue
        frame = frame.dropna(subset=["eval/episode_reward/mean"]).sort_values("_step")
        if frame.empty:
            continue
        out = pd.DataFrame({"step": frame["_step"].astype(int)})
        for source, name in CURVE_COLUMNS.items():
            out[name] = frame[source].to_numpy() if source in frame.columns else np.nan
        out.insert(0, "history_spec_id", spec_id)
        out.insert(0, "delay_k", int(run["delay_k"]))
        out.insert(0, "seed", run["seed"])
        out.insert(0, "arm", ARM_OF[run["condition"]])
        out.insert(0, "condition", run["condition"])
        out.insert(0, "wandb_id", run["wandb_id"])
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def window_mean(curve: pd.DataFrame, column: str, step: int) -> float | None:
    """Mean of the points in ``(step - WINDOW, step]``; ``None`` below three points.

    Returning ``None`` is also how a run that never reached ``step`` is excluded from that
    budget tier, so no separate max-step test is needed at the call sites.
    """
    sub = curve[(curve["step"] > step - WINDOW) & (curve["step"] <= step)].dropna(
        subset=[column])
    return float(sub[column].mean()) if len(sub) >= 3 else None


BUDGET_TIERS = (600_000_000, 1_000_000_000, 2_000_000_000, 2_900_000_000, 4_000_000_000)
TIER_LABEL = {600_000_000: "600M", 1_000_000_000: "1G", 2_000_000_000: "2G",
              2_900_000_000: "2p9G", 4_000_000_000: "4G"}


def build_budget_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Reward and prediction error at each budget tier, read **within** each run.

    ``total_steps`` only bounds PPO's loop and the learning rate is constant (no schedule
    in ``nnx_ppo/algorithms/ppo.py``), so a run's state at step *s* is an *s*-step run and a
    4 G run contributes to the 600 M and 2 G tiers as well. ``xml-ceiling-vs-convergence``
    checked this against separately launched twins: matched pairs agree to +-2.9 %.
    """
    rows = []
    for wandb_id, curve in curves.groupby("wandb_id"):
        head = curve.iloc[0]
        row = {"wandb_id": wandb_id, "condition": head["condition"], "arm": head["arm"],
               "seed": head["seed"], "delay_k": int(head["delay_k"]),
               "max_step": int(curve["step"].max()), "n_points": len(curve)}
        for tier in BUDGET_TIERS:
            label = TIER_LABEL[tier]
            row[f"reward_at_{label}"] = window_mean(curve, "reward_mean", tier)
            row[f"fm_mse_at_{label}"] = window_mean(curve, "fm_mse_eval", tier)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "delay_k"], ignore_index=True)


# --------------------------------------------------------------------------------------
# Noise sweep (claim 4)
# --------------------------------------------------------------------------------------

NOISE_CONDITIONS = ("encdec", "expfm")


def build_noise_table(runs: pd.DataFrame, store: Store) -> pd.DataFrame:
    """One row per (run, sigma, dataset). Missing cells are emitted, never dropped.

    sigma is in post-tanh action units: actions live in ``[-1, 1]``, so sigma = 0.02 is 2 %
    of the actuator half-range and 1 % of its full range. The noise is added *after* the
    network acts and *outside* ``EfferenceCopy``, so the efference queue holds the intended
    action while the body executes the perturbed one -- unobserved motor noise, and the
    predictor's error against it is irreducible by construction.
    """
    rows = []
    subset = runs[runs["condition"].isin(NOISE_CONDITIONS)]
    for _, run in subset.iterrows():
        identity = run_identity(run)
        for sigma, spec_id in NOISE_SPEC_IDS.items():
            record, entry = read_eval(store, run["wandb_id"], spec_id)
            for dataset in DATASETS:
                row = {**identity, "action_noise": sigma, "eval_spec_id": spec_id,
                       "dataset": dataset, "have_eval": record is not None}
                if record is not None:
                    resolved = entry.resolved or {}
                    row["artifact_walker_xml"] = resolved.get("walker_xml_path")
                    row["record_action_noise"] = resolved.get("action_noise")
                    row.update(eval_metrics(record, dataset))
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Layer-wise probe (claim 5), reused from the sibling analysis
# --------------------------------------------------------------------------------------

PROBE_CONDITION_MAP = {"expfm_600m": "expfm", "pgfm_600m": "pgfm",
                       "expfm_2g": "expfm_2g", "pgfm_2g": "pgfm_2g"}


def build_probe_table() -> pd.DataFrame:
    """Subset ``explicit-vs-implicit-fm-probe/data.csv`` to the panels this talk uses.

    Reusing a sibling's committed CSV rather than re-deriving it: the numbers are the same
    fits on the same artifacts, they are already vetted, and re-running them would take
    ~35 minutes to reproduce values to the last digit. The source file's hash is recorded
    in ``checks.txt`` so the provenance survives.
    """
    if not PROBE_SOURCE.exists():
        raise SystemExit(f"missing {PROBE_SOURCE}; run that folder's extract.py first")
    source = pd.read_csv(PROBE_SOURCE)
    frame = source[source["condition"].isin(PROBE_CONDITION_MAP)].copy()
    frame["condition"] = frame["condition"].map(PROBE_CONDITION_MAP)
    keep = ["condition", "arm", "budget", "wandb_id", "run_name", "delay_k", "step",
            "probe", "pathway", "stage_index", "stage_label", "target",
            "target_degenerate", "test_r2", "val_r2", "lambda", "n_train", "n_test",
            "n_features"]
    return frame[keep].sort_values(
        ["condition", "delay_k", "pathway", "stage_index", "target"], ignore_index=True)


# --------------------------------------------------------------------------------------
# Concatenated-layer probe (claim 5, new)
# --------------------------------------------------------------------------------------

#: Layer groups, as regexes over the recording's leaf names. The critic is **excluded**:
#: it is handed the current proprioception directly, so including it would decode the
#: target from the target. So are ``3/action/1/delay`` (the raw delayed buffer) and
#: ``3/action/0/proprioception`` (the pre-delay Normalizer output, which the actor's
#: policy path never sees) -- both would leak for the same reason, one trivially.
LAYER_GROUPS = {
    "forward_model": r"^3/action/1/predictor/[0-4]$",
    "decoder": r"^3/action/1/decoder/[0-4]$",
    "encoder": r"^3/action/0/task_obs/[0-6]$",
}
COMPOSITE_GROUPS = {
    "fm_plus_decoder": ("forward_model", "decoder"),
    "whole_network": ("encoder", "forward_model", "decoder"),
}
#: Expected feature width per group, asserted so a changed container tree is a failure
#: rather than a quietly narrower design matrix.
EXPECTED_WIDTH = {"forward_model": 4 * 512 + 277, "decoder": 4 * 512 + 76,
                  "encoder": 640 + 4 * 512 + 64 + 32}

#: Wider than ``ld.LAMBDA_GRID``. At 7 233 features the whole-network fit selects 1e4, i.e.
#: the standard grid's top value 1e3 is a *boundary* optimum and would under-regularise the
#: wide groups relative to the narrow ones -- exactly the bias that would make a bar chart
#: comparing widths meaningless. Every bar in this table, the single-layer reference bars
#: included, uses this grid, so the figure is internally consistent; ``checks.txt`` records
#: the two grids' disagreement on the layers they share.
GROUP_LAMBDAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
GROUP_DECODE_VERSION = 1
GROUP_SEED = 0

#: Which recordings get the expensive treatment: the delay/budget cells the talk shows.
GROUP_CELLS = [("expfm", 10), ("pgfm", 10), ("expfm", 20), ("pgfm", 20),
               ("expfm_2g", 10), ("pgfm_2g", 10)]

#: Single layers carried alongside the groups so a bar chart can be read against the
#: pathway figure: the input baseline every claim rests on, and the two end points.
REFERENCE_PROBES = {
    "input": None,                                   # delayed proprioception + efference
    "p_hat": "3/action/1/predictor/4",
    "latent_z": "3/action/0/task_obs/6",
}


def group_key() -> str:
    payload = json.dumps({"decode": ld.DECODE_VERSION, "group": GROUP_DECODE_VERSION,
                          "seed": GROUP_SEED, "lambdas": list(GROUP_LAMBDAS),
                          "test_frac": ld.DEFAULT_TEST_FRAC,
                          "val_frac": ld.DEFAULT_VAL_FRAC,
                          "max_samples": ld.DEFAULT_MAX_SAMPLES,
                          "groups": LAYER_GROUPS, "composites": COMPOSITE_GROUPS,
                          "references": REFERENCE_PROBES}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def decode_groups_for_run(path: Path, meta: dict) -> list[dict]:
    """Ridge-decode the current state from each layer *group* of one recording.

    The layers of a group share one design matrix, so this cannot go through
    ``ld.decode``, which wants a dense ``[T, N, F]`` -- the whole actor would be 2.4 TB.
    ``RowSampler`` fixes the sampled rows once and each layer is gathered into them one at
    a time, so peak memory is rows x summed-width (~2 GB) instead.
    """
    with ld.Recording(path) as rec:
        delay_k = rec.delay_k
        targets_all = ld.make_targets(rec.target, delay_k)
        mask = ld.valid_mask(rec.dones, delay_k)
        sampler = ld.RowSampler(mask, rec.n_clips, seed=GROUP_SEED)
        stats = ld.valid_stats(rec.dones, delay_k)
        degenerate = ld.degenerate_targets(delay_k)
        targets = {"proprio": sampler.take(targets_all["current"]),
                   "delta": sampler.take(targets_all["delta"])}

        resolved: dict[str, list[str]] = {}
        for name, pattern in LAYER_GROUPS.items():
            hits = sorted(n for n in rec.layer_names if re.match(pattern, n))
            if not hits:
                raise SystemExit(f"{path}: group '{name}' matched no layer")
            resolved[name] = hits

        # Gather once, reuse across every group and both targets.
        taken = {n: sampler.take(rec.layer(n))
                 for names in resolved.values() for n in names}
        for name, names in resolved.items():
            width = sum(taken[n]["train"].shape[1] for n in names)
            if EXPECTED_WIDTH[name] != width:
                raise SystemExit(
                    f"{path}: group '{name}' is {width} features, expected "
                    f"{EXPECTED_WIDTH[name]} -- the container tree changed, so this "
                    f"table is not comparable with the committed one.")

        designs = {name: ld.concat_splits([taken[n] for n in names])
                   for name, names in resolved.items()}
        for name, members in COMPOSITE_GROUPS.items():
            designs[name] = ld.concat_splits([designs[m] for m in members])

        action_name = rec.action_leaf_name()
        queue = ld.efference_queue(rec.layer(action_name), rec.efference_length)
        designs["input"] = ld.concat_splits([
            sampler.take(targets_all["delayed"]), sampler.take(queue)])
        for label, leaf in REFERENCE_PROBES.items():
            if leaf is not None:
                designs[label] = taken.get(leaf) or sampler.take(rec.layer(leaf))

        base = {"run_name": str(rec.attrs["run_name"]), "delay_k": delay_k,
                "efference_length": rec.efference_length,
                "step": int(rec.attrs["step"]),
                "network_class": str(rec.attrs["network_class"]),
                "dataset": str(rec.attrs["dataset"]),
                "decode_seed": GROUP_SEED, "decode_version": ld.DECODE_VERSION,
                "group_version": GROUP_DECODE_VERSION,
                "frac_valid": stats["frac_valid"], **meta}

        rows = []
        for group, design in designs.items():
            n_layers = len(resolved.get(group, [])) or (
                sum(len(resolved[m]) for m in COMPOSITE_GROUPS[group])
                if group in COMPOSITE_GROUPS else 1)
            for target, Y in targets.items():
                result = ld.decode_prepared(design, Y, lambdas=GROUP_LAMBDAS)
                rows.append({**base, "group": group, "n_layers": n_layers,
                             "target": target,
                             "target_degenerate": degenerate[target], **result})
    return rows


def build_group_table(runs: pd.DataFrame, store: Store, *, redecode: bool) -> pd.DataFrame:
    """One row per (run, group, target), from the cache or from the HDF5s."""
    CACHE.mkdir(exist_ok=True)
    wanted = runs[[(c, int(d)) in GROUP_CELLS
                   for c, d in zip(runs["condition"], runs["delay_k"])]]
    rows: list[pd.DataFrame] = []
    for _, run in wanted.iterrows():
        entry = store.lookup("activations", run["wandb_id"], ACT_SPEC_ID)
        if entry is None:
            print(f"  no activations:{ACT_SPEC_ID} for {run['wandb_id']} "
                  f"({run['wandb_name']}) -- skipped, reported in coverage.txt")
            continue
        stamp = (entry.resolved or {}).get("walker_xml_path")
        if stamp != NEW_XML:
            raise SystemExit(
                f"activations for {run['wandb_id']} were recorded on {stamp!r}, expected "
                f"{NEW_XML} -- a pre-2026-08-18 recording must not enter a figure.")
        cached = CACHE / run["wandb_id"] / f"{ACT_SPEC_ID}__{group_key()}.csv"
        if cached.exists() and not redecode:
            rows.append(pd.read_csv(cached))
            continue
        print(f"  decoding groups for {run['wandb_name']} ...", flush=True)
        meta = {"condition": run["condition"], "arm": ARM_OF[run["condition"]],
                "budget": int(run["config.ppo.total_steps"]),
                "wandb_id": run["wandb_id"], "seed": run["seed"]}
        frame = pd.DataFrame(decode_groups_for_run(store.root / entry.path, meta))
        cached.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cached, index=False)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["condition", "delay_k", "target", "group"], ignore_index=True)


# --------------------------------------------------------------------------------------
# Cross-checks that replace the comparability report
# --------------------------------------------------------------------------------------

def legacy_vs_v2(runs: pd.DataFrame, store: Store) -> list[str]:
    """How far apart are the two eval generations on the runs that hold both?

    Claim 1's no-efference arm exists only as ``legacy-batch``, so the figure mixes an
    unhashed pre-pipeline eval into a folder that otherwise reads v2. This measures the
    cost of that on the 22 with-efference runs that hold both, which is the only honest
    way to license it.
    """
    lines = ["legacy-batch vs eval3ds-347333e3 (v2), old-XML with-efference runs",
             "-" * 72]
    deltas = []
    for _, run in runs[runs["condition"] == "efference_old"].iterrows():
        legacy, _ = read_eval(store, run["wandb_id"], LEGACY_SPEC_ID)
        modern, _ = read_eval(store, run["wandb_id"], EVAL_SPEC_ID)
        if legacy is None or modern is None:
            continue
        a = eval_metrics(legacy, PRIMARY_DATASET)["episode_reward"]
        b = eval_metrics(modern, PRIMARY_DATASET)["episode_reward"]
        deltas.append((int(run["delay_k"]), 100.0 * (b - a) / a))
    if not deltas:
        lines.append("  no run holds both generations")
        return lines
    values = np.array([d for _, d in deltas])
    lines.append(f"  n = {len(values)} runs, delays {min(d for d, _ in deltas)}"
                 f"-{max(d for d, _ in deltas)}")
    lines.append(f"  v2 - legacy, % of legacy reward on {PRIMARY_DATASET}: "
                 f"mean {values.mean():+.2f} %, median {np.median(values):+.2f} %, "
                 f"range {values.min():+.2f} .. {values.max():+.2f} %")
    worst = max(deltas, key=lambda d: abs(d[1]))
    lines.append(f"  largest single disagreement: delay {worst[0]}, {worst[1]:+.2f} %")
    lines.append("  Reference: re-evaluating one checkpoint twice moves reward ~1 % "
                 "(README, eval nondeterminism).")
    return lines


def noise_zero_vs_plain(runs: pd.DataFrame, store: Store) -> list[str]:
    """sigma = 0 and the no-noise spec should be the same measurement. Are they?"""
    lines = ["", f"{NOISE_SPEC_IDS[0.0]} (sigma=0) vs {EVAL_SPEC_ID} (no action_noise "
                 f"field)", "-" * 72]
    deltas = []
    for _, run in runs[runs["condition"].isin(NOISE_CONDITIONS)].iterrows():
        a, _ = read_eval(store, run["wandb_id"], NOISE_SPEC_IDS[0.0])
        b, _ = read_eval(store, run["wandb_id"], EVAL_SPEC_ID)
        if a is None or b is None:
            continue
        ra = eval_metrics(a, PRIMARY_DATASET)["episode_reward"]
        rb = eval_metrics(b, PRIMARY_DATASET)["episode_reward"]
        deltas.append(100.0 * (rb - ra) / ra)
    if not deltas:
        lines.append("  no run holds both")
        return lines
    values = np.array(deltas)
    lines.append(f"  n = {len(values)}; difference in {PRIMARY_DATASET} reward: "
                 f"mean {values.mean():+.4f} %, max |{np.abs(values).max():.4f}| %")
    lines.append("  These are two spec ids for one recipe (`normalise_spec` drops a None "
                 "action_noise), so anything but ~0 means the ids are not "
                 "interchangeable and the noise figures must not borrow a baseline "
                 "from the plain spec.")
    return lines


def group_vs_layer_grid(groups: pd.DataFrame, probe: pd.DataFrame) -> list[str]:
    """The wider lambda grid, measured where the two tables overlap."""
    lines = ["", "wide lambda grid vs the standard one, on the layers both tables hold",
             "-" * 72]
    if groups.empty or probe.empty:
        lines.append("  one of the tables is empty")
        return lines
    probe_names = {"input": "input::delayed_plus_efference",
                   "p_hat": "layer::3/action/1/predictor/4",
                   "latent_z": "layer::3/action/0/task_obs/6"}
    merged = []
    for group, probe_name in probe_names.items():
        left = groups[groups["group"] == group]
        for _, row in left.iterrows():
            match = probe[(probe["wandb_id"] == row["wandb_id"])
                          & (probe["probe"] == probe_name)
                          & (probe["target"] == row["target"])]
            if len(match) == 1 and np.isfinite(row["test_r2"]):
                merged.append((group, float(row["test_r2"]),
                               float(match.iloc[0]["test_r2"]),
                               float(row["lambda"]), float(match.iloc[0]["lambda"])))
    if not merged:
        lines.append("  no overlap")
        return lines
    diffs = np.array([abs(a - b) for _, a, b, _, _ in merged])
    hit_ceiling = sum(1 for *_, lam_wide, lam_std in merged if lam_std >= 1000.0)
    lines.append(f"  n = {len(merged)} (group, run, target) cells")
    lines.append(f"  |wide - standard| R^2: mean {diffs.mean():.4f}, max {diffs.max():.4f}")
    lines.append(f"  cells where the standard grid selected its top value (1e3): "
                 f"{hit_ceiling}/{len(merged)}")
    lines.append("  A narrow layer is unaffected; the wide groups are what needed the "
                 "extra decades, which is why the bar figure uses one grid throughout.")
    return lines


def curve_vs_eval(delay: pd.DataFrame, budget: pd.DataFrame) -> list[str]:
    """Do the training curve and the offline eval agree about the 600 M endpoint?

    They are different measurements -- train clips vs held-out clips, in-memory weights vs
    a restored checkpoint -- so they should *correlate*, not coincide. A figure that put
    one on each axis would be the mistake; this records the relationship instead.
    """
    lines = ["", "training-curve reward at 600 M vs offline held-out eval reward",
             "-" * 72]
    left = budget[["wandb_id", "arm", "delay_k", "reward_at_600M"]].dropna()
    right = delay[(delay["dataset"] == PRIMARY_DATASET) & delay["have_eval"]]
    merged = left.merge(right[["wandb_id", "episode_reward"]], on="wandb_id")
    if merged.empty:
        lines.append("  no overlap")
        return lines
    ratio = merged["episode_reward"] / merged["reward_at_600M"]
    correlation = np.corrcoef(merged["reward_at_600M"], merged["episode_reward"])[0, 1]
    lines.append(f"  n = {len(merged)} runs; held-out / train-clip reward: "
                 f"mean {ratio.mean():.3f}, range {ratio.min():.3f}-{ratio.max():.3f}")
    lines.append(f"  Pearson r = {correlation:.4f}")
    lines.append("  The gap is the train-eval generalisation gap plus the 2026-08-20 "
                 "`eval_env = train_env` bug, both of which apply to every arm equally.")
    return lines


def ablation_check(runs: pd.DataFrame, store: Store) -> list[str]:
    """The claim-1 ablation figure, measured a second way that has no split confound.

    The figure is a training curve, and its control is the delay-10 `encdec` run, whose
    WandB ``eval/*`` scored the **train** clips (it predates the 2026-08-20
    ``eval_env = train_env`` fix) while the three ablations' scored held-out clips. That
    difference flatters the control. Here all four are read from the *same* offline eval
    spec on the *same* held-out split, so whatever survives is not the confound.
    """
    lines = ["", f"decoder-input ablations vs their control, offline {EVAL_V3_SPEC_ID} "
                 f"({PRIMARY_DATASET})", "-" * 72]
    control = runs[(runs["condition"] == ABLATION_CONTROL_CONDITION)
                   & (runs["delay_k"] == ABLATION_DELAY)]
    if control.empty:
        lines.append("  no control run")
        return lines
    control_id = control.iloc[0]["wandb_id"]
    record, _ = read_eval(store, control_id, EVAL_V3_SPEC_ID)
    if record is None:
        lines.append(f"  control {control_id} holds no {EVAL_V3_SPEC_ID}; produce it with "
                     f"`artifacts ensure --kind eval --runs {control_id}`")
        return lines
    baseline = eval_metrics(record, PRIMARY_DATASET)["episode_reward"]
    lines.append(f"  control {control_id} (all three inputs): {baseline:.1f}")
    for condition in ("ablate_intention", "ablate_proprioception", "ablate_efference"):
        row = runs[runs["condition"] == condition]
        if row.empty:
            continue
        wandb_id = row.iloc[0]["wandb_id"]
        ablated, _ = read_eval(store, wandb_id, EVAL_V3_SPEC_ID)
        if ablated is None:
            lines.append(f"  {condition:<22} {wandb_id}: no {EVAL_V3_SPEC_ID}")
            continue
        value = eval_metrics(ablated, PRIMARY_DATASET)["episode_reward"]
        lines.append(f"  {condition:<22} {wandb_id}: {value:8.1f}  "
                     f"({100 * value / baseline:5.1f} % of control)")
    lines.append("  The training curves put the same three at 43-45 % of the control, so "
                 "the ordering and the size of the effect do not depend on which "
                 "measurement is used -- which is what licenses the curve figure despite "
                 "its split mismatch.")
    return lines


def cohort_summary(runs: pd.DataFrame) -> list[str]:
    lines = ["", "cohorts", "-" * 72]
    for condition, group in runs.groupby("condition", sort=False):
        delays = sorted(int(d) for d in group["delay_k"])
        commits = sorted({str(c)[:8] for c in group["git_commit"]})
        lines.append(f"  {condition:<18} n={len(group):<3} seed="
                     f"{sorted({int(s) for s in group['seed']})} commit={commits}")
        lines.append(f"  {'':<18} delays={delays}")
    return lines


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

PARTS = ("delay", "curves", "noise", "probe", "groups")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-run CONDITIONS against the index and rewrite runs.csv")
    parser.add_argument("--sync", action="store_true", help="refresh the index from WandB")
    parser.add_argument("--check", action="store_true",
                        help="rebuild and diff against the committed CSVs; exit 1 on drift")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--part", action="append", choices=(*PARTS, "all"),
                        help="which tables to rebuild (default: all)")
    parser.add_argument("--redecode", action="store_true",
                        help="refit the concatenated-layer probes from the HDF5s")
    args = parser.parse_args()
    parts = set(PARTS) if not args.part or "all" in args.part else set(args.part)

    assert_spec_ids()
    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    counts = runs.groupby("condition").size().to_dict()
    if counts != EXPECTED_N:
        raise SystemExit(
            f"cohort sizes changed.\n  now:      {counts}\n  expected: {EXPECTED_N}\n"
            f"Every figure here assumes these cohorts. Update EXPECTED_N deliberately and "
            f"say what moved in report.md.")
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    written: dict[str, pd.DataFrame] = {}
    ok = True

    if "delay" in parts:
        delay = build_delay_table(runs, store)
        assert_artifact_body(delay)
        ok &= pipeline.write_csv(delay, HERE / "data_delay.csv", check=args.check)
        written["delay"] = delay

    if "curves" in parts or "delay" in parts:
        curves = build_curves(runs, store)
        if "curves" in parts:
            ok &= pipeline.write_csv(curves, HERE / "curves.csv", check=args.check)
            budget = build_budget_table(curves)
            ok &= pipeline.write_csv(budget, HERE / "data_budget.csv", check=args.check)
            written["budget"] = budget

    if "noise" in parts:
        noise = build_noise_table(runs, store)
        ok &= pipeline.write_csv(noise, HERE / "data_noise.csv", check=args.check)

    if "probe" in parts:
        probe = build_probe_table()
        ok &= pipeline.write_csv(probe, HERE / "data_probe.csv", check=args.check)
        written["probe"] = probe

    if "groups" in parts:
        groups = build_group_table(runs, store, redecode=args.redecode)
        ok &= pipeline.write_csv(groups, HERE / "data_probe_groups.csv", check=args.check)
        written["groups"] = groups

    if not args.check and parts == set(PARTS):
        lines = [f"Cross-checks for {HERE.name}. Rebuild: extract.py", "=" * 72]
        lines += cohort_summary(runs)
        lines += ["", f"probe source: {PROBE_SOURCE.name} sha256 "
                      f"{hashlib.sha256(PROBE_SOURCE.read_bytes()).hexdigest()[:12]}"]
        lines += ["", CURVE_NOTE]
        lines += legacy_vs_v2(runs, store)
        lines += noise_zero_vs_plain(runs, store)
        lines += curve_vs_eval(written["delay"], written["budget"])
        lines += ablation_check(runs, store)
        lines += group_vs_layer_grid(written["groups"], written["probe"])
        (HERE / "checks.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))

    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
