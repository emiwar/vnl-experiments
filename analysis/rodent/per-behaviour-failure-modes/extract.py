"""Which behaviours does position-control-without-proprioception actually fail at?

Parent question: [`../efference-copy-vs-proprioception/`](../efference-copy-vs-proprioception/)
found that with the decoder's proprioception stream ablated, a **position**-actuated policy
with a 2-step efference copy reaches 88 % of the proprioception-intact baseline on held-out
5 s clips, while the torque arm reaches 33 %. Taken at face value that puts the
encoder-decoder premise in doubt: if handing intentions straight to local joint servos
nearly works, the decoder may not be doing much.

A cross-clip mean is the wrong statistic to settle that on, because the clip set is
behaviourally heterogeneous. This folder decomposes the same numbers three ways:

1. **by reward term and termination reason** -- is the deficit a tracking deficit or a
   falling-over deficit, and which of the ten reward terms carries it;
2. **by behaviour** -- the hypothesis under test is that the ablated position policy sits
   at the intact ceiling on sedentary behaviour and far below it on locomotion and rearing,
   so that 88 % is a mixture of ~100 % and ~70 % rather than a uniform 88 %;
3. **at the moment of failure** -- what the reference animal was doing when the episode
   terminated, which is not the same question as which clips scored badly.

Behaviour labels come from ``behaviour_labels.py`` in this folder; read its docstring
before touching anything here, because the two clip sets are labelled by unrelated means
and their taxonomies are different partitions.

The interpretation rule, fixed before looking at the numbers
------------------------------------------------------------
``pos_nointent`` is in the cohort for a reason. It has proprioception but no imitation
target, so it bounds the reward available for merely standing around in a plausible pose.
**If the task-blind policy also scores near the intact ceiling on the sedentary bins, then
"near-ceiling on sedentary behaviour" is a fact about the reward function and not about the
policy**, and the finding is that the reward is easy to collect while still -- not that the
servos substitute for proprioception. That reading is committed to here rather than chosen
once the bars are drawn.

Staging
-------
This script is written to be useful before the expensive data exists, because the cohort's
checkpoints live on the cluster and the per-clip artifacts need a round-trip.

* **Stage A (this version)** needs the run index only. The inline end-of-training eval
  already pushes per-reason ``termination_rate``, all ten ``reward_terms`` and six
  ``errors`` into each run's WandB summary, for all three datasets -- so questions (1)
  above is answerable with no new compute at all.
* **Stage B** adds ``clips.csv`` / ``behaviour.csv`` from ``eval`` artifacts produced with
  the ``per_clip`` spec key, which answers (2).
* **Stage C** adds the per-frame behaviour join from ``trace`` artifacts, which answers
  (3).

``REQUIRES`` grows at each stage, so ``coverage.txt`` states the gap rather than hiding it.

Run it
------
    ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/extract.py
    ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/extract.py --sync --refresh
    ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/extract.py --check

Frozen is the default: ``data.csv`` is rebuilt from exactly the runs in the committed
``runs.csv``. This is the only script in the folder that reads the index or the artifact
store; ``plot.py`` reads the CSVs it writes and nothing else.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: Stage A needs nothing but the index. Stage B appends `eval:<per-clip spec>`.
REQUIRES = ["index"]

#: The eval spec the producer currently defaults to, pinned so this folder keeps resolving
#: these exact files after a future VERSION bump (analysis/README.md §2). Used for exactly
#: one run -- the 2026-08-11 cross-check, which died in its inline eval and so has no
#: `final_eval/*` keys.
EVAL_SPEC_ID = "eval3ds-382e9e69"

#: The per-clip eval spec (`EvalProducer` with `per_clip=True`) and the `new_eval` trace
#: spec, both pinned. Neither changed `VERSION`, so these ids are stable; pinning them
#: means a future bump leaves this folder resolving its own files (analysis/README.md §2).
EVAL_PC_SPEC_ID = "eval3ds-pc-fac96053"
TRACE_SPEC_ID = "trace-new_eval-657f97da"

#: How far back from a termination to look when asking what the animal was doing when it
#: failed. Half a second is ~2 MotionMapper transitions at this session's rate, so the
#: modal behaviour over the window is a description of the approach to the failure rather
#: than of the single frame it happened on -- which at 100 Hz is noise.
FAILURE_WINDOW_S = 0.5

XML_ROOT = ("/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/vnl-playground/"
            "vnl_playground/tasks/rodent/xmls")
NEW_XML = f"{XML_ROOT}/rodent_no_tail_collisions.xml"

#: The clip set every run must share for per-clip pairing to mean anything. See
#: `_pairable` below.
REF_DATA = ("/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/vnl-playground/"
            "vnl_playground/tasks/rodent/reference_data/reference_clips.h5")
CLIP_LENGTH = 250

#: ``eval_env = train_env`` until this date, so ``eval/*`` logged before it is a
#: *train-split* number. This folder reads `final_eval/*` and offline artifacts rather than
#: the training curves, so the gate is not strictly needed for the metric -- it is kept
#: because it also pins the cohort to one experimental era, and dropping it would let in
#: runs whose held-out split means something different.
HELD_OUT_EVAL_SINCE = pd.Timestamp("2026-08-20", tz="UTC")

#: The 2026-08-11 sweep's commit. Its runs trained the full 600 M steps and then died in
#: `run_final_eval`, so `state == "failed"` and there are no `final_eval/*` keys.
AUG11_COMMIT = "ef060b73c049f0242100f38ecf427083c2cf8ab1"

#: Identical to the parent folder's list, and for the same reasons -- see its docstring.
#: `total_steps` and `weight_decay` are deliberately absent.
PPO_PARAMS = {
    "clip_range": 0.2,
    "combine_advantages": False,
    "critic_loss_weight": 1,
    "discounting_factor": 0.95,
    "gae_lambda": 0.95,
    "gradient_clipping": 1,
    "learning_rate": 0.0001,
    "n_envs": 4096,
    "n_epochs": 4,
    "n_minibatches": 8,
    "normalize_advantages": True,
    "rollout_length": 20,
}

DATASETS = ("train", "old_eval", "new_eval")

#: Every reward term `AbsoluteImitation` configures, in the order they are reported.
#: `joints_vel` and `bodies_pos` have weight 0 in this env config and are kept precisely so
#: that shows up as a zero column rather than as an absence a reader has to wonder about.
REWARD_TERMS = ("root_pos", "root_quat", "joints", "joints_vel", "bodies_pos", "end_eff",
                "torso_z_range", "control_cost", "control_diff_cost", "energy_cost")

#: The six per-step error metrics the eval whitelists, mapped to short column names.
ERRORS = {
    "root_pos_distance": "root_pos_m",
    "root_angular_error": "root_angular_deg",
    "joint_l2_error": "joint_l2",
    "joint_vel_l2_error": "joint_vel_l2",
    "body_errors/total": "body_total_m",
    "body_errors/end_eff_total": "body_end_eff_m",
}

#: The four failure terminations, plus the two derived rates. `survived` is *not* only
#: "did not fail": the env sets `done` on clip-end truncation without setting any
#: `terminations/*` flag, so `survived = 1 - any` means "reached the end of the clip
#: without failing" (vnl-playground rodent/imitation.py:187-224).
TERMINATIONS = ("root_too_far", "root_too_rotated", "pose_error", "nan_termination")

#: Control steps per second: `ctrl_dt = 0.01`. Used for the hazard rate.
STEPS_PER_S = 100.0


# --------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------

def has_tag(tags: object, tag: str) -> bool:
    """Whether ``tag`` is in an index ``tags`` cell, which may be a list or its repr."""
    if isinstance(tags, str):
        return tag in [t.strip() for t in tags.strip("[]").replace("'", "").split(",")]
    if isinstance(tags, (list, tuple)):
        return tag in tags
    return False


def sound_training_mask(df: pd.DataFrame) -> pd.Series:
    """Exclude runs known to have trained wrongly: the ``BUG`` tag, or a broken commit.

    The union of the project's own marker and the code-derived test for the 2026-08-21/24
    runs that trained with ``entropy_weight``/``kl_weight``/``min_std`` silently zeroed.
    Same rule as the parent folder, where it is shown the two agree exactly.
    """
    return ~(df["tags"].apply(has_tag, tag="BUG")
             | df["git_commit"].isin(pipeline.UNREGULARIZED_COMMITS))


def completed_training(df: pd.DataFrame) -> pd.Series:
    """Runs that trained to their full budget, ``state`` notwithstanding.

    ``state == "finished"`` would drop the 2026-08-11 sweep, which trained all 600 M steps
    and then died in the end-of-training eval. Copied from
    ``../position-control-open-loop/extract.py``, where the gate is shown to be exactly
    discriminating on this XML and frame.
    """
    reached = df["summary._step"] >= df["config.ppo.total_steps"]
    return (df["state"] == "finished") | reached.fillna(False)


def _pairable(df: pd.DataFrame) -> pd.Series:
    """The precondition for comparing clip *i* of one run against clip *i* of another.

    ``ReferenceClips.split()`` is ``RandomState(0).permutation(n_clips)`` over
    ``qpos.shape[0] // clip_length`` clips of the file at ``reference_data_path``, with
    ``keep_clips_idx`` selecting a subset first. If any of those three differ between two
    runs, their clip axes are different clips in a different order and every paired figure
    downstream is silently wrong -- so this is a *gate*, not a comparability note. It
    happens to be satisfied by every run in the cohort, which is why it can be a gate.
    """
    return ((df["env_params.reference_data_path"] == REF_DATA)
            & (df["env_params.clip_length"] == CLIP_LENGTH)
            & df["env_params.keep_clips_idx"].isna())


def _standard(df: pd.DataFrame) -> pd.Series:
    """Everything that must hold for a run to be in this cohort at all.

    The parent folder's ``_standard``, plus ``completed_training`` and ``_pairable``.
    Expressed on ``env_params``/``net_params`` rather than on tags or the run name: the
    cluster working copy drifts from the committed script, so the logged config is the only
    record of what ran (analysis/README.md §6).
    """
    mask = (
        (df["env"] == "AbsoluteImitation")
        & (df["env_params.walker_xml_path"] == NEW_XML)
        & (df["env_params.body_target_frame"] == "reference_root")
        & (df["net_params.network_class"] == "RodentEncDecDelays")
        & (df["net_params.enc_hidden_sizes"] == "[512, 512, 512, 512]")
        & (df["net_params.dec_hidden_sizes"] == "[512, 512, 512, 512]")
        & (df["net_params.critic_hidden_sizes"] == "[1024, 1024]")
        & (df["net_params.latent_size"] == 32)
        & (df["net_params.kl_weight"] == 0.001)
        & (df["net_params.entropy_weight"] == 0.01)
        & (df["net_params.min_std"] == 0.1)
        & (df["seed"] == 42)
        & sound_training_mask(df)
        & completed_training(df)
        & _pairable(df)
        & (pd.to_datetime(df["created_at"], utc=True) >= HELD_OUT_EVAL_SINCE)
    )
    for key, value in PPO_PARAMS.items():
        mask &= df[f"config.ppo.{key}"] == value
    return mask


def _mode(df: pd.DataFrame, torque: bool) -> pd.Series:
    return df["env_params.torque_actuators"] == torque


def _noproprio(df: pd.DataFrame, eff: int) -> pd.Series:
    """Proprioception ablated, intention kept, at one efference-queue length."""
    return ((df["net_params.dec_use_proprioception"] == False)  # noqa: E712
            & (df["net_params.dec_use_intention"] != False)     # noqa: E712
            & (df["net_params.efference_length"] == eff))


def _intact(df: pd.DataFrame) -> pd.Series:
    """All three decoder streams, no delay, no efference copy: the ceiling."""
    return (pipeline.full_decoder_inputs_mask(df)
            & (df["delay_k"] == 0)
            & (df["net_params.efference_length"] == 0))


def _delayed(df: pd.DataFrame, k: int) -> pd.Series:
    """Proprioception present but delayed by ``k`` steps, with a matching efference copy.

    This is the *reward-matched control*: at k = 10 a torque policy scores within a few
    percent of the ablated position policy on held-out 5 s clips, so the two can be asked
    whether equal means imply equal behaviour profiles.
    """
    return (pipeline.full_decoder_inputs_mask(df)
            & (df["delay_k"] == k)
            & (df["net_params.efference_length"] == k))


def _nointent(df: pd.DataFrame) -> pd.Series:
    """Intention ablated instead: the task-blind floor. See the docstring's rule."""
    return df["net_params.dec_use_intention"] == False  # noqa: E712


def _aug11_torque_delay10(df: pd.DataFrame) -> pd.Series:
    """The one 2026-08-11 run kept as a cross-check, selected against a weaker record.

    This run predates ``net_params.network_class`` / ``efference_length`` /
    ``dec_use_*`` being logged at all, so ``_standard`` cannot select it and the only
    thing separating it from the ``RodentForwardModel`` run at the same delay in the same
    launch is the ``EncDec`` tag. That is a **documented departure** from "names and tags
    are not evidence": the tag is the only surviving record, so the choice is between
    using it and dropping the run. The run is therefore kept in its own condition and
    excluded from every primary figure -- it exists to answer "does a three-week-older
    torque delay-10 run have the same behaviour profile?", and nothing rests on it.

    Everything that *is* logged is still checked: the env, body, frame, actuator, delay,
    regularisation, seed, PPO settings, clip set, and that it reached ``total_steps``.
    """
    mask = (
        (df["git_commit"] == AUG11_COMMIT)
        & df["tags"].apply(has_tag, tag="EncDec")
        & (df["env"] == "AbsoluteImitation")
        & (df["env_params.walker_xml_path"] == NEW_XML)
        & (df["env_params.body_target_frame"] == "reference_root")
        & (df["env_params.torque_actuators"] == True)   # noqa: E712
        & (df["delay_k"] == 10)
        & (df["net_params.kl_weight"] == 0.001)
        & (df["net_params.entropy_weight"] == 0.01)
        & (df["net_params.min_std"] == 0.1)
        & (df["seed"] == 42)
        & sound_training_mask(df)
        & completed_training(df)
        & _pairable(df)
    )
    for key, value in PPO_PARAMS.items():
        mask &= df[f"config.ppo.{key}"] == value
    return mask


# One entry per experimental cell; `select_conditions` raises if a run matches two.
#
# `_standard` carries `created_at >= 2026-08-20`, so the aug-11 cell cannot collide with
# any of the others even though it is selected by a different route.
CONDITIONS = {
    # the subject: the 88 %-of-intact result being decomposed
    "pos_noproprio_eff2": lambda df: _standard(df) & _mode(df, False) & _noproprio(df, 2),
    # the floor the efference queue improves on -- what the queue itself buys, per behaviour
    "pos_noproprio_eff0": lambda df: _standard(df) & _mode(df, False) & _noproprio(df, 0),
    # the ceiling, and the per-clip reference every paired figure is taken against
    "pos_intact": lambda df: _standard(df) & _mode(df, False) & _intact(df),
    "torque_intact": lambda df: _standard(df) & _mode(df, True) & _intact(df),
    # the same ablation under the other actuator: isolates actuator from ablation
    "torque_noproprio_eff2": lambda df: _standard(df) & _mode(df, True) & _noproprio(df, 2),
    # reward-matched control: same mean reward as the subject, different mechanism
    "torque_delay10": lambda df: _standard(df) & _mode(df, True) & _delayed(df, 10),
    # task-blind floors (see the interpretation rule in the module docstring). One per
    # actuator: the control costs differ by ~6x between them, so a single floor borrowed
    # across actuators would not bound the right thing.
    "pos_nointent": lambda df: _standard(df) & _mode(df, False) & _nointent(df),
    "torque_nointent": lambda df: _standard(df) & _mode(df, True) & _nointent(df),
    # cross-check only, excluded from primary figures
    "torque_delay10_aug11": _aug11_torque_delay10,
}

# Must be constant for the comparison to be fair; comparability.txt flags what varies.
# The manipulations are excluded: `env_params.torque_actuators`,
# `net_params.efference_length`, `net_params.dec_use_*` and `delay_k`.
#
# `env_params.reference_data_path` / `clip_length` / `keep_clips_idx` appear here *as well
# as* in `_pairable`. That is deliberate duplication: the gate stops a non-pairable run
# entering, and the invariant makes the fact that they are all equal visible in a committed
# file rather than only implied by the gate not having fired.
INVARIANTS = [
    "env",
    "env_params.walker_xml_path",
    "env_params.body_target_frame",
    "env_params.ctrl_dt",
    "env_params.mocap_hz",
    "env_params.clip_length",
    "env_params.reference_data_path",
    "env_params.keep_clips_idx",
    "env_params.termination_criteria.root_too_far.max_distance",
    "env_params.termination_criteria.root_too_rotated.max_degrees",
    "env_params.termination_criteria.pose_error.max_l2_error",
    "net_params.network_class",
    "net_params.enc_hidden_sizes",
    "net_params.dec_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.latent_size",
    "net_params.kl_weight",
    "net_params.entropy_weight",
    "net_params.min_std",
    "config.ppo.total_steps",
    "summary._step",
    "seed",
    "git_commit",
    "cuda_version",
    "os",
    "gpu",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "repos.nnx_ppo.dirty",
    "repos.vnl_playground.dirty",
] + [f"config.ppo.{k}" for k in PPO_PARAMS]


# --------------------------------------------------------------------------------------
# reading one run's measurements
# --------------------------------------------------------------------------------------

def _num(value: object) -> float | None:
    """A finite float, or ``None`` -- so a NaN summary cell reads as absent, not as 0."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load_eval(store: Store, run: pd.Series) -> dict | None:
    """The pinned offline ``eval`` record for one run, or ``None`` if it has none.

    Only consulted for runs with no inline eval, which in this cohort is the single
    2026-08-11 cross-check run. Two independently written records are compared before the
    numbers are used -- the artifact's ``resolved.walker_xml_path`` and its ``env_class``
    and ``step`` against the run's own ``env_params`` and ``summary._step``. This is the
    ``assert_artifact_body`` pattern from analysis/README.md, and it is not ceremony: an
    eval that silently re-simulated a run on the local default body is exactly how
    ``collision-model-xml`` earned a retraction, and comparing a run's config against
    itself could not have caught it.
    """
    if _num(run.get("summary.final_eval/old_eval/episode_reward/mean")) is not None:
        return None
    entry = store.lookup("eval", run["wandb_id"], EVAL_SPEC_ID)
    if entry is None:
        return None
    record = json.loads((store.root / entry.path).read_text())

    stamped = (entry.resolved or {}).get("walker_xml_path")
    if stamped is None:
        raise ValueError(
            f"{run['wandb_id']}: eval artifact has no resolved.walker_xml_path, so it "
            f"predates the 2026-08-18 fix and may have been simulated on the wrong body")
    if Path(str(stamped)).name != Path(str(run["env_params.walker_xml_path"])).name:
        raise ValueError(f"{run['wandb_id']}: eval artifact used {stamped}, run trained "
                         f"on {run['env_params.walker_xml_path']}")
    if record.get("env_class") != run["env"]:
        raise ValueError(f"{run['wandb_id']}: eval artifact env_class "
                         f"{record.get('env_class')} != run env {run['env']}")
    step, actual = record.get("step"), _num(run.get("summary._step"))
    if actual is not None and step != int(actual):
        raise ValueError(f"{run['wandb_id']}: eval artifact restored step {step}, run "
                         f"reached {int(actual)}")
    return record


#: Groups whose ``{mean, std}`` leaf ``evaluation.flat_summary`` collapses to the mean when
#: it writes the inline WandB keys: it emits ``<base>/errors/<name>`` and
#: ``<base>/reward_terms/<name>`` directly, while keeping ``<stat>/mean`` and
#: ``<stat>/std`` for ``episode_reward`` / ``lifespan_steps`` / ``lifespan_s``. The two
#: sources therefore have *different key shapes for the same numbers*, which is exactly
#: the sort of asymmetry that reads as "this run has no reward terms" if it is not
#: handled -- so `DatasetReader` mirrors `flat_summary` rather than guessing.
_INLINE_MEAN_ONLY = ("errors", "reward_terms")


class DatasetReader:
    """Reads one dataset's measurements from whichever record the run has.

    Paths are always given in the **artifact's** shape --
    ``("reward_terms", "joints", "mean")`` -- and translated for the inline source, so
    call sites do not have to know which one a given run uses.

    The two sources are the *same computation* (`evaluation.evaluate_networks` for both)
    but not the same measurement: the inline one runs on the in-memory weights at
    ``total_steps`` while the offline one restores the newest checkpoint, and MuJoCo Warp
    is not bit-reproducible either way. Which source a row came from is recorded in
    ``reward_source``, and no figure mixes the two within a series.
    """

    def __init__(self, run: pd.Series, record: dict | None, dataset: str):
        self.dataset = dataset
        if record is not None:
            self._block = record.get("datasets", {}).get(dataset, {})
            self.source = f"eval:{EVAL_SPEC_ID}"
        else:
            self._block = None
            self.source = "final_eval"
        self._run = run

    def _inline_key(self, path: tuple[str, ...]) -> str | None:
        """The WandB summary key ``flat_summary`` would have written, or ``None``.

        ``None`` means the inline record genuinely does not carry that number -- the
        ``std`` of a reward term, say -- rather than that it is missing for this run.
        """
        if path and path[0] in _INLINE_MEAN_ONLY:
            if path[-1] == "std":
                return None                      # flat_summary drops it
            if path[-1] == "mean":
                path = path[:-1]
        return "summary.final_eval/" + self.dataset + "/" + "/".join(path)

    def get(self, *path: str) -> float | None:
        """One scalar, from the artifact if there is one and the summary otherwise."""
        if self._block is None:
            key = self._inline_key(path)
            return None if key is None else _num(self._run.get(key))
        node: object = self._block
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return _num(node)

    @property
    def present(self) -> bool:
        return self.get("episode_reward", "mean") is not None


def hazard_rate(survived: float | None, lifespan_steps: float | None) -> float | None:
    """Failures per second of simulated time, treating clip-end truncation as censored.

    The constant-hazard MLE: with ``d`` failures over total exposure ``T``, the rate is
    ``d / T``, which for a per-episode survival fraction ``s`` and mean alive time ``t``
    is ``(1 - s) / t``. This is the only lifespan statistic comparable *across* the
    datasets, because raw survival penalises the 30 s clips for having six times as many
    chances to fail (analysis/rodent/README.md). Formula as in
    ``../forward-model-vs-encdec-seeds/extract.py``.
    """
    if survived is None or lifespan_steps is None or lifespan_steps <= 0:
        return None
    return (1.0 - survived) / (lifespan_steps / STEPS_PER_S)


def control_mode(run: pd.Series) -> str:
    """Read from ``env_params``, the only record of what the env actually used."""
    return "torque" if bool(run["env_params.torque_actuators"]) else "position"


def build_rows(run: pd.Series, record: dict | None) -> list[dict]:
    """One row per (run, dataset). Long rather than wide: ~30 metrics x 3 datasets."""
    rows = []
    for dataset in DATASETS:
        r = DatasetReader(run, record, dataset)
        if not r.present:
            # Recorded as a row with nulls rather than dropped: a missing dataset is a
            # coverage fact, and silently shortening the frame is how a cohort gets
            # reported on the subset that happened to have data.
            rows.append({"condition": run["condition"], "wandb_id": run["wandb_id"],
                         "dataset": dataset, "reward_source": "none"})
            continue

        survived = r.get("termination_rate", "survived")
        lifespan = r.get("lifespan_steps", "mean")
        reward = r.get("episode_reward", "mean")

        row = {
            "condition": run["condition"],
            "wandb_id": run["wandb_id"],
            "wandb_name": run["wandb_name"],
            "mode": control_mode(run),
            "efference_length": _num(run.get("net_params.efference_length")),
            "delay_k": _num(run.get("delay_k")),
            "use_intention": run.get("net_params.dec_use_intention"),
            "use_proprioception": run.get("net_params.dec_use_proprioception"),
            "state": run["state"],
            "created_at": run["created_at"],
            "git_commit": str(run["git_commit"])[:8],
            "gpu": run["gpu"],
            "cuda_version": run.get("cuda_version"),
            "actual_step": _num(run.get("summary._step")),
            "dataset": dataset,
            "reward_source": r.source,
            "n_clips": r.get("n_clips"),
            "reward": reward,
            "reward_std": r.get("episode_reward", "std"),
            "lifespan_steps": lifespan,
            "lifespan_s": r.get("lifespan_s", "mean"),
            # Reward per *alive* step: the axis that separates "tracks the reference worse"
            # from "falls over sooner", and the only reward axis comparable across datasets.
            "reward_per_step": (None if reward is None or not lifespan
                                else reward / lifespan),
            "survived": survived,
            "term_any": r.get("termination_rate", "any"),
            "hazard_rate": hazard_rate(survived, lifespan),
        }
        for reason in TERMINATIONS:
            row[f"term_{reason}"] = r.get("termination_rate", reason)
        for key, short in ERRORS.items():
            # `key` may itself contain a slash (`body_errors/total`): in the artifact it is
            # one dict key, not two levels of nesting.
            row[f"err_{short}"] = r.get("errors", key, "mean")
        for term in REWARD_TERMS:
            total = r.get("reward_terms", term, "mean")
            row[f"rt_{term}"] = total
            # The stored terms are masked episode *totals*, so they are confounded with
            # lifespan: a policy that dies at 60 % of the clip banks 60 % of every term.
            # Every composition figure must use the per-step column.
            row[f"rtps_{term}"] = (None if total is None or not lifespan
                                   else total / lifespan)
        rows.append(row)
    return rows


def assert_reward_terms_sum(df: pd.DataFrame) -> str:
    """Check the ten reward terms reconstruct the episode reward, and report the residual.

    A free and surprisingly strong test that nothing was misread: the env's total reward is
    the plain sum over its configured terms, so the terms must add up to
    ``episode_reward`` per run and dataset. It is also the check that would catch a term
    being added to the env without being added to ``REWARD_TERMS`` here -- which would
    otherwise show up as a quietly incomplete decomposition.
    """
    columns = [f"rt_{t}" for t in REWARD_TERMS]
    have = df[df.reward.notna()].copy()
    # `min_count=len(columns)` rather than 1: a row missing even one term must come out
    # NaN, not silently sum the rest. The coverage assertion below then catches it. An
    # earlier version used min_count=1 and reported "48 rows" while actually validating
    # three, because the inline summary keys had been read under the artifact's key shape.
    total = have[columns].sum(axis=1, min_count=len(columns))
    resid = (total - have["reward"]).abs() / have["reward"].abs()

    checked = int(resid.notna().sum())
    if checked < len(have):
        missing = have.loc[resid.isna(), ["wandb_id", "dataset"]]
        raise SystemExit(
            f"reward_terms incomplete for {len(have) - checked} of {len(have)} "
            f"(run, dataset) rows that have a reward: "
            + ", ".join(f"{r.wandb_id}/{r.dataset}" for _, r in missing.iterrows())
            + "\nA row with a reward must have all ten terms; a NaN here means a key was "
              "read under the wrong shape, not that the run lacks the measurement.")

    worst = float(resid.max()) if checked else 0.0
    if worst > 1e-6:
        offenders = have.loc[resid > 1e-6, ["wandb_id", "dataset"]]
        raise SystemExit(
            f"reward_terms do not sum to episode_reward (max relative residual "
            f"{worst:.2e}): "
            + ", ".join(f"{r.wandb_id}/{r.dataset}" for _, r in offenders.iterrows()))
    return (f"reward_terms sum vs episode_reward: max relative residual {worst:.2e} "
            f"over {checked}/{len(have)} (run, dataset) rows with a reward")


# --------------------------------------------------------------------------------------
# Stage B: per-clip rows, from the `per_clip` eval artifacts
# --------------------------------------------------------------------------------------

def load_behaviour_labels():
    """The committed labels from ``behaviour_labels.py``. No h5 is opened here.

    Returns ``(clip rows, [32, 1500] cluster ids, [100] names, {name: coarse group})``.
    The frame-level arrays are only meaningful for ``new_eval``; the clip rows cover all
    three datasets.

    The coarse grouping is read from the ``.npz`` rather than restated here, so the
    partition a figure bins by is the same object the labels were written with and cannot
    drift from it.
    """
    clips = pd.read_csv(HERE / "behaviour_labels_clips.csv")
    with np.load(HERE / "behaviour_labels_frames.npz", allow_pickle=False) as z:
        names = z["names"].astype(str)
        coarse = dict(zip(names, z["coarse"].astype(str)))
        return clips, z["labels"], names, coarse


def load_per_clip_eval(store: Store, run: pd.Series) -> dict | None:
    """The per-clip ``eval`` record for one run, with the same provenance checks.

    Returns ``None`` when the artifact has not been produced yet, so Stage A stays
    runnable and ``coverage.txt`` reports the gap rather than the script failing.
    """
    entry = store.lookup("eval", run["wandb_id"], EVAL_PC_SPEC_ID)
    if entry is None:
        return None
    record = json.loads((store.root / entry.path).read_text())

    stamped = (entry.resolved or {}).get("walker_xml_path")
    if stamped is None:
        raise ValueError(
            f"{run['wandb_id']}: per-clip eval has no resolved.walker_xml_path")
    if Path(str(stamped)).name != Path(str(run["env_params.walker_xml_path"])).name:
        raise ValueError(f"{run['wandb_id']}: per-clip eval used {stamped}, run trained "
                         f"on {run['env_params.walker_xml_path']}")
    if record.get("env_class") != run["env"]:
        raise ValueError(f"{run['wandb_id']}: per-clip eval env_class "
                         f"{record.get('env_class')} != run env {run['env']}")
    step, actual = record.get("step"), _num(run.get("summary._step"))
    if actual is not None and step != int(actual):
        raise ValueError(f"{run['wandb_id']}: per-clip eval restored step {step}, run "
                         f"reached {int(actual)}")
    # Forward-looking companion to `assert_artifact_actuator`: artifacts produced from
    # 2026-09-23 stamp the actuator the env was rebuilt with, so it can be compared
    # against the independently written WandB config at load time rather than inferred
    # from a physical consequence. Absence means the artifact predates the stamp, which
    # is not an error -- the physical check covers those.
    stamped_actuator = (entry.resolved or {}).get("torque_actuators")
    if stamped_actuator is not None:
        trained = bool(run["env_params.torque_actuators"])
        if bool(stamped_actuator) != trained:
            raise ValueError(
                f"{run['wandb_id']}: per-clip eval was simulated with "
                f"torque_actuators={bool(stamped_actuator)} but the run trained with "
                f"{trained}. The checkpoint's config.json disagrees with WandB; the "
                f"index is authoritative (see `assert_artifact_actuator`).")

    # Per-clip quantities are the ones eval nondeterminism hits hardest, and on this
    # laptop the same checkpoint spans ~1.2 % between passes while the cluster is exact
    # (analysis/README.md §6). A cohort with one laptop-produced member would carry noise
    # nothing in the CSVs records, so refuse it outright.
    gpu = (entry.producer or {}).get("gpu", "")
    if "RTX 4060" in str(gpu):
        raise ValueError(
            f"{run['wandb_id']}: per-clip eval was produced on {gpu}, where the eval is "
            f"not reproducible. Re-produce it on the cluster.")
    return record


def term_reason(flags: dict[str, float], survived: bool) -> str:
    """The reason this clip ended, as one string -- a genuine partition of the episodes.

    The env evaluates every termination predicate independently on the terminating step
    (``rodent/base.py`` ORs them into ``any``), so **more than one can fire at once** and
    the per-reason rates in ``termination_rate`` are *not* a partition: summed with
    ``survived`` they exceed 1 in 19 of this cohort's 51 (run, dataset) cells, by up to
    0.6 %.

    Measured over 13 984 episodes here: 72.75 % survived, 21.81 % ``root_too_far``,
    4.85 % ``root_too_rotated``, 0.38 % ``pose_error``, and **0.21 % fired
    ``root_too_far`` and ``root_too_rotated`` together** (no other combination, and
    ``nan_termination`` never fired at all). Rather than pick one by precedence -- the
    reasons are not causally ordered, and a silent ``argmax`` would misattribute a
    concurrent failure to whichever happened to be listed first -- a co-occurrence gets
    its own ``a+b`` label. That keeps the column an exact partition, so a stacked bar
    built from it sums to 1 by construction, and leaves the overlap visible as its own
    (thin) band rather than hidden.
    """
    fired = [r for r in TERMINATIONS if flags.get(r, 0.0) > 0.5]
    if fired:
        return "+".join(fired)
    if survived:
        return "survived"
    # `terminations/any` fired but no named reason did -- only possible if the env gains
    # a reason this folder does not know about, which is worth seeing rather than hiding.
    return "unknown"


def build_clip_rows(run: pd.Series, record: dict, labels: pd.DataFrame) -> list[dict]:
    """One row per (run, dataset, clip) from a per-clip eval record."""
    rows = []
    for dataset in DATASETS:
        block = record.get("datasets", {}).get(dataset, {}).get("per_clip")
        if block is None:
            continue
        n = len(block["clip_ids"])
        env = block["env"]
        label_rows = labels[labels.dataset == dataset].set_index("clip_index")

        # The artifact's own `clip_names`, stamped by the loader on the cluster, against
        # the committed labels derived independently on the laptop. Two records written
        # by different code from the same source: if they disagree, one of them is
        # describing a different clip set and no per-behaviour number means anything.
        stamped = block.get("clip_names")
        if stamped is not None:
            expected = [str(label_rows.loc[i, "behaviour"]) for i in range(n)]
            if stamped[:n] != expected:
                bad = [i for i in range(n) if stamped[i] != expected[i]]
                raise ValueError(
                    f"{run['wandb_id']}/{dataset}: artifact clip_names disagree with "
                    f"behaviour_labels_clips.csv at {len(bad)} of {n} clips "
                    f"(first: clip {bad[0]}, artifact {stamped[bad[0]]!r} vs committed "
                    f"{expected[bad[0]]!r}). Re-run behaviour_labels.py --verify.")

        for i in range(n):
            lifespan = float(block["lifespan_steps"][i])
            reward = float(block["episode_reward"][i])
            flags = {r: float(env[f"terminations/{r}"][i]) for r in TERMINATIONS
                     if f"terminations/{r}" in env}
            survived = float(env["terminations/any"][i]) < 0.5
            denom = max(lifespan, 1.0)
            row = {
                "condition": run["condition"],
                "wandb_id": run["wandb_id"],
                "mode": control_mode(run),
                "dataset": dataset,
                "clip_index": i,
                "behaviour": label_rows.loc[i, "behaviour"],
                "coarse": label_rows.loc[i, "coarse"],
                "modal_frac": label_rows.loc[i, "modal_frac"],
                "episode_reward": reward,
                "lifespan_steps": lifespan,
                "lifespan_s": lifespan / STEPS_PER_S,
                "reward_per_step": reward / denom,
                "survived": int(survived),
                "term_reason": term_reason(flags, survived),
            }
            for key, short in ERRORS.items():
                row[f"err_{short}"] = (float(env[key][i]) / denom
                                       if key in env else None)
            for term in REWARD_TERMS:
                key = f"rewards/{term}"
                if key in env:
                    # Only the per-alive-step form. The episode total is exactly
                    # `rtps_<term> * lifespan_steps`, so carrying both would add ten
                    # redundant columns to a 14 000-row committed CSV. `data.csv` keeps
                    # the totals at the run level, where they are 51 rows.
                    row[f"rtps_{term}"] = float(env[key][i]) / denom
            rows.append(row)
    return rows


def assert_clips_are_paired(clips: pd.DataFrame) -> str:
    """Every run must have seen the same clips, in the same order, per dataset.

    This is the licence for every paired figure in ``plot.py``. It is asserted on the
    data rather than inferred from the config gates in ``_pairable``, because the gates
    bound what *entered* the cohort while this bounds what is actually in the CSV.
    """
    lines = []
    for dataset, sub in clips.groupby("dataset", observed=True):
        keys = sub.groupby("wandb_id").apply(
            lambda g: tuple(zip(g.clip_index, g.behaviour)), include_groups=False)
        distinct = set(keys)
        if len(distinct) != 1:
            raise SystemExit(
                f"{dataset}: runs disagree on the clip axis "
                f"({len(distinct)} distinct clip/label sequences across "
                f"{len(keys)} runs) -- paired comparison is not valid")
        lines.append(f"  {dataset}: {len(keys)} runs share one clip axis of "
                     f"{len(next(iter(distinct)))} clips")
    return "clip axis identical across runs:\n" + "\n".join(lines)


#: How far a per-clip artifact may sit from the independent inline ``final_eval`` number
#: before it counts as a discrepancy rather than as eval nondeterminism -- **per dataset**,
#: because what one clip can do to the mean differs by an order of magnitude between them.
#:
#: The two are different measurements of the same weights: inline runs on the in-memory
#: network at ``total_steps``, the artifact restores the newest checkpoint (asserted equal
#: in `load_per_clip_eval`), and MuJoCo Warp's GPU physics is not bit-reproducible, so a
#: clip near the failure boundary can survive in one pass and die in the other.
#:
#: Measured across this cohort: ``train`` (673 clips) agrees to 0.90 % worst case and
#: 0.12 % median, ``old_eval`` (169 clips) to 1.53 % / 0.22 %, but ``new_eval`` (32 clips
#: of 30 s) to only 15.2 % / 3.0 %. That is not a different kind of error, it is the same
#: error with a 21x smaller denominator: a surviving 30 s clip banks ~12 000 reward against
#: a few hundred for an early death, so one clip of 32 flipping moves the mean ~10 %. The
#: survival bound below is what pins that reading -- the three cells that exceed 8 % differ
#: by 1, 1 and 3 clips of survival, and reward delta correlates 0.70 with survival delta.
REWARD_TOLERANCE = {"train": 0.03, "old_eval": 0.03, "new_eval": 0.20}

#: Episodes whose survival may differ between the two passes. This is the mechanistic
#: quantity -- what actually varies is *which clips lived* -- so it is the bound that says
#: the two passes measured the same policy, independently of how much the reward mean
#: happened to move.
#:
#: Neither a flat count nor a flat fraction works, because the two regimes are different:
#: on 32 clips a single flip is already 3.1 % (discreteness dominates), while on 673 clips
#: flips accumulate roughly in proportion to n. So the bound is the larger of a small
#: absolute floor and a fraction. Measured maxima across this cohort: 3 of 32
#: (``new_eval``), 6 of 169 (``old_eval``), 10 of 673 (``train``) -- i.e. 9.4 %, 3.6 % and
#: 1.5 %, all comfortably inside it, while a pass that had restored a different
#: checkpoint would flip far more.
SURVIVAL_FLOOR_CLIPS = 4
SURVIVAL_TOLERANCE_FRAC = 0.05


def survival_tolerance(n_clips: int) -> float:
    return max(SURVIVAL_FLOOR_CLIPS, SURVIVAL_TOLERANCE_FRAC * n_clips)


def assert_per_clip_matches_aggregate(clips: pd.DataFrame, data: pd.DataFrame) -> str:
    """Cross-check the per-clip artifacts against the inline numbers in ``data.csv``.

    The per-clip block is an independent *run* of the same measurement on the same
    weights, so it should land on the inline value. This is the check that the artifact
    evaluated the checkpoint it claims to: the provenance asserts in
    `load_per_clip_eval` compare recorded strings, while this compares numbers, and only
    the second would catch a checkpoint directory holding another run's weights.

    Two bounds, because the reward mean alone cannot distinguish "measured something else"
    from "one long clip fell over this time" -- see `REWARD_TOLERANCE`.
    """
    idx = ["wandb_id", "dataset"]
    inline = data[data.reward.notna()].set_index(idx)[["reward", "survived", "n_clips"]]
    got = clips.groupby(idx, observed=True).agg(
        pc_reward=("episode_reward", "mean"), pc_survived=("survived", "mean"),
        pc_n=("clip_index", "size"))
    j = inline.join(got, how="inner").dropna(subset=["reward", "pc_reward"])
    if j.empty:
        return "per-clip vs inline: no overlapping (run, dataset) pairs to compare"

    j["rel"] = (j.pc_reward - j.reward).abs() / j.reward.abs()
    j["tol"] = [REWARD_TOLERANCE[d] for _, d in j.index]
    j["surv_clips"] = (j.pc_survived - j.survived).abs() * j.pc_n

    bad_reward = j[j.rel > j.tol]
    j["surv_tol"] = [survival_tolerance(int(n)) for n in j.pc_n]
    bad_surv = j[j.surv_clips > j.surv_tol]
    if len(bad_reward) or len(bad_surv):
        raise SystemExit(
            "per-clip eval disagrees with the inline final_eval beyond what eval\n"
            "nondeterminism explains:\n"
            + pd.concat([bad_reward, bad_surv]).drop_duplicates().to_string())

    lines = ["per-clip vs inline final_eval (independent passes, same weights):"]
    for dataset, sub in j.groupby(level="dataset", observed=True):
        lines.append(
            f"  {dataset:9s} n_clips={int(sub.pc_n.iloc[0]):4d}  reward within "
            f"{100 * sub.rel.max():5.2f}% (median {100 * sub.rel.median():4.2f}%, "
            f"tol {100 * sub.tol.iloc[0]:.0f}%)  survival within "
            f"{sub.surv_clips.max():.0f} clip(s) (tol {sub.surv_tol.iloc[0]:.0f})")
    return "\n".join(lines)


#: Control cost per alive step separates the two actuators by a factor of ~6 with no
#: overlap: position runs span -0.123..-0.099 and torque runs -0.020..-0.005 across this
#: cohort. The gap is physical -- `control_cost` is `0.02 * sum(action^2)` and a position
#: action is a joint-angle target held near the reference while a torque action idles near
#: zero -- so it is a *measurement* of which actuator was simulated, independent of any
#: recorded config.
ACTUATOR_BOUNDARY = -0.05


def assert_artifact_actuator(frame: pd.DataFrame, *, label: str,
                             cost_col: str = "rtps_control_cost",
                             dataset: str | None = "old_eval") -> str:
    """Assert each artifact was simulated with the actuator the run actually trained on.

    This exists because of a near-miss on 2026-09-23. ``7w26do00``'s checkpoint
    ``config.json`` had been overwritten in place by a *torque* run's config -- 61 of its
    62 config leaves still matched the WandB record, and the 62nd was
    ``torque_actuators``. ``parse_env_config`` reads that field from the checkpoint, so an
    offline rebuild would have simulated a position-trained policy with torque actuators
    and filed it under ``pos_noproprio_eff2``, in the one folder whose parent question is
    position versus torque. It was caught only because the same corruption left a stray
    trailing byte and the producer raised ``JSONDecodeError``; a clean overwrite would
    have produced silently.

    The provenance asserts in `load_per_clip_eval` could not have caught it -- they compare
    strings the *same* file supplied. This compares a physical consequence of the choice
    against the run's WandB-logged actuator, which is two independently written records,
    and it works on artifacts produced before the ``resolved.torque_actuators`` stamp
    existed.
    """
    sub = frame if dataset is None else frame[frame.dataset == dataset]
    modes = sub.groupby("wandb_id", observed=True)["mode"].first()
    cost = sub.groupby("wandb_id", observed=True)[cost_col].mean()
    joined = pd.concat([modes.rename("claimed"), cost.rename("cost")], axis=1).dropna()
    joined["implied"] = np.where(joined.cost > ACTUATOR_BOUNDARY, "torque", "position")
    bad = joined[joined.claimed != joined.implied]
    if len(bad):
        raise SystemExit(
            f"a {label} artifact was simulated with the wrong actuator -- its control cost "
            "per alive step does not match the actuator the run trained with:\n"
            + bad.to_string()
            + "\n\nThe run's `env_params.torque_actuators` in the index is authoritative; "
              "the checkpoint's config.json is not (see this function's docstring).")
    # Report the margin, so a future cohort that narrows the gap is visible rather than
    # silently relying on a threshold that no longer separates.
    pos = joined.loc[joined.implied == "position", "cost"]
    tor = joined.loc[joined.implied == "torque", "cost"]
    margin = (f"position {pos.min():.4f}..{pos.max():.4f}, "
              f"torque {tor.min():.4f}..{tor.max():.4f}") if len(pos) and len(tor) else "n/a"
    return (f"{label}: actuator matches the trained actuator for all {len(joined)} runs "
            f"(control cost per alive step: {margin}; boundary {ACTUATOR_BOUNDARY})")


def build_behaviour_rows(clips: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, dataset, grouping level, bin) from ``clips.csv``.

    Three levels in one long frame -- ``behaviour`` (the 6 snippet classes or the modal
    MotionMapper name), ``coarse`` (groom / still / rear / locomote) and ``all`` -- so a
    figure picks its resolution with a filter instead of a different CSV.

    Committed even though it is derivable, because the aggregation rule is a decision:
    the clip is the unit *within* a run, so a bin's value is the mean over that run's
    clips, and a later across-run mean weights runs equally rather than pooling the clips
    of a three-run cell against those of a one-run cell.
    """
    metric_cols = (["episode_reward", "lifespan_s", "reward_per_step", "survived"]
                   + [c for c in clips.columns if c.startswith(("err_", "rtps_"))])
    out = []
    for (run_id, dataset), sub in clips.groupby(["wandb_id", "dataset"], observed=True):
        bins = [("behaviour", name, g)
                for name, g in sub.groupby("behaviour", observed=True)]
        bins += [("coarse", name, g) for name, g in sub.groupby("coarse", observed=True)]
        bins += [("all", "all", sub)]
        for level, name, g in bins:
            row = {
                "condition": sub["condition"].iloc[0],
                "wandb_id": run_id,
                "mode": sub["mode"].iloc[0],
                "dataset": dataset,
                "level": level,
                "behaviour": name,
                "n_clips": len(g),
            }
            for col in metric_cols:
                row[f"{col}_mean"] = float(g[col].mean())
            row["episode_reward_median"] = float(g["episode_reward"].median())
            row["episode_reward_sd"] = (float(g["episode_reward"].std(ddof=1))
                                        if len(g) > 1 else None)
            # Standard error over the clips in this bin, for this run. Labelled that way
            # in the figures: it is the clip-level spread, not the run-level one, and the
            # run-level one is not estimable from 1-3 runs.
            row["episode_reward_sem"] = (float(g["episode_reward"].sem())
                                         if len(g) > 1 else None)
            # Failures per second of simulated time in this bin, truncations censored.
            alive = float(g["lifespan_s"].sum())
            n_fail = len(g) - int(g["survived"].sum())
            row["hazard_rate"] = n_fail / alive if alive else None
            for reason in list(TERMINATIONS) + ["survived"]:
                row[f"frac_{reason}"] = float((g["term_reason"] == reason).mean())
            out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# Stage C: per-frame behaviour, from the `trace` artifacts
# --------------------------------------------------------------------------------------

def load_trace(store: Store, run: pd.Series):
    """Open one run's ``new_eval`` trace, or ``None`` if it has not been produced."""
    entry = store.lookup("trace", run["wandb_id"], TRACE_SPEC_ID)
    if entry is None:
        return None
    path = store.root / entry.path
    if not path.exists():
        return None
    stamped = (entry.resolved or {}).get("walker_xml_path")
    if stamped is None:
        raise ValueError(f"{run['wandb_id']}: trace has no resolved.walker_xml_path")
    if Path(str(stamped)).name != Path(str(run["env_params.walker_xml_path"])).name:
        raise ValueError(f"{run['wandb_id']}: trace used {stamped}, run trained on "
                         f"{run['env_params.walker_xml_path']}")
    if (entry.resolved or {}).get("truncated"):
        raise ValueError(
            f"{run['wandb_id']}: trace is truncated (max_steps was set), so it is an "
            f"incomplete record of the episode and must not be pooled with full ones")
    return path


def behaviour_of_each_step(frame: np.ndarray, labels: np.ndarray,
                           names: np.ndarray) -> np.ndarray:
    """Map each traced step to the behaviour of the reference frame it was tracking.

    ``frame`` is the env's own ``current_frame``, not ``step * ctrl_dt * mocap_hz``: the
    two differ by up to half a frame, and reading the value the env used removes a source
    of error that would be invisible.
    """
    idx = np.clip(np.rint(frame).astype(int), 0, labels.shape[1] - 1)
    clip_ids = np.arange(frame.shape[0])[:, None]
    # `- 1`: the MotionMapper ids are 1-based (behaviour_labels.py asserts it).
    return names[labels[clip_ids, idx] - 1]


def build_frame_rows(run: pd.Series, path, labels: np.ndarray, names: np.ndarray,
                     coarse_of: dict) -> tuple[list[dict], list[dict]]:
    """``(behaviour_frames rows, failure_context rows)`` for one run's new_eval trace.

    This is the measurement a per-clip number cannot give: the 30 s clips average 13
    MotionMapper behaviours each, so "reward on this clip" is a mixture, while "reward
    per step while the reference was walking" is not.
    """
    import h5py

    with h5py.File(path, "r") as f:
        n_clips = int(f.attrs["n_clips"])
        ctrl_dt = float(f.attrs["ctrl_dt"])
        frame = np.asarray(f["current_frame"])[:n_clips]
        alive = np.asarray(f["alive"])[:n_clips] > 0
        reward = np.asarray(f["reward"])[:n_clips]
        term = {r: np.asarray(f[f"terminations/{r}"])[:n_clips] for r in TERMINATIONS
                if f"terminations/{r}" in f}
        any_term = np.asarray(f["terminations/any"])[:n_clips]
        terms = {t: np.asarray(f[f"rewards/{t}"])[:n_clips] for t in REWARD_TERMS
                 if f"rewards/{t}" in f}
        errs = {short: np.asarray(f[key])[:n_clips]
                for key, short in ERRORS.items() if key in f}

    beh = behaviour_of_each_step(frame, labels, names)

    # Where the episode ended, and what the reference was doing there.
    frames_rows, failure_rows = [], []
    for c in range(n_clips):
        hit = np.flatnonzero(any_term[c] > 0.5)
        if not len(hit):
            continue
        t = int(hit[0])
        reason = next((r for r, a in term.items() if a[c, t] > 0.5), "unknown")
        window = slice(max(0, t - int(FAILURE_WINDOW_S / ctrl_dt)), t + 1)
        vals, counts = np.unique(beh[c, window], return_counts=True)
        failure_rows.append({
            "condition": run["condition"], "wandb_id": run["wandb_id"],
            "mode": control_mode(run), "clip_index": c,
            "term_step": t, "term_s": t * ctrl_dt,
            "term_frame": int(round(float(frame[c, t]))),
            "term_reason": reason,
            "behaviour_at_term": str(beh[c, t]),
            "behaviour_window_modal": str(vals[counts.argmax()]),
            "window_modal_frac": float(counts.max() / counts.sum()),
        })

    # Terminations attributed to the behaviour they happened in, for the hazard.
    term_beh = {str(r["behaviour_at_term"]): 0 for r in failure_rows}
    for r in failure_rows:
        term_beh[str(r["behaviour_at_term"])] += 1

    for name in sorted(set(beh[alive])):
        sel = alive & (beh == name)
        n_steps = int(sel.sum())
        alive_s = n_steps * ctrl_dt
        row = {
            "condition": run["condition"], "wandb_id": run["wandb_id"],
            "mode": control_mode(run), "dataset": "new_eval",
            "behaviour": name, "coarse": coarse_of.get(name, "exclude"),
            "alive_steps": n_steps, "alive_s": alive_s,
            "reward_per_step": float(reward[sel].sum() / n_steps),
            "n_terminations": term_beh.get(name, 0),
            # Failures per second *of exposure to this behaviour*. The right denominator:
            # a behaviour the policy spends ten times as long in has ten times as many
            # chances to fail, and a raw count would read that as fragility.
            "hazard_per_s": term_beh.get(name, 0) / alive_s if alive_s else None,
        }
        for term, arr in terms.items():
            row[f"rtps_{term}"] = float(arr[sel].sum() / n_steps)
        for short, arr in errs.items():
            row[f"err_{short}"] = float(arr[sel].sum() / n_steps)
        frames_rows.append(row)
    return frames_rows, failure_rows


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)

    store = Store()

    # --- Stage A: the index tier -----------------------------------------------------
    rows: list[dict] = []
    for _, run in runs.iterrows():
        rows.extend(build_rows(run, load_eval(store, run)))
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    df = df.sort_values(["condition", "wandb_id", "dataset"], ignore_index=True)
    verdicts = [assert_reward_terms_sum(df)]

    # --- Stage B: per-clip, where the artifacts exist ---------------------------------
    labels, frame_labels, names, coarse_of = load_behaviour_labels()
    per_clip_records = {run["wandb_id"]: load_per_clip_eval(store, run)
                        for _, run in runs.iterrows()}
    have_per_clip = [k for k, v in per_clip_records.items() if v is not None]

    clips = behaviour = None
    if have_per_clip:
        clip_rows = []
        for _, run in runs.iterrows():
            record = per_clip_records[run["wandb_id"]]
            if record is not None:
                clip_rows.extend(build_clip_rows(run, record, labels))
        clips = pd.DataFrame(clip_rows)
        clips["dataset"] = pd.Categorical(clips["dataset"], categories=DATASETS,
                                          ordered=True)
        clips = clips.sort_values(["condition", "wandb_id", "dataset", "clip_index"],
                                  ignore_index=True)
        verdicts.append(assert_clips_are_paired(clips))
        verdicts.append(assert_per_clip_matches_aggregate(clips, df))
        verdicts.append(assert_artifact_actuator(clips, label="per-clip eval"))
        behaviour = build_behaviour_rows(clips).sort_values(
            ["condition", "wandb_id", "dataset", "level", "behaviour"],
            ignore_index=True)

    # --- Stage C: per-frame behaviour, where the traces exist -------------------------
    frames = failures = None
    trace_paths = {run["wandb_id"]: load_trace(store, run) for _, run in runs.iterrows()}
    have_trace = [k for k, v in trace_paths.items() if v is not None]
    if have_trace:
        frame_rows, failure_rows = [], []
        for _, run in runs.iterrows():
            path = trace_paths[run["wandb_id"]]
            if path is None:
                continue
            fr, fa = build_frame_rows(run, path, frame_labels, names, coarse_of)
            frame_rows.extend(fr)
            failure_rows.extend(fa)
        frames = pd.DataFrame(frame_rows).sort_values(
            ["condition", "wandb_id", "behaviour"], ignore_index=True)
        # The traces are a *separate* offline rebuild from the per-clip evals, reading the
        # same checkpoint config, so they need the same physical check rather than
        # inheriting the evals' verdict. Weighted by alive time so the per-behaviour rows
        # aggregate back to a per-run control cost.
        weighted = frames.assign(_w=frames.alive_steps * frames.rtps_control_cost)
        per_run = (weighted.groupby("wandb_id", observed=True)
                   .apply(lambda g: g._w.sum() / g.alive_steps.sum(),
                          include_groups=False).rename("rtps_control_cost").reset_index())
        per_run = per_run.merge(
            frames.groupby("wandb_id", observed=True)["mode"].first().reset_index(),
            on="wandb_id")
        verdicts.append(assert_artifact_actuator(per_run, label="trace", dataset=None))
        failures = pd.DataFrame(failure_rows).sort_values(
            ["condition", "wandb_id", "clip_index"], ignore_index=True)

    # `REQUIRES` grows only once the data is there, so `coverage.txt` reads as a plan
    # while the cluster round-trip is outstanding and as a guarantee afterwards. Runs are
    # never dropped for a missing artifact -- the stage is skipped for everyone.
    requires = list(REQUIRES)
    if have_per_clip:
        requires.append(f"eval:{EVAL_PC_SPEC_ID}")
    if have_trace:
        requires.append(f"trace:{TRACE_SPEC_ID}")
    pipeline.write_coverage(runs, requires, HERE)
    verdicts.append(
        f"stages: A (index) always; B (per-clip) {len(have_per_clip)}/{len(runs)} runs; "
        f"C (trace) {len(have_trace)}/{len(runs)} runs")

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    report = "\n\n".join(verdicts) + "\n\n" + report
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
    print(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    for frame, name in ((clips, "clips.csv"), (behaviour, "behaviour.csv"),
                        (frames, "behaviour_frames.csv"),
                        (failures, "failure_context.csv")):
        if frame is not None:
            ok &= pipeline.write_csv(frame.round(6), HERE / name, check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
