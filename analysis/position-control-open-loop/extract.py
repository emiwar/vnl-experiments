"""Position control, re-checked on the new body and reference frame -- and is it open loop?

The question(s)
---------------
[`position-vs-torque-control/`](../position-vs-torque-control/) concluded that **position
control is nearly delay-invariant out to 100 control steps (1 s)** while torque control
collapses. A second of dead time is a lot for a 100 Hz imitation task, so the result was
suspicious enough to be worth re-running on the configuration we now use everywhere:
``rodent_no_tail_collisions.xml`` (new body) and ``body_target_frame=reference_root``
(pure target pose, no current-state leak into the target). The 2026-08-31/09-01 batch does
that, and adds the ablations that ask *why*:

1. **Q1** -- does position control's delay tolerance survive the new XML + reference frame?
2. **Q2** -- WandB suggests position control also works with **no proprioception at all**
   (``dec_use_proprioception=False``), i.e. open loop. Is that real?
   * **Q2a** -- is that open-loop performance overfitted to the training clips?
   * **Q2b** -- does the *delay* length matter when there is no proprioception?
   * **Q2c** -- is it an artefact of ``kl_weight`` being too small (a wide-open latent)?

What the architecture actually does, because two of these questions are answered by it
-------------------------------------------------------------------------------------
``delays.network_builders.build_delay_network`` puts the ``Delay`` layer **inside the
proprioception branch only**::

    actor = Concat({ task_obs:      encoder -> VariationalBottleneck(latent),
                     proprioception: Delay(k_steps=delay_k) })   -> EfferenceCopy(queue=eff)

Three consequences this analysis leans on:

* ``task_obs`` -- the imitation target -- reaches the actor **undelayed**, through the
  latent. It is the only path by which the reference reaches the policy.
* With ``dec_use_proprioception=False`` the proprioception branch is *not constructed*, so
  ``delay_k`` appears nowhere in the network. **Delay is architecturally inert in the
  no-proprioception arm** (Q2b is decided by code, and the runs measure the noise floor
  around it). What still varies across those runs is ``efference_length``, which the run
  names tie to ``delay_k`` and which is therefore easy to mistake for a delay effect.
* Under ``reference_root``, ``AbsoluteImitation``'s ``joint`` and ``body`` targets are
  absolute, but ``root`` and ``quat`` stay expressed in the *current* root frame
  (``envs/absolute_imitation.py`` says so, and calls it unavoidable for an egocentric
  representation). So a "no proprioception" run still receives an undelayed **root-pose
  error** signal. It is joint-level feedback that is missing, not all feedback -- the
  report does not call these runs feedback-free.

And what position control means in this model (``tasks/rodent/base.py`` + the XML): the
actuators are MuJoCo affine-bias servos, ``force = gainprm0 * act + bias0 - kp * qpos``
with ``kp = -biasprm1``, which for **29 of the 38** actuators is exactly
``kp * (mid_j + half_range_j * ctrl - qpos_j)``: **the action is a target joint angle,
normalised to that joint's own range**. ``torque_actuators=True`` instead sets
``gainprm0`` to that actuator's ``forcerange`` maximum and drops the bias entirely, i.e.
``force = gain * ctrl`` with no state term. The reference joint angles the policy is asked
to imitate are, up to the per-joint affine map, the same quantity as a position action --
which is the mechanism the report argues from. ``actuator_map.py`` in this folder derives
all of that from the XML and writes ``actuator_map.txt``; it also shows that the two modes
do *not* have matched force authority, which report.md carries as a caveat.

Conditions
----------
Ten cells on the new setup (new XML, ``reference_root``, seed 42, standard architecture,
``latent_size=32``, ``min_std=0.1``, ``AbsoluteImitation``, trained to budget, regularised
commits only), plus the two previous-setup cells pinned to the commits
[`position-vs-torque-control/`](../position-vs-torque-control/) used, so Q1 is a
comparison against that analysis' own runs rather than a re-selected approximation of them.

The torque arm of Q1 is ``torque_efference_aug11``, the 2026-08-11 sweep -- the only
complete torque delay sweep on this body and frame. It is kept apart from the later
``torque_efference`` batches on purpose: separated, it is the one condition here with no
flagged invariant at all, and it is the only one whose numbers come from an offline eval
artifact rather than the run's own inline eval. See ``completed_training`` for why it needs
a gate other than ``state == "finished"``, and ``EVAL_SPEC_ID`` for the artifact.

Metrics, and one trap in them
-----------------------------
The primary numbers are the **inline end-of-training eval** (``final_eval/*``), which
evaluates the same weights on all three datasets -- ``train`` (the 80 % split it trained
on), ``old_eval`` (held-out, same 250-frame clips) and ``new_eval`` (32 unseen 1500-frame
clips). Having ``train`` and ``old_eval`` from one pass is what makes Q2a answerable
without an offline eval.

The WandB ``eval/*`` series is *also* recorded (``window_reward``, the mean of the eval
points in the last 50 M steps, per analysis/README.md §6) but it means **different things
across this cohort**: before the 2026-08-20 fix ``eval_env = train_env``, so the
previous-setup runs' ``eval/*`` is a train-split curve while the new runs' is genuinely
held out. ``inline_split`` records which, and no figure differences the two.

Run it
------
    ../.venv/bin/python analysis/position-control-open-loop/extract.py            # frozen
    ../.venv/bin/python analysis/position-control-open-loop/extract.py --refresh
    ../.venv/bin/python analysis/position-control-open-loop/extract.py --check
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: Non-default ``history`` spec: the enc-dec metric names (the default spec carries the
#: forward model's ``.../1/decoder/5/sigma`` path, which an enc-dec run never logs) plus
#: both the pre- and post-rename reward keys, since this cohort spans that rename.
#: Reproduce with::
#:
#:   artifacts ensure --kind history --runs position-control-open-loop/runs.csv \
#:       --set keys='["eval/episode_reward/mean", "eval/lifespan/mean",
#:         "eval/net/3/action/0/task_obs/6/kl_divergence/mean",
#:         "eval/net/3/action/1/5/sigma/mean",
#:         "eval/net/3/action/1/decoder/5/sigma/mean",
#:         "eval/env/joint_l2_error/mean", "eval/env/root_pos_distance/mean",
#:         "eval/env/terminations/any/mean", "eval/env/terminations/root_too_far/mean",
#:         "episode_reward/mean", "lifespan_mean"]'
HISTORY_SPEC_ID = "hist2000-533d4b5c"

#: Offline ``eval`` artifacts, pinned. Needed only by ``torque_efference_aug11``, whose runs
#: died in their end-of-training eval and so have no ``final_eval/*`` keys; every other run
#: here reports its own inline eval and needs no artifact. This is the **VERSION 2** spec
#: rather than the producer's current default (`eval3ds-382e9e69`, VERSION 3) because V2 is
#: the one that covers all 46 runs of that sweep, and a uniform source within a series
#: matters more than being on the newest version: only 4 of the 23 have V3. What it costs is
#: measured rather than assumed -- see ``eval_calibration.py`` and report.md.
EVAL_SPEC_ID = "eval3ds-347333e3"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}", f"eval:{EVAL_SPEC_ID}"]

#: The 2026-08-11 torque sweep: one commit, one launch, delays 0..100 in 23 steps, `eff ==
#: delay`, `rollout_length` 20, all 600,064,000 steps, all dead in the final eval. Kept as
#: its own condition rather than merged into ``torque_efference`` so that each series is
#: internally uniform in commit, eval source, PPO settings and cluster stack -- which is
#: what makes it readable as a curve. ``torque_efference`` stays the same-week, same-commit
#: comparator for the position runs at delays 0-20.
AUG11_COMMIT = "ef060b73"

NEW_XML = "rodent_no_tail_collisions.xml"
OLD_XML = "rodent.xml"

#: The architecture every cell shares. Stored as JSON strings in the index.
STD_ARCH = {
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
}
STD_KL = 0.001

#: The two commits [`position-vs-torque-control/`](../position-vs-torque-control/) drew
#: its position-efference sweep from, and the one it drew its torque baseline from. Pinning
#: these makes the "previous setup" cells *that* analysis' runs, not a superset: the same
#: gates without the pin also admit 16 further torque and 1 further position run from other
#: commits, which would quietly change the curve we are checking ourselves against.
PREV_POS_COMMITS = ("891cd0d3", "b18513ae")
PREV_TORQUE_COMMITS = ("1cd5838f",)

#: `eval_env = train_env` was live until this date, so `eval/*` before it is a train-split
#: curve. See analysis/README.md §6.
HELD_OUT_EVAL_SINCE = pd.Timestamp("2026-08-20", tz="UTC")

#: Averaging window for the curve-derived reward: eval runs every 10 M steps and a single
#: point moves a few percent, so five points are averaged rather than the last one.
WINDOW = 50_000_000

CURVE_COLUMNS = {
    "reward_mean": ("eval/episode_reward/mean", "episode_reward/mean"),
    "lifespan_mean": ("eval/lifespan/mean", "lifespan_mean"),
    "encoder_kl": ("eval/net/3/action/0/task_obs/6/kl_divergence/mean",),
    "action_sigma": ("eval/net/3/action/1/5/sigma/mean",),
    "joint_l2_error": ("eval/env/joint_l2_error/mean",),
    "root_pos_distance": ("eval/env/root_pos_distance/mean",),
    "termination_rate": ("eval/env/terminations/any/mean",),
    "term_root_too_far": ("eval/env/terminations/root_too_far/mean",),
}

EVAL_DATASETS = ("train", "old_eval", "new_eval")
CTRL_DT = 0.01
TERMINATIONS = ("survived", "pose_error", "root_too_far", "root_too_rotated",
                "nan_termination")


# --------------------------------------------------------------------------------------
# condition selectors
# --------------------------------------------------------------------------------------


def _has_tag(df: pd.DataFrame, tag: str) -> pd.Series:
    return df["tags"].fillna("").str.split(",").apply(lambda tags: tag in tags)


def completed_training(df: pd.DataFrame) -> pd.Series:
    """Runs that trained to their full budget, ``state`` notwithstanding.

    ``state == "finished"`` is the usual gate, but it also drops the 2026-08-11 torque sweep
    (46 runs, `ef060b73`, note "New XML + reference_root."), which trained all 600 M steps
    and then **died in the end-of-training eval**. Those runs are the only complete torque
    delay sweep on the new XML and reference frame -- 0 to 100 in 23 steps -- so excluding
    them is what left the first version of this analysis with no torque arm past delay 20.

    The gate is ``summary._step >= config.ppo.total_steps`` rather than the date or the
    commit, because that is the property being asserted: the run got to the end. It is also
    exactly discriminating on this XML + frame -- of the 126 non-finished runs there, the 46
    from 2026-08-11 are the only ones that reach ``total_steps``, and none of them carries a
    ``final_eval/*`` key, which is the signature of dying in that eval. Everything else
    stopped mid-training and is correctly excluded (the crashed delay-200 position run
    ``nzaltgrr``, at 426 M of 600 M, among them).

    The cost is that these runs have no inline end-of-training eval, so their held-out
    numbers have to come from an offline ``eval`` artifact instead -- see
    ``EVAL_SPEC_ID`` and ``reward_source``.
    """
    reached = df["summary._step"] >= df["config.ppo.total_steps"]
    return (df["state"] == "finished") | reached.fillna(False)


def _base(df: pd.DataFrame, *, xml: str, frame: str) -> pd.Series:
    """Everything every cell holds fixed.

    The network is identified by **tags**, not ``network_class``: that column was added to
    the WandB config later and is null on every run before 2026-08, which is the whole
    previous-setup half of this cohort (and the 2026-08-11 sweep).

    ``regularized_training_mask`` drops the 2026-08-21..24 runs whose regularisation was
    silently zeroed. Three of them (``kbpfipxq`` delay 0, ``ul9wzzpl`` delay 5,
    ``4d06k2pq`` delay 10) would otherwise join ``torque_efference`` and one
    (``zllge7px``) ``torque_noproprio``; ``ul9wzzpl`` reports a reward of exactly 0.
    Nothing in ``net_params`` distinguishes them -- see analysis/README.md §6.
    """
    mask = (
        (df["env"] == "AbsoluteImitation")
        & completed_training(df)
        & (df["seed"] == 42)
        & _has_tag(df, "EncDec")
        & ~_has_tag(df, "ForwardModel")
        & df["env_params.walker_xml_path"].astype(str).str.endswith("/" + xml)
        & (df["env_params.body_target_frame"] == frame)
        & (df["net_params.latent_size"] == 32)
        & (df["net_params.min_std"] == 0.1)
        & pipeline.regularized_training_mask(df)
    )
    for column, value in STD_ARCH.items():
        mask &= df[column] == value
    return mask


def _new(df: pd.DataFrame) -> pd.Series:
    return _base(df, xml=NEW_XML, frame="reference_root")


def _mode(df: pd.DataFrame, torque: bool) -> pd.Series:
    return df["env_params.torque_actuators"] == torque


def _commit_in(df: pd.DataFrame, prefixes) -> pd.Series:
    commits = df["git_commit"].fillna("")
    mask = pd.Series(False, index=df.index)
    for prefix in prefixes:
        mask |= commits.str.startswith(prefix[:8])
    return mask


def _efference_baseline(df: pd.DataFrame, *, torque: bool) -> pd.Series:
    """Standard cell: all three decoder streams, efference matched to the delay."""
    return (_new(df) & _mode(df, torque)
            & pipeline.full_decoder_inputs_mask(df)
            & (df["net_params.kl_weight"] == STD_KL)
            & (df["delay_k"] == df["efference_length"]))


def _later_torque(df: pd.DataFrame) -> pd.Series:
    """``torque_efference`` proper: the baseline minus the separately-kept 2026-08-11 sweep."""
    return _efference_baseline(df, torque=True) & ~_commit_in(df, (AUG11_COMMIT,))


def _no_efference(df: pd.DataFrame, *, torque: bool) -> pd.Series:
    """Delayed proprioception, no efference copy. `efference_length == 0`, `delay_k > 0`."""
    return (_new(df) & _mode(df, torque)
            & pipeline.full_decoder_inputs_mask(df)
            & (df["net_params.kl_weight"] == STD_KL)
            & (df["delay_k"] > 0) & (df["efference_length"] == 0))


def _ablation(df: pd.DataFrame, *, torque: bool, flag: str) -> pd.Series:
    return (_new(df) & _mode(df, torque)
            & (df[f"net_params.{flag}"] == False))  # noqa: E712


def _kl_sweep(df: pd.DataFrame) -> pd.Series:
    """Position, delay 0, all streams, `kl_weight` above the standard 0.001."""
    return (_new(df) & _mode(df, torque=False)
            & pipeline.full_decoder_inputs_mask(df)
            & (df["net_params.kl_weight"] != STD_KL))


def _aug11_torque(df: pd.DataFrame) -> pd.Series:
    """The 2026-08-11 torque sweep: same gates as ``torque_efference``, pinned to its commit."""
    return _efference_baseline(df, torque=True) & _commit_in(df, (AUG11_COMMIT,))


def _previous(df: pd.DataFrame, *, torque: bool) -> pd.Series:
    return (_base(df, xml=OLD_XML, frame="current_root")
            & _mode(df, torque)
            & pipeline.full_decoder_inputs_mask(df)
            & (df["net_params.kl_weight"] == STD_KL)
            & (df["delay_k"] == df["efference_length"])
            & _commit_in(df, PREV_TORQUE_COMMITS if torque else PREV_POS_COMMITS))


# Conditions must be mutually exclusive; `select_conditions` raises if they are not. The
# no-efference and kl-sweep cells are carved out of the efference baseline by
# `efference_length == 0, delay_k > 0` and `kl_weight != 0.001` respectively, and the two
# decoder-input ablations by their flags -- so nothing overlaps.
CONDITIONS = {
    # --- new setup: the delay sweep -----------------------------------------------------
    "pos_efference": lambda df: _efference_baseline(df, torque=False),
    "torque_efference": _later_torque,
    "torque_efference_aug11": _aug11_torque,
    # --- new setup: the open-loop question ----------------------------------------------
    "pos_noproprio": lambda df: _ablation(df, torque=False,
                                          flag="dec_use_proprioception"),
    "torque_noproprio": lambda df: _ablation(df, torque=True,
                                             flag="dec_use_proprioception"),
    "pos_nointent": lambda df: _ablation(df, torque=False, flag="dec_use_intention"),
    "torque_nointent": lambda df: _ablation(df, torque=True, flag="dec_use_intention"),
    "pos_no_efference": lambda df: _no_efference(df, torque=False),
    "torque_no_efference": lambda df: _no_efference(df, torque=True),
    # --- new setup: the bottleneck sweep ------------------------------------------------
    "pos_kl_sweep": _kl_sweep,
    # --- the previous analysis' own runs (old XML, current_root) -------------------------
    "prev_pos_efference": lambda df: _previous(df, torque=False),
    "prev_torque_efference": lambda df: _previous(df, torque=True),
}

#: Must hold *within* every condition. `env_params.torque_actuators`,
#: `walker_xml_path` and `body_target_frame` vary *between* conditions by design and are
#: listed so comparability.txt shows where. `git_commit` varies within the new-setup cells
#: (2c147a6 / f4992e2 / b7c4b32) and within `prev_pos_efference` (891cd0d3 / b18513ae);
#: both diffs are checked by hand in report.md.
INVARIANTS = [
    "env", "seed",
    "net_params.latent_size", "net_params.min_std", "net_params.latent_min_std",
    "net_params.entropy_weight", "net_params.std_scale", "net_params.normalize_obs",
    "net_params.enc_hidden_sizes", "net_params.dec_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "env_params.clip_length", "env_params.ctrl_dt", "env_params.sim_dt",
    "env_params.solver", "env_params.iterations", "env_params.ls_iterations",
    "env_params.rescale_factor", "env_params.mujoco_impl",
    "env_params.reference_length", "env_params.reference_stride",
    "env_params.qvel_init", "env_params.start_frame_range",
    "env_params.reward_terms.joints.weight", "env_params.reward_terms.joints.exp_scale",
    "env_params.reward_terms.control_cost.weight",
    "env_params.reward_terms.energy_cost.weight",
    "env_params.termination_criteria.pose_error.max_l2_error",
    "env_params.termination_criteria.root_too_far.max_distance",
    "env_params.walker_xml_path", "env_params.body_target_frame",
    "env_params.torque_actuators",
    "config.ppo.n_envs", "config.ppo.learning_rate", "config.ppo.rollout_length",
    "config.ppo.n_epochs", "config.ppo.n_minibatches", "config.ppo.clip_range",
    "config.ppo.discounting_factor", "config.ppo.gae_lambda",
    "config.ppo.total_steps", "config.ppo.rollout_length", "summary._step",
    "net_params.kl_weight", "git_commit",
    "os", "cuda_version",
]

#: Columns the report's own prose depends on and that no invariant covers, so that a
#: rebuild would notice if they moved.
ARCH_FLAGS = ("dec_use_intention", "dec_use_proprioception")


# --------------------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------------------


def load_curve(store: Store, wandb_id: str) -> pd.DataFrame | None:
    """Tidy curve for one run, or ``None`` if it logged no eval series.

    Column names are coalesced across the mid-project logging rename, so a previous-setup
    run (``episode_reward/mean``) and a new one (``eval/episode_reward/mean``) come back
    with the same column names. What those numbers *mean* still differs -- see
    ``inline_split`` -- and this function deliberately does not hide that, it only spells
    the columns the same way.
    """
    entry = store.lookup("history", wandb_id, HISTORY_SPEC_ID)
    if entry is None:
        raise FileNotFoundError(
            f"no history:{HISTORY_SPEC_ID} for {wandb_id}; see this module's docstring "
            f"for the `artifacts ensure` command that makes it")
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None

    out = pd.DataFrame({"step": []})
    for name, sources in CURVE_COLUMNS.items():
        present = [s for s in sources if s in frame.columns]
        if name == "reward_mean":
            if not present:
                return None
            frame = frame.dropna(subset=[present[0]]).sort_values("_step")
            if frame.empty:
                return None
            out = pd.DataFrame({"step": frame["_step"].astype(int)})
        out[name] = frame[present[0]].to_numpy() if present else np.nan
    return out.reset_index(drop=True)


def load_eval(store: Store, run: pd.Series) -> dict | None:
    """The pinned offline ``eval`` record for one run, or ``None`` if it has none.

    Two independently written records are compared before the numbers are used: the
    artifact's ``resolved.walker_xml_path`` / ``env_class`` against the run's own
    ``env_params``, and its ``resolved.checkpoint_step`` against ``summary._step``. This is
    the ``assert_artifact_body`` pattern from analysis/README.md, and it is not ceremony
    here: an eval that silently re-simulated a run on the local default body is exactly how
    ``collision-model-xml`` got a retracted headline. Comparing the run's config against
    itself, as that folder originally did, could not have caught it.

    The step check earns its place too. These runs have no inline eval to cross-check
    against, so "this artifact evaluated the weights the run finished with" rests entirely
    on the checkpoint the producer happened to restore.
    """
    # Only where there is no inline eval to use, and only on the new setup. The
    # previous-setup cells are deliberately left on `window_reward` even though the torque
    # half of them *does* hold V2 eval artifacts: the position half holds none, and giving
    # one arm of a paired comparison a better metric than the other is worse than giving
    # both the same worse one. See caveat 8 in report.md.
    if run["condition"].startswith("prev_"):
        return None
    if _num(run.get("summary.final_eval/old_eval/episode_reward/mean")):
        return None

    entry = store.lookup("eval", run["wandb_id"], EVAL_SPEC_ID)
    if entry is None:
        return None
    record = json.loads((store.root / entry.path).read_text())

    stamped = (entry.resolved or {}).get("walker_xml_path")
    if stamped is None:
        raise ValueError(
            f"{run['wandb_id']}: eval artifact has no resolved.walker_xml_path, which means "
            f"it predates the 2026-08-18 fix and may have been simulated on the wrong body")
    if Path(str(stamped)).name != Path(str(run["env_params.walker_xml_path"])).name:
        raise ValueError(f"{run['wandb_id']}: eval artifact used {stamped}, run trained on "
                         f"{run['env_params.walker_xml_path']}")
    if record.get("env_class") != run["env"]:
        raise ValueError(f"{run['wandb_id']}: eval artifact env_class "
                         f"{record.get('env_class')} != run env {run['env']}")
    step, actual = record.get("step"), run.get("summary._step")
    if actual is not None and step != int(actual):
        raise ValueError(f"{run['wandb_id']}: eval artifact restored step {step}, run "
                         f"reached {int(actual)}")
    return record


def window_mean(curve: pd.DataFrame, column: str, step: int) -> float | None:
    """Mean of the eval points in ``(step - WINDOW, step]``; ``None`` if fewer than three."""
    sub = curve[(curve["step"] > step - WINDOW) & (curve["step"] <= step)]
    value = float(sub[column].mean()) if len(sub) >= 3 else None
    return None if value is not None and not np.isfinite(value) else value


# --------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------


def control_mode(run: pd.Series) -> str:
    """Read from ``env_params``, which is the only record of what the env actually used."""
    return "torque" if bool(run["env_params.torque_actuators"]) else "position"


def decoder_streams(run: pd.Series) -> str:
    """Which of the decoder's three input streams this run had.

    Spelled out rather than left implicit in the condition name because it is the axis Q2
    is about, and because ``efference_length == 0`` is a third ablation that no
    ``dec_use_*`` flag records.
    """
    streams = []
    if run.get("net_params.dec_use_intention") is not False:
        streams.append("intention")
    if run.get("net_params.dec_use_proprioception") is not False:
        streams.append("proprioception")
    if int(run["efference_length"]) > 0:
        streams.append("efference")
    return "+".join(streams)


def build_row(run: pd.Series, curve: pd.DataFrame | None,
              evaluation: dict | None) -> dict:
    created = pd.Timestamp(run["created_at"])
    row = {
        "condition": run["condition"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "wandb_project": PROJECT,
        "git_commit": (run["git_commit"] or "")[:8],
        "created_at": run["created_at"],
        "gpu": run["gpu"],
        "setup": "previous" if run["condition"].startswith("prev_") else "new",
        "control_mode": control_mode(run),
        "walker_xml": str(run["env_params.walker_xml_path"]).rsplit("/", 1)[-1],
        "body_target_frame": run["env_params.body_target_frame"],
        "delay_k": int(run["delay_k"]),
        "efference_length": int(run["efference_length"]),
        "kl_weight": run["net_params.kl_weight"],
        "dec_use_intention": run.get("net_params.dec_use_intention"),
        "dec_use_proprioception": run.get("net_params.dec_use_proprioception"),
        "decoder_streams": decoder_streams(run),
        # `delay_k` cannot reach the network at all when the proprioception branch is not
        # built: it is that branch that holds the Delay. Recorded per row so a figure can
        # assert it rather than the reader having to remember it.
        "delay_is_effective": run.get("net_params.dec_use_proprioception") is not False,
        "seed": int(run["seed"]),
        "actual_step": run.get("summary._step"),
        "restart_count": run.get("requeue.restart_count"),
        # The one PPO knob that is not constant across this cohort: three torque runs
        # (commit 4245ae4, including the only delay-20 one) trained with rollout_length
        # 60 rather than 20. Kept as a column so a figure can mark them and the report
        # can price the difference off the delays where both settings exist.
        "rollout_length": run.get("config.ppo.rollout_length"),
        "n_envs": run.get("config.ppo.n_envs"),
        "os": run.get("os"),
        "cuda_version": run.get("cuda_version"),
        # `repos.*.dirty=True` voids the commit as an identifier of the code that ran
        # (analysis/README.md §4). True for every new-setup run, absent before 2026-08-24.
        "repos_dirty": _any_dirty(run),
        # Whether this run's WandB `eval/*` series is held out or on the training clips.
        "inline_split": ("held_out" if created >= HELD_OUT_EVAL_SINCE else "train"),
        "inline_reward": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
        "inline_lifespan": pipeline.first_present(
            run, "summary.eval/lifespan/mean", "summary.lifespan_mean"),
        "encoder_kl": run.get(
            "summary.eval/net/3/action/0/task_obs/6/kl_divergence/mean"),
        "action_sigma": run.get("summary.eval/net/3/action/1/5/sigma/mean"),
    }

    row["n_eval_points"] = 0 if curve is None else len(curve)
    row["max_step"] = None if curve is None else int(curve["step"].max())
    if curve is not None:
        last = int(curve["step"].max())
        for column in CURVE_COLUMNS:
            row[f"window_{column}"] = window_mean(curve, column, last)
    else:
        for column in CURVE_COLUMNS:
            row[f"window_{column}"] = None
    row["window_reward"] = row.pop("window_reward_mean")

    # Where the three-dataset numbers come from. Preference order, and why:
    #
    # 1. the run's own **inline** end-of-training eval (`final_eval/*`) -- one pass over
    #    all three datasets on the weights the run finished with;
    # 2. the pinned **offline** eval artifact, for the 2026-08-11 sweep, which died in
    #    that eval and has no `final_eval` at all.
    #
    # These are different code paths and must not be mixed silently, so `reward_source`
    # records which one every row used. They are not, however, far apart: on the runs that
    # hold both, the offline V2 number is +0.33 % from the inline one on average (median
    # +0.47 %, worst 1.2 %) -- an order of magnitude below the 3 % replicate noise floor
    # and two below the effects being measured. `eval_calibration.py` in this folder
    # measures that and writes `eval_calibration.txt`; report.md carries it as a caveat.
    # The weights are the same object in both: `total_steps` is a multiple of
    # `checkpoint_every_steps`, so the last checkpoint *is* the final state, and
    # `load_eval` asserts the artifact restored exactly `summary._step`.
    if evaluation is not None:
        row["reward_source"] = f"eval:{EVAL_SPEC_ID}"
    elif _num(run.get("summary.final_eval/old_eval/episode_reward/mean")):
        row["reward_source"] = "final_eval"
    else:
        row["reward_source"] = "none"

    for dataset in EVAL_DATASETS:
        record = (evaluation or {}).get("datasets", {}).get(dataset)
        if record is not None:
            def field(*path, _r=record):
                node = _r
                for key in path:
                    if not isinstance(node, dict) or key not in node:
                        return None
                    node = node[key]
                return node
        else:
            prefix = f"summary.final_eval/{dataset}"

            def field(*path, _p=prefix):
                return run.get(_p + "/" + "/".join(str(k) for k in path))

        reward = field("episode_reward", "mean")
        life = field("lifespan_steps", "mean")
        survived = field("termination_rate", "survived")
        row[f"{dataset}_reward"] = reward
        row[f"{dataset}_lifespan_steps"] = life
        # Cumulative reward is not comparable across the 5 s and 30 s datasets; these two
        # are (see analysis/README.md §6).
        row[f"{dataset}_reward_per_step"] = (
            reward / life if _num(reward) and _num(life) else None)
        row[f"{dataset}_hazard"] = (
            (1.0 - survived) / (life * CTRL_DT)
            if _num(survived, zero_ok=True) and _num(life) else None)
        for termination in TERMINATIONS:
            row[f"{dataset}_term_{termination}"] = field("termination_rate", termination)
        row[f"{dataset}_joint_l2_error"] = field("errors", "joint_l2_error")
        row[f"{dataset}_root_pos_distance"] = field("errors", "root_pos_distance")
        row[f"{dataset}_encoder_kl"] = field(
            "net_metrics", "3", "action", "0", "task_obs", "6", "kl_divergence")
        row[f"{dataset}_action_sigma"] = field(
            "net_metrics", "3", "action", "1", "5", "sigma")

    # The generalisation gap Q2a is about: same weights, same clip length, unseen clips.
    if _num(row["train_reward"]) and _num(row["old_eval_reward"]):
        row["heldout_gap_frac"] = 1.0 - row["old_eval_reward"] / row["train_reward"]
    else:
        row["heldout_gap_frac"] = None
    return row


def _any_dirty(run: pd.Series):
    """``True`` if any of the three repos was dirty; ``None`` if unrecorded."""
    flags = [run.get(f"repos.{repo}.dirty")
             for repo in ("nnx_ppo", "vnl_playground", "vnl_experiments")]
    present = [f for f in flags if f is not None and f is not np.nan and f == f]
    return None if not present else bool(any(present))


def _num(value, *, zero_ok: bool = False) -> bool:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    return True if zero_ok else bool(value)


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    rows, curves = [], []
    for _, run in runs.iterrows():
        curve = load_curve(store, run["wandb_id"])
        rows.append(build_row(run, curve, load_eval(store, run)))
        if curve is None:
            continue
        tagged = curve.copy()
        tagged.insert(0, "efference_length", int(run["efference_length"]))
        tagged.insert(0, "delay_k", int(run["delay_k"]))
        tagged.insert(0, "condition", run["condition"])
        tagged.insert(0, "wandb_id", run["wandb_id"])
        curves.append(tagged)

    df = pd.DataFrame(rows).sort_values(
        ["condition", "delay_k", "efference_length", "wandb_id"], ignore_index=True)
    curves_df = pd.concat(curves, ignore_index=True).sort_values(
        ["condition", "delay_k", "wandb_id", "step"], ignore_index=True)

    report = comparability_report(runs, invariant_cols=INVARIANTS,
                                  group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report + "\n")

    print("\nWhere each condition's three-dataset numbers came from:")
    for (condition, source), group in df.groupby(["condition", "reward_source"]):
        print(f"  {condition:24s} {source:24s} n={len(group)}")

    print("\nCohort:")
    for condition, sub in df.groupby("condition"):
        print(f"  {condition:22s} n={len(sub):2d}  {sub['control_mode'].iat[0]:8s} "
              f"streams={sorted(set(sub['decoder_streams']))}")
        print(f"  {'':22s} delay={sorted(sub['delay_k'])} "
              f"eff={sorted(sub['efference_length'])} "
              f"kl={sorted(set(sub['kl_weight']))}")

    # A run whose network never saw `delay_k` must not be read as a delay data point.
    inert = df[~df["delay_is_effective"] & (df["delay_k"] > 0)]
    if not inert.empty:
        print("\nRuns whose delay_k is architecturally inert (no proprioception branch, "
              "so no Delay layer). Their x-axis is efference_length:")
        print(inert[["wandb_id", "wandb_name", "delay_k", "efference_length"]]
              .to_string(index=False))

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
