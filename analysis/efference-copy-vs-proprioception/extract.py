"""Can an efference copy replace ablated proprioception -- and does its length matter?

The question
------------
Ablating the decoder's proprioception stream (``net_params.dec_use_proprioception=False``)
leaves the policy with no sensory feedback about its own body at all. The remaining input
is the *intention* latent -- the imitation target, encoded -- plus, optionally, an
**efference copy**: a queue of the policy's own ``efference_length`` most recent actions.
So:

1. How much of the ablation's cost does an efference copy buy back?
2. Does the **length** of the queue matter, and where does it saturate?
3. Does the answer differ between **position** and **torque** actuators?

Why the answer could plausibly differ by control mode
-----------------------------------------------------
``build_delay_network`` gives the decoder ``latent_size + efference_length * action_size``
inputs once proprioception is gone (32 + 38L here). Under **position** control the action
*is* a target joint angle -- ``force = kp * (offset + scale * ctrl - qpos)`` -- so the
recent-action queue is a proxy for recent *commanded posture*, and in a servo whose
40 ms FILTER lag dominates, commanded posture is a decent estimate of actual posture. Under
**torque** control the action is a generalised force; recovering configuration from a torque
history requires integrating twice through unknown contact, so the same queue carries far
less information about where the body is. The efference copy is therefore expected to
substitute for proprioception much better under position control -- this analysis is the
test of that.

What "no proprioception" does and does not remove
-------------------------------------------------
It removes the proprioception *branch* of the actor. It does **not** make the policy open
loop, because ``task_obs`` still reaches it undelayed through the encoder, and under
``body_target_frame=reference_root`` the ``root`` and ``quat`` sub-keys of ``task_obs`` are
the reference root pose relative to the **current** root -- an undelayed root position and
orientation error. See [`../position-control-open-loop/frame_leak.py`](
../position-control-open-loop/frame_leak.py), which measures it, and README §6. Whole-body
*configuration* is genuinely gone; global root error is not. Nothing here turns on the
difference, but "open loop" would be the wrong word for these runs.

Why runs with different ``delay_k`` are pooled
----------------------------------------------
``build_delay_network`` puts the ``Delay`` layer *inside* the proprioception branch, and
``dec_use_proprioception=False`` does not construct that branch -- so ``delay_k`` reaches
nothing and a ``delay10_eff10_noproprio`` run is bit-identical to a ``delay0_eff10_noproprio``
one (asserted in [`../position-control-open-loop/check_delay_inert.py`](
../position-control-open-loop/check_delay_inert.py)). Pooling across ``delay_k`` at fixed
``efference_length`` is therefore sound, and this cohort *re-tests* it empirically: it holds
three pairs that differ only in that inert knob (README §6, and ``delay_inertness`` in
report.md).

Why the readout is at 600 M steps
---------------------------------
Every arm is read at the **full 600 M** budget: the mean of the eval points in (550 M,
600 M]. This was not possible in the first version of this analysis -- the 2026-09-02
position sweep lost six runs to cluster-filesystem errors partway through training, so a
600 M readout would have had no position arm above ``efference_length = 2``, and everything
had to be read at 400 M instead. The 2026-09-03 relaunch closed that: both arms are now
complete at 600 M except position ``efference_length = 3`` and torque
``efference_length = 50``, which are interior points rather than anchors.

``reward_400M`` is still computed for every run, for two reasons that outlive the gap it was
invented for. It keeps the crashed runs contributing something (they are replicates at 400 M
of points the relaunch also covers, so they are a free check on the relaunch), and the
difference between the two columns *is* a result: the gain from 400 M to 600 M grows with
``efference_length``, because the queue is concatenated onto the decoder's input and a
3 832-wide first layer learns more slowly than a 108-wide one. Reading at 400 M understated
one end of the x-axis, which is why the earlier version could not settle the shape of the
curve past the plateau and this one can.

Run it
------
    ../.venv/bin/python analysis/efference-copy-vs-proprioception/extract.py
    ../.venv/bin/python analysis/efference-copy-vs-proprioception/extract.py --sync --refresh
    ../.venv/bin/python analysis/efference-copy-vs-proprioception/extract.py --check

Frozen is the default. This is the only script in the folder that reads the index or the
artifact store; ``plot.py`` reads the CSVs it writes and nothing else.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: The history spec the producer currently defaults to. Pinned, so this folder keeps
#: resolving these exact files after a future VERSION bump (README §2).
HISTORY_SPEC_ID = "hist2000-09fea177"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

XML_ROOT = ("/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/vnl-playground/"
            "vnl_playground/tasks/rodent/xmls")
NEW_XML = f"{XML_ROOT}/rodent_no_tail_collisions.xml"

#: ``eval_env = train_env`` until this date, so ``eval/*`` logged before it is a
#: *train-split* number and not comparable with a later one. Every run here is after it,
#: which is why the whole cohort can be read off the training curves at all.
#:
#: This is a gate on what the **metric means**, not on which code version produced it, so
#: it stays even though the stack-version gates below have been relaxed. It currently
#: excludes no otherwise-eligible run: the decoder-input ablation flags were only added on
#: 2026-08-21, so no ablated run predates it (``selection_audit.py``).
HELD_OUT_EVAL_SINCE = pd.Timestamp("2026-08-20", tz="UTC")

#: Every PPO hyperparameter that shapes the optimisation. The cohort is required to match
#: on all of them, not on the two that happened to get filtered first: "the parameters
#: match" is the standard this analysis is held to, and ``rollout_length`` and
#: ``n_minibatches`` really do vary elsewhere in the project (60/40 and 16), so pinning
#: them is a live filter rather than a no-op.
#:
#: ``total_steps`` is deliberately **absent**. It sets how long a run goes, not how it
#: learns -- ``nnx_ppo.algorithms.ppo`` passes ``learning_rate`` to optax as a constant
#: with no schedule, so nothing about the optimisation depends on the configured horizon,
#: and a run budgeted for 2 G steps is directly comparable at 400 M with one budgeted for
#: 600 M. ``weight_decay`` is absent too: it is ``None`` for all 684 runs in the project,
#: so an equality filter on it would only add a None-handling branch.
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


def has_tag(tags: object, tag: str) -> bool:
    """Whether ``tag`` is in an index ``tags`` cell, which may be a list or its repr."""
    if isinstance(tags, str):
        return tag in [t.strip() for t in tags.strip("[]").replace("'", "").split(",")]
    if isinstance(tags, (list, tuple)):
        return tag in tags
    return False


def sound_training_mask(df: pd.DataFrame) -> pd.Series:
    """Exclude runs known to have trained wrongly: the ``BUG`` tag, or a broken commit.

    Two rules for one set, on purpose. The ``BUG`` tag is the project's own marker and the
    one to honour; ``UNREGULARIZED_COMMITS`` is the code-derived test for the 2026-08-21/24
    runs that trained with ``entropy_weight``/``kl_weight``/``min_std`` silently zeroed. As
    of this cohort the two agree **exactly** -- 21 runs each, no run in one and not the
    other, asserted in ``selection_audit.py`` -- so taking the union costs nothing now and
    catches whichever of the two a future run is missing.
    """
    return ~(df["tags"].apply(has_tag, tag="BUG")
             | df["git_commit"].isin(pipeline.UNREGULARIZED_COMMITS))

#: The primary training budget and the earlier one kept for the stability check, plus the
#: width of the averaging window ending at each. A single eval point is noisy and the last
#: one is not a measurement (README §6), so every reward here is the mean of the points in
#: ``(step - WINDOW, step]``. At one eval per 10 M steps that is five points per window.
BUDGET = 600_000_000
EARLY = 400_000_000
WINDOW = 50_000_000
MIN_POINTS = 3

#: ``(column suffix, budget)`` in the order the columns are written. The first entry is the
#: primary readout: ``usable_<suffix>`` is derived from it and every figure defaults to it.
READOUTS = (("600M", BUDGET), ("400M", EARLY))

#: Observation and action widths of this env/body, used only to report the decoder's
#: input width per run (``32 + 38 * efference_length`` once proprioception is gone).
#: Measured in ``../position-control-open-loop/`` on the same XML.
TASK_OBS_SIZE = 640
PROPRIO_SIZE = 277
ACTION_SIZE = 38


def _standard(df: pd.DataFrame) -> pd.Series:
    """Everything that must hold for a run to be in this cohort at all.

    Deliberately strict and deliberately expressed on ``env_params`` / ``net_params``
    rather than on tags or the run name: the cluster working copy drifts from the
    committed script, so the logged config is the only record of what ran (README §6).
    """
    mask = (
        (df["env"] == "AbsoluteImitation")
        & (df["env_params.walker_xml_path"] == NEW_XML)
        & (df["env_params.body_target_frame"] == "reference_root")
        # Not relaxed to "this value or absent", even though ~490 older runs never recorded
        # the key: `RodentForwardModel` runs did not record it either, so accepting absence
        # would silently admit a different architecture. Only the run *name* separates them
        # there, and README §6 is explicit that names are not evidence.
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
        & (pd.to_datetime(df["created_at"], utc=True) >= HELD_OUT_EVAL_SINCE)
    )
    for key, value in PPO_PARAMS.items():
        mask &= df[f"config.ppo.{key}"] == value
    return mask


def _mode(df: pd.DataFrame, torque: bool) -> pd.Series:
    return df["env_params.torque_actuators"] == torque


def _noproprio(df: pd.DataFrame) -> pd.Series:
    """Proprioception ablated, intention kept -- the arm this question sweeps."""
    return ((df["net_params.dec_use_proprioception"] == False)  # noqa: E712
            & (df["net_params.dec_use_intention"] != False))    # noqa: E712


def _intact(df: pd.DataFrame) -> pd.Series:
    """The performance the ablation has to recover: all streams, no delay, no efference.

    ``full_decoder_inputs_mask`` rather than a filter kwarg because the flags are *absent*
    on every run predating them, which a ``== True`` filter would read as a mismatch.
    """
    return (pipeline.full_decoder_inputs_mask(df)
            & (df["delay_k"] == 0)
            & (df["net_params.efference_length"] == 0))


def _nointent(df: pd.DataFrame) -> pd.Series:
    """Intention ablated instead: a task-blind reference, standing in for a missing point.

    Not the same manipulation as this question's arm and not on its x-axis, but the torque
    arm has no ``efference_length = 0`` run, so it has no measured left anchor. These runs
    (proprioception present and delayed by 10, efference 10, no imitation target) bound how
    much reward a policy scores when it cannot know *what* to imitate, which is the scale
    the flat torque line has to be judged against.
    """
    return df["net_params.dec_use_intention"] == False  # noqa: E712


# One entry per experimental cell; `select_conditions` raises if a run matches two.
CONDITIONS = {
    "pos_noproprio": lambda df: _standard(df) & _mode(df, False) & _noproprio(df),
    "torque_noproprio": lambda df: _standard(df) & _mode(df, True) & _noproprio(df),
    "pos_intact": lambda df: _standard(df) & _mode(df, False) & _intact(df),
    "torque_intact": lambda df: _standard(df) & _mode(df, True) & _intact(df),
    "pos_nointent": lambda df: _standard(df) & _mode(df, False) & _nointent(df),
    "torque_nointent": lambda df: _standard(df) & _mode(df, True) & _nointent(df),
}

# Must be constant for the comparison to be fair; comparability.txt flags what varies.
# `env_params.torque_actuators` is the manipulation and `net_params.efference_length` the
# x-axis, so neither is here.
#
# `repos.nnx_ppo.*` and `repos.vnl_playground.*` are deliberately **not** here, which
# departs from README §4. The owner of these repos has confirmed that no
# compatibility-breaking change landed across the span this cohort covers, so a differing
# nnx-ppo commit is not evidence of a differing experiment. The evidence is kept rather
# than discarded: `pooling_check.txt` still measures two groups that straddle the nnx-ppo
# commit and the CUDA upgrade, and bounds their combined effect at <= 1.06 %.
INVARIANTS = [
    "env",
    "env_params.walker_xml_path",
    "env_params.body_target_frame",
    "env_params.ctrl_dt",
    "env_params.clip_length",
    "env_params.reference_data_path",
    "net_params.network_class",
    "net_params.enc_hidden_sizes",
    "net_params.dec_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.latent_size",
    "net_params.kl_weight",
    "net_params.entropy_weight",
    "net_params.min_std",
    "config.ppo.total_steps",
    "seed",
    "git_commit",
    "cuda_version",
    "os",
    "gpu",
] + [f"config.ppo.{key}" for key in PPO_PARAMS]

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")
LIFESPAN_KEYS = ("eval/lifespan/mean", "lifespan_mean")


def load_curve(store: Store, wandb_id: str) -> pd.DataFrame | None:
    """The run's sampled eval curve as ``step / reward / lifespan``, or None.

    Returns None rather than raising for a run that produced no history rows at all --
    three runs in this cohort died before their first eval, which is a fact to report in
    coverage.txt and data.csv, not an error.
    """
    entry = store.lookup("history", wandb_id, HISTORY_SPEC_ID)
    if entry is None:
        return None
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None
    if "_step" not in frame.columns or frame.empty:
        return None
    reward = next((k for k in REWARD_KEYS if k in frame.columns), None)
    if reward is None:
        return None
    lifespan = next((k for k in LIFESPAN_KEYS if k in frame.columns), None)
    out = pd.DataFrame({"step": frame["_step"], "reward": frame[reward]})
    out["lifespan"] = frame[lifespan] if lifespan else np.nan
    return out.dropna(subset=["reward"]).sort_values("step", ignore_index=True)


def window_stats(curve: pd.DataFrame | None, end: int) -> dict:
    """Mean reward/lifespan over the eval points in ``(end - WINDOW, end]``.

    Reports the window it actually used -- how many points, and where they stop -- so a
    truncated window (a run that died inside it) is visible in the committed CSV instead
    of being averaged away silently.
    """
    blank = {"reward": None, "lifespan": None, "points": 0,
             "last_step": None, "complete": False}
    if curve is None:
        return blank
    sub = curve[(curve["step"] > end - WINDOW) & (curve["step"] <= end)]
    if len(sub) < MIN_POINTS:
        return blank
    return {
        "reward": round(float(sub["reward"].mean()), 2),
        "lifespan": (round(float(sub["lifespan"].mean()), 3)
                     if sub["lifespan"].notna().any() else None),
        "points": int(len(sub)),
        "last_step": int(sub["step"].max()),
        # True when the run survived to the end of the window, so the mean covers the
        # whole 50 M rather than a truncated head of it.
        "complete": bool(curve["step"].max() >= end),
    }


def build_row(run: pd.Series, store: Store) -> dict:
    curve = load_curve(store, run["wandb_id"])
    readouts = {label: window_stats(curve, end) for label, end in READOUTS}

    efference = run.get("net_params.efference_length")
    use_intention = run.get("net_params.dec_use_intention") is not False
    use_proprio = run.get("net_params.dec_use_proprioception") is not False

    row = {
        "condition": run["condition"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "state": run["state"],
        "created_at": run["created_at"],
        "git_commit": str(run["git_commit"])[:8],
        "gpu": run["gpu"],
        "mode": "torque" if run["env_params.torque_actuators"] else "position",
        "efference_length": efference,
        # Inert in the noproprio arm -- the Delay layer lives in the branch that is not
        # built -- and kept only so the pooling can be checked rather than asserted.
        "delay_k": run.get("delay_k"),
        "use_intention": use_intention,
        "use_proprioception": use_proprio,
        # What the decoder's first layer actually sees, which is the confound at the long
        # end of the sweep: 32 + 38 * L once proprioception is gone, so 3 832 at L = 100.
        "decoder_in_width": (
            ((32 if use_intention else 0)
             + (PROPRIO_SIZE if use_proprio else 0)
             + int(efference * ACTION_SIZE))
            if pd.notna(efference) else None),
        "actual_step": run.get("summary._step"),
        "hist_max_step": int(curve["step"].max()) if curve is not None else None,
        "eval_points": int(len(curve)) if curve is not None else 0,
        # Every run is post-2026-08-20, so `eval/*` is the held-out split throughout.
        "eval_split": ("held_out"
                       if pd.to_datetime(run["created_at"], utc=True)
                       >= HELD_OUT_EVAL_SINCE else "train"),
        "repos_dirty": bool(run.get("repos.nnx_ppo.dirty") is True
                            or run.get("repos.vnl_playground.dirty") is True),
    }
    for label, _ in READOUTS:
        stats = readouts[label]
        row[f"reward_{label}"] = stats["reward"]
        row[f"lifespan_{label}"] = stats["lifespan"]
        # Reward per surviving step separates "tracks the reference worse" from "falls
        # over sooner"; total episode reward confounds the two.
        row[f"reward_per_step_{label}"] = (
            round(stats["reward"] / stats["lifespan"], 4)
            if stats["reward"] is not None and stats["lifespan"] else None)
        row[f"window_points_{label}"] = stats["points"]
        row[f"window_last_step_{label}"] = stats["last_step"]
        row[f"window_complete_{label}"] = stats["complete"]
    primary = READOUTS[0][0]
    row[f"usable_{primary}"] = row[f"reward_{primary}"] is not None
    row["usable_400M"] = row["reward_400M"] is not None
    return row


def build_curves(runs: pd.DataFrame, store: Store) -> pd.DataFrame:
    """Long-form eval curves for every cohort run, for the training-curve figure."""
    frames = []
    for _, run in runs.iterrows():
        curve = load_curve(store, run["wandb_id"])
        if curve is None:
            continue
        curve = curve.assign(condition=run["condition"], wandb_id=run["wandb_id"],
                             efference_length=run.get("net_params.efference_length"),
                             mode="torque" if run["env_params.torque_actuators"]
                             else "position")
        frames.append(curve)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.sort_values(["condition", "wandb_id", "step"], ignore_index=True)


def main() -> None:
    args = pipeline.parse_args(__doc__)
    store = Store()

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    df = pd.DataFrame([build_row(run, store) for _, run in runs.iterrows()])
    df = df.sort_values(["condition", "efference_length", "wandb_id"],
                        ignore_index=True)
    curves = build_curves(runs, store)

    report = comparability_report(runs, invariant_cols=INVARIANTS,
                                  group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
    print(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
