"""On WalkerWalk, do a joint servo and an explicit forward model *add up* against delay,
or are they two ways of buying the same thing?

Two manipulations have each been shown to buy delay tolerance in this project, and they
are motivated by opposite stories about where the missing information comes from:

* **the joint servo** (``../walker-joint-stiffness/``) is peripheral and mechanical. The
  actuator becomes ``tau = kp*(q* - q) - kv*qdot``, so it corrects position error inside
  every ``sim_dt`` substep -- the only *undelayed* feedback path in an otherwise delayed
  system. It does not estimate anything; it just does not have to wait.
* **the explicit forward model** (``../walker-forward-model-1b/``) is central and
  computational. A supervised predictor maps (delayed obs, actions in flight) -> current
  obs and hands the estimate to the actor, so the actor acts on a *guess* about now
  rather than on a fact about then.

If both work because they close the same gap -- the state at the moment of acting -- the
second one added should buy much less than the first, and the 2x2 should be strongly
sub-additive. If they close different gaps, the effects should stack.

This folder is the 2x2. It is a genuine crossing: WalkerWalk x ``servo_kp`` in {0, 64} x
``network_class`` in {DelayedMLP, FlatForwardModel}, all at 1e9 steps, over delays
0/2/3/5/7/10/12/15/20/25 -- complete in all four arms at eight of those ten.

Why ``servo_kp = 64`` and why 1e9
---------------------------------
``servo_kp = 64`` is not a new choice: ``../walker-joint-stiffness/`` swept
0.25-128 at 4.8e8 and 64 is the stiffness with the most runs, at the stiff end where the
sweep's effect was monotone and largest. Fixing it makes this folder a 2x2 rather than a
2xN, at the cost of saying nothing about whether the interaction depends on stiffness.

**1e9 is forced, not preferred**: there is no ``FlatForwardModel`` x ``servo_kp = 64`` run
at 4.8e8 at all, so the 2x2 exists only at this budget. That is fortunate rather than
awkward. The stiffness folder's central caveat was that 4.8e8 is short of where the torque
arm transitions, so its +2.7e2 gap at delay 10 was largely a *rate* difference; at 1e9 the
sibling forward-model folder measured that same delay-10 arm gap fall from +2.7e2 to
+2.6e1. Every run here has the longer budget, so none of the four arms is being read
mid-transition in the way that trap describes -- see ``late_gain`` for the per-run check.

The four arms
-------------
=================  ==================  ===============================================
condition          ``servo_kp``        network
=================  ==================  ===============================================
``mlp_torque``     0 (shipped plant)   ``DelayedMLP``          -- the doubly-baseline arm
``mlp_servo``      64                  ``DelayedMLP``
``fm_torque``      0 (shipped plant)   ``FlatForwardModel``
``fm_servo``       64                  ``FlatForwardModel``
=================  ==================  ===============================================

``efference_length == delay_k`` in every run of all four arms, so the two network arms are
given exactly the same information and differ only in whether the state estimate is built
explicitly. ``fm_loss_weight = 1`` and ``detach_prediction = True`` throughout: the
predictor is trained by its own regression loss and not by the policy gradient.

Force parity is enforced on the servo side (``forcerange = [-1, 1]`` in *actuator* space,
``envs/servo_control.md`` §1.2), so a servo arm has exactly the torque authority a torque
arm has and cannot win by being stronger. It is still a **different plant**, not the same
actuator with a different interface -- the servo adds ``-kp*q - kv*qdot`` inside every
substep. That is the manipulation, not a confound, but it is why "torque" and "servo" are
conditions here and not a readout column.

The two readouts, and why both
------------------------------
``reward_final`` -- mean of the eval points in the last 5e7 steps, not the run's last
point (README §6: one inline eval point moves by a few percent). Every run has the same
1e9 budget, so this needs no cross-budget alignment.

``steps_to_900`` -- the first step at which the trailing 2.4e7-step mean of eval reward
reaches 900, on a common 4.8e6 grid. The stiffness pilot showed why both are needed: at
delay 0 the torque arm *ends* level with the servo arms and yet reaches criterion twice as
fast, so a level-only readout would have reported "no effect" where the honest answer is
"same ceiling, different rate".

Two definitional choices that a reader should be able to check:

* **The common grid is not optional and it is not cosmetic.** Eval cadence is
  ``6e5`` on most torque runs and ``4.8e6`` on *every* servo run -- so cadence is
  confounded with the manipulation. A trailing window defined in steps would average 8x
  more samples on the torque side, making its criterion strictly harder to trigger on an
  upward fluctuation, which biases in favour of the servo. :func:`on_common_grid` puts
  every run on the coarse grid (nearest single sample, so each grid point is one
  256-episode eval in both cadences) and ``main`` reports what the correction moved.
  ``steps_to_900_native`` keeps the un-regridded number so the size of the correction is
  in ``data.csv`` rather than only in a log.
* **900 is this project's convention, not a published standard**, shared with both sibling
  folders. dm_control returns are bounded in [0, 1000] by construction, and the benchmark
  literature reports curves rather than declaring a task solved. 900 sits clearly above
  where these runs plateau when they fail and clearly below where they plateau when they
  succeed. ``steps_to_800`` and ``steps_to_950`` are extracted so ``plot.py`` can show the
  ordering does not depend on it.

A run that never reaches a criterion is **censored, not missing**: ``solved_*`` is False,
``steps_to_*`` is blank and ``smooth_max`` records how close it got, so ``plot.py`` can
draw it at the budget rather than let a line stop (which reads as absent data).

Selection gates that no run name would reveal
---------------------------------------------
* **budget.** ``total_steps == 1e9`` exactly, *and* the run must have reached it. The
  4.8e8, 2e9 and 4e9 WalkerWalk runs all exist and are all excluded: 4.8e8 has no
  ``fm_servo`` cell, and the longer runs exist only on the torque side, so admitting
  either would make budget the confound in a comparison whose whole point is a clean
  crossing.

  Seven candidate runs are dropped by the reached-it half of that gate, and they are
  **all in one arm** (``fm_servo``, seed 53), which is exactly the correlation README §6
  says to check before reading anything into a gap. ``attrition.py`` checks it and
  answers it with throughput: all seven ran at 5.3-5.4e4 ``train_sps`` against 1.7-2.2e5
  for every run that finished, a uniform 3.3-4x slowdown across delays 2 to 20, and all
  seven stopped after 2.9 h at ~4.5e8 steps. They hit a wall clock on contended nodes.
  The attrition is a throughput artefact, it does not vary with delay, and nothing about
  it involves the manipulation.
* **delay.** 0/2/3/5/7/10/12/15/20/25 -- every delay where at least three of the four
  arms exist. Delay **30** is excluded because only ``fm_servo`` was run there; one line
  of four running alone to the right edge of a panel reads as the other three failing
  rather than as never having been launched, and its single run is quoted in report.md
  instead. The two remaining holes, ``mlp_servo`` at delays 3 and 25, are reported by
  :func:`design_grid` and leave the *interaction* undefined at those delays while the
  simple effect of the servo *within the forward-model arm* is still measurable -- which
  at delay 25 is the point, since that is the first delay where the forward model alone
  leaves real headroom (``../walker-forward-model-1b/`` measures it at 8.5e2 there).
* **min_std.** The project contains a ``min_std`` sweep (0.01-0.2) whose runs are named
  exactly like the defaults. Gated to 0.001.
* **the eval-env fix (``971ab99``, 2026-09-09).** Before it the eval env inherited the
  training wrappers, so ``eval/*`` was scaled x10 *and* truncated by ``EpisodeWrapper``'s
  random phase (track README). Gated on the commit, not the date. This also excludes the
  pre-Hydra era outright, which has no ``net_params.*`` at all.
* **``servo_kp`` is read with absent == 0.** A run predating ``540e356`` has no
  ``env_params.servo_kp`` at all and its env is the unpatched Playground task, which is
  exactly what ``servo_kp = 0`` means: ``apply_servo`` returns at ``if kp == 0.0`` before
  writing any field or re-running ``put_model``. The two spellings are unified in
  :func:`_kp` and nowhere else.
* **``servo_center`` is *not* a gate.** The servo runs carry ``range`` and the torque runs
  carry ``qpos0`` or nothing, but at ``servo_kp = 0`` the field is never read, so on the
  torque side it is a launch label and not a physical property. It is kept as a column.

What is confounded with the manipulation, and what is done about it
-------------------------------------------------------------------
* **Seed.** The servo arms are seeds 51/52/53 and the torque arms 43/46/47/48: no seed
  appears in both. Unavoidable -- the servo runs are a later launch -- and it means an
  arm difference smaller than the within-arm seed spread is not readable. ``seed_spread``
  writes that spread per cell so the noise floor comes from this cohort's own data rather
  than from an assumption. Seeds are grouped on the **pair** ``i{seed}/p{config.seed}``
  (track README): ``seed`` alone sets only the network init, ``config.seed`` keys the env
  resets, rollouts and eval, and one batch here ran i43/p12345.
* **Eval cadence**, handled above by :func:`on_common_grid`.
* **Commit, dirty flag and GPU.** ``repos.vnl_experiments.dirty`` is True on part of the
  torque side, which per README §6 voids the commit hash there outright, so the hashes
  cannot be the comparability argument. ``divergence_check.py`` replaces it with a
  behavioural one, and ``assert_servo_identity`` below pins the servo subsystem from the
  run records. GPU is a throughput confound, not a reward one (README §6), and no readout
  here is a speed-in-seconds measure -- ``steps_to_900`` is counted in environment steps.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-servo-x-forward-model/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-servo-x-forward-model/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-servo-x-forward-model/extract.py --check

Writes ``data.csv`` (one row per run) and ``curves.csv`` (the eval series on the common
grid). ``plot.py`` reads only those. ``divergence_check.py`` lives beside them, needs
WandB, and is not part of the rebuild.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: The `history` producer takes the project as a **spec field** defaulting to the *rodent*
#: project, so a dm_control history artifact only exists under an explicit override:
#:
#:     python -m vnl_experiments.artifacts ensure --kind history \
#:         --runs analysis/dm_control_suite/walker-servo-x-forward-model/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
#:
#: Same id as both sibling WalkerWalk folders, so a curve here is the same generation of
#: data those analyses read. Asserted in `main`.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

TASK = "WalkerWalk"
BUDGET = 1_000_000_000

#: The one stiffness this folder crosses with the network axis. See the docstring.
SERVO_KP = 64.0

#: Delays with at least three of the four arms. The 2026-09-21 extension added delay 12
#: (all four arms), filled the delay-0 ``fm_servo`` hole, and added ``fm_servo`` at 3, 25
#: and 30. **Delay 30 is excluded**: only ``fm_servo`` was ever run there, so drawing it
#: would extend one line of four alone to the right edge of every panel, which reads as
#: the other three failing rather than as never having been launched. It is quoted in
#: report.md as context for the ``fm_servo`` arm's own ceiling instead of plotted.
DELAYS = (0, 2, 3, 5, 7, 10, 12, 15, 20, 25)

#: Default exploration width; the project sweeps this elsewhere under identical run names.
MIN_STD = 0.001

#: Runs at or before this commit evaluated on the *training* env, wrappers and all.
PRE_EVAL_FIX_COMMIT = "d168093"

#: End-of-training reward is the mean of the eval points in the last this-many steps.
FINAL_WINDOW_STEPS = 50_000_000

#: `late_gain` compares the last window of this width against the one two windows earlier,
#: i.e. "how much did the run still gain over its final 1e8 steps". This is the per-run
#: check on the budget trap, which is what killed the 4.8e8 reading of this question.
LATE_WINDOW_STEPS = 50_000_000

#: Common eval grid for the time-to-criterion readout: the *coarsest* cadence in the
#: cohort. Not optional -- cadence is confounded with the manipulation here.
GRID_STEPS = 4_800_000

#: Trailing window the criterion is read off, in steps: 5 grid points after regridding.
#: Matches ``../walker-joint-stiffness/``; ``steps_to_900_native`` is also recorded, on
#: the native cadence with a 1e7 window, which is ``../walker-forward-model-1b/``'s
#: definition -- so a number here can be put beside either sibling.
SMOOTH_WINDOW_STEPS = 24_000_000
NATIVE_WINDOW_STEPS = 10_000_000

#: Reward criteria for time-to-solve. None is a published standard; see the docstring.
THRESHOLDS = (800, 900, 950)
HEADLINE_THRESHOLD = 900

#: dm_control returns are bounded in [0, 1000] by construction.
TASK_MAX = 1000


def _kp(df: pd.DataFrame) -> pd.Series:
    """``servo_kp``, with *absent* read as 0 -- the one place the two spellings meet."""
    return df["env_params.servo_kp"].fillna(0.0)


def _cohort(df: pd.DataFrame) -> pd.Series:
    """Everything the 2x2 could draw on, before the two arm gates."""
    return (
        df["env"].eq(TASK)
        & df["net_params.min_std"].eq(MIN_STD)
        & df["config.ppo.total_steps"].eq(BUDGET)
        & df["net_params.delay_k"].isin(DELAYS)
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
        # `state == "finished"` alone silently drops runs that trained fully and died in
        # the final eval (README §6). Gate on the property meant: reached its budget.
        & (df["state"].eq("finished") | df["summary._step"].ge(BUDGET))
    )


def _arm(network_class: str, kp: float):
    def selector(df: pd.DataFrame) -> pd.Series:
        return (_cohort(df)
                & df["net_params.network_class"].eq(network_class)
                & _kp(df).eq(kp))
    return selector


#: The 2x2. Delay is a **column**, not part of the condition, so an arm keeps one colour
#: across every panel (README §7).
CONDITIONS = {
    "mlp_torque": _arm("DelayedMLP", 0.0),
    "mlp_servo": _arm("DelayedMLP", SERVO_KP),
    "fm_torque": _arm("FlatForwardModel", 0.0),
    "fm_servo": _arm("FlatForwardModel", SERVO_KP),
}

#: The two factors, for the interaction readout. Kept here so `plot.py` and `report.md`
#: cannot disagree with `CONDITIONS` about which arm is which.
ARM_FACTORS = {
    "mlp_torque": ("mlp", "torque"),
    "mlp_servo": ("mlp", "servo"),
    "fm_torque": ("fm", "torque"),
    "fm_servo": ("fm", "servo"),
}

#: Must hold across the whole cohort. Anything flagged here is a fairness problem until
#: explained. `servo_kp`, `servo_center` and `network_class` are absent because they are
#: the swept axes; `servo_damping_ratio` is here because it is the other knob that changes
#: what the servo *is*, and it must not co-vary with anything.
INVARIANTS = [
    "config.ppo.total_steps",
    "summary._step",
    "config.ppo.n_envs",
    "config.ppo.rollout_length",
    "config.ppo.learning_rate",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "config.ppo.clip_range",
    "config.ppo.discounting_factor",
    "config.ppo.gae_lambda",
    "config.ppo.gradient_clipping",
    "config.ppo.weight_decay",
    "config.ppo.normalize_advantages",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "env_params.sim_dt",
    "env_params.impl",
    "env_params.servo_damping_ratio",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.normalize_obs",
    "net_params.activation",
    "net_params.initializer_scale",
    "net_params.std_scale",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "repos.vnl_playground.commit",
]

#: Expected to vary. Reported in their own section so that a ``*** VARIES ***`` in the
#: INVARIANTS section stays a signal rather than routine noise. Every one of these is
#: discussed in report.md: the first four are the design, the rest are the cohort's
#: unavoidable heterogeneity.
DESIGN_AXES = [
    "env_params.servo_kp",
    "env_params.servo_center",
    "net_params.network_class",
    "net_params.delay_k",
    "net_params.efference_length",
    "config.eval.every_steps",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_experiments.dirty",
    "gpu",
    "seed",
    "config.seed",
]

#: Asserted per arm rather than reported: these are what *make* an arm, so a stray value
#: means the condition label does not describe the network that ran, which is a bug and
#: not a caveat.
ARM_SIGNATURE = {
    "fm_torque": {"net_params.fm_loss_weight": 1.0,
                  "net_params.detach_prediction": True,
                  "net_params.predictor_hidden_sizes": "[256, 256, 256, 256]"},
    "fm_servo": {"net_params.fm_loss_weight": 1.0,
                 "net_params.detach_prediction": True,
                 "net_params.predictor_hidden_sizes": "[256, 256, 256, 256]"},
    "mlp_torque": {"net_params.fm_loss_weight": None,
                   "net_params.detach_prediction": None,
                   "net_params.predictor_hidden_sizes": None},
    "mlp_servo": {"net_params.fm_loss_weight": None,
                  "net_params.detach_prediction": None,
                  "net_params.predictor_hidden_sizes": None},
}

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")

#: Per-joint servo parameters `train.py` records from `actuator_mode.servo_report`.
#: Carried into data.csv so the physics of a run is recoverable from the CSV alone --
#: `servo_kp` is normalised, so it does not by itself say what stiffness a joint had.
SERVO_JOINTS = ("right_hip", "right_knee", "right_ankle",
                "left_hip", "left_knee", "left_ankle")


def history_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def reward_series(hist: pd.DataFrame | None) -> pd.DataFrame | None:
    """The eval-reward series as a tidy step/value frame, or None."""
    if hist is None:
        return None
    key = next((k for k in REWARD_KEYS if k in hist.columns), None)
    if key is None:
        return None
    out = hist.dropna(subset=[key])[["_step", key]]
    return out.rename(columns={"_step": "step", key: "value"}).sort_values("step")


def on_common_grid(series: pd.DataFrame, cadence: int = GRID_STEPS) -> pd.DataFrame:
    """One eval point per ``cadence`` steps, whichever sample lies nearest each grid step.

    **This is what makes the time-to-criterion comparable across the arms.** Every
    ``servo_kp = 64`` run logs eval every 4.8e6 steps; most torque runs log every 6e5 --
    so cadence is confounded with the manipulation, not merely heterogeneous. A trailing
    mean defined in steps would average 8x more samples on the torque side, making its
    criterion strictly harder to trigger on an upward fluctuation, which flatters the
    servo arms. That is the direction this analysis must not be wrong in.

    Picking the *nearest single sample* rather than averaging within the bin keeps the
    noise properties identical too: every grid point is one 256-episode eval whichever
    cadence the run used. ``steps_to_*`` is then quantised to 4.8e6 steps (0.5 % of the
    budget) for every run equally, and ``main`` reports how far this moved the fine-cadence
    runs' answers.
    """
    steps = series["step"].to_numpy()
    grid = np.arange(cadence, steps.max() + 1, cadence)
    idx = np.abs(steps[None, :] - grid[:, None]).argmin(axis=1)
    keep = series.iloc[np.unique(idx)]
    return keep.sort_values("step").reset_index(drop=True)


def trailing_mean(steps: np.ndarray, values: np.ndarray,
                  window: int = SMOOTH_WINDOW_STEPS) -> np.ndarray:
    """Mean of every sample in ``(s - window, s]``, for each s in ``steps``.

    Trailing rather than centred: a centred window reads the future, and a criterion read
    off raw eval points fires on noise. Positions whose window is not yet fully covered by
    the run are NaN, so a run whose first eval point happens to land high cannot "solve"
    the task at step 0.
    """
    lo = np.searchsorted(steps, steps - window, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= window, out, np.nan)


def time_to_criterion(series: pd.DataFrame) -> dict:
    """``steps_to_<thr>`` / ``solved_<thr>`` / ``smooth_max``, on the common grid."""
    grid = on_common_grid(series)
    steps = grid["step"].to_numpy()
    smooth = trailing_mean(steps, grid["value"].to_numpy())
    out = {"smooth_max": float(np.nanmax(smooth)) if np.isfinite(smooth).any() else None,
           "grid_points": int(len(steps))}
    for thr in THRESHOLDS:
        reached = np.where(smooth >= thr)[0]
        out[f"solved_{thr}"] = bool(len(reached))
        out[f"steps_to_{thr}"] = int(steps[reached[0]]) if len(reached) else None
    return out


def native_time_to_criterion(series: pd.DataFrame) -> dict:
    """The sibling folder's definition: native cadence, 1e7 window.

    Two jobs. It measures what the regridding cost (``main`` prints the paired shift), and
    it lets a number in this folder be put beside ``../walker-forward-model-1b/``, whose
    torque-side runs are literally the same runs under the other definition.
    """
    steps = series["step"].to_numpy()
    smooth = trailing_mean(steps, series["value"].to_numpy(), window=NATIVE_WINDOW_STEPS)
    reached = np.where(smooth >= HEADLINE_THRESHOLD)[0]
    return {f"steps_to_{HEADLINE_THRESHOLD}_native":
            int(steps[reached[0]]) if len(reached) else None}


def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    network, actuator = ARM_FACTORS[run["condition"]]
    row = {
        "condition": run["condition"],
        # The two factors as their own columns, so the interaction can be computed from
        # data.csv without re-parsing the condition name.
        "network": network,
        "actuator": actuator,
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "nnx_ppo_commit": run.get("repos.nnx_ppo.commit"),
        "vnl_experiments_dirty": run.get("repos.vnl_experiments.dirty"),
        "gpu": run["gpu"],
        "requeued": "requeue" in (run.get("tags") or []),
        "budget": int(run["config.ppo.total_steps"]),
        "delay": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        # The init seed and the PPO seed are different draws and only the pair identifies
        # a replicate (track README). `seed_pair` is what plot.py groups on.
        "seed": int(run["seed"]),
        "ppo_seed": int(run["config.seed"]),
        "seed_pair": f"i{int(run['seed'])}/p{int(run['config.seed'])}",
        "actual_step": run.get("summary._step"),
        "ctrl_dt": run.get("env_params.ctrl_dt"),
        "servo_kp": 0.0 if pd.isna(run.get("env_params.servo_kp"))
                    else float(run["env_params.servo_kp"]),
        # Inert at servo_kp = 0 -- a launch label there, never a physical property.
        "servo_center": run.get("env_params.servo_center"),
        # What the config says. Confounded with the manipulation, which is why
        # `on_common_grid` exists; recorded so that fact stays visible in data.csv.
        "eval_every_steps": run.get("config.eval.every_steps"),
        # Post-`971ab99` only, so one value -- recorded as a column so a future cohort
        # that straddles the eval-wrapper fix cannot pool the two silently.
        "reward_source": "dmc_eval",
        # The run summary's single final eval point. Reference only; figures use
        # `reward_final`.
        "reward_last_point": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
    }
    # The derived physics, straight from the run record. Absent (NaN) on torque runs,
    # which is correct: there is no servo to describe.
    for joint in SERVO_JOINTS:
        row[f"kp_{joint}"] = run.get(f"servo.{joint}.kp")
        row[f"settling_{joint}"] = run.get(f"servo.{joint}.settling_time")
        row[f"dead_{joint}"] = run.get(f"servo.{joint}.dead_ctrl_fraction")

    blank = {"reward_final": None, "reward_final_n": 0, "reward_max": None,
             "late_gain": None, "curve_points": 0, "curve_max_step": None,
             "eval_spacing_median": None, "eval_spacing_tail": None,
             "smooth_max": None, "grid_points": 0,
             f"steps_to_{HEADLINE_THRESHOLD}_native": None}
    blank.update({f"steps_to_{t}": None for t in THRESHOLDS})
    blank.update({f"solved_{t}": None for t in THRESHOLDS})
    if series is None or series.empty:
        row.update(blank)
        return row

    steps = series["step"].to_numpy()
    values = series["value"].to_numpy()
    window = series[series["step"] > steps.max() - FINAL_WINDOW_STEPS]

    # Is the run still moving at the budget, and in which direction? This is the per-run
    # form of README §6's "common mid-training budget" trap -- the trap that made the
    # 4.8e8 reading of this question unusable. A large positive `late_gain` in one arm and
    # not another means the level readout is partly a rate difference.
    late = series[series["step"] > steps.max() - LATE_WINDOW_STEPS]["value"]
    earlier = series[(series["step"] > steps.max() - 3 * LATE_WINDOW_STEPS)
                     & (series["step"] <= steps.max() - 2 * LATE_WINDOW_STEPS)]["value"]

    row.update(
        reward_final=float(window["value"].mean()),
        reward_final_n=int(len(window)),
        reward_max=float(values.max()),
        late_gain=float(late.mean() - earlier.mean()) if len(earlier) else None,
        curve_points=int(len(steps)),
        curve_max_step=int(steps.max()),
        eval_spacing_median=(int(np.median(np.diff(steps))) if len(steps) > 1 else None),
        # The cadence at the *end* of the curve. For a requeued run that straddled the
        # 2026-09-15 cadence change the median is its early fine cadence and says nothing
        # about how far short of `total_steps` the last eval legitimately falls -- which
        # is what the staleness check in `main` needs.
        eval_spacing_tail=(int(np.median(np.diff(steps)[-5:])) if len(steps) > 1
                           else None),
    )
    row.update(time_to_criterion(series))
    row.update(native_time_to_criterion(series))
    return row


def build_curves(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    """The eval series on the common grid, plus the trailing mean the criterion uses.

    ``reward_smooth`` is written here rather than recomputed in ``plot.py`` so the curve a
    figure draws and the number ``steps_to_900`` reports are the same object.
    """
    if series is None or series.empty:
        return []
    grid = on_common_grid(series)
    steps = grid["step"].to_numpy()
    values = grid["value"].to_numpy()
    smooth = trailing_mean(steps, values)
    network, actuator = ARM_FACTORS[run["condition"]]
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "network": network, "actuator": actuator,
             "delay": int(run["net_params.delay_k"]),
             "seed_pair": f"i{int(run['seed'])}/p{int(run['config.seed'])}",
             "step": int(s), "reward_mean": float(v),
             "reward_smooth": None if not np.isfinite(m) else float(m)}
            for s, v, m in zip(steps, values, smooth)]


def assert_arm_signature(runs: pd.DataFrame) -> None:
    """The arm-defining net_params really do split along the condition label."""
    for condition, expected in ARM_SIGNATURE.items():
        sub = runs[runs["condition"] == condition]
        for col, want in expected.items():
            got = sub[col].dropna().unique() if col in sub else np.array([])
            if want is None:
                if len(got):
                    raise SystemExit(
                        f"{condition}: expected {col} to be absent, found {got!r}. The "
                        f"condition label does not describe the network that ran.")
            elif list(got) != [want]:
                raise SystemExit(
                    f"{condition}: expected {col} == {want!r}, found {got!r}. The "
                    f"condition label does not describe the network that ran.")


def assert_servo_identity(df: pd.DataFrame) -> str:
    """Did every servo run simulate the same actuator, and every torque run none?

    ``repos.vnl_experiments.dirty`` is True on part of this cohort, so ``git_commit`` does
    not identify the code that ran (README §4) -- and the servo was being actively
    developed in the days around these launches. The hash therefore cannot be the
    comparability argument for the servo half of the 2x2.

    This checks the thing that matters instead. ``train.py`` records
    ``actuator_mode.servo_report`` into the run config: the *derived* per-joint physics,
    read back off the built model. If the servo code had changed between the
    ``mlp_servo`` and ``fm_servo`` launches, the same ``servo_kp`` would have produced
    different per-joint values, and the network contrast inside the servo half would be
    confounded with a change of plant.

    Torque runs must carry no servo parameters at all, which is the check that
    ``servo_kp = 0`` really did leave the model unpatched rather than writing a "neutral"
    servo.

    Raises rather than reports: a servo half built from two different plants is not a
    caveat, it is a different experiment.
    """
    servo, torque = df[df.actuator == "servo"], df[df.actuator == "torque"]
    lines = [f"\nservo runs: {len(servo)}   torque runs: {len(torque)}",
             f"per-joint servo physics, over every servo run (servo_kp = {SERVO_KP:g})"]
    for joint in SERVO_JOINTS:
        kp = servo[f"kp_{joint}"].round(6).unique()
        dead = servo[f"dead_{joint}"].round(9).unique()
        settle = servo[f"settling_{joint}"].round(6).unique()
        if len(kp) != 1 or len(dead) != 1 or len(settle) != 1:
            raise SystemExit(
                f"servo runs disagree about joint {joint}: kp={kp}, "
                f"dead_ctrl_fraction={dead}, settling_time={settle}. The two servo arms "
                f"did not simulate the same actuator, so the network contrast inside the "
                f"servo half is confounded with a change of plant.")
        if len(torque[f"kp_{joint}"].dropna()):
            raise SystemExit(
                f"a torque run carries a servo parameter for {joint}. `servo_kp = 0` is "
                f"supposed to leave the model unpatched.")
        lines.append(f"   {joint:12s} kp={kp[0]:9.3f}  t_settle={settle[0] * 1e3:6.1f} ms"
                     f"  dead_ctrl={dead[0]:.3f}")
    # The interpretable axis (servo_control.md §2.1): the servo helps, on the EP story,
    # when the periphery corrects faster than the central loop can.
    slowest = max(SERVO_JOINTS, key=lambda j: servo[f"settling_{j}"].iloc[0])
    t_ms = servo[f"settling_{slowest}"].iloc[0] * 1e3
    lines.append(f"\n   slowest joint is {slowest}, t_settle = {t_ms:.1f} ms; "
                 f"one delay step is 25 ms")
    for delay in sorted(df.delay.unique()):
        if delay:
            lines.append(f"     delay {delay:2d} = {delay * 25:3d} ms   "
                         + ("servo faster than the loop" if t_ms < delay * 25
                            else "servo SLOWER than the loop"))
    return "\n".join(lines) + "\n"


def design_grid(df: pd.DataFrame) -> str:
    """Runs per (arm x delay), so a hole in the 2x2 is visible as a hole."""
    table = pd.crosstab(df["condition"], df["delay"])
    holes = [(c, d) for c in table.index for d in table.columns if table.loc[c, d] == 0]
    complete = [d for d in table.columns if (table[d] > 0).all()]
    singles = [(c, d) for c in table.index for d in table.columns
               if table.loc[c, d] == 1]
    return (f"\nruns per arm x delay\n{table}\n"
            f"\n  delays with all four arms present : {complete}"
            f"\n  empty cells                       : {holes if holes else 'none'}"
            f"\n  n=1   cells                       : {len(singles)} of "
            f"{table.size}\n"
            f"\n  Only the complete delays support an interaction readout; a hole means\n"
            f"  one corner of the 2x2 was never run, not that it failed.\n")


def seed_spread(df: pd.DataFrame) -> str:
    """Within-cell spread, the noise floor every n=1 comparison is read against."""
    g = df.groupby(["condition", "delay"])["reward_final"]
    tab = g.agg(["count", "mean", "min", "max"])
    tab["spread"] = tab["max"] - tab["min"]
    multi = tab[tab["count"] > 1]
    tail = (f"\n  largest spread among them: {multi['spread'].max():.1f} reward\n"
            f"  median spread            : {multi['spread'].median():.1f} reward\n"
            if len(multi) else "\n  no multi-seed cells\n")
    return (f"\nreward_final by cell\n{tab.to_string()}\n"
            f"\n  cells with >1 seed: {len(multi)} of {len(tab)}" + tail
            + "\n  Variance is strongly regime-dependent here -- a solved cell's seeds\n"
              "  agree to a few reward, a transitioning one's can differ by hundreds --\n"
              "  so read a comparison against the spread in *its own* regime, not the\n"
              "  maximum.\n")


def interaction_table(df: pd.DataFrame) -> str:
    """The 2x2 itself: cell means, the two simple effects, and their difference.

    The interaction is ``(fm_servo - fm_torque) - (mlp_servo - mlp_torque)``: how much
    *less* (or more) the servo buys when a forward model is already present. Computed only
    at delays where all four corners exist, because a 2x2 with a hole is not a 2x2.
    """
    lines = []
    for readout, scale, unit in [("reward_final", 1.0, "reward"),
                                 ("steps_to_900", 1e-6, "e6 steps")]:
        lines.append(f"\n{readout} ({unit}); cells are means over seeds, "
                     f"'-' = some run in the cell never reached the criterion")
        lines.append(f"{'delay':>6} {'mlp_tq':>9} {'mlp_sv':>9} {'fm_tq':>9} "
                     f"{'fm_sv':>9} | {'servo|mlp':>10} {'servo|fm':>10} "
                     f"{'interaction':>12}")
        for delay in sorted(df.delay.unique()):
            cell, ok = {}, True
            for arm in CONDITIONS:
                sub = df[(df.condition == arm) & (df.delay == delay)]
                # A mean over only the runs that happened to reach the criterion is the
                # one number this table must not print.
                if sub.empty or sub[readout].isna().any():
                    ok = False
                    cell[arm] = None
                else:
                    cell[arm] = float(sub[readout].mean()) * scale
            fmt = (lambda v: f"{v:9.1f}" if v is not None else f"{'-':>9}")
            row = f"{delay:>6} " + " ".join(fmt(cell[a]) for a in CONDITIONS)
            if ok:
                s_mlp = cell["mlp_servo"] - cell["mlp_torque"]
                s_fm = cell["fm_servo"] - cell["fm_torque"]
                row += f" | {s_mlp:>+10.1f} {s_fm:>+10.1f} {s_fm - s_mlp:>+12.1f}"
            else:
                row += f" | {'-':>10} {'-':>10} {'-':>12}"
            lines.append(row)
    lines.append(
        "\n  Read the interaction column as: how much the servo buys *on top of* a\n"
        "  forward model, minus how much it buys without one. Strongly negative means\n"
        "  the two are substitutes; near zero means they add; positive means they\n"
        "  compound. Most cells are n=1, so compare against the seed spread above\n"
        "  before reading a sign.\n")
    return "\n".join(lines)


def regrid_cost(df: pd.DataFrame) -> str:
    """What putting every run on the common 4.8e6 grid moved, per run that had a choice.

    Only the fine-cadence runs are affected; the coarse ones are already on the grid. If
    this were large it would be a finding rather than a footnote, because cadence is
    confounded with the manipulation.
    """
    both = df[df[f"steps_to_{HEADLINE_THRESHOLD}"].notna()
              & df[f"steps_to_{HEADLINE_THRESHOLD}_native"].notna()].copy()
    both["shift"] = (both[f"steps_to_{HEADLINE_THRESHOLD}"]
                     - both[f"steps_to_{HEADLINE_THRESHOLD}_native"])
    is_fine = both["eval_spacing_median"] < GRID_STEPS / 2
    fine, coarse = both[is_fine], both[~is_fine]
    lines = [f"\nregridding cost on steps_to_{HEADLINE_THRESHOLD} "
             f"(common grid minus native cadence)",
             f"  runs with both readouts : {len(both)}  "
             f"({len(fine)} fine cadence, {len(coarse)} coarse)"]
    for name, g in [("all", both), ("fine cadence (6e5)", fine),
                    ("coarse cadence (4.8e6)", coarse)]:
        if len(g):
            lines.append(f"  {name:<24}: mean {g['shift'].mean() / 1e6:+.2f}e6  "
                         f"median {g['shift'].median() / 1e6:+.2f}e6  "
                         f"range [{g['shift'].min() / 1e6:+.2f}e6, "
                         f"{g['shift'].max() / 1e6:+.2f}e6]")
    if len(fine) and len(coarse):
        lines.append(
            f"\n  The number that matters is the *difference* between those two groups,\n"
            f"  because cadence is confounded with the manipulation: every servo run is\n"
            f"  coarse and most torque runs are fine. It is "
            f"{(fine['shift'].mean() - coarse['shift'].mean()) / 1e6:+.2f}e6 steps --\n"
            f"  i.e. under the native definition the torque arms' crossings would sit\n"
            f"  that much earlier *relative to* the servo arms than they do here.\n"
            f"  Against a 1e9-step budget and arm differences of 1e8-5e8 steps that is\n"
            f"  a percent-scale correction, but it points the wrong way if left out.")
    lines.append(f"\n  The two definitions also differ in window (2.4e7 vs 1e7), and a\n"
                 f"  wider trailing window lags a rising curve, so the shift above is\n"
                 f"  the *combined* cost of matching both siblings' conventions rather\n"
                 f"  than grid granularity alone.\n")
    return "\n".join(lines)


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)
    assert_arm_signature(runs)

    store = Store()
    producer = get_producer("history")
    history_spec_id = producer.spec_id(producer.spec(project=PROJECT))
    if history_spec_id != HISTORY_SPEC_ID:
        raise SystemExit(
            f"history spec_id has drifted: got {history_spec_id}, expected "
            f"{HISTORY_SPEC_ID}.\nThe curves this analysis reads were made by a different "
            f"spec or producer VERSION. Update HISTORY_SPEC_ID deliberately, re-produce, "
            f"and say so in report.md -- do not silently repoint at a different "
            f"generation of data.")

    rows, curves = [], []
    for _, run in runs.iterrows():
        series = reward_series(history_of(store, run["wandb_id"], history_spec_id))
        rows.append(build_row(run, series))
        curves.extend(build_curves(run, series))

    df = pd.DataFrame(rows).sort_values(
        ["delay", "condition", "seed_pair", "wandb_id"], ignore_index=True)
    curves_df = pd.DataFrame(curves).sort_values(
        ["delay", "condition", "seed_pair", "step"], ignore_index=True)

    missing = int((df["curve_points"] == 0).sum())
    if missing:
        raise SystemExit(
            f"\n*** {missing}/{len(df)} runs have no history artifact. Runs are never "
            f"silently dropped for a missing artifact (README §3); produce them:\n"
            f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
            f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
            f"        --set project='\"{PROJECT}\"'\n")

    # A `history` artifact made while a run was still training is a snapshot, and `ensure`
    # will not replace it (README §6). The tolerance comes from the eval cadence, not from
    # a fixed fraction: the last eval lands on the last cadence multiple below
    # `total_steps`, so a complete 1e9 run at the 4.8e6 cadence legitimately stops ~1.7e6
    # short, which a 0.1 % rule would read as truncated. It must be the *tail* cadence:
    # the requeued runs that straddled the 2026-09-15 change have a fine-cadence median
    # that describes only their early portion.
    slack = df["eval_spacing_tail"].fillna(0) * 3 + 1
    stale = df[df["actual_step"] - df["curve_max_step"] > slack]
    if len(stale):
        print("\n*** history artifacts fall more than three tail eval intervals short of "
              "the run's own _step -- produced mid-training. Re-produce with "
              "`artifacts ensure --override`:")
        print(stale[["wandb_id", "curve_max_step", "actual_step",
                     "eval_spacing_tail"]].to_string(index=False))
        raise SystemExit(1)

    servo = assert_servo_identity(df)
    grid = design_grid(df)
    spread = seed_spread(df)
    inter = interaction_table(df)
    cost = regrid_cost(df)
    for block in (grid, spread, inter, cost, servo):
        print(block)

    report = (comparability_report(runs, invariant_cols=INVARIANTS,
                                   group_col="condition")
              + "\n\n--- DESIGN AXES: expected to vary; see report.md ------------------\n"
              + comparability_report(runs, invariant_cols=DESIGN_AXES,
                                     group_col="condition")
              + "\n\n" + "#" * 78 + "\n# DESIGN GRID\n" + "#" * 78 + grid
              + "\n" + "#" * 78 + "\n# WITHIN-CELL SPREAD\n" + "#" * 78 + spread
              + "\n" + "#" * 78 + "\n# THE 2x2\n" + "#" * 78 + inter
              + "\n" + "#" * 78 + "\n# REGRIDDING COST\n" + "#" * 78 + cost
              + "\n" + "#" * 78 + "\n# SERVO IDENTITY\n" + "#" * 78 + servo)
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
