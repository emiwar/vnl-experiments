"""On WalkerWalk, how much delay does an explicit forward model absorb, and how much
sooner does it solve the task?

The sibling folder [`../explicit-forward-model/`](../explicit-forward-model/) asked the
same architecture question across three tasks at 4.8e8 steps and came back with a
WalkerWalk answer it could not defend: the baseline was *still climbing at the end of
training*, so the gap it measured was "the forward model gets there sooner", not "the
baseline cannot get there". This folder is that follow-up -- WalkerWalk only, at 1e9 steps
and beyond, with the readout the speed claim actually needs, which is a time-to-criterion
rather than a level.

The two arms, matched in everything but the state estimate
----------------------------------------------------------
* ``DelayedMLP`` -- the actor sees the ``delay_k``-step-stale observation plus an
  efference copy of the ``delay_k`` actions still in flight; the critic sees the fresh
  observation. Any state estimate is implicit in the actor MLP.
* ``FlatForwardModel`` -- identical, plus a supervised predictor
  (delayed obs, action buffer) -> current obs whose output is handed to the actor.
  ``fm_loss_weight = 1`` and ``detach_prediction = True``, so the predictor is trained by
  its own regression loss and *not* by the policy gradient.

``efference_length == delay_k`` in every run, so the two arms are given exactly the same
information and differ only in whether the state estimate is built explicitly.

The three readouts
------------------
``reward_at_1b`` -- the mean of the eval points in ``(1e9 - 5e7, 1e9]``. **This is what the
main figure plots**, and it is what makes the 1e9, 2e9 and 4e9 runs one comparable cohort:
the optimiser is plain Adam at a *constant* ``learning_rate = 1e-4``
(``nnx_ppo.algorithms.ppo`` builds ``optax.adam(learning_rate=<float>)``, no schedule), and
``total_steps`` enters nothing but the loop bound, so a 4e9 run's weights at 1e9 come from
exactly the process a native 1e9 run ran. Reading every run at one step is therefore a
like-for-like comparison, not an extrapolation. Cross-check that held: the 2e9 MLP at delay
15 reads 7.8e2 at 1e9 against the native 1e9 run's 7.5e2, and the 4e9 MLP at delay 20 reads
6.1e2 against 5.6e2-6.0e2.

``reward_final`` -- the same window at the *end* of whatever the run actually ran. Equal to
``reward_at_1b`` for a 1e9 run; for the longer runs the pair is the interesting object (MLP
delay 15: 7.8e2 at 1e9, 9.6e2 at 2e9), and it is what the 4.8e8 panel uses.

``steps_to_<thr>`` -- the first step at which the **trailing 1e7-step mean** of eval reward
reaches ``thr``, computed over the run's **full** length however long that is. This is the
"how long to solve it" axis. Four things about it:

* **The 900 criterion is this project's convention, not a published standard.** dm_control
  returns are bounded in [0, 1000] by construction (1000 steps x a reward in [0, 1]), and
  the benchmark literature reports curves rather than declaring a task solved, so there is
  no number to inherit. 900 is chosen because it sits clearly above where these runs
  plateau when they fail (5.2e2-7.8e2) and clearly below where they plateau when they
  succeed (9.4e2-9.8e2), and the delay-0 runs of *both* arms reach ~9.8e2 -- so it is not
  a bar that the architecture question itself decides. ``steps_to_800`` and
  ``steps_to_950`` are extracted alongside so ``plot.py`` can show that the ordering of
  the arms does not depend on the choice.
* **Trailing, not centred, and not a single point.** A criterion read off raw eval points
  fires on noise; a centred window reads the future. The trailing window means
  "by this step, recent average performance is at criterion", which is what a
  time-to-solve is supposed to say. The window is defined in *steps*, not samples, so it
  means the same thing at every eval cadence -- which matters here, see below.
* **A run that never reaches the criterion is censored, not missing.** ``solved_<thr>`` is
  False and ``steps_to_<thr>`` is blank; ``smooth_max`` records how close it got and
  ``curve_max_step`` is the bound the censoring is relative to. Since the cohort now spans
  1e9 to 4e9, *how much* budget a censored run had is part of the claim -- "did not solve
  it in 4e9 steps" is a far stronger statement than "did not solve it in 1e9" -- so
  ``plot.py`` parks each censored run at its **own** budget rather than on a shared line.
* **The eval cadence is heterogeneous, and it does not matter here.** Three *measured*
  spacings are in the cohort, and none of them equals the configured value, because the
  trainer evaluates on iteration boundaries of 8192 x 30 = 245 760 steps: a config of
  ``6e5`` gives 4.92e5 (2 iterations), one of ``4.8e6`` gives 4.92e6 (20), and the *early*
  part of a requeued run that straddled the change sits at 7.37e5 (3) -- see the resume
  note below. So ``eval_spacing_median`` is recorded from the curve and
  ``eval_every_steps`` from the config, and they are different columns on purpose.
  A trailing window in steps averages whatever falls inside it,
  so the readout means the same thing throughout, but it is built from ~17 points at the
  fine cadence and ~2 at the coarse one. ``cadence_bias.py`` measures the consequence
  directly, by recomputing every fine-cadence run's crossing on a coarse grid: the median
  shift is **+2.0e6 steps** (range -2.5e6 to +5.9e6), and it is *positive* because a coarse
  grid can only report a crossing on its own grid points, i.e. granularity rather than
  noise. Against effects of 1e8-2.3e9 steps that is under 1 %, so the definition is left
  alone and the measurement is committed as ``cadence_bias.txt``.

Cohorts
-------
**Primary: every run with at least 1e9 steps** (1e9, 2e9 and 4e9), read at 1e9 for reward
and over full length for time-to-criterion. The 2e9/4e9 runs exist to settle the one
question the 1e9 budget could not: whether the MLP's failure at high delay is slowness or
incapacity. It is slowness -- MLP delay 15 crosses at 1.24e9 and one of the two MLP delay-20
runs at 2.27e9 -- so the extended runs change the *conclusion*, not just the error bars,
and they are in the primary cohort rather than a supplement.

**Support: the 4.8e8 cohort** (seed 42, the sibling folder's dataset). Kept as its own
`cohort` value and its own figure, for one job: it is a complete single-launch grid over
delays 0/2/5/7/10/15/20 for both arms on a third seed. **It is never pooled with the
primary cohort**, because 4.8e8 is short of where the MLP transitions at almost every
delay -- exactly README §6's "common mid-training budget" trap, and the reason this folder
exists. Measured: the MLP at delay 10 finishes 4.8e8 at 6.9e2 and 1e9 at 9.4e2, because its
curve steps up at ~6.2e8, past the shorter budget. The arm gap at delay 10 reads +2.7e2 at
4.8e8 against +2.6e1 at 1e9 -- an order of magnitude, nearly all of it budget.

Selection gates that no run name would reveal
---------------------------------------------
* **servo control (physics, hard gate).** Since ``540e356`` the dm_control envs take
  ``servo_kp``, which rewrites the torque motors as position servos. That is a different
  plant, so ``servo_kp > 0`` cannot enter this cohort. ``servo_kp == 0`` *is* admissible and
  is not a judgement call: ``actuator_mode.apply_servo`` returns at ``if kp == 0.0: return
  env`` before writing any field or re-running ``put_model``, so it is the shipped env
  bit-for-bit -- identical to the older runs where the key is absent entirely.
* **the ``walker-joint-stiffness`` cohort (design gate).** That folder's ``servo_kp`` sweep
  includes its own ``servo_kp = 0`` baselines, which the physics gate above would *admit*.
  They are excluded anyway, because they are MLP-only at delays 0 and 5 on one seed at
  4.8e8: letting them in would add n=5 and n=4 to two cells of one arm at the two delays
  where nothing happens, while the arm they would be compared against has n=1 there. They
  are recognisable two ways -- the ``env-override`` tag and ``servo_center == "range"``
  against the delay sweep's ``"qpos0"`` -- and ``_kp_sweep`` asserts the two agree, so a
  drift in either is loud rather than silent. Several of these runs are named *exactly*
  like the defaults (``WalkerWalk_DelayedMLP_delay0_eff0``), which is the same trap as
  ``min_std`` below in a new guise.
* **min_std.** The project contains a ``min_std`` sweep (0.01-0.2) whose runs are named
  exactly like the defaults. Gated to ``min_std == 0.001``.
* **The eval-env fix (``971ab99``, 2026-09-09).** Before it the eval env inherited the
  training wrappers, so ``eval/*`` was reward-scaled x10 *and* the episodes were truncated
  by ``EpisodeWrapper``'s random phase (track README). Gated on the commit, not the date --
  both sides of the fix were created on 2026-09-09. This also excludes the entire pre-Hydra
  era, whose WalkerWalk runs have no ``net_params.*`` at all.

Requeue, resumes, and what they did not do
------------------------------------------
Seven runs come from ``slurm_dmc_requeue.sh`` and were preempted and resumed up to three
times. Two consequences, both checked rather than assumed:

* A light checkpoint omits the env states, so on resume the episode phases are redrawn and
  the actor's delay/efference queues start empty (track README). Both transients are
  bounded by one episode (<= 1000 steps) against a >= 1e9-step budget. Looked for directly
  in the eval series at each recorded resume step: ``wh2fofyc`` reads 962-964 continuously
  across its resume at 9.3e8, with no step and no dip. Nothing here needs excluding.
* **A requeued run's ``config.eval.every_steps`` describes only its last attempt.** The
  three runs that straddled the 2026-09-15 cadence change log ``4.8e6`` but evaluated their
  early portion every 3 iterations (7.4e5) -- ``ayruh7m9`` switches at 1.45e9,
  ``wh2fofyc`` at 9.3e8, ``o3lsdt1n`` at ~1.0e9. So ``eval_every_steps`` from the config is
  recorded here *and* the measured ``eval_spacing_median``, because only the second one
  describes the curve. Neither is used by the readouts (the windows are in steps), but a
  future question about curve *shape* at fine resolution needs the measured one.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/extract.py --check

Writes ``data.csv`` (one row per run) and ``curves.csv`` (the eval series). ``plot.py``
reads only those. Two comparability checks live beside them and are not part of the
rebuild: ``nan_guard_inert.py`` (needs WandB) and ``cadence_bias.py`` (reads ``curves.csv``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: The `history` producer takes the project as a **spec field** defaulting to the rodent
#: one, so a dm_control history artifact only exists under an explicit override:
#:
#:     python -m vnl_experiments.artifacts ensure --kind history \
#:         --runs analysis/dm_control_suite/walker-forward-model-1b/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
#:
#: Because the project is hashed into the spec, control-suite and rodent curves can never
#: be pooled by accident. Same id as the sibling folder, so the 4.8e8 curves here are
#: literally the same bytes that analysis read. Asserted in `main`.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

TASK = "WalkerWalk"

#: Default exploration width; the project sweeps this elsewhere under identical run names.
MIN_STD = 0.001

#: Runs at or before this commit evaluated on the *training* env, wrappers and all.
PRE_EVAL_FIX_COMMIT = "d168093"

#: The step at which reward is read for the primary cohort, so 1e9 / 2e9 / 4e9 runs are
#: directly comparable. Valid because the learning rate is a constant (see the docstring).
READOUT_STEP = 1_000_000_000

#: Admitted budgets. Anything >= READOUT_STEP is "primary"; 4.8e8 is the support cohort.
SUPPORT_BUDGET = 480_000_000
BUDGETS = (SUPPORT_BUDGET, 1_000_000_000, 2_000_000_000, 4_000_000_000)

#: Reward windows are this wide, ending at READOUT_STEP or at the run's own last step.
FINAL_WINDOW_STEPS = 50_000_000

#: Width of the trailing window that defines "solved", in environment steps. Defined in
#: steps, not samples, so it means the same thing at all three eval cadences in the cohort;
#: `cadence_bias.py` measures what that costs (median +2.0e6 steps, i.e. grid granularity).
SMOOTH_WINDOW_STEPS = 10_000_000

#: Reward criteria for time-to-solve. 900 is the headline; the other two are extracted so
#: `plot.py` can show the arm ordering does not depend on the choice. See the docstring --
#: none of these is a published standard.
THRESHOLDS = (800, 900, 950)
HEADLINE_THRESHOLD = 900

#: `servo_kp > 0` is a different plant. 0 is a documented no-op (`apply_servo` returns
#: before touching the model), and the key is absent on runs predating `540e356`.
ADMITTED_SERVO_KP = 0.0


def _candidates(df: pd.DataFrame) -> pd.Series:
    """Everything this question could draw on, before the two servo gates.

    Task, exploration width, budget, the eval-env fix, and completion. Kept separate from
    the servo gates so ``_kp_sweep`` can be checked against exactly this pool and no
    wider -- see its docstring.
    """
    return (
        df["env"].eq(TASK)
        & df["net_params.min_std"].eq(MIN_STD)
        & df["config.ppo.total_steps"].isin(BUDGETS)
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
        & df["state"].eq("finished")
    )


def _kp_sweep(df: pd.DataFrame) -> pd.Series:
    """The `walker-joint-stiffness` launch, by two independent fingerprints.

    The tag is what the launch set; ``servo_center == "range"`` is what its config shows.
    Within this question's candidate pool they must agree -- if a future WalkerWalk launch
    sets one without the other, this raises rather than quietly changing who is in the
    cohort.

    **Scoped to the candidate pool on purpose, and it is not a formality.** Checked
    project-wide it already fails: ``9cihzefa`` and ``a5xc6e16`` are **HopperHop** runs
    carrying ``env-override`` with ``servo_center = "qpos0"``, so on that task the tag and
    the config fingerprint genuinely pick out different sets and the tag alone would be
    the wrong gate. That is a fact about HopperHop, not about this cohort, and a check
    wide enough to trip on it would have to be disabled -- which is how a guard stops
    being read. Scope it to what it protects, and let the next question re-derive its own.
    """
    pool = _candidates(df)
    tagged = df["tags"].apply(lambda t: "env-override" in (t or [])) & pool
    centred = df["env_params.servo_center"].eq("range") & pool
    if not tagged.equals(centred):
        disagree = df.loc[(tagged != centred) & pool, "wandb_id"].tolist()
        raise SystemExit(
            f"the two kp-sweep fingerprints disagree on {disagree}: within this "
            f"question's candidate pool the `env-override` tag and "
            f"`servo_center == 'range'` no longer pick out the same runs. Decide which "
            f"one defines the sweep, and say so here -- do not let the cohort change "
            f"silently.")
    return tagged


def _arm(network_class: str):
    """One architecture, with every selection gate applied."""
    def selector(df: pd.DataFrame) -> pd.Series:
        servo_ok = (df["env_params.servo_kp"].isna()
                    | df["env_params.servo_kp"].eq(ADMITTED_SERVO_KP))
        return (
            df["net_params.network_class"].eq(network_class)
            & _candidates(df)
            & servo_ok
            & ~_kp_sweep(df)
        )
    return selector


#: The two arms. `cohort` and `budget` are **columns**, not part of the condition, so an
#: arm keeps one colour everywhere (README §7).
CONDITIONS = {
    "delayed_mlp": _arm("DelayedMLP"),
    "flat_forward_model": _arm("FlatForwardModel"),
}

#: Must hold within a cohort. Anything varying here is a fairness problem until explained.
INVARIANTS = [
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
    "env_params.servo_kp",
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

#: Expected to vary, by design. Reported in their own section of `comparability.txt` so
#: that the INVARIANTS section stays a list of things that are *wrong* if they vary --
#: burying the designed axes in with them is how a real flag gets read as routine.
DESIGN_AXES = [
    "config.ppo.total_steps",
    "summary._step",
    "config.eval.every_steps",
    "env_params.servo_center",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_experiments.dirty",
    "gpu",
    "seed",
]

#: Asserted per arm rather than reported: these are what *make* an arm, so a stray value
#: here would mean the condition label is wrong, which is a bug and not a caveat.
ARM_SIGNATURE = {
    "flat_forward_model": {"net_params.fm_loss_weight": 1.0,
                           "net_params.detach_prediction": True,
                           "net_params.predictor_hidden_sizes": "[256, 256, 256, 256]"},
    "delayed_mlp": {"net_params.fm_loss_weight": None,
                    "net_params.detach_prediction": None,
                    "net_params.predictor_hidden_sizes": None},
}

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")


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


def trailing_mean(steps: np.ndarray, values: np.ndarray,
                  window: int = SMOOTH_WINDOW_STEPS) -> np.ndarray:
    """Mean of every sample in ``(s - window, s]``, for each s in ``steps``.

    A window in *steps* rather than samples, so the smoothing means the same thing at
    each of the three eval cadences in this cohort. Positions whose window is not yet
    fully covered by the run (``s < window``) are NaN, so a criterion cannot be met on two
    early samples -- without that, a run whose first eval point happens to land high
    "solves" the task at step 0.
    """
    lo = np.searchsorted(steps, steps - window, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= window, out, np.nan)


def window_mean(series: pd.DataFrame, end: float) -> tuple[float | None, int]:
    """Mean eval reward in ``(end - FINAL_WINDOW_STEPS, end]``, and how many points."""
    w = series[(series["step"] > end - FINAL_WINDOW_STEPS) & (series["step"] <= end)]
    return (float(w["value"].mean()) if len(w) else None), int(len(w))


def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    budget = int(run["config.ppo.total_steps"])
    row = {
        "condition": run["condition"],
        "cohort": "primary" if budget >= READOUT_STEP else "support",
        "budget": budget,
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "nnx_ppo_commit": run.get("repos.nnx_ppo.commit"),
        "vnl_experiments_dirty": run.get("repos.vnl_experiments.dirty"),
        "gpu": run["gpu"],
        "requeued": "requeue" in (run.get("tags") or []),
        "delay": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        "seed": int(run["seed"]),
        "min_std": run["net_params.min_std"],
        "actual_step": run.get("summary._step"),
        "ctrl_dt": run.get("env_params.ctrl_dt"),
        # 0 or absent, both meaning the shipped torque plant -- see the docstring's gate.
        "servo_kp": run.get("env_params.servo_kp"),
        "servo_center": run.get("env_params.servo_center"),
        # What the config *says*, which for a requeued run describes only its last
        # attempt. `eval_spacing_median` below is what the curve actually has.
        "eval_every_steps": run.get("config.eval.every_steps"),
        # Post-`971ab99` only, so one value -- recorded as a column so a future cohort
        # that straddles the fix cannot pool the two silently (track README).
        "reward_source": "dmc_eval",
        # The run summary's single final eval point. Kept for reference only.
        "reward_last_point": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
    }
    blank = {"reward_at_1b": None, "reward_at_1b_n": 0,
             "reward_final": None, "reward_final_n": 0,
             "reward_max": None, "reward_max_step": None, "smooth_max": None,
             "curve_points": 0, "curve_max_step": None, "eval_spacing_median": None,
             "eval_spacing_tail": None}
    blank.update({f"steps_to_{t}": None for t in THRESHOLDS})
    blank.update({f"solved_{t}": None for t in THRESHOLDS})
    if series is None or series.empty:
        row.update(blank)
        return row

    steps = series["step"].to_numpy()
    values = series["value"].to_numpy()
    smooth = trailing_mean(steps, values)

    final, final_n = window_mean(series, steps.max())
    # Only defined for a run that actually *trained* to the readout step; a 4.8e8 run has
    # no value at 1e9 and must come out blank rather than silently reusing its endpoint.
    # Keyed on the budget, not on the curve's last point: at the 4.8e6 cadence a complete
    # 1e9 run's last eval lands at 9.985e8, so a `steps.max() >= 1e9` test would discard
    # the three runs whose window is in fact full (the window is 5e7 wide).
    at_1b, at_1b_n = ((None, 0) if budget < READOUT_STEP
                      else window_mean(series, READOUT_STEP))
    row.update(
        reward_at_1b=at_1b,
        reward_at_1b_n=at_1b_n,
        reward_final=final,
        reward_final_n=final_n,
        reward_max=float(values.max()),
        reward_max_step=int(steps[values.argmax()]),
        # How high the *smoothed* curve ever got: the y-value a censored run should be
        # read at, and what says whether it missed the criterion narrowly or by 4e2.
        smooth_max=float(np.nanmax(smooth)) if np.isfinite(smooth).any() else None,
        curve_points=int(len(steps)),
        curve_max_step=int(steps.max()),
        eval_spacing_median=(int(np.median(np.diff(steps))) if len(steps) > 1 else None),
        # The cadence at the *end* of the curve. For a requeued run that straddled the
        # 2026-09-15 change the median is its early fine cadence and says nothing about
        # how far short of `total_steps` the last eval legitimately falls -- which is what
        # the staleness check in `main` needs.
        eval_spacing_tail=(int(np.median(np.diff(steps)[-5:])) if len(steps) > 1
                           else None),
    )
    for thr in THRESHOLDS:
        reached = np.where(smooth >= thr)[0]
        row[f"solved_{thr}"] = bool(len(reached))
        row[f"steps_to_{thr}"] = int(steps[reached[0]]) if len(reached) else None
    return row


def build_curves(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    """The eval series, plus the trailing mean the criterion is read off.

    ``reward_smooth`` is written here rather than recomputed in ``plot.py`` so that the
    curve a figure draws and the number ``steps_to_900`` reports are the same object.
    """
    if series is None or series.empty:
        return []
    steps = series["step"].to_numpy()
    values = series["value"].to_numpy()
    smooth = trailing_mean(steps, values)
    budget = int(run["config.ppo.total_steps"])
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "cohort": "primary" if budget >= READOUT_STEP else "support",
             "budget": budget,
             "delay": int(run["net_params.delay_k"]), "seed": int(run["seed"]),
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


def comparability(runs: pd.DataFrame) -> str:
    """One section per cohort, invariants and designed axes reported separately.

    Per cohort is where the invariants have to hold, because that is the only grouping a
    figure ever averages within. Splitting INVARIANTS from DESIGN_AXES is the point: a
    ``*** VARIES ***`` in the first section is a problem, one in the second is the
    experiment.
    """
    parts = []
    for cohort in ("primary", "support"):
        sub = runs[runs["cohort"] == cohort]
        parts += [
            f"\n{'#' * 78}\n# cohort = {cohort}  n={len(sub)}  "
            f"(budgets: {sorted(sub['config.ppo.total_steps'].unique())})\n{'#' * 78}",
            "\n--- INVARIANTS: anything flagged here is a fairness problem -------------",
            comparability_report(sub, invariant_cols=INVARIANTS, group_col="condition"),
            "\n--- DESIGN AXES: expected to vary; see report.md ------------------------",
            comparability_report(sub, invariant_cols=DESIGN_AXES, group_col="condition"),
        ]
    return "\n".join(parts)


def grid_report(df: pd.DataFrame) -> str:
    """Runs per (cohort, arm, delay), so a hole in the design is visible as a hole."""
    parts = []
    for cohort in ("primary", "support"):
        sub = df[df["cohort"] == cohort]
        table = pd.crosstab(sub["condition"], sub["delay"])
        parts.append(f"\ncohort = {cohort}  (runs per arm x delay)\n{table}")
        holes = [(c, d) for c in table.index for d in table.columns
                 if table.loc[c, d] == 0]
        parts.append(f"  holes: {holes if holes else 'none'}")
        ext = sub[sub["budget"] > READOUT_STEP]
        if len(ext):
            parts.append("  runs past the 1e9 readout: " + ", ".join(
                f"{r.wandb_id} {r.condition.replace('flat_forward_model','FM')
                                 .replace('delayed_mlp','MLP')}"
                f" d{r.delay} s{r.seed} {r.budget // 10**6}M"
                for r in ext.itertuples()))
    return "\n".join(parts)


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)
    assert_arm_signature(runs)
    runs = runs.assign(cohort=np.where(
        runs["config.ppo.total_steps"] >= READOUT_STEP, "primary", "support"))

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
        ["cohort", "condition", "delay", "seed", "wandb_id"], ignore_index=True)
    curves_df = pd.DataFrame(curves).sort_values(
        ["cohort", "condition", "delay", "seed", "step"], ignore_index=True)

    missing = int((df["curve_points"] == 0).sum())
    if missing:
        raise SystemExit(
            f"\n*** {missing}/{len(df)} runs have no history artifact. Runs are never "
            f"silently dropped for a missing artifact (README §3); produce them:\n"
            f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
            f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
            f"        --set project='\"{PROJECT}\"'\n")

    # A `history` artifact made while a run was still training is a snapshot, and `ensure`
    # will not replace it (README §6). The tolerance has to come from the eval cadence,
    # not from a fixed fraction: the last eval lands on the last cadence multiple below
    # `total_steps`, so a complete 1e9 run at the 4.8e6 cadence legitimately stops 1.7e6
    # short -- which a 0.1 % rule reads as a truncated run. And it has to be the *tail*
    # cadence, not the median, or the three requeued runs that straddled the 2026-09-15
    # change are judged against their early fine cadence and all three false-positive.
    # Three intervals of slack: a genuinely stale artifact is short by a large fraction of
    # the run (the README's example reached 190e6 of 600e6), not by one interval.
    slack = df["eval_spacing_tail"].fillna(0) * 3 + 1
    stale = df[df["actual_step"] - df["curve_max_step"] > slack]
    if len(stale):
        print("\n*** history artifacts fall more than three tail eval intervals short of "
              "the run's own _step -- produced mid-training. Re-produce with "
              "`artifacts ensure --override`:")
        print(stale[["wandb_id", "curve_max_step", "actual_step",
                     "eval_spacing_tail"]].to_string(index=False))
        raise SystemExit(1)

    # Every primary-cohort run must have a readout at 1e9, or the main figure is quietly
    # comparing different step counts.
    short = df[(df["cohort"] == "primary") & (df["reward_at_1b"].isna())]
    if len(short):
        raise SystemExit(
            f"\n*** primary-cohort runs with no reward at {READOUT_STEP:,}:\n"
            f"{short[['wandb_id', 'budget', 'curve_max_step']].to_string(index=False)}")

    grid = grid_report(df)
    print(grid)
    solved = df.groupby(["cohort", "condition"])[f"solved_{HEADLINE_THRESHOLD}"].agg(
        ["sum", "count"])
    print(f"\nruns reaching the {HEADLINE_THRESHOLD} criterion:\n{solved}")
    print(f"\neval cadences present (config vs measured median):\n"
          f"{df.groupby(['eval_every_steps'])['eval_spacing_median'].agg(['min', 'max', 'count'])}\n")

    report = comparability(runs) + "\n\n" + "#" * 78 + "\n# DESIGN GRID\n" + \
        "#" * 78 + "\n" + grid + "\n"
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
