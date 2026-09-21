"""On WalkerWalk, is a joint position servo (equilibrium-point control) a viable action
space, and does its stiffness buy any tolerance to sensorimotor delay?

The delay experiments so far vary a *network-side* delay: the actor sees a ``delay_k``-step
stale observation. The equilibrium-point hypothesis says the periphery should absorb some
of that, because muscle viscoelasticity is an instantaneous spring rather than a loop --
so the actuator would be the only *undelayed* feedback path in an otherwise delayed system.
``servo_kp`` (``envs/servo_control.md``) turns WalkerWalk's torque motors into position
servos to test it.

Started as a viability pilot (delays 0 and 5 only, where it could not have answered the
hypothesis -- torque control loses just 1.2 % over that range, so there was no delay damage
to repair). The 2026-09-17 extension added ``servo_kp`` 32/48/64 and **delay 10**, which is
where torque control first falls apart, and that is where the comparison became informative.

The two conditions
------------------
* ``torque`` -- ``servo_kp = 0``. The model is not patched at all: ``apply_servo`` returns
  the env untouched, so this is WalkerWalk exactly as MuJoCo Playground ships it, with the
  action a normalised joint torque. This is the arm every previous delay run in the project
  used.
* ``servo`` -- ``servo_kp > 0``. Every ``<motor>`` becomes an affine position servo,
  ``tau = kp*(q* - q) - kv*qdot`` with the setpoint ``q* = center + half*ctrl``. The action
  keeps the same dimension and the same ``[-1, 1]`` range; only its meaning changes, from
  "fraction of maximum torque" to "position within the joint's range".

``servo_kp`` is **dimensionless**, normalised per joint by ``kp_ref = gear/half`` -- the
stiffness at which a full-range position error saturates the actuator. WalkerWalk's gears
are 100/50/20 (hip/knee/ankle) and its ``kp_ref`` 57.3/19.1/25.5 N*m/rad, so a raw
stiffness would mean something different on every joint. ``servo_kp = 1`` means "full-range
error saturates"; ``servo_kp = 16`` means "a sixteenth of the range saturates".

Force parity is enforced (``forcerange = [-1, 1]`` in actuator space), so the servo arm has
exactly the torque authority the torque arm has and cannot win by being stronger.

Why ``servo_center = range`` is on every run, including the torque ones
----------------------------------------------------------------------
``servo_center`` decides the setpoint at ``ctrl = 0``. Under ``qpos0`` (the default)
WalkerWalk's knee centre lands on its own upper joint limit, so **all positive ctrl on both
knees is dead** -- 50 % of the policy's output on that joint, and 40 % on the hip. Under
``range`` every joint's usable band is the full ``[-1, 1]``. This pilot uses ``range``
throughout, so the servo arm is not handicapped by an unreachable half of its action space.

**At ``servo_kp = 0`` the field is inert**: ``apply_servo`` returns before reading it, so
the torque runs carry ``servo_center = range`` as a launch label and nothing more. It is
used here as the selector that identifies this one launch, *not* as a physical property. A
future cohort must not read it as one.

Readout
-------
``reward_final`` -- the mean of the eval points in the last 5e7 steps, not the run's final
point (README §6: one inline eval point moves by a few percent). At the 4.8e6 eval cadence
that is ~11 points. ``reward_last_point`` is kept for reference only.

Reward is ``eval/*``, which is post-``971ab99`` here: unscaled and full-length. Note that a
dm_control "eval" is a fresh episode of the *same* task -- there is no held-out set in this
track, so this is not a generalisation measure.

The budget caveat, which is the main thing limiting the conclusion
-----------------------------------------------------------------
At 4.8e8 the **torque** arm at delay 10 is the one that has not converged: it ends at 6.95e2
and is still gaining +4.5e1 per 1e8 steps, while ``servo_kp`` 48/64 have settled (+2.7,
+2.6) at 9.6e2. So the +2.7e2 gap at matched budget is in large part a *rate* difference,
which is README §6's "common mid-training budget" trap -- here pointing in the servo's
favour.

Torque runs at 1e9 exist outside this cohort and reach ~9.44e2 at delay 10 (``e9e1text``
9.46e2, ``biin3y95`` 9.43e2, ``lrrbnmz9`` 9.43e2, run-summary final points, not windowed).
They are deliberately **not** pooled in -- the servo arm was never run at 1e9, so admitting
them would make budget the confound -- and are quoted in report.md as a ceiling instead. Read
against that ceiling the servo's advantage is ~+2e1 rather than +2.7e2, at half the budget.

Design holes, and single-seed cells
-----------------------------------
Only ``servo_kp = 0`` has more than one seed (4 at delay 0, 4 at delay 5, 1 at each of 7/10/
15/20); ``servo_kp = 1`` has two at delay 0; **every other cell is n=1**. The design is not
rectangular either: the soft stiffnesses (0.25-1) were run only at delays 0/5, the stiff ones
(32-64) at 0/5/10, and the torque arm alone reaches delays 15 and 20. All of this is in
``comparability.txt``'s design grid rather than smoothed over. The largest seed spread
anywhere in the cohort is 3.4e2 (``servo_kp = 1``, delay 0: 4.76e2 vs 8.19e2), but variance
is strongly regime-dependent -- in the solved regime the torque seeds span < 5 -- so the
noise floor for a given comparison has to be read from its own regime, not from that
maximum.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-joint-stiffness/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-joint-stiffness/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-joint-stiffness/extract.py --check

Writes ``data.csv`` (one row per run) and ``curves.csv`` (the eval series). ``plot.py``
reads only those.
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
#:         --runs analysis/dm_control_suite/walker-joint-stiffness/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
#:
#: Same id as the sibling WalkerWalk folders, so these curves are the same generation of
#: data those analyses read. Asserted in `main`.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

TASK = "WalkerWalk"
NETWORK = "DelayedMLP"
BUDGET = 480_000_000

#: Identifies the stiffness-sweep launch. Inert at servo_kp = 0 -- see the module
#: docstring -- so it is a launch label, never a physical property.
LAUNCH_CENTER = "range"

#: Default exploration width; the project sweeps this elsewhere under identical run names.
MIN_STD = 0.001

#: Runs at or before this commit evaluated on the *training* env, wrappers and all.
PRE_EVAL_FIX_COMMIT = "d168093"

#: End-of-training reward is the mean of the eval points in the last this-many steps.
FINAL_WINDOW_STEPS = 50_000_000

#: `late_gain` compares the last window of this width against the one two windows earlier,
#: i.e. "how much did the run improve over its final 1e8 steps".
LATE_WINDOW_STEPS = 50_000_000

#: Window over which `max_drawdown_late` looks for a peak-to-trough fall.
DRAWDOWN_WINDOW_STEPS = 200_000_000

#: Common eval grid for the time-to-criterion readout: the *coarsest* cadence in the cohort
#: (the stiffness sweep's 4.8e6). See `on_common_grid` for why this is not optional.
GRID_STEPS = 4_800_000

#: Trailing window the criterion is read off, in steps: 5 grid points after regridding.
SMOOTH_WINDOW_STEPS = 24_000_000

#: Reward criteria for time-to-solve. dm_control returns are bounded in [0, 1000] by
#: construction (1000 steps x a per-step reward in [0, 1]) and the benchmark literature
#: reports curves rather than declaring a task solved, so there is no published number to
#: inherit. 900 is this project's convention, shared with `walker-forward-model-1b/`: it
#: sits clearly above where these runs plateau when they fail (6e1-8.1e2) and clearly below
#: where they plateau when they succeed (9.4e2-9.8e2). 800 and 950 are extracted alongside
#: so `plot.py` and the report can show the ordering does not depend on the choice.
THRESHOLDS = (800, 900, 950)
HEADLINE_THRESHOLD = 900


def _cohort(df: pd.DataFrame) -> pd.Series:
    """Every WalkerWalk MLP run at the 4.8e8 budget that is comparable on reward.

    Gated on the *configuration*, never on the run name or the WandB note. **Budget is
    pinned to 4.8e8 rather than made a facet**: the 1e9 / 2e9 / 4e9 torque runs exist and
    are tempting as extra baseline, but the servo arm was never run at those budgets, and
    the servo learns at a visibly different rate -- so admitting them would be README §6's
    "common mid-training budget" trap with the budget itself as the confound. The 1e9
    torque points are quoted in report.md as a *ceiling* caveat instead of plotted here.

    Two launches are pooled, which is what gives delay 10 a torque baseline at all:

    * the stiffness sweep (``7bedb5cc``, 2026-09-16/17), which carries ``servo_center``;
    * the torque delay sweep (``971ab99e``, 2026-09-09), which predates ``servo_kp`` and
      so has no servo fields -- its env is the unpatched task, i.e. torque control.

    That pooling is licensed two ways. The sibling
    ``walker-forward-model-1b/nan_guard_inert.py`` already read every diff between these
    commits and found only render-path, diagnostics-only and ``servo_kp`` changes (inert at
    0), and proved no run in that cohort ever diverged. And it is checked *empirically*
    here: the two launches overlap at delays 0 and 5, and ``launch_overlap`` in `main`
    compares them.
    """
    return (
        df["env"].eq(TASK)
        & df["net_params.network_class"].eq(NETWORK)
        & df["config.ppo.total_steps"].eq(BUDGET)
        & df["net_params.min_std"].eq(MIN_STD)
        # Pre-fix runs evaluated on the *training* env: `eval/*` scaled x10 and truncated
        # by EpisodeWrapper's random phase. Gated on the commit, not the date -- both sides
        # of the fix were created on 2026-09-09 (track README).
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
        # Excludes the pre-Hydra 9e1-run era outright: those have no `net_params.*` at all.
        & df["net_params.delay_k"].notna()
        # `state == "finished"` alone silently drops runs that trained fully and died in
        # the final eval (README §6). Gate on the property meant: reached its budget.
        & (df["state"].eq("finished") | df["summary._step"].ge(BUDGET))
    )


def _kp(df: pd.DataFrame) -> pd.Series:
    """``servo_kp``, with *absent* read as 0.

    A run predating ``540e356`` has no ``env_params.servo_kp`` column value at all, and its
    env is the unpatched Playground task -- which is exactly what ``servo_kp = 0`` means.
    This is the one place the two spellings of "torque control" are unified; everywhere
    else the column is already numeric.
    """
    return df["env_params.servo_kp"].fillna(0.0)


CONDITIONS = {
    "torque": lambda df: _cohort(df) & _kp(df).eq(0.0),
    "servo": lambda df: _cohort(df) & _kp(df).gt(0.0),
}

#: Must be constant for a kp-to-kp comparison to be fair. `servo_kp` is of course absent:
#: it is the swept axis. `servo_damping_ratio` and `servo_center` are here because they are
#: the other two knobs that change what the servo *is*, and `episode_length` /
#: `reward_scale` / `ctrl_dt` because they change what "reward" means.
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
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "env_params.sim_dt",
    "env_params.servo_damping_ratio",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.normalize_obs",
    "config.eval.every_steps",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "repos.vnl_experiments.dirty",
]

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")

#: Per-joint servo parameters `train.py` records from `actuator_mode.servo_report`. Carried
#: into data.csv so the physics of a run is recoverable from the CSV alone -- `servo_kp` is
#: normalised, so it does not by itself say what stiffness any joint actually had.
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

    **This is what makes the time-to-criterion comparable across the two launches.** The
    2026-09-09 torque delay sweep logged eval every 6e5 steps (800 points) and the stiffness
    sweep every 4.8e6 (100 points) -- an 8x difference. A trailing mean defined in *steps*
    would therefore average 8x more samples for the torque arm, making its criterion
    strictly harder to trigger on an upward fluctuation. Since torque is the baseline arm at
    every delay past 5, that bias would flatter the servo, which is the direction this
    analysis must not be wrong in.

    Picking the *nearest single sample* rather than averaging within the bin keeps the noise
    properties identical too: every grid point is one 256-episode eval in both launches.
    ``steps_to_*`` is then quantised to 4.8e6 steps (1 % of the budget) for every run
    equally. ``main`` reports how far this moves the dense runs' answers.
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
    off raw eval points fires on noise. The window is in *steps*, so it means the same thing
    at any budget -- and after :func:`on_common_grid` it also spans the same number of
    samples in both launches. Positions whose window is not yet fully covered by the run are
    NaN, so a run whose first eval point happens to land high cannot "solve" the task at
    step 0.
    """
    lo = np.searchsorted(steps, steps - window, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= window, out, np.nan)


def time_to_criterion(series: pd.DataFrame) -> dict:
    """``steps_to_<thr>`` / ``solved_<thr>`` / ``smooth_max``, on the common grid.

    A run that never reaches a criterion is **censored, not missing**: ``solved_*`` is False
    and ``steps_to_*`` is blank, while ``smooth_max`` records how close it got, so the plots
    can draw it rather than let the line simply stop (which reads as absent data). At delay
    10 that distinction *is* the result -- torque never reaches 900 and the stiff servos do.
    """
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
    """The same thing without regridding -- kept only to measure what regridding cost."""
    steps = series["step"].to_numpy()
    smooth = trailing_mean(steps, series["value"].to_numpy())
    reached = np.where(smooth >= HEADLINE_THRESHOLD)[0]
    return {f"steps_to_{HEADLINE_THRESHOLD}_native":
            int(steps[reached[0]]) if len(reached) else None}


def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    row = {
        "condition": run["condition"],
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "gpu": run["gpu"],
        "servo_kp": 0.0 if pd.isna(run.get("env_params.servo_kp"))
                    else float(run["env_params.servo_kp"]),
        "servo_center": run.get("env_params.servo_center"),
        # Which launch a run came from. The two are pooled deliberately (see _cohort) and
        # `launch_overlap` checks they agree where they overlap, so this column is what
        # makes that check -- and any future doubt about it -- possible from data.csv alone.
        "launch": ("kp_sweep" if run.get("env_params.servo_center") == LAUNCH_CENTER
                   else "delay_sweep_0909"),
        "eval_every_steps": run.get("config.eval.every_steps"),
        "servo_damping_ratio": run.get("env_params.servo_damping_ratio"),
        "delay": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        "seed": int(run["seed"]),
        "actual_step": run.get("summary._step"),
        "ctrl_dt": run.get("env_params.ctrl_dt"),
        # Post-`971ab99` only, so one value -- recorded as a column so a future cohort that
        # straddles the eval-wrapper fix cannot pool the two silently (track README).
        "reward_source": "dmc_eval",
        # The run summary's single final eval point. Reference only; the figures use
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
             "late_gain": None, "max_drawdown_late": None,
             "curve_points": 0, "curve_max_step": None,
             "smooth_max": None, "grid_points": 0,
             f"steps_to_{HEADLINE_THRESHOLD}_native": None}
    blank.update({f"steps_to_{t}": None for t in THRESHOLDS})
    blank.update({f"solved_{t}": None for t in THRESHOLDS})
    if series is None or series.empty:
        row.update(blank)
        return row

    steps = series["step"].to_numpy()
    values = series["value"].to_numpy()
    window = series[series["step"] >= steps.max() - FINAL_WINDOW_STEPS]

    # Is the run still moving at the budget, and in which direction? The servo arm learns
    # more slowly than torque, so a shared 4.8e8 readout is biased against it -- README §6's
    # "common mid-training budget" trap, and the reason this analysis cannot report a
    # converged stiffness ordering.
    late = series[series["step"] >= steps.max() - LATE_WINDOW_STEPS]["value"]
    earlier = series[(series["step"] >= steps.max() - 3 * LATE_WINDOW_STEPS)
                     & (series["step"] < steps.max() - 2 * LATE_WINDOW_STEPS)]["value"]

    # Largest peak-to-trough fall in the last 2e8 steps. A *negative* late gain can mean
    # either "converged and drifting" or "collapsed just before the readout", and those are
    # different facts about the run; the drawdown separates them.
    tail = values[steps >= steps.max() - DRAWDOWN_WINDOW_STEPS]
    drawdown = float(np.max(np.maximum.accumulate(tail) - tail)) if len(tail) else None

    row.update(
        reward_final=float(window["value"].mean()),
        reward_final_n=int(len(window)),
        reward_max=float(values.max()),
        late_gain=float(late.mean() - earlier.mean()) if len(earlier) else None,
        max_drawdown_late=drawdown,
        curve_points=int(len(steps)),
        curve_max_step=int(steps.max()),
    )
    row.update(time_to_criterion(series))
    row.update(native_time_to_criterion(series))
    return row


def build_curves(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    """The eval series on the common grid, plus the trailing mean the criterion uses.

    ``reward_smooth`` is written here rather than recomputed in ``plot.py`` so the curve a
    figure draws and the number ``steps_to_900`` reports are the same object. The grid is
    the same one :func:`time_to_criterion` reads, so a curve in this file is directly the
    evidence for that run's time-to-criterion.
    """
    if series is None or series.empty:
        return []
    grid = on_common_grid(series)
    steps = grid["step"].to_numpy()
    values = grid["value"].to_numpy()
    smooth = trailing_mean(steps, values)
    kp = (0.0 if pd.isna(run.get("env_params.servo_kp"))
          else float(run["env_params.servo_kp"]))
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "servo_kp": kp,
             "delay": int(run["net_params.delay_k"]), "seed": int(run["seed"]),
             "step": int(s), "reward_mean": float(v),
             "reward_smooth": None if not np.isfinite(m) else float(m)}
            for s, v, m in zip(steps, values, smooth)]


def launch_overlap(df: pd.DataFrame) -> str:
    """Do the two pooled launches agree where they overlap?

    This is the load-bearing comparability check of the whole analysis. Delay 10 has a
    torque baseline only because the 2026-09-09 torque delay sweep is pooled with the
    2026-09-16/17 stiffness sweep, and those launches differ in commit, in eval cadence
    (6e5 vs 4.8e6) and in whether the ``servo_*`` fields exist at all. They also overlap at
    delays 0 and 5 -- so the agreement there is a direct measurement of what the pooling
    costs, rather than an argument from reading diffs.
    """
    t = df[df.condition == "torque"]
    lines = ["\ntorque reward_final by (delay, launch)"]
    tab = t.pivot_table(index="delay", columns="launch", values="reward_final",
                        aggfunc=["mean", "count"])
    lines.append(tab.to_string())
    both = t.groupby("delay")["launch"].nunique()
    shared = both[both > 1].index.tolist()
    lines.append(f"\n  delays present in both launches: {shared}")
    for delay in shared:
        g = t[t.delay == delay].groupby("launch")["reward_final"].mean()
        diff = g.max() - g.min()
        lines.append(f"    delay {delay:2d}: "
                     + "  ".join(f"{k}={v:.1f}" for k, v in g.items())
                     + f"   |diff| = {diff:.1f}")
    lines.append("\n  Read against the within-launch seed spread at the same delay: if the\n"
                 "  between-launch difference is no larger, the launches are poolable on\n"
                 "  reward and the delay-10 baseline stands.\n")
    return "\n".join(lines)


def design_grid(df: pd.DataFrame) -> str:
    """Runs per (servo_kp x delay), so a hole in the design is visible as a hole."""
    table = pd.crosstab(df["servo_kp"], df["delay"])
    holes = [(kp, d) for kp in table.index for d in table.columns
             if table.loc[kp, d] == 0]
    singles = [(kp, d) for kp in table.index for d in table.columns
               if table.loc[kp, d] == 1]
    return (f"\nruns per servo_kp x delay\n{table}\n"
            f"\n  empty cells : {holes if holes else 'none'}"
            f"\n  n=1   cells : {singles if singles else 'none'}"
            f"\n\n  A single-seed cell has no spread of its own. The largest seed spread "
            f"observed\n  anywhere in this cohort is the reference for what counts as "
            f"resolved.\n")


def seed_spread(df: pd.DataFrame) -> str:
    """Within-cell spread, which is the noise floor every n=1 comparison is read against."""
    g = df.groupby(["servo_kp", "delay"])["reward_final"]
    tab = g.agg(["count", "mean", "min", "max"])
    tab["spread"] = tab["max"] - tab["min"]
    multi = tab[tab["count"] > 1]
    tail = (f"\n  largest spread among them: {multi['spread'].max():.1f}\n"
            if len(multi) else "\n  no multi-seed cells\n")
    return (f"\nreward_final by cell\n{tab.to_string()}\n"
            f"\n  cells with >1 seed: {len(multi)}" + tail)


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

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
        ["delay", "servo_kp", "seed", "wandb_id"], ignore_index=True)
    curves_df = pd.DataFrame(curves).sort_values(
        ["delay", "servo_kp", "seed", "step"], ignore_index=True)

    # A `history` artifact made while a run was still training is a snapshot, and `ensure`
    # will not replace it (README §6). Compare what the curve reaches against the index.
    stale = df[df["curve_max_step"].notna()
               & (df["curve_max_step"] < df["actual_step"] * 0.999)]
    if len(stale):
        print("\n*** history artifacts are short of the run's own _step -- produced "
              "mid-training. Re-produce with `artifacts ensure --override`:")
        print(stale[["wandb_id", "curve_max_step", "actual_step"]].to_string(index=False))
        raise SystemExit(1)

    missing = int((df["curve_points"] == 0).sum())
    if missing:
        raise SystemExit(
            f"\n*** {missing}/{len(df)} runs have no history artifact. Runs are never "
            f"silently dropped for a missing artifact (README §3); produce them:\n"
            f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
            f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
            f"        --set project='\"{PROJECT}\"'\n")

    grid = design_grid(df)
    spread = seed_spread(df)
    overlap = launch_overlap(df)
    print(grid)
    print(spread)
    print(overlap)

    report = (comparability_report(runs, invariant_cols=INVARIANTS,
                                   group_col="condition")
              + "\n\n" + "#" * 78 + "\n# DESIGN GRID\n" + "#" * 78 + grid
              + "\n" + "#" * 78 + "\n# WITHIN-CELL SPREAD\n" + "#" * 78 + spread
              + "\n" + "#" * 78 + "\n# CROSS-LAUNCH OVERLAP\n" + "#" * 78 + overlap + "\n")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
