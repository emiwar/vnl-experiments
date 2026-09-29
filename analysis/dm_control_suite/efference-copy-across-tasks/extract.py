"""Does a delayed policy need an efference copy? Every dm_control task with enough data.

Every delay run in this track has been *efference-matched* by default:
``efference_length == delay_k``, so the actor is handed the whole queue of actions taken
since the observation it is acting on. That is a design decision inherited from the rodent,
not a measurement. This folder tests it, on every task where both arms exist.

The comparison is always the same two arms at a fixed delay -- ``efference_length ==
delay_k`` against ``efference_length = 0`` -- and the question is whether the answer is a
property of *delay* or a property of the *task*.

Panels, not conditions
----------------------
Raw reward is not comparable across these tasks (track README), so every figure is small
multiples with one panel per task and raw reward on each y-axis. A "panel" here is a
**task plus an actuator**, because `../walker-servo-x-forward-model/` and
`../walker-joint-stiffness/` establish that turning WalkerWalk's torque motors into
position servos (``servo_kp = 64``) changes the control problem rather than parameterising
it -- the servo closes a position loop inside every ``sim_dt`` substep, which is the only
undelayed feedback path in an otherwise delayed system. WalkerWalk therefore appears twice
and is never pooled.

=========================  ================  ==========  ========  ===============
panel                      task              servo_kp    budget    1 delay step
=========================  ================  ==========  ========  ===============
``cartpole_swingup``       CartpoleSwingup   0           4.8e8     10 ms
``cheetah_run``            CheetahRun        0           4.8e8     10 ms
``reacher_hard``           ReacherHard       0           4.8e8     20 ms
``humanoid_walk``          HumanoidWalk      0           4.8e8     25 ms
``walker_walk_torque``     WalkerWalk        0           1e9       25 ms
``walker_walk_servo``      WalkerWalk        64          1e9       25 ms
=========================  ================  ==========  ========  ===============

``HopperHop`` is excluded: three efference-matched runs and **no** no-copy arm at all, so
there is nothing to compare. ``WalkerRun``, ``WalkerStand``, ``BallInCup``,
``CartpoleBalance`` and ``HumanoidStand`` have no no-copy runs either. The other
``servo_kp`` values (0.25-128) belong to ``../walker-joint-stiffness/`` and have no no-copy
arm; only 64 does.

**The budget is per panel, and that is a property of the launches, not a choice.** The
no-copy arm exists at exactly one budget per panel -- 4.8e8 for the four single-actuator
tasks, 1e9 for both WalkerWalk panels -- so the budget at which both arms exist is forced.
Within a panel nothing is mixed; across panels the step axis of the time-to-criterion
figure means different things, which is why that figure carries a per-panel budget line.
A WalkerWalk panel therefore must not be read as "slower than CheetahRun": it had twice the
budget to be slow in.

The criterion is per task, and derived
--------------------------------------
``steps_to_*`` needs a reward level to reach, and this track's 900 convention is a
*WalkerWalk* convention. dm_control returns are bounded in [0, 1000] by construction, but
what a competent policy actually reaches is task-specific: CartpoleSwingup in this cohort
never exceeds 8.8e2 at any delay, so a 900 criterion would censor every run in that panel
and report nothing. Each panel's criterion is therefore a fraction of **its own delay-0
anchor** -- the mean ``reward_final`` of that panel's ``delay_k = 0`` runs, which is what an
undelayed policy achieves on that task at that budget. The headline fraction is 0.9;
0.8 and 0.95 are extracted so ``plot.py`` can show the ordering does not depend on it, and
the absolute level each fraction resolves to is written into ``data.csv`` (``thr_*``) and
printed by :func:`anchor_table` rather than left implicit.

Two consequences a reader has to hold:

* **A panel's criterion is only as good as its anchor.** ``anchor_table`` prints the spread
  of the delay-0 runs behind each one; where that spread is large the criterion inherits it,
  and the report says so rather than quoting a time.
* ``steps_to_900`` is **also** kept, for the two WalkerWalk panels, so a number here can
  still be put beside ``../walker-joint-stiffness/`` and
  ``../walker-servo-x-forward-model/``, whose runs are partly the same runs.

The three arms
--------------
``efference`` (``efference_length == delay_k``), ``no_efference`` (``efference_length = 0``)
and ``undelayed`` (``delay_k = 0``). The third is not a point on the sweep: at
``delay_k = 0`` an efference-matched run *is* an ``efference_length = 0`` run, one
experiment under two names, so assigning it to either arm would be a fiction and assigning
it to both is what ``select_conditions`` refuses. It is drawn as a reference level and used
for the criterion.

``condition`` is ``{panel}_{arm}`` -- 18 cells -- so ``comparability_report`` and
``write_coverage`` work per panel *and* per arm, which is where a fairness problem would
live. ``plot.py`` colours by ``arm`` alone, reusing ``CONDITION_STYLE``'s ``efference`` /
``no_efference`` entries so the manipulation is the same colour here as in the rodent's
``../../rodent/proprioceptive-delay-efference/``.

Selection gates that no run name would reveal
---------------------------------------------
* **network.** ``DelayedMLP`` only. ``FlatForwardModel`` has no no-copy arm anywhere, and
  the one ``FlatRecurrent`` no-copy run (``8debmjui``, WalkerWalk delay 10) belongs to the
  "can recurrence replace the copy" question, not this one -- pooling it would put an LSTM
  in a feedforward cell. It is quoted in report.md instead.
* **budget**, per panel, as above -- ``total_steps`` exactly, *not* "reached it".
* **not** "reached the budget". Runs short of their panel's budget are kept with
  ``usable = False`` and reported by :func:`attrition`, never dropped: in three panels the
  attrition is concentrated in the no-copy arm, and turning "these cells are not measured"
  into "these delays have no no-copy arm" is the failure mode README §3 exists to prevent.
* **min_std.** The project contains a ``min_std`` sweep (0.01-0.2) whose runs are named
  exactly like the defaults. Gated to 0.001.
* **the eval-env fix (``971ab99``, 2026-09-09).** Before it the eval env inherited the
  training wrappers, so ``eval/*`` was scaled x10 *and* truncated by ``EpisodeWrapper``'s
  random phase (track README). Gated on the commit, not the date. Also excludes the
  pre-Hydra era outright, which has no ``net_params.*`` at all.
* **``servo_kp`` is read with absent == 0**, the one place the two spellings of "torque
  control" meet. A run predating ``540e356`` has no field and its env is the unpatched task.

Is the manipulation real?
-------------------------
If the queue were inert, misaligned or never built, "no copy needed" is exactly what this
experiment would return on every task at once. ``efference_identity.py`` tests that
directly -- per task, since ``obs_size`` and ``action_size`` differ -- and report.md carries
its verdict. Read it before reading any null result here.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/efference-copy-across-tasks/extract.py
    ../.venv/bin/python analysis/dm_control_suite/efference-copy-across-tasks/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/efference-copy-across-tasks/extract.py --check

Writes ``data.csv`` (one row per run) and ``curves.csv`` (the eval series on each panel's
common grid). ``plot.py`` reads only those.
"""

from dataclasses import dataclass
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
#:         --runs analysis/dm_control_suite/efference-copy-across-tasks/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"'
#:
#: Same id as every sibling WalkerWalk folder, so a curve here is the same generation of
#: data those analyses read. Asserted in `main`.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

NETWORK = "DelayedMLP"

#: Default exploration width; the project sweeps this elsewhere under identical run names.
MIN_STD = 0.001

#: Runs at or before this commit evaluated on the *training* env, wrappers and all.
PRE_EVAL_FIX_COMMIT = "d168093"


@dataclass(frozen=True)
class Panel:
    """One small multiple: a task, an actuator, and the budget both arms share.

    ``ctrl_dt_ms`` is carried rather than looked up so ``data.csv`` alone determines the
    millisecond axis; it is **asserted** against ``env_params.ctrl_dt`` in `main`, because
    the track README's table is the current registry and not a promise.
    """

    key: str
    task: str
    servo_kp: float
    budget: int
    ctrl_dt_ms: float
    label: str


PANELS = (
    Panel("cartpole_swingup", "CartpoleSwingup", 0.0, 480_000_000, 10,
          "CartpoleSwingup"),
    Panel("cheetah_run", "CheetahRun", 0.0, 480_000_000, 10, "CheetahRun"),
    Panel("reacher_hard", "ReacherHard", 0.0, 480_000_000, 20, "ReacherHard"),
    Panel("humanoid_walk", "HumanoidWalk", 0.0, 480_000_000, 25, "HumanoidWalk"),
    Panel("walker_walk_torque", "WalkerWalk", 0.0, 1_000_000_000, 25,
          "WalkerWalk, torque"),
    Panel("walker_walk_servo", "WalkerWalk", 64.0, 1_000_000_000, 25,
          "WalkerWalk, servo (kp = 64)"),
)
PANEL_BY_KEY = {p.key: p for p in PANELS}

ARMS = ("efference", "no_efference", "undelayed")

#: End-of-training reward is the mean of the eval points in the last this-many steps of the
#: panel's budget -- an absolute window, so a run that stopped early gets no readout rather
#: than one that looks like everyone else's.
FINAL_WINDOW_STEPS = 50_000_000

#: `late_gain` compares the last window against the one two windows earlier: how much the
#: run still gained over its final 1e8 steps. The per-run form of README §6's
#: mid-training-budget trap.
LATE_WINDOW_STEPS = 50_000_000

#: Common eval grid: the *coarsest* cadence in the cohort. Not optional -- cadence is partly
#: confounded with the arm (every no-copy run logs every 4.8e6; many efference runs log
#: every 6e5), and a trailing window defined in steps would average 8x more samples on the
#: fine-cadence side.
GRID_STEPS = 4_800_000

#: Trailing window the criterion is read off, in steps: 5 grid points after regridding.
#: Matches both sibling WalkerWalk folders. The native-cadence variant uses their other
#: definition so a number here can be put beside either.
SMOOTH_WINDOW_STEPS = 24_000_000
NATIVE_WINDOW_STEPS = 10_000_000

#: Time-to-criterion levels, as fractions of each panel's own delay-0 anchor. See the
#: module docstring for why this is not a fixed reward number.
FRACTIONS = (0.8, 0.9, 0.95)
HEADLINE_FRACTION = 0.9

#: Kept *in addition*, and only meaningful on the WalkerWalk panels, so that a number here
#: can be put beside `../walker-joint-stiffness/` and `../walker-servo-x-forward-model/`.
LEGACY_THRESHOLD = 900

#: A run whose trailing mean spends longer than this between the 0.8 and 0.9 criteria has
#: *stalled* rather than passed through. Inherited from `../walker-servo-x-forward-model/`.
STALL_DWELL_STEPS = 100_000_000

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")

#: Per-joint servo parameters `train.py` records from `actuator_mode.servo_report`.
SERVO_JOINTS = ("right_hip", "right_knee", "right_ankle",
                "left_hip", "left_knee", "left_ankle")


def _kp(df: pd.DataFrame) -> pd.Series:
    """``servo_kp``, with *absent* read as 0 -- the one place the two spellings meet."""
    return df["env_params.servo_kp"].fillna(0.0)


#: The columns that, together, say "this is the same experiment". Two runs agreeing on all
#: of them are not independent replicates -- they are the same draw, repeated, differing
#: only in GPU nondeterminism. Copied field for field from
#: ``../walker-servo-x-forward-model/`` so the two folders cannot disagree about which run
#: of a pair is in a figure.
DEDUPE_KEY = [
    "env",
    "net_params.network_class",
    "net_params.delay_k",
    "net_params.efference_length",
    "net_params.min_std",
    "config.ppo.total_steps",
    "env_params.servo_kp",
    "env_params.servo_damping_ratio",
    "env_params.servo_center",
    "seed",
    "config.seed",
]


def _first_of_duplicates(df: pd.DataFrame, candidates: pd.Series) -> pd.Series:
    """One run per (params, both seeds) group: the earliest launched.

    Two runs agreeing on every entry of :data:`DEDUPE_KEY` -- including **both** seeds, the
    network-initialisation seed and the PPO seed that keys env resets, rollouts and eval
    (track README) -- are one experiment run twice. Counting both weights that seed double
    in every cell mean and spread, which matters most where cells are n=1, i.e. everywhere
    in the no-copy arm.

    ``env`` is in the key because this cohort spans six panels: without it two runs of
    different *tasks* sharing every other field would collapse into one.

    **Earliest launched, tie-broken on ``wandb_id``**, so a duplicate arriving later never
    changes which run is kept and ``runs.csv`` does not churn on a refresh.
    """
    pool = df[candidates].copy()
    for col in ("env_params.servo_kp", "env_params.servo_damping_ratio"):
        pool[col] = pool[col].fillna(0.0)
    pool["env_params.servo_center"] = pool["env_params.servo_center"].fillna("absent")
    keep = (pool.sort_values(["created_at", "wandb_id"])
                .groupby(DEDUPE_KEY, dropna=False)["wandb_id"].first())
    return df["wandb_id"].isin(set(keep))


def _base(df: pd.DataFrame) -> pd.Series:
    """Gates that apply to every panel, before task / actuator / budget / arm."""
    return (
        df["net_params.network_class"].eq(NETWORK)
        & df["net_params.min_std"].eq(MIN_STD)
        # Excludes the pre-Hydra era outright: those have no `net_params.*` at all.
        & df["net_params.delay_k"].notna()
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
    )


def _cohort(df: pd.DataFrame) -> pd.Series:
    """Everything any panel could draw on. De-duplicated last, so an excluded run cannot
    displace an admissible twin."""
    base = _base(df)
    return base & _first_of_duplicates(df, base)


def _cell(panel: Panel, arm: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        mask = (_cohort(df)
                & df["env"].eq(panel.task)
                & _kp(df).eq(panel.servo_kp)
                & df["config.ppo.total_steps"].eq(panel.budget))
        if arm == "undelayed":
            return mask & df["net_params.delay_k"].eq(0)
        mask &= df["net_params.delay_k"].gt(0)
        if arm == "efference":
            return mask & df["net_params.efference_length"].eq(df["net_params.delay_k"])
        return mask & df["net_params.efference_length"].eq(0)
    return selector


CONDITIONS = {f"{p.key}_{arm}": _cell(p, arm) for p in PANELS for arm in ARMS}

#: Must hold *within a panel*. Across panels almost all of them vary by design, which is
#: why `panel_invariants` reports per panel and `comparability_report` is grouped by the
#: `{panel}_{arm}` condition rather than by the arm alone.
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
    "env_params.action_repeat",
    "env_params.impl",
    "env_params.nconmax",
    "env_params.njmax",
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

#: Expected to vary. Reported separately so a ``*** VARIES ***`` above stays a signal.
DESIGN_AXES = [
    "env",
    "env_params.servo_kp",
    "net_params.delay_k",
    "net_params.efference_length",
    "env_params.servo_center",
    "config.eval.every_steps",
    "state",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_experiments.dirty",
    "requeue.restart_count",
    "gpu",
    "seed",
    "config.seed",
]


# ---------------------------------------------------------------------------
# Curve readouts
# ---------------------------------------------------------------------------

def history_of(store: Store, wandb_id: str,
               spec_id: str) -> tuple[pd.DataFrame | None, str, str | None]:
    """``(frame, status)`` with status one of ``absent`` / ``empty`` / ``ok``.

    The three are different facts and only one of them is a problem. **Absent** means the
    artifact was never produced, and `main` refuses to continue -- README §3, runs are never
    silently dropped for a missing artifact. **Empty** means the artifact exists and the run
    logged no eval rows at all, because it died before its first eval: three runs here did
    (two CartpoleSwingup ``failed`` at step 0, one CheetahRun ``crashed``). That is a
    property of the run, not a gap in the store, and re-producing it would produce the same
    zero bytes forever. Those runs stay in ``data.csv`` with ``usable = False`` and blank
    readouts, and :func:`attrition` names them.
    """
    entry = store.lookup("history", wandb_id, spec_id)
    if entry is None:
        return None, "absent", None
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None, "empty", entry.created_at
    return ((None, "empty", entry.created_at) if frame.empty
            else (frame, "ok", entry.created_at))


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

    Picking the *nearest single sample* rather than averaging within the bin keeps the noise
    properties identical across cadences: every grid point is one eval of ``eval.n_envs``
    episodes whichever cadence the run used.
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
    off raw eval points fires on noise. Positions whose window is not yet fully covered are
    NaN, so a run whose first eval point lands high cannot "solve" at step 0.
    """
    lo = np.searchsorted(steps, steps - window, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= window, out, np.nan)


def window_mean(series: pd.DataFrame, upper: int, width: int) -> tuple[float | None, int]:
    """Mean of the eval points in ``(upper - width, upper]``, and how many there were."""
    sub = series[(series["step"] > upper - width) & (series["step"] <= upper)]
    return (float(sub["value"].mean()), len(sub)) if len(sub) else (None, 0)


def curve_stats(series: pd.DataFrame, budget: int) -> dict:
    """Everything derived from a run's eval curve that does not need the panel's anchor."""
    steps = series["step"].to_numpy()
    values = series["value"].to_numpy()
    final, final_n = window_mean(series, budget, FINAL_WINDOW_STEPS)
    late, _ = window_mean(series, budget, LATE_WINDOW_STEPS)
    earlier, _ = window_mean(series, budget - 2 * LATE_WINDOW_STEPS, LATE_WINDOW_STEPS)
    grid = on_common_grid(series)
    smooth = trailing_mean(grid["step"].to_numpy(), grid["value"].to_numpy())
    return {
        "reward_final": final,
        "reward_final_n": final_n,
        "reward_max": float(values.max()),
        "late_gain": (None if late is None or earlier is None else float(late - earlier)),
        "curve_points": int(len(steps)),
        "curve_max_step": int(steps.max()),
        "eval_spacing_median": (int(np.median(np.diff(steps))) if len(steps) > 1
                                else None),
        # The cadence at the *end* of the curve. For a requeued run that straddled the
        # 2026-09-15 cadence change the median describes only its early portion.
        "eval_spacing_tail": (int(np.median(np.diff(steps)[-5:])) if len(steps) > 1
                              else None),
        "smooth_max": (float(np.nanmax(smooth)) if np.isfinite(smooth).any() else None),
        "grid_points": int(len(grid)),
    }


def time_to_level(series: pd.DataFrame, level: float) -> int | None:
    """First step whose trailing mean on the common grid reaches ``level``, or None.

    None is **censored, not missing**: the run did not get there within its budget, and
    ``smooth_max`` says how close it came. ``plot.py`` draws it on the budget line rather
    than letting a line stop, which would read as absent data.
    """
    grid = on_common_grid(series)
    steps = grid["step"].to_numpy()
    smooth = trailing_mean(steps, grid["value"].to_numpy())
    reached = np.where(smooth >= level)[0]
    return int(steps[reached[0]]) if len(reached) else None


def time_to_level_native(series: pd.DataFrame, level: float) -> int | None:
    """The sibling folders' other definition: native cadence, 1e7 window."""
    steps = series["step"].to_numpy()
    smooth = trailing_mean(steps, series["value"].to_numpy(), window=NATIVE_WINDOW_STEPS)
    reached = np.where(smooth >= level)[0]
    return int(steps[reached[0]]) if len(reached) else None


def band_dwell(series: pd.DataFrame, lo: float, hi: float) -> int:
    """How long the trailing mean sat between the lowest and the headline criterion.

    These tasks' learning curves are bimodal: a run reaches a plateau and then either steps
    up promptly or sits there for hundreds of millions of steps. The headline criterion
    falls in the gap, so a cell whose seeds stalled reads as "slow to learn" when it is
    "slow to leave the lower mode". Without this column those are indistinguishable.
    """
    grid = on_common_grid(series)
    smooth = trailing_mean(grid["step"].to_numpy(), grid["value"].to_numpy())
    in_band = np.isfinite(smooth) & (smooth >= lo) & (smooth < hi)
    return int(in_band.sum()) * GRID_STEPS


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    panel_key, arm = _split(run["condition"])
    panel = PANEL_BY_KEY[panel_key]
    delay = int(run["net_params.delay_k"])
    eff_len = int(run["net_params.efference_length"])
    actual = run.get("summary._step")
    row = {
        "condition": run["condition"],
        "panel": panel_key,
        "panel_label": panel.label,
        # `efference` / `no_efference` / `undelayed`. plot.py colours on this, reusing
        # CONDITION_STYLE's shared entries.
        "arm": arm,
        "task": run["env"],
        "servo_kp": 0.0 if pd.isna(run.get("env_params.servo_kp"))
                    else float(run["env_params.servo_kp"]),
        "ctrl_dt_ms": panel.ctrl_dt_ms,
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "state": run["state"],
        "git_commit": run["git_commit"],
        "nnx_ppo_commit": run.get("repos.nnx_ppo.commit"),
        "vnl_experiments_dirty": run.get("repos.vnl_experiments.dirty"),
        "gpu": run["gpu"],
        "restart_count": run.get("requeue.restart_count"),
        "budget": panel.budget,
        "delay": delay,
        "delay_ms": delay * panel.ctrl_dt_ms,
        "efference_length": eff_len,
        # The init seed and the PPO seed are different draws and only the pair identifies a
        # replicate (track README) -- ReacherHard in particular has one batch at i42/p1234
        # and one at i42/p42, which grouping on `seed` alone would pool.
        "seed": int(run["seed"]),
        "ppo_seed": int(run["config.seed"]),
        "seed_pair": f"i{int(run['seed'])}/p{int(run['config.seed'])}",
        "actual_step": actual,
        # Two different failures, kept apart because they have different causes and
        # different consequences. `reached_budget` is about *training*: the run stopped
        # short. `has_readout` is about *logging*: three HumanoidWalk runs trained the full
        # 4.8e8 and are recorded `finished`, but their `eval/*` series stops at 6.6e7 /
        # 2.7e8 / 4.0e8 -- verified against WandB directly, with a single-key fetch, so it
        # is the run and not the multi-key intersection trap. A run like that would pass
        # any "reached its budget" gate and then contribute a blank readout, which is the
        # quietest way to lose a cell. `usable` requires both.
        "reached_budget": bool(pd.notna(actual) and actual >= panel.budget),
        "eval_every_steps": run.get("config.eval.every_steps"),
        # Post-`971ab99` only, so one value -- recorded so a future cohort straddling the
        # eval-wrapper fix cannot pool the two silently.
        "reward_source": "dmc_eval",
        "reward_last_point": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
        "servo_center": run.get("env_params.servo_center"),
        # When the artifact was made, and when the run last said anything. Their order is
        # what separates "this artifact is a mid-training snapshot" from "this run's eval
        # series really does end there"; see the staleness check in `main`.
        "artifact_made_at": run.get("_artifact_created_at"),
        "heartbeat_at": run.get("heartbeat_at"),
    }
    for joint in SERVO_JOINTS:
        row[f"kp_{joint}"] = run.get(f"servo.{joint}.kp")

    blank = {"reward_final": None, "reward_final_n": 0, "reward_max": None,
             "late_gain": None, "curve_points": 0, "curve_max_step": None,
             "eval_spacing_median": None, "eval_spacing_tail": None,
             "smooth_max": None, "grid_points": 0}
    row.update(blank if series is None or series.empty
               else curve_stats(series, panel.budget))
    row["has_readout"] = row["reward_final"] is not None
    row["usable"] = bool(row["reached_budget"] and row["has_readout"])
    return row


def _split(condition: str) -> tuple[str, str]:
    """``'{panel}_{arm}'`` -> ``(panel, arm)``. Split on the arm, not on the last ``_``.

    Two traps, both of which this hit in turn. ``rsplit('_', 1)`` returns
    ``('walker_walk_torque_no', 'efference')`` -- a panel key matching nothing and the wrong
    arm. And matching arms in declaration order is no better, because
    ``'..._no_efference'.endswith('_efference')`` is **true**: every no-copy run would be
    read as a with-copy run of a panel that does not exist. Longest suffix first is what
    makes the two arms distinguishable, so it is sorted here rather than relying on the
    order :data:`ARMS` happens to be written in.
    """
    for arm in sorted(ARMS, key=len, reverse=True):
        if condition.endswith("_" + arm):
            return condition[: -len(arm) - 1], arm
    raise KeyError(f"condition {condition!r} does not end in one of {ARMS}")


def build_curves(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    """The eval series on the common grid, plus the trailing mean the criterion uses.

    ``reward_smooth`` is written here rather than recomputed in ``plot.py`` so the curve a
    figure draws and the number ``steps_to_*`` reports are the same object.
    """
    if series is None or series.empty:
        return []
    panel_key, arm = _split(run["condition"])
    grid = on_common_grid(series)
    steps = grid["step"].to_numpy()
    values = grid["value"].to_numpy()
    smooth = trailing_mean(steps, values)
    return [{"wandb_id": run["wandb_id"], "panel": panel_key, "arm": arm,
             "delay": int(run["net_params.delay_k"]),
             "seed_pair": f"i{int(run['seed'])}/p{int(run['config.seed'])}",
             "step": int(s), "reward_mean": float(v),
             "reward_smooth": None if not np.isfinite(m) else float(m)}
            for s, v, m in zip(steps, values, smooth)]


def add_criteria(df: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """Per-panel anchors, the levels they imply, and every run's time to each.

    Two passes are needed and that is the point: a panel's criterion is a property of the
    *panel* (what an undelayed policy reaches on that task at that budget), so it cannot be
    computed while walking runs one at a time. Anchors use only ``usable`` delay-0 runs --
    a criterion set by a run that stopped early would be too low for every run it is then
    applied to.
    """
    df = df.copy()
    for col in ([f"thr_{int(f * 100)}" for f in FRACTIONS]
                + [f"steps_to_{int(f * 100)}" for f in FRACTIONS]
                + [f"solved_{int(f * 100)}" for f in FRACTIONS]
                + ["anchor", "anchor_n", "band_dwell",
                   f"steps_to_{LEGACY_THRESHOLD}",
                   f"solved_{LEGACY_THRESHOLD}",
                   f"steps_to_{LEGACY_THRESHOLD}_native"]):
        df[col] = None

    for panel_key in df["panel"].unique():
        rows = df["panel"] == panel_key
        anchors = df[rows & df["arm"].eq("undelayed") & df["usable"]
                     & df["reward_final"].notna()]["reward_final"]
        anchor = float(anchors.mean()) if len(anchors) else None
        df.loc[rows, "anchor"] = anchor
        df.loc[rows, "anchor_n"] = len(anchors)
        levels = {f: (None if anchor is None else anchor * f) for f in FRACTIONS}
        for f, level in levels.items():
            df.loc[rows, f"thr_{int(f * 100)}"] = level

        for idx in df.index[rows]:
            series = curves.get(df.at[idx, "wandb_id"])
            if series is None or series.empty:
                continue
            for f, level in levels.items():
                if level is None:
                    continue
                hit = time_to_level(series, level)
                df.at[idx, f"steps_to_{int(f * 100)}"] = hit
                df.at[idx, f"solved_{int(f * 100)}"] = hit is not None
            lo, hi = levels[FRACTIONS[0]], levels[HEADLINE_FRACTION]
            if lo is not None and hi is not None:
                df.at[idx, "band_dwell"] = band_dwell(series, lo, hi)
            # The absolute 900, kept only so a WalkerWalk number here can be put beside the
            # sibling folders'. Meaningless on CartpoleSwingup, whose runs never reach it --
            # which is exactly why the headline criterion is a fraction of the anchor.
            hit900 = time_to_level(series, LEGACY_THRESHOLD)
            df.at[idx, f"steps_to_{LEGACY_THRESHOLD}"] = hit900
            df.at[idx, f"solved_{LEGACY_THRESHOLD}"] = hit900 is not None
            df.at[idx, f"steps_to_{LEGACY_THRESHOLD}_native"] = time_to_level_native(
                series, LEGACY_THRESHOLD)
    return df


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def mode_rate(df: pd.DataFrame) -> str:
    """How many runs of each cell reached the headline criterion, not what they averaged.

    Added 2026-09-29, because HumanoidWalk made the cell mean unusable. From delay 5 onward
    that panel's runs are **bimodal in both arms** -- within-cell spreads of 4.7e2 to 9.0e2
    reward, with individual runs at 4.4e1 and 9.4e1 beside others at 9.4e2 and 9.7e2 -- so a
    cell mean is a weighted average of "solved" and "never got going" and moves with the
    ratio of the two rather than with anything about the policy. A mean of 3.1e2 does not
    describe any run in the cell.

    Where a cell is bimodal the informative statistic is the **rate**: of the runs in this
    cell, how many crossed the criterion. It is still a small-n statistic and it is reported
    as a fraction rather than dressed up as a probability, but unlike the mean it is a
    quantity every run in the cell actually has.

    Printed for every panel, not just the bimodal ones, so that a reader can see which
    panels the distinction matters for -- in ReacherHard and the WalkerWalk panels almost
    every cell is 1/1 or 3/3 and the mean is fine.
    """
    key = f"solved_{int(HEADLINE_FRACTION * 100)}"
    usable = df[df["usable"] & df["arm"].isin(("efference", "no_efference"))]
    lines = [f"\nfraction of each cell's runs reaching the {HEADLINE_FRACTION:.0%} criterion",
             "  (the readout to prefer wherever WITHIN-CELL SPREAD shows the cell is "
             "bimodal)"]
    for p in PANELS:
        sub = usable[usable["panel"] == p.key]
        if sub.empty:
            continue
        lines.append(f"\n  {p.label}")
        lines.append(f"  {'delay':>6} {'with copy':>12} {'no copy':>12}   spread(copy) "
                     f"spread(none)")
        for delay in sorted(int(d) for d in sub["delay"].unique()):
            cells = {}
            for arm in ("efference", "no_efference"):
                c = sub[(sub["delay"] == delay) & (sub["arm"] == arm)]
                if c.empty:
                    cells[arm] = ("-", "")
                    continue
                solved = int(c[key].astype("boolean").fillna(False).sum())
                rng = c["reward_final"]
                cells[arm] = (f"{solved}/{len(c)}",
                              f"{rng.max() - rng.min():.0f}" if len(c) > 1 else "")
            lines.append(f"  {delay:>6} {cells['efference'][0]:>12} "
                         f"{cells['no_efference'][0]:>12}   "
                         f"{cells['efference'][1]:>12} {cells['no_efference'][1]:>12}")
    lines.append(
        "\n  A cell at 1/3 or 2/4 is not a policy that scores its cell mean; it is a coin\n"
        "  flip whose bias this many runs cannot measure. Where both arms of a delay are\n"
        "  fractional, that delay says nothing about the efference copy however large the\n"
        "  difference of means happens to be.\n")
    return "\n".join(lines)


def paired_contrast(df: pd.DataFrame) -> str:
    """Every comparison that shares a seed pair: the same draw, with and without the copy.

    This is the strongest evidence in the folder and it exists by accident rather than by
    design -- the no-copy arms were launched at whatever seed the panel's first batch used,
    and in five panels that seed also appears in the efference arm. Where it does, the
    difference is **paired**: same network initialisation, same PPO stream keying the env
    resets, rollouts and eval, same task, same delay, same budget, differing only in
    whether the actor sees its own recent actions.

    Why it matters more than the cell means above: `seed_spread` shows ReacherHard's
    efference cells spanning 1.1e2-1.8e2 reward at the long delays, which is *larger* than
    the cell-mean differences there. A cell-mean contrast at that panel therefore cannot
    resolve its own effect, and a paired one can, because the seed variation is differenced
    out rather than averaged over.

    What it still is not: one pair per delay is one observation, not a paired *test*. What
    it buys is that a paired sign disagreeing with the cell-mean sign at the same delay
    exposes that cell's effect as seed noise -- and the per-panel sign count below is a
    sign test over delays, whose caveat is that the delays of one panel share a single
    no-copy seed and so are not independent draws of it.
    """
    usable = df[df["usable"] & df["arm"].isin(("efference", "no_efference"))]
    lines = ["\npaired contrasts: same panel, same delay, SAME seed pair, both arms"]
    total = 0
    for p in PANELS:
        sub = usable[usable["panel"] == p.key]
        rows = []
        for (delay, seed), g in sub.groupby(["delay", "seed_pair"]):
            if set(g["arm"]) != {"efference", "no_efference"}:
                continue
            eff = g[g["arm"] == "efference"].iloc[0]
            non = g[g["arm"] == "no_efference"].iloc[0]
            rows.append((int(delay), seed, eff["reward_final"], non["reward_final"]))
        if not rows:
            lines.append(f"\n  {p.label}: no delay has both arms at one seed")
            continue
        total += len(rows)
        lines.append(f"\n  {p.label}   ({len(rows)} paired delay(s))")
        lines.append(f"  {'delay':>6} {'seed':>11} {'with copy':>11} {'no copy':>11} "
                     f"{'difference':>12}")
        signs = []
        for delay, seed, e, n in sorted(rows):
            signs.append(np.sign(n - e))
            lines.append(f"  {delay:>6} {seed:>11} {e:>11.1f} {n:>11.1f} "
                         f"{n - e:>+12.1f}")
        pos, neg = int(sum(s > 0 for s in signs)), int(sum(s < 0 for s in signs))
        lines.append(f"    sign: {pos} of {len(signs)} delays favour NO copy, "
                     f"{neg} favour the copy")
    lines.append(
        f"\n  {total} paired comparison(s) in all. Compare each against the cell-mean\n"
        f"  difference for the same panel and delay in THE CONTRAST: REWARD. Agreement in\n"
        f"  *sign* is what pairing buys; a disagreement means that cell's effect is seed\n"
        f"  noise, which nothing else here could detect.\n")
    return "\n".join(lines)


def design_grid(df: pd.DataFrame) -> str:
    """Runs per (panel, arm, delay), and how many of them reached the budget."""
    lines = []
    for p in PANELS:
        sub = df[df["panel"] == p.key]
        if sub.empty:
            lines.append(f"\n{p.label}: no runs")
            continue
        all_t = pd.crosstab(sub["arm"], sub["delay"])
        ok = pd.crosstab(sub[sub.usable]["arm"], sub[sub.usable]["delay"]).reindex(
            index=all_t.index, columns=all_t.columns, fill_value=0)
        both = [int(d) for d in all_t.columns
                if all_t.get(d, pd.Series()).get("efference", 0) > 0
                and all_t.get(d, pd.Series()).get("no_efference", 0) > 0]
        both_ok = [int(d) for d in ok.columns
                   if ok.get(d, pd.Series()).get("efference", 0) > 0
                   and ok.get(d, pd.Series()).get("no_efference", 0) > 0]
        lines.append(f"\n=== {p.label}  (budget {p.budget / 1e6:.0f}e6, "
                     f"1 step = {p.ctrl_dt_ms:g} ms)")
        lines.append("all runs:\n" + all_t.to_string())
        lines.append("reached the budget:\n" + ok.to_string())
        lines.append(f"  delays with BOTH arms present : {both}")
        lines.append(f"  delays with BOTH arms usable  : {both_ok}")
        if set(both) - set(both_ok):
            lines.append(f"  *** {sorted(set(both) - set(both_ok))} have both arms but "
                         f"not both finished -- not a hole in the design, a hole in the "
                         f"data ***")
    lines.append("\n  Only the 'usable' delays support a contrast. A delay present above "
                 "and absent\n  below is measured-but-unfinished, which is a different "
                 "statement from never run.\n")
    return "\n".join(lines)


def anchor_table(df: pd.DataFrame) -> str:
    """Each panel's delay-0 anchor, its spread, and the criteria it implies.

    The criterion is only as good as the anchor, so the spread of the runs behind it is
    printed beside it. A panel whose delay-0 runs disagree by more than a few percent
    cannot support a quoted time-to-criterion, and report.md says so rather than quoting
    one anyway.
    """
    lines = ["\nper-panel delay-0 anchor and the criteria it sets",
             f"{'panel':<22} {'n':>2} {'anchor':>8} {'min':>8} {'max':>8} {'spread%':>8} "
             + " ".join(f"{'thr' + str(int(f * 100)):>8}" for f in FRACTIONS)]
    for p in PANELS:
        sub = df[(df["panel"] == p.key) & df["arm"].eq("undelayed") & df["usable"]]
        vals = sub["reward_final"].dropna()
        if not len(vals):
            lines.append(f"{p.label:<22} {'0':>2} {'-- no usable delay-0 run --':>50}")
            continue
        anchor = vals.mean()
        spread = 100 * (vals.max() - vals.min()) / anchor if anchor else float("nan")
        lines.append(f"{p.label:<22} {len(vals):>2} {anchor:>8.1f} {vals.min():>8.1f} "
                     f"{vals.max():>8.1f} {spread:>7.1f}% "
                     + " ".join(f"{anchor * f:>8.1f}" for f in FRACTIONS))
    lines.append("\n  'anchor' is the mean reward_final of that panel's delay-0 runs that "
                 "reached the\n  budget. 'spread%' is (max-min)/anchor over those runs: "
                 "where it is large the\n  criterion inherits that noise and a time-to-"
                 "criterion is not quotable.\n")
    return "\n".join(lines)


def contrast_table(df: pd.DataFrame, readout: str, scale: float, unit: str) -> str:
    """Per panel and delay: with the copy, without it, and the difference.

    A cell mean over "the runs that happened to finish or happened to reach the criterion"
    is the one number this must not print, so any cell with a missing value shows a dash.
    """
    usable = df[df["usable"]]
    lines = [f"\n{readout} ({unit}); cells are means over seeds, "
             f"'-' = cell empty, unfinished, or some run censored"]
    for p in PANELS:
        sub = usable[usable["panel"] == p.key]
        if sub.empty:
            continue
        ref = sub[sub["arm"].eq("undelayed")][readout]
        ref_txt = (f"{float(ref.mean()) * scale:.1f} (n={len(ref)})"
                   if len(ref) and ref.notna().all() else "-")
        lines.append(f"\n  {p.label}   delay 0 reference: {ref_txt}")
        lines.append(f"  {'delay':>6} {'ms':>5} {'with copy':>11} {'n':>3} "
                     f"{'no copy':>11} {'n':>3} {'difference':>12}")
        sweep = sub[sub["arm"].ne("undelayed")]
        for delay in sorted(int(d) for d in sweep["delay"].unique()):
            cells = {}
            for arm in ("efference", "no_efference"):
                c = sweep[(sweep["delay"] == delay) & (sweep["arm"] == arm)]
                cells[arm] = (None if c.empty or c[readout].isna().any()
                              else (float(c[readout].mean()) * scale, len(c)))
            fmt = lambda v: (f"{v[0]:>11.1f} {v[1]:>3}" if v else f"{'-':>11} {'0':>3}")
            row = (f"  {delay:>6} {delay * p.ctrl_dt_ms:>5.0f} "
                   f"{fmt(cells['efference'])} {fmt(cells['no_efference'])}")
            if cells["efference"] and cells["no_efference"]:
                row += f" {cells['no_efference'][0] - cells['efference'][0]:>+12.1f}"
            else:
                row += f" {'-':>12}"
            lines.append(row)
    lines.append(
        "\n  'difference' is (no copy - with copy), so a negative reward difference and a\n"
        "  positive steps difference both mean the efference copy helped. Read each\n"
        "  difference against the spread in WITHIN-CELL SPREAD for the same panel and\n"
        "  delay before calling it real -- most no-copy cells are a single run.\n")
    return "\n".join(lines)


def seed_spread(df: pd.DataFrame) -> str:
    """Within-cell spread per panel: the noise floor every n=1 comparison is read against."""
    usable = df[df["usable"] & df["arm"].ne("undelayed")]
    lines = ["\nreward_final spread within each (panel, arm, delay) cell"]
    for p in PANELS:
        sub = usable[usable["panel"] == p.key]
        if sub.empty:
            continue
        tab = sub.groupby(["arm", "delay"])["reward_final"].agg(
            ["count", "mean", "min", "max"])
        tab["spread"] = tab["max"] - tab["min"]
        multi = tab[tab["count"] > 1]
        lines.append(f"\n  {p.label}: {len(multi)} of {len(tab)} cells have >1 seed")
        if len(multi):
            lines.append(f"    spread over those cells: median {multi['spread'].median():.1f}"
                         f"  max {multi['spread'].max():.1f} reward")
            lines.append("    " + multi.to_string().replace("\n", "\n    "))
        else:
            lines.append("    no multi-seed cell -- this panel supplies no noise floor of "
                         "its own")
    lines.append("\n  Variance is strongly regime-dependent: a solved cell's seeds agree to "
                 "a few\n  reward, a transitioning one's can differ by hundreds. Read a "
                 "comparison against\n  the spread in *its own* regime and its own panel, "
                 "never the cohort maximum.\n")
    return "\n".join(lines)


def attrition(df: pd.DataFrame) -> str:
    """Which runs fall short of their panel's budget, and is that correlated with the arm?

    README §6: a cohort whose attrition correlates with the swept variable cannot be read
    at a common budget as if the gap were noise. Here it does correlate, in three panels,
    and always in the same direction -- the no-copy arm was launched last everywhere.
    """
    short = df[~df["usable"]]
    lines = [f"\nruns without a usable readout at their panel's budget: "
             f"{len(short)} of {len(df)}",
             f"  of which: stopped training early {int((~short['reached_budget']).sum())}, "
             f"trained fully but stopped logging eval "
             f"{int((short['reached_budget'] & ~short['has_readout']).sum())}"]
    if short.empty:
        return "\n".join(lines + ["   none\n"])
    for p in PANELS:
        sub = short[short["panel"] == p.key]
        if sub.empty:
            continue
        total = df[df["panel"] == p.key].groupby("arm").size()
        lost = sub.groupby("arm").size()
        lines.append(f"\n  {p.label}: {len(sub)} short "
                     + "  ".join(f"{a}={int(lost.get(a, 0))}/{int(total.get(a, 0))}"
                                 for a in ARMS))
        for _, r in sub.sort_values(["arm", "delay"]).iterrows():
            why = ("stopped training" if not r["reached_budget"]
                   else "trained fully, eval series ends early")
            last = (r["curve_max_step"] / 1e6 if pd.notna(r["curve_max_step"])
                    else float("nan"))
            lines.append(f"    {r['wandb_id']}  {r['arm']:<13} delay {int(r['delay']):>2}  "
                         f"{r['seed_pair']:<11} state={r['state']:<9} trained to "
                         f"{(r['actual_step'] or 0) / 1e6:7.1f}e6, last eval at "
                         f"{last:7.1f}e6   {why}")
    lines.append(
        "\n  Where the losses sit in one arm, the missing cells are not missing at random.\n"
        "  They are not an outcome either -- the no-copy arms were launched last in every\n"
        "  panel -- but they do decide which delays can be contrasted at all.\n")
    return "\n".join(lines)


def bimodality(df: pd.DataFrame) -> str:
    """How often does a run stall below the headline criterion, and in which arm?"""
    key = f"solved_{int(HEADLINE_FRACTION * 100)}"
    ok = df[df["usable"] & df[key].astype("boolean").fillna(False)
            & df["band_dwell"].notna()]
    if ok.empty:
        return "\n  no run both reached the headline criterion and its budget\n"
    stuck = ok[ok["band_dwell"].astype(float) > STALL_DWELL_STEPS]
    lines = [f"\ndwell between the {FRACTIONS[0]:.0%} and {HEADLINE_FRACTION:.0%} criteria, "
             f"for runs that later exceeded the {HEADLINE_FRACTION:.0%} one",
             f"  stalled (> {STALL_DWELL_STEPS / 1e6:.0f}e6 in the band): "
             f"{len(stuck)} of {len(ok)}"]
    if len(stuck):
        lines.append(f"\n{pd.crosstab(stuck['panel'], stuck['arm']).to_string()}")
        for _, r in stuck.sort_values(["panel", "delay"]).iterrows():
            lines.append(f"    {r['wandb_id']}  {r['panel']:<20} {r['arm']:<13} delay "
                         f"{int(r['delay']):>2}  {r['seed_pair']:<11} dwell "
                         f"{float(r['band_dwell']) / 1e6:5.0f}e6")
    lines.append(
        "\n  A cell's time-to-criterion is dominated by how many of its seeds stalled, so\n"
        "  check its steps_to_80 in data.csv before calling it slow: if that moved little\n"
        "  and steps_to_90 moved a lot, what was measured is a stall, not a learning rate.\n")
    return "\n".join(lines)


def requeue_confound(df: pd.DataFrame) -> str:
    """Is preemption confounded with the arm, and does it land on the time readout?

    A light checkpoint does not save env states, so on resume each env redraws its episode
    phase and **the actor's delay and efference queues are zeroed** (track README). That
    eats a transient a never-preempted run does not, and it would make the efference arm
    look slower -- the direction in which this folder reports a no-copy advantage on time.
    """
    key = f"steps_to_{int(HEADLINE_FRACTION * 100)}"
    lines = ["\nrequeue counts by arm (-1 = the field predates the run)"]
    d = df.copy()
    d["restarts"] = d["restart_count"].fillna(-1)
    lines.append(pd.crosstab(d["arm"], d["restarts"]).to_string())
    req = d[(d["restarts"] > 0) & d["usable"] & d["arm"].eq("efference")]
    lines.append(f"\n  requeued efference runs: {len(req)}; their {key} beside their cell")
    for _, r in req.sort_values(["panel", "delay"]).iterrows():
        cell = d[(d.panel == r.panel) & (d.delay == r.delay) & d.arm.eq("efference")
                 & (d.wandb_id != r.wandb_id) & d[key].notna()]
        others = ", ".join(f"{float(v) / 1e6:.0f}" for v in sorted(cell[key])) or "none"
        mine = f"{float(r[key]) / 1e6:.0f}e6" if pd.notna(r[key]) else "censored"
        slowest = (len(cell) and pd.notna(r[key])
                   and float(r[key]) > cell[key].astype(float).max())
        lines.append(f"    {r['wandb_id']}  {r['panel']:<20} delay {int(r['delay']):>2}  "
                     f"restarts {int(r['restarts'])}  {mine:>9}   cell-mates {others}"
                     + ("   slowest in its cell" if slowest else ""))
    lines.append(
        "\n  Read this before reading any time difference. Where the no-copy arm has zero\n"
        "  restarts and the efference arm does not, the two cannot be separated here.\n")
    return "\n".join(lines)


def panel_invariants(runs: pd.DataFrame) -> str:
    """Within each panel, are the invariants single-valued across the two arms?

    ``comparability_report`` is grouped by ``{panel}_{arm}``, which shows each cell's values
    but not whether the *arms of one panel* agree -- and that is the comparison every figure
    makes. This collapses it to one line per panel per column and flags only the columns
    that differ between arms of the same panel.
    """
    lines = ["\nwithin-panel invariant check: does any INVARIANT differ between the arms "
             "of one panel?"]
    clean = True
    for p in PANELS:
        sub = runs[runs["condition"].str.startswith(p.key + "_")]
        if sub.empty:
            continue
        flagged = []
        for col in INVARIANTS:
            if col not in sub or col == "summary._step":
                continue
            per_arm = {c: sorted(map(str, sub[sub["condition"] == c][col].dropna()
                                     .unique()))
                       for c in sub["condition"].unique()}
            values = {tuple(v) for v in per_arm.values() if v}
            if len(values) > 1:
                flagged.append(f"{col}: " + "; ".join(
                    f"{c.removeprefix(p.key + '_')}={v}" for c, v in per_arm.items() if v))
        if flagged:
            clean = False
            lines.append(f"\n  *** {p.label} ***")
            lines += [f"      {f}" for f in flagged]
        else:
            lines.append(f"  {p.label:<26} all invariants agree across arms")
    lines.append("\n  `summary._step` is excluded here and reported by ATTRITION instead: "
                 "it is\n  expected to differ and its differences are the point.\n"
                 if clean else
                 "\n  A flag above is a fairness problem until explained in report.md.\n")
    return "\n".join(lines)


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    # The ms axis is drawn from `ctrl_dt_ms` in data.csv, so a drifted PANELS table would
    # mislabel silently. The registry is the truth; the table is a copy of it.
    for p in PANELS:
        got = runs[runs["condition"].str.startswith(p.key + "_")][
            "env_params.ctrl_dt"].dropna().unique()
        if len(got) and sorted(got) != [p.ctrl_dt_ms / 1000]:
            raise SystemExit(
                f"{p.key}: PANELS says ctrl_dt_ms = {p.ctrl_dt_ms} but the runs carry "
                f"env_params.ctrl_dt = {sorted(got)}. Fix PANELS, not the axis label.")

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

    rows, curve_rows, series_by_id = [], [], {}
    absent, empty = [], []
    for _, run in runs.iterrows():
        frame, status, made_at = history_of(store, run["wandb_id"], history_spec_id)
        run = run.copy()
        run["_artifact_created_at"] = made_at
        (absent if status == "absent" else empty if status == "empty" else []).append(
            run["wandb_id"])
        series = reward_series(frame)
        series_by_id[run["wandb_id"]] = series
        rows.append(build_row(run, series))
        curve_rows.extend(build_curves(run, series))

    df = pd.DataFrame(rows)
    if absent:
        raise SystemExit(
            f"\n*** {len(absent)}/{len(df)} runs have no history artifact. Runs are never "
            f"silently dropped for a missing artifact (README §3); produce them:\n"
            f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
            f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
            f"        --set project='\"{PROJECT}\"'\n")
    if empty:
        # Not an error: the artifact exists and the run has no eval rows to put in it.
        print(f"\nnote: {len(empty)} run(s) logged no eval points at all and carry an "
              f"empty history artifact; they are usable = False and feed no readout:")
        print(df[df.wandb_id.isin(empty)][
            ["wandb_id", "condition", "state", "actual_step"]].to_string(index=False))

    # A `history` artifact made while a run was still training is a snapshot, and `ensure`
    # will not replace it (README §6). The tolerance is three tail eval intervals, because
    # a complete run's last eval lands on the last cadence multiple below its own step.
    # A still-*running* run is exempt: both sides of the comparison move, and it already
    # carries usable = False and feeds no headline.
    # A curve that stops short of the run's own `_step` has two possible causes, and only
    # one of them is fixable. It is a **snapshot** (README §6) if the artifact was made
    # while the run was still going, i.e. before the run's last heartbeat -- re-produce
    # those. It is a **short eval series** if the artifact postdates the run: WandB has no
    # more rows, so re-producing returns the same bytes for ever. Three HumanoidWalk runs
    # are in the second class; their `eval/*` stops at 6.6e7 / 2.7e8 / 4.0e8 while training
    # ran to 4.8e8, confirmed against WandB with a single-key fetch so it is not the
    # multi-key intersection trap the track README warns about. Without this split the
    # check tells you to run `--override` in a loop that can never clear it.
    slack = df["eval_spacing_tail"].fillna(0) * 3 + 1
    stale = df[df["actual_step"] - df["curve_max_step"] > slack].copy()
    made = pd.to_datetime(stale["artifact_made_at"], errors="coerce", utc=True)
    beat = pd.to_datetime(stale["heartbeat_at"], errors="coerce", utc=True)
    snapshot = stale[made < beat]
    truncated = stale[~(made < beat)]
    if len(truncated):
        print("\nnote: these runs trained past their last eval point -- the artifact "
              "postdates the run, so WandB has no more rows and re-producing changes "
              "nothing. They carry has_readout = False:")
        print(truncated[["wandb_id", "panel", "arm", "state", "curve_max_step",
                         "actual_step"]].to_string(index=False))
    if len(snapshot):
        print("\n*** history artifacts made BEFORE their run's last heartbeat and short "
              "of its _step: mid-training snapshots. Re-produce with "
              "`artifacts ensure --override`:")
        print(snapshot[["wandb_id", "panel", "state", "curve_max_step", "actual_step",
                        "artifact_made_at", "heartbeat_at"]].to_string(index=False))
        raise SystemExit(1)

    df = add_criteria(df, series_by_id)
    panel_order = {p.key: i for i, p in enumerate(PANELS)}
    df["_o"] = df["panel"].map(panel_order)
    df = df.sort_values(["_o", "delay", "arm", "seed_pair", "wandb_id"]).drop(
        columns="_o").reset_index(drop=True)
    curves_df = pd.DataFrame(curve_rows)
    curves_df["_o"] = curves_df["panel"].map(panel_order)
    curves_df = curves_df.sort_values(
        ["_o", "delay", "arm", "seed_pair", "step"]).drop(columns="_o").reset_index(
        drop=True)

    grid = design_grid(df)
    anchors = anchor_table(df)
    short = attrition(df)
    spread = seed_spread(df)
    reward = contrast_table(df, "reward_final", 1.0, "reward")
    steps = contrast_table(df, f"steps_to_{int(HEADLINE_FRACTION * 100)}", 1e-6,
                           "e6 steps")
    rates = mode_rate(df)
    paired = paired_contrast(df)
    modes = bimodality(df)
    requeue = requeue_confound(df)
    invariants = panel_invariants(runs)
    for block in (grid, anchors, short, spread, reward, steps, rates, paired, modes,
                  requeue, invariants):
        print(block)

    report = (comparability_report(runs, invariant_cols=INVARIANTS,
                                   group_col="condition")
              + "\n\n--- DESIGN AXES: expected to vary; see report.md ------------------\n"
              + comparability_report(runs, invariant_cols=DESIGN_AXES,
                                     group_col="condition")
              + "\n\n" + "#" * 78 + "\n# DESIGN GRID\n" + "#" * 78 + grid
              + "\n" + "#" * 78 + "\n# PANEL ANCHORS AND CRITERIA\n" + "#" * 78 + anchors
              + "\n" + "#" * 78 + "\n# ATTRITION\n" + "#" * 78 + short
              + "\n" + "#" * 78 + "\n# WITHIN-CELL SPREAD\n" + "#" * 78 + spread
              + "\n" + "#" * 78 + "\n# THE CONTRAST: REWARD\n" + "#" * 78 + reward
              + "\n" + "#" * 78 + "\n# THE CONTRAST: TIME\n" + "#" * 78 + steps
              + "\n" + "#" * 78 + "\n# MODE RATES\n" + "#" * 78 + rates
              + "\n" + "#" * 78 + "\n# PAIRED CONTRASTS\n" + "#" * 78 + paired
              + "\n" + "#" * 78 + "\n# BIMODALITY\n" + "#" * 78 + modes
              + "\n" + "#" * 78 + "\n# REQUEUE CONFOUND\n" + "#" * 78 + requeue
              + "\n" + "#" * 78 + "\n# WITHIN-PANEL INVARIANTS\n" + "#" * 78 + invariants)
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
