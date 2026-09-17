"""On WalkerWalk, is a joint position servo (equilibrium-point control) a viable action
space, and does its stiffness buy any tolerance to sensorimotor delay?

The delay experiments so far vary a *network-side* delay: the actor sees a ``delay_k``-step
stale observation. The equilibrium-point hypothesis says the periphery should absorb some
of that, because muscle viscoelasticity is an instantaneous spring rather than a loop --
so the actuator would be the only *undelayed* feedback path in an otherwise delayed system.
``servo_kp`` (``envs/servo_control.md``) turns WalkerWalk's torque motors into position
servos to test it.

This is the **first pilot**: is the servo trainable at all, over what stiffness range, and
is there any sign of a delay interaction. It is not a test of the hypothesis, for the
reason in "What this pilot cannot answer" below.

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

What this pilot cannot answer
-----------------------------
**The delays are too small for the hypothesis to be testable.** Torque control loses only
~1.3 % going from delay 0 to delay 5 (9.77e2 -> 9.64e2). There is essentially no delay
damage for peripheral stiffness to repair, so "does stiffness buy delay tolerance" cannot
be answered from these runs in either direction. WalkerWalk at 25 ms/step needs delay >= 15
before torque control is meaningfully hurt, and delay 20 needs ~4e9 steps to train
(``o3lsdt1n``, ``owmz54wv`` -- outside this cohort). What this pilot *does* establish is the
viable stiffness range and the cost of the servo at matched delay, which is what the next
sweep needs in order to be affordable.

Design holes, and single-seed cells
-----------------------------------
Only ``servo_kp`` 0 has three seeds; 1 has two at delay 0; every other cell is n=1. The
design is also not rectangular: ``servo_kp = 0.25`` was run at delay 5 only. Both are
reported in ``comparability.txt``'s design grid rather than smoothed over, and the report
treats any single-cell difference smaller than the seed spread at ``servo_kp = 1`` (3.4e2,
the largest observed) as unresolved.

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

#: The launch marker for this pilot. Inert at servo_kp = 0 -- see the module docstring.
LAUNCH_CENTER = "range"

#: End-of-training reward is the mean of the eval points in the last this-many steps.
FINAL_WINDOW_STEPS = 50_000_000

#: `late_gain` compares the last window of this width against the one two windows earlier,
#: i.e. "how much did the run improve over its final 1e8 steps".
LATE_WINDOW_STEPS = 50_000_000

#: Window over which `max_drawdown_late` looks for a peak-to-trough fall.
DRAWDOWN_WINDOW_STEPS = 200_000_000


def _cohort(df: pd.DataFrame) -> pd.Series:
    """Everything in this one launch, before splitting on stiffness.

    Gated on the *configuration*, not on the run name or the WandB notes. The WalkerWalk
    runs this deliberately excludes are all torque runs from other launches at other
    budgets (1e9/2e9/4e9) and other seeds, which would otherwise enter as extra `torque`
    rows at a budget the servo arm was never run at -- README §6's "common mid-training
    budget" trap, pointing the wrong way.
    """
    return (
        df["env"].eq(TASK)
        & df["net_params.network_class"].eq(NETWORK)
        & df["config.ppo.total_steps"].eq(BUDGET)
        & df["env_params.servo_center"].eq(LAUNCH_CENTER)
        # `state == "finished"` alone silently drops runs that trained fully and died in
        # the final eval (README §6). Gate on the property meant: reached its budget.
        & (df["state"].eq("finished") | df["summary._step"].ge(BUDGET))
    )


CONDITIONS = {
    "torque": lambda df: _cohort(df) & df["env_params.servo_kp"].eq(0.0),
    "servo": lambda df: _cohort(df) & df["env_params.servo_kp"].gt(0.0),
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
    "env_params.servo_center",
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


def build_row(run: pd.Series, series: pd.DataFrame | None) -> dict:
    row = {
        "condition": run["condition"],
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "gpu": run["gpu"],
        "servo_kp": float(run["env_params.servo_kp"]),
        "servo_center": run["env_params.servo_center"],
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
             "curve_points": 0, "curve_max_step": None}
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
    return row


def build_curves(run: pd.Series, series: pd.DataFrame | None) -> list[dict]:
    if series is None or series.empty:
        return []
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "servo_kp": float(run["env_params.servo_kp"]),
             "delay": int(run["net_params.delay_k"]), "seed": int(run["seed"]),
             "step": int(s), "reward_mean": float(v)}
            for s, v in zip(series["step"], series["value"])]


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
    print(grid)
    print(spread)

    report = (comparability_report(runs, invariant_cols=INVARIANTS,
                                   group_col="condition")
              + "\n\n" + "#" * 78 + "\n# DESIGN GRID\n" + "#" * 78 + grid
              + "\n" + "#" * 78 + "\n# WITHIN-CELL SPREAD\n" + "#" * 78 + spread + "\n")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
