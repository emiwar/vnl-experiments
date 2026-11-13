"""Where does WalkerWalkTerrain become interesting, and is a lower reward a plateau or slow learning?

First look at the obstacle-block variant of WalkerWalk (``vnl_experiments/envs/walker_terrain.py``):
box obstacles regenerated every episode, mean top height ``terrain.height``, height and gap
drawn as ``mean * (1 + irregularity * U[-1, 1])``. The preliminary sweeps are three slices
through (delay, height), not a grid:

* **delay sweeps at height 0.1 m and 0.3 m** (irregularity 1, blind), delays 0-25, seed
  pair i42/p1234;
* **height sweep at delay 5** (0.05-0.3 m) and a sparser one at **delay 0** (0.1-0.4 m);
* an **observed-terrain** delay sweep at 0.3 m (delays 0, 2, 5, 7, 10), one **periodic**
  (irregularity 0) run at height 0.1, delay 0, and one blind delay-5 / 0.3 m run trained
  to 2 G steps.

Conditions
----------
* ``flat_ground``: plain WalkerWalk, ``DelayedMLP`` with ``efference_length == delay``, the
  same network / PPO settings, every seed pair available. It is the ``height = 0`` end of
  the height axis and the reference curve for the delay sweep. Mostly MJX-JAX runs from
  before the 2026-11-09 stack update (README: WalkerWalk reward is unchanged across it);
  ``impl`` is a column so the split is visible.
* ``terrain_blind`` / ``terrain_observed`` / ``terrain_periodic``: WalkerWalkTerrain by
  ``terrain.observed`` and ``terrain.irregularity``. Height and delay are columns.

Readouts (all from the eval curve, ``eval/episode_reward/mean``, unscaled)
-------------------------------------------------------------------------
* ``reward``: mean of the eval points in ``(READ_AT - 50 M, READ_AT]`` (README §6).
* ``reward_at_<N>M``: the same 50 M window ending at 100/200/300 M, for the curve shape.
* ``late_gain``: ``reward`` minus the window ending at ``READ_AT - 100 M``.
* ``late_slope`` (+ ``late_slope_se``): OLS slope of the eval points in the last
  :data:`SLOPE_SPAN` steps, in reward per 100 M steps. The plateau test: a run that has
  converged has a slope indistinguishable from zero; a slow learner is still climbing.
* ``t50`` / ``t90``: first eval step at which the reward reaches 50 % / 90 % of the run's
  *own* ``reward``. "Slower learning" moves these right; "an earlier plateau" (a lower
  ceiling reached as fast) leaves them where flat ground has them.
* ``steps_to_{500,800,900}``: first eval step at or above those absolute levels; NaN if
  never reached.
* ``fit_rinf`` / ``fit_a`` / ``fit_tau``: a saturating exponential ``R(t) = Rinf - A exp(-t / tau)``
  fitted to the eval points in ``[FIT_FROM, READ_AT]``. ``Rinf`` is the extrapolated
  ceiling, so "earlier plateau" predicts ``Rinf`` below flat ground's and "slower
  learning" predicts a flat-ground ``Rinf`` with a longer ``tau``. ``Rinf`` is bounded
  to [0, 1000] (the task maximum); ``fit_at_bound`` flags a fit that hit it, i.e. one
  where the ceiling is not identified by the data.
* ``backtest_err``: the same fit on points up to :data:`BACKTEST_AT` only, used to
  predict ``reward``; the error is how far the extrapolation can be trusted -- if it is
  large against a run's distance to its fitted ceiling, the ceiling is not a measurement.
* ``reward_at_{1000,2000}M``: the same 50 M window ending at 1 G / 2 G, for runs trained
  that long (NaN otherwise). The direct check on ``fit_rinf``, which only sees <= 480 M.

``long_curves.csv`` holds the full eval curve of every run trained past ``READ_AT`` at a
delay where a terrain run was trained past it (the flat-ground runs there are the
reference for the long terrain run).

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/extract.py --check
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: Eval curves: the id every sibling folder reads.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

NETWORK = "DelayedMLP"
MIN_STD = 0.001
PRE_EVAL_FIX_COMMIT = "d168093"
CTRL_DT = 0.025

READ_AT = 480_000_000
WINDOW = 50_000_000
SHAPE_AT = (100_000_000, 200_000_000, 300_000_000)
LONG_AT = (1_000_000_000, 2_000_000_000)
SLOPE_SPAN = 150_000_000
SOLVE_LEVELS = (500, 800, 900)
FRACTIONS = (0.5, 0.9)
FIT_FROM = 10_000_000
BACKTEST_AT = 330_000_000
#: Walker's per-step reward is in [0, 1] over 1000 steps.
REWARD_MAX = 1000.0

KEY = "eval/episode_reward/mean"


def _completed(df: pd.DataFrame) -> pd.Series:
    """Trained to its budget (README §6: `finished` alone drops final-eval crashes)."""
    return (df["state"].eq("finished")
            | df["summary._step"].ge(df["config.ppo.total_steps"]))


def _noise(df: pd.DataFrame) -> pd.Series:
    if "net_params.motor_noise_std" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df["net_params.motor_noise_std"], errors="coerce").fillna(0.0)


def _base(df: pd.DataFrame) -> pd.Series:
    return (
        df["net_params.network_class"].eq(NETWORK)
        & df["net_params.min_std"].eq(MIN_STD)
        & df["net_params.efference_length"].eq(df["net_params.delay_k"])
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
        & df["env_params.servo_kp"].fillna(0.0).eq(0.0)
        & _noise(df).eq(0.0)
        & df["net_params.privileged_critic"].map(lambda v: v is not False)
        & df["distill.teacher"].isna()
        & df["config.ppo.total_steps"].ge(READ_AT)
        & _completed(df)
    )


def _flat(df: pd.DataFrame) -> pd.Series:
    return _base(df) & df["env"].eq("WalkerWalk")


def _terrain(observed: bool, irregularity: float):
    def selector(df: pd.DataFrame) -> pd.Series:
        return (_base(df) & df["env"].eq("WalkerWalkTerrain")
                & df["env_params.terrain.observed"].eq(observed)
                & df["env_params.terrain.irregularity"].eq(irregularity))
    return selector


CONDITIONS = {
    "flat_ground": _flat,
    "terrain_blind": _terrain(False, 1.0),
    "terrain_observed": _terrain(True, 1.0),
    "terrain_periodic": _terrain(False, 0.0),
}

INVARIANTS = [
    "config.ppo.n_envs",
    "config.ppo.rollout_length",
    "config.ppo.learning_rate",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "config.ppo.clip_range",
    "config.ppo.discounting_factor",
    "config.ppo.gae_lambda",
    "config.ppo.normalize_advantages",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "env_params.action_repeat",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.normalize_obs",
    "net_params.std_scale",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    # Terrain geometry other than the swept height / irregularity / observed. Absent on
    # flat_ground, which comparability reports per condition.
    "env_params.terrain.gap",
    "env_params.terrain.block_length",
    "env_params.terrain.block_width",
    "env_params.terrain.block_halfheight",
    "env_params.terrain.num_blocks",
    "env_params.terrain.start_gap",
    "env_params.terrain.back_margin",
    "env_params.terrain.obs_blocks",
    "env_params.terrain.relative_stand_height",
    "env_params.naconmax",
    "env_params.njmax",
]

DESIGN_AXES = [
    "env",
    "net_params.delay_k",
    "env_params.terrain.height",
    "env_params.terrain.irregularity",
    "env_params.terrain.observed",
    "env_params.impl",
    "config.ppo.total_steps",
    "config.eval.every_steps",
    "state",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "repos.vnl_experiments.dirty",
    "gpu",
    "seed",
    "config.seed",
]


def _series(store: Store, wandb_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, HISTORY_SPEC_ID)
    if entry is None:
        raise KeyError(wandb_id)
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None
    if frame.empty or KEY not in frame.columns:
        return None
    out = frame.dropna(subset=[KEY])[["_step", KEY]]
    return out.rename(columns={"_step": "step", KEY: "value"}).sort_values("step")


def _window(s: pd.DataFrame | None, upper: int) -> tuple[float, int]:
    if s is None:
        return np.nan, 0
    sub = s[(s.step > upper - WINDOW) & (s.step <= upper)]
    return (float(sub.value.mean()), len(sub)) if len(sub) else (np.nan, 0)


def _slope(s: pd.DataFrame | None) -> tuple[float, float]:
    """OLS slope over the last SLOPE_SPAN steps, reward per 100 M steps, and its s.e."""
    if s is None:
        return np.nan, np.nan
    sub = s[(s.step > READ_AT - SLOPE_SPAN) & (s.step <= READ_AT)]
    if len(sub) < 5:
        return np.nan, np.nan
    x = sub.step.to_numpy() / 1e8
    y = sub.value.to_numpy()
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    resid = y - y.mean() - b * xc
    se = float(np.sqrt((resid ** 2).sum() / (len(x) - 2) / (xc ** 2).sum()))
    return b, se


def _fit(s: pd.DataFrame | None, upper: int) -> tuple[float, float, float, bool]:
    """``(Rinf, A, tau, at_bound)`` of ``Rinf - A exp(-t / tau)`` on ``[FIT_FROM, upper]``."""
    from scipy.optimize import curve_fit

    nan = (np.nan, np.nan, np.nan, False)
    if s is None:
        return nan
    sub = s[(s.step >= FIT_FROM) & (s.step <= upper)]
    if len(sub) < 8:
        return nan
    t = sub.step.to_numpy() / 1e8
    y = sub.value.to_numpy()

    def f(t, rinf, a, tau):
        return rinf - a * np.exp(-t / tau)

    p0 = (min(y.max() + 10, REWARD_MAX), max(y.max() - y.min(), 1.0), 1.0)
    try:
        (rinf, a, tau), _ = curve_fit(f, t, y, p0=p0, maxfev=20000,
                                      bounds=([0, 0, 0.01], [REWARD_MAX, 5000, 100]))
    except RuntimeError:
        return nan
    at_bound = bool(rinf >= REWARD_MAX - 1e-3 or tau >= 100 - 1e-3)
    return float(rinf), float(a), float(tau * 1e8), at_bound


def _first_at(s: pd.DataFrame | None, level: float) -> float:
    if s is None or not np.isfinite(level):
        return np.nan
    hit = s[(s.value >= level) & (s.step <= READ_AT)]
    return float(hit.step.iloc[0]) if len(hit) else np.nan


def build_row(run: pd.Series, s: pd.DataFrame | None) -> dict:
    cond = run["condition"]
    flat = cond == "flat_ground"
    reward, n = _window(s, READ_AT)
    early, _ = _window(s, READ_AT - 2 * WINDOW)
    slope, slope_se = _slope(s)
    row = {
        "condition": cond,
        "height": 0.0 if flat else float(run["env_params.terrain.height"]),
        "irregularity": np.nan if flat else float(run["env_params.terrain.irregularity"]),
        "observed": False if flat else bool(run["env_params.terrain.observed"]),
        "delay_k": int(run["net_params.delay_k"]),
        "delay_ms": int(run["net_params.delay_k"]) * CTRL_DT * 1000,
        "seed_pair": f"i{int(run['seed'])}/p{int(run['config.seed'])}",
        "impl": run.get("env_params.impl") if isinstance(run.get("env_params.impl"), str)
        else "jax",
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "state": run["state"],
        "created_at": run["created_at"],
        "budget": int(run["config.ppo.total_steps"]),
        "git_commit": str(run["git_commit"])[:7],
        "dirty": run.get("repos.vnl_experiments.dirty"),
        "gpu": run["gpu"],
        "reward_source": "dmc_eval",
        "reward": reward,
        "reward_n": n,
        "late_gain": reward - early if n else np.nan,
        "late_slope": slope,
        "late_slope_se": slope_se,
    }
    for at in SHAPE_AT:
        row[f"reward_at_{at // 1_000_000}M"] = _window(s, at)[0]
    for at in LONG_AT:
        row[f"reward_at_{at // 1_000_000}M"] = (
            _window(s, at)[0] if row["budget"] >= at else np.nan)
    rinf, amp, tau, at_bound = _fit(s, READ_AT)
    row.update(fit_rinf=rinf, fit_a=amp, fit_tau=tau, fit_at_bound=at_bound)
    b_rinf, b_a, b_tau, _ = _fit(s, BACKTEST_AT)
    if np.isfinite(b_rinf) and s is not None:
        win = s[(s.step > READ_AT - WINDOW) & (s.step <= READ_AT)].step.to_numpy()
        pred = float(np.mean(b_rinf - b_a * np.exp(-win / b_tau)))
        row["backtest_err"] = pred - reward
    else:
        row["backtest_err"] = np.nan
    for f in FRACTIONS:
        row[f"t{int(f * 100)}"] = _first_at(s, f * reward)
    for level in SOLVE_LEVELS:
        row[f"steps_to_{level}"] = _first_at(s, level)
    return row


def summary(df: pd.DataFrame) -> str:
    cols = ["reward", "late_gain", "late_slope", "late_slope_se", "t50", "t90",
            "fit_rinf", "fit_tau", "backtest_err"]
    lines = []
    flat = df[df.condition == "flat_ground"]
    per_seed = flat.groupby(["delay_k", "seed_pair"])[cols].mean().reset_index()
    agg = per_seed.groupby("delay_k").agg(
        reward=("reward", "mean"), lo=("reward", "min"), hi=("reward", "max"),
        n=("reward", "size"), slope=("late_slope", "mean"), t90=("t90", "mean"),
        rinf=("fit_rinf", "mean"), backtest=("backtest_err", "mean"))
    lines.append(f"\nflat_ground by delay (mean over seed pairs; range; n seeds):\n"
                 + agg.round(1).to_string())
    ter = df[df.condition != "flat_ground"].sort_values(
        ["condition", "delay_k", "height", "seed_pair"])
    lines.append("\nterrain runs:\n" + ter[
        ["condition", "delay_k", "height", "seed_pair", *cols, "fit_at_bound"]
    ].round(2).to_string(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = pipeline.parse_args(__doc__)
    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    got = sorted(runs["env_params.ctrl_dt"].dropna().unique())
    if got != [CTRL_DT]:
        raise SystemExit(f"ctrl_dt: expected {CTRL_DT}, runs carry {got}")
    producer = get_producer("history")
    if (sid := producer.spec_id(producer.spec(project=PROJECT))) != HISTORY_SPEC_ID:
        raise SystemExit(f"history spec id drifted: {sid} != {HISTORY_SPEC_ID}")

    store = Store()
    rows, curves, absent, full = [], [], [], {}
    for _, run in runs.iterrows():
        try:
            s = _series(store, run["wandb_id"])
        except KeyError:
            absent.append(run["wandb_id"])
            continue
        row = build_row(run, s)
        rows.append(row)
        if s is not None and row["budget"] > READ_AT:
            full[row["wandb_id"]] = s
        if s is not None:
            for p in s[s.step <= READ_AT].itertuples():
                curves.append({k: row[k] for k in
                               ("wandb_id", "condition", "delay_k", "height",
                                "seed_pair")} | {"step": int(p.step), "reward": p.value})
    if absent:
        raise SystemExit(
            f"{len(absent)} runs lack a history artifact; produce them with:\n"
            f"  python -m vnl_experiments.artifacts ensure --kind history --runs "
            f"analysis/dm_control_suite/{HERE.name}/runs.csv "
            f"--set project='\"{PROJECT}\"'")

    df = (pd.DataFrame(rows)
          .sort_values(["condition", "delay_k", "height", "seed_pair", "wandb_id"])
          .reset_index(drop=True))
    curves_df = (pd.DataFrame(curves)
                 .sort_values(["condition", "delay_k", "height", "seed_pair",
                               "wandb_id", "step"]).reset_index(drop=True))

    long_runs = df[df.wandb_id.isin(full)]
    long_delays = set(long_runs[long_runs.condition != "flat_ground"].delay_k)
    keep = long_runs[long_runs.delay_k.isin(long_delays)]
    long_df = pd.DataFrame(
        [{k: r[k] for k in ("wandb_id", "condition", "delay_k", "height", "seed_pair",
                            "budget")} | {"step": int(p.step), "reward": p.value}
         for r in keep.to_dict("records") for p in full[r["wandb_id"]].itertuples()],
        columns=["wandb_id", "condition", "delay_k", "height", "seed_pair", "budget",
                 "step", "reward"],
    ).sort_values(["condition", "delay_k", "height", "seed_pair", "wandb_id", "step"]
                  ).reset_index(drop=True)

    text = summary(df)
    print(text)
    report = (comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
              + "\n\n--- DESIGN AXES: expected to vary ---\n"
              + comparability_report(runs, invariant_cols=DESIGN_AXES,
                                     group_col="condition")
              + "\n" + text)
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    ok &= pipeline.write_csv(long_df, HERE / "long_curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
