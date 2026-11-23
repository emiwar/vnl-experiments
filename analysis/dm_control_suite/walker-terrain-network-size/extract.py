"""Does actor size matter on WalkerWalkTerrain (0.3 m blocks, delay 5) -- and is it width or depth?

A preliminary sweep of the **actor** MLP's hidden shape (depth x width); the critic is
held at 5 x 256 throughout. Everything else is the ``walker-terrain-factors`` grid's
setup: ``WalkerWalkTerrain``, ``terrain.height == 0.3``, ``irregularity == 1``,
``delay_k == 5`` (125 ms), ``min_std == 0.001``, no motor noise, no distillation, 480 M
steps.

Two cohorts, both read from the same index and history artifacts:

* ``sweep`` -- the main sweep, all in the ``eff+priv`` cell of walker-terrain-factors
  (``DelayedMLP``, efference 5, blind, no servo, privileged critic). 21 non-default
  shapes, mostly **one run each**, plus the factors analysis' 256x4 runs of that cell,
  which are the best-replicated shape. A 2026-11-22 batch ("smaller network longer
  train") added second seeds at 2 G budget for 4x128, 4x1024, 8x512 and the new
  8x1024; several of those crashed early and stay in with ``complete`` False.
* ``factors`` -- depth-4 widths 64-1024 in the other 15 privileged-critic cells of the
  factor grid (DelayedMLP and FlatForwardModel, efference 0/5, sensor, servo), set
  beside those cells' 256x4 runs from walker-terrain-factors. Batches: "sweeping
  factors for smaller network" (128), "... larger network" (512, one FM 1024),
  "... even smaller network" (FM 64). Unprivileged-critic cells got no size runs and
  are not selected.

A condition is ``<cell>|<depth>x<width>``. Every hidden layer in this sweep has the
same width, so ``depth`` and ``width`` are the whole shape; extract.py stops if that
ever stops being true.

``actor_params`` is computed, not logged (current runs do not log it): an MLP
``[obs + efference*act] + hidden + [2*act]`` with biases, obs 24 blind / 28 observed,
act 6. The formula reproduces the legacy WalkerWalk runs' logged ``n_actor_params``
exactly (eff 5, 256x4 -> 214 540). For ``FlatForwardModel`` it counts the decoder only
(input obs, no efference) and ``predictor_params`` counts the predictor (input
obs + efference*act, output obs); the predictor is 256x4 in every FM run here.

Readouts, the same definitions as ``walker-terrain-factors/data.csv``:

* ``reward_480M``: mean eval reward over (430 M, 480 M].
* ``reward_{50,100,200}M``: the same 50 M window ending there -- early-learning
  readouts. ``reward_50M`` is (0, 50 M]. Every window readout is NaN unless the run's
  history reaches the window's end (less 5 M, one eval interval).
* ``steps_to_{900,800,700,600}``: first eval step at or above that level, up to 480 M;
  NaN if never reached by 480 M.
* ``reward_{1000,2000}M``: the same window at 1 G / 2 G, only for runs whose history
  reaches it (the three 2 G baselines); NaN otherwise.
* ``late_gain``: ``reward_480M`` minus the (330 M, 380 M] window: is the run still
  rising at the end of the budget? Descriptive only.

``curves.csv`` holds every eval point of every run, including the 2 G runs' tails.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-network-size/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-network-size/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-network-size/extract.py --check

Writes ``runs.csv``, ``data.csv``, ``curves.csv``, ``comparability.txt``, ``coverage.txt``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-delays"
#: Eval curves: the id walker-terrain-first-look / -factors read.
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]
KEY = "eval/episode_reward/mean"

HEIGHT = 0.3
DELAY = 5
NETWORKS = ("DelayedMLP", "FlatForwardModel")
MIN_STD = 0.001
PRE_EVAL_FIX_COMMIT = "d168093"
CTRL_DT = 0.025
READ_AT = 480_000_000
WINDOW = 50_000_000
EARLY_AT = (50_000_000, 100_000_000, 200_000_000)
LATE_REF_AT = 380_000_000
#: Long-budget windows, for runs whose history gets there (2 G runs log their last
#: eval at 1.9968 G; evals are 4.8 M apart, so "gets there" allows 5 M).
LONG_AT = (1_000_000_000, 2_000_000_000)
LONG_SLACK = 5_000_000
LEVELS = (900, 800, 700, 600)

OBS_BLIND, OBS_TERRAIN_EXTRA, ACT = 24, 4, 6
CRITIC = "[256, 256, 256, 256, 256]"
DEFAULT_ACTOR = "[256, 256, 256, 256]"

#: The cell the main sweep lives in.
SWEEP_CELL = "eff+priv"
#: Sweep shapes in the SWEEP_CELL, (depth, width). 4x256 is the factors' default.
SWEEP_SHAPES = [
    (1, 8), (1, 256), (1, 512), (1, 1024), (1, 2048), (1, 4096),
    (2, 8), (2, 64), (2, 128), (2, 256), (2, 1024),
    (4, 64), (4, 128), (4, 256), (4, 512), (4, 1024), (4, 2048),
    (8, 8), (8, 16), (8, 64), (8, 128), (8, 512), (8, 1024),
]
#: Every other privileged-critic cell of the factor grid (names as walker-terrain-
#: factors' cell_name: eff, fm, servo, sensor, priv in that order).
FACTOR_CELLS = ["+".join(x for x in (e, f, sv, sn, "priv") if x)
                for e in ("eff", "") for f in ("fm", "") for sv in ("servo", "")
                for sn in ("sensor", "")]
FACTOR_CELLS.remove(SWEEP_CELL)
FACTOR_SHAPES = [(4, 64), (4, 128), (4, 256), (4, 512), (4, 1024)]

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
    "env_params.terrain.irregularity",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.critic_hidden_sizes",
    "net_params.normalize_obs",
    "net_params.std_scale",
    "net_params.delay_k",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "config.eval.every_steps",
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
    "env_params.impl",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
]
DESIGN_AXES = [
    "net_params.actor_hidden_sizes",
    "net_params.predictor_hidden_sizes",
    "net_params.network_class",
    "net_params.efference_length",
    "net_params.privileged_critic",
    "env_params.servo_kp",
    "env_params.terrain.observed",
    "config.ppo.total_steps",
    "summary._step",
    "state",
    "git_commit",
    "repos.vnl_experiments.dirty",
    "gpu",
    "seed",
    "config.seed",
]


# --------------------------------------------------------------------------------------
# selection -- the factor definitions are walker-terrain-factors' (copied, not imported:
# that folder is not a package)
# --------------------------------------------------------------------------------------

def _noise(df: pd.DataFrame) -> pd.Series:
    if "net_params.motor_noise_std" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df["net_params.motor_noise_std"], errors="coerce").fillna(0.0)


def cell_of(df: pd.DataFrame) -> pd.Series:
    """walker-terrain-factors' cell name for every row of an index frame."""
    fm = df["net_params.network_class"].eq("FlatForwardModel")
    priv = fm | df["net_params.privileged_critic"].map(lambda v: v is not False)
    eff = pd.to_numeric(df["net_params.efference_length"]).fillna(0).gt(0)
    servo = pd.to_numeric(df["env_params.servo_kp"]).fillna(0.0).gt(0)
    sensor = df["env_params.terrain.observed"].fillna(False).astype(bool)
    parts = [eff.map({True: "eff", False: ""}), fm.map({True: "fm", False: ""}),
             servo.map({True: "servo", False: ""}), sensor.map({True: "sensor", False: ""}),
             priv.map({True: "priv", False: ""})]
    joined = pd.concat(parts, axis=1).apply(lambda r: "+".join(x for x in r if x), axis=1)
    return joined.replace("", "none")


def shape_of(sizes: str) -> tuple[int, int] | None:
    widths = json.loads(sizes) if isinstance(sizes, str) else None
    if not widths or len(set(widths)) != 1:
        return None
    return len(widths), widths[0]


def _grid(df: pd.DataFrame) -> pd.Series:
    return (df["env"].eq("WalkerWalkTerrain")
            & df["env_params.terrain.height"].eq(HEIGHT)
            & df["net_params.delay_k"].eq(DELAY)
            & df["env_params.terrain.irregularity"].eq(1.0)
            & df["net_params.network_class"].isin(NETWORKS)
            & df["net_params.min_std"].eq(MIN_STD)
            & df["net_params.critic_hidden_sizes"].eq(CRITIC)
            & _noise(df).eq(0.0)
            & df["distill.teacher"].isna()
            & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
            & df["summary._step"].notna())


def shape_name(depth: int, width: int) -> str:
    return f"{depth}x{width}"


def _cond(cell: str, depth: int, width: int):
    sizes = json.dumps([width] * depth)

    def selector(df: pd.DataFrame) -> pd.Series:
        m = _grid(df)
        return m & df["net_params.actor_hidden_sizes"].eq(sizes) & cell_of(df).eq(cell)
    return selector


CONDITIONS = {
    **{f"{SWEEP_CELL}|{shape_name(d, w)}": _cond(SWEEP_CELL, d, w)
       for d, w in SWEEP_SHAPES},
    **{f"{c}|{shape_name(d, w)}": _cond(c, d, w)
       for c in FACTOR_CELLS for d, w in FACTOR_SHAPES},
}


# --------------------------------------------------------------------------------------
# parameter counts
# --------------------------------------------------------------------------------------

def mlp_params(sizes: list[int]) -> int:
    return int(sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:])))


def param_counts(run: pd.Series) -> tuple[int, int]:
    """(actor params, predictor params) -- see the module docstring."""
    hidden = json.loads(run["net_params.actor_hidden_sizes"])
    eff = int(run["net_params.efference_length"])
    obs = OBS_BLIND + (OBS_TERRAIN_EXTRA if run["env_params.terrain.observed"] else 0)
    if run["net_params.network_class"] == "FlatForwardModel":
        pred = json.loads(run["net_params.predictor_hidden_sizes"])
        if pred != [256] * 4:
            raise SystemExit(f"{run['wandb_id']}: predictor {pred}, report assumes 4x256")
        return (mlp_params([obs] + hidden + [2 * ACT]),
                mlp_params([obs + eff * ACT] + pred + [obs]))
    return mlp_params([obs + eff * ACT] + hidden + [2 * ACT]), 0


# --------------------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------------------

def _series(store: Store, wandb_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, HISTORY_SPEC_ID)
    if entry is None:
        return None
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None
    if frame.empty or KEY not in frame.columns:
        return None
    out = frame.dropna(subset=[KEY])[["_step", KEY]]
    out = out.rename(columns={"_step": "step", KEY: "value"}).sort_values("step")
    # A requeued attempt can leave a killed attempt's points in the history
    # (requeue.init_wandb_resumable); keep the last one logged at each step.
    return out.drop_duplicates("step", keep="last")


def _window(s: pd.DataFrame | None, upper: int) -> float:
    # A window counts only if the run got to its end: a run that crashed at 72 M
    # must not report a "100 M" reward from (50 M, 72 M].
    if s is None or s.step.max() < upper - LONG_SLACK:
        return np.nan
    sub = s[(s.step > upper - WINDOW) & (s.step <= upper)]
    return float(sub.value.mean()) if len(sub) else np.nan


def _first_at(s: pd.DataFrame | None, level: float) -> float:
    if s is None:
        return np.nan
    hit = s[(s.value >= level) & (s.step <= READ_AT)]
    return float(hit.step.iloc[0]) if len(hit) else np.nan


def check_invariants(runs: pd.DataFrame) -> None:
    varying = {c: sorted(map(str, runs[c].dropna().unique())) for c in INVARIANTS
               if c in runs.columns and runs[c].dropna().astype(str).nunique() > 1}
    if varying:
        raise SystemExit(f"non-design settings vary across the cohort: {varying}")


def build(runs: pd.DataFrame, store: Store) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for _, r in runs.iterrows():
        shape = shape_of(r["net_params.actor_hidden_sizes"])
        if shape is None:
            raise SystemExit(f"{r['wandb_id']}: non-uniform actor widths")
        cell, _ = r["condition"].split("|")
        s = _series(store, r["wandb_id"])
        actor, predictor = param_counts(r)
        rows.append({
            "wandb_id": r["wandb_id"],
            "condition": r["condition"],
            "cohort": "sweep" if cell == SWEEP_CELL else "factors",
            "cell": cell,
            "depth": shape[0],
            "width": shape[1],
            "actor_params": actor,
            "predictor_params": predictor,
            "seed_pair": f"i{int(r['seed'])}/p{int(r['config.seed'])}",
            "reward_480M": _window(s, READ_AT),
            **{f"reward_{u // 1_000_000}M": _window(s, u) for u in EARLY_AT},
            "late_gain": _window(s, READ_AT) - _window(s, LATE_REF_AT),
            **{f"reward_{u // 1_000_000}M": (
                _window(s, u) if s is not None and s.step.max() >= u - LONG_SLACK
                else np.nan) for u in LONG_AT},
            **{f"steps_to_{lv}": _first_at(s, lv) for lv in LEVELS},
            "network_class": r["net_params.network_class"],
            "state": r["state"],
            "complete": s is not None and s.step.max() >= READ_AT,
            "steps_trained": float(s.step.max()) if s is not None else np.nan,
            "budget": int(r["config.ppo.total_steps"]),
            "requeued": "Requeue-safe run: attempt" in str(r.get("notes", "")),
            "git_commit": r["git_commit"],
            "dirty": r.get("repos.vnl_experiments.dirty"),
            "gpu": r["gpu"],
            "created_at": r["created_at"],
        })
        if s is not None:
            c = s.copy()
            c.insert(0, "wandb_id", r["wandb_id"])
            curves.append(c)
    df = (pd.DataFrame(rows)
          .sort_values(["cohort", "cell", "depth", "width", "seed_pair", "wandb_id"])
          .reset_index(drop=True))
    cv = pd.concat(curves).sort_values(["wandb_id", "step"]).reset_index(drop=True)
    return df, cv


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
    check_invariants(runs)

    df, curves = build(runs, Store())
    text = (df[["condition", "seed_pair", "actor_params", "reward_50M", "reward_100M",
                "reward_200M", "reward_480M", "steps_to_800", "steps_to_900",
                "complete", "wandb_id"]].to_string()
            + f"\n\n{len(df)} runs, {int(df.complete.sum())} complete")
    print(text)
    report = (comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
              + "\n\n--- DESIGN AXES: expected to vary ---\n"
              + comparability_report(runs, invariant_cols=DESIGN_AXES,
                                     group_col="condition")
              + "\n" + text + "\n")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
