"""Which additions help WalkerWalkTerrain at 0.3 m blocks and delay 5? A five-factor grid.

Split out of ``walker-terrain-first-look/``, whose delay/height sweeps hold every factor
here at its baseline. Locked: ``WalkerWalkTerrain``, ``terrain.height == 0.3``,
``delay_k == 5`` (125 ms), ``terrain.irregularity == 1``, no motor noise, no
distillation, ``min_std == 0.001``.

Factors (parametric settings collapsed to true/false; the raw value is kept beside each):

* ``efference``: ``efference_length > 0`` (raw: ``efference_length``, 0 or 5). In
  ``DelayedMLP`` the action queue goes to the actor; in ``FlatForwardModel`` it goes to
  the predictor.
* ``forward_model``: ``network_class == "FlatForwardModel"`` (else ``DelayedMLP``).
* ``blind``: ``not terrain.observed`` (``False`` = the block sensor is on).
* ``privileged_critic``: ``FlatForwardModel`` has no such key; its critic always gets
  the undelayed observation (``make_forward_model_actor_critic``), so it is ``True``.
  Forward model x unprivileged critic is therefore not a cell that can exist: 24 of the
  32 combinations are possible.
* ``servo``: ``servo_kp > 0`` (raw: ``servo_kp``).

Each combination is its own condition, named by the factors that are on
(``eff+fm+servo+sensor+priv``; ``none`` if all are off), so ``coverage.txt`` and
``comparability.txt`` are per cell.

Readouts, the same definitions as ``walker-terrain-first-look/data.csv``:

* ``reward_480M``: mean eval reward over (430 M, 480 M]; NaN if the run has not got there.
* ``reward_2B``: mean eval reward over (1.95 G, 2 G], only for runs whose history reaches
  1.99 G (2 G runs log their last eval at 1.9968 G); NaN otherwise, including for every
  480 M run and for 2 G runs that stopped or are still running. Never compare it with
  ``reward_480M`` across cells without checking which runs have it.
* ``steps_to_{900,800,700,600}``: first eval step at or above that level, up to 480 M;
  NaN if not reached within ``min(steps_trained, 480 M)``. Read ``steps_trained`` before
  treating NaN as "never".

``complete`` means the run's history reaches 480 M, i.e. every readout is final. That
includes still-running and crashed runs that got past 480 M (a 2 G run is a valid 480 M
run); runs that stopped short stay in the selection with ``complete`` False rather than
being dropped. Runs trained past 480 M are read at 480 M (``budget`` says which).

The history artifact of a ``running`` run is a snapshot taken when it was pulled. Past
480 M that does not matter; below it, re-pull that run with ``artifacts ensure ...
--override`` once it has got there, or its readouts stay NaN. Runs that never
logged a step carry no data and are excluded by the selector (``kwwavcon``: failed at
launch, with default PPO settings). Every non-factor setting in :data:`INVARIANTS` must
be single-valued across the grid; ``extract.py`` stops otherwise.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-factors/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-factors/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-factors/extract.py --check

Writes ``runs.csv``, ``data.csv``, ``comparability.txt`` and ``coverage.txt``.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-delays"
#: Eval curves: the same id walker-terrain-first-look reads.
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
#: reward_2B's window end, and how far a run must get for it to count.
LONG_READ_AT = 2_000_000_000
LONG_REACHED = 1_990_000_000
LEVELS = (900, 800, 700, 600)

FACTORS = ["efference", "forward_model", "blind", "privileged_critic", "servo"]
#: Cell-name slug of each factor when it is *on* (blind is on when it is False).
SLUGS = {"efference": "eff", "forward_model": "fm", "servo": "servo",
         "blind": "sensor", "privileged_critic": "priv"}

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
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
]
#: Expected to vary: the factors' raw settings, budget, seeds, state.
DESIGN_AXES = [
    "net_params.network_class",
    "net_params.efference_length",
    "net_params.privileged_critic",
    "env_params.servo_kp",
    "env_params.terrain.observed",
    "net_params.predictor_hidden_sizes",
    "config.ppo.total_steps",
    "summary._step",
    "state",
    "repos.vnl_experiments.dirty",
    "gpu",
    "seed",
    "config.seed",
]


# --------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------

def _noise(df: pd.DataFrame) -> pd.Series:
    if "net_params.motor_noise_std" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df["net_params.motor_noise_std"], errors="coerce").fillna(0.0)


def factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The five true/false factors for every row of an index frame."""
    fm = df["net_params.network_class"].eq("FlatForwardModel")
    priv = df["net_params.privileged_critic"].map(lambda v: v is not False)
    return pd.DataFrame({
        "efference": pd.to_numeric(df["net_params.efference_length"]).fillna(0).gt(0),
        "forward_model": fm,
        "blind": ~df["env_params.terrain.observed"].fillna(False).astype(bool),
        "privileged_critic": fm | priv,
        "servo": pd.to_numeric(df["env_params.servo_kp"]).fillna(0.0).gt(0),
    }, index=df.index)


def _grid(df: pd.DataFrame) -> pd.Series:
    return (df["env"].eq("WalkerWalkTerrain")
            & df["env_params.terrain.height"].eq(HEIGHT)
            & df["net_params.delay_k"].eq(DELAY)
            & df["env_params.terrain.irregularity"].eq(1.0)
            & df["net_params.network_class"].isin(NETWORKS)
            & df["net_params.min_std"].eq(MIN_STD)
            & _noise(df).eq(0.0)
            & df["distill.teacher"].isna()
            & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
            & df["summary._step"].notna())


def cell_name(flags: dict) -> str:
    on = [SLUGS[f] for f in ("efference", "forward_model", "servo", "blind",
                             "privileged_critic")
          if (not flags[f] if f == "blind" else flags[f])]
    return "+".join(on) or "none"


def _cell(flags: dict):
    def selector(df: pd.DataFrame) -> pd.Series:
        f = factor_frame(df)
        m = _grid(df)
        for k, v in flags.items():
            m &= f[k].eq(v)
        return m
    return selector


CONDITIONS = {cell_name(flags): _cell(flags)
              for values in itertools.product((True, False), repeat=len(FACTORS))
              for flags in [dict(zip(FACTORS, values))]
              if not (flags["forward_model"] and not flags["privileged_critic"])}


# --------------------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------------------

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


def _first_at(s: pd.DataFrame | None, level: float) -> float:
    if s is None:
        return np.nan
    hit = s[(s.value >= level) & (s.step <= READ_AT)]
    return float(hit.step.iloc[0]) if len(hit) else np.nan


def check_invariants(runs: pd.DataFrame) -> None:
    """Every non-factor setting must be single-valued across the grid."""
    varying = {c: sorted(map(str, runs[c].dropna().unique())) for c in INVARIANTS
               if c in runs.columns and runs[c].dropna().astype(str).nunique() > 1}
    if varying:
        raise SystemExit(f"non-factor settings vary across the grid: {varying}")


def build(runs: pd.DataFrame, store: Store) -> pd.DataFrame:
    factors = factor_frame(runs)
    rows = []
    for i, r in runs.iterrows():
        try:
            s = _series(store, r["wandb_id"])
        except KeyError:
            # coverage.txt names the gap and the command that closes it.
            s = None
        reward, n = _window(s, READ_AT)
        long_done = s is not None and s.step.max() >= LONG_REACHED
        reward_2b = _window(s, LONG_READ_AT)[0] if long_done else np.nan
        rows.append({
            "wandb_id": r["wandb_id"],
            "condition": r["condition"],
            **{f: bool(factors.at[i, f]) for f in FACTORS},
            "seed_pair": f"i{int(r['seed'])}/p{int(r['config.seed'])}",
            "reward_480M": reward,
            "reward_2B": reward_2b,
            **{f"steps_to_{lv}": _first_at(s, lv) for lv in LEVELS},
            "efference_length": int(r["net_params.efference_length"]),
            "servo_kp": float(pd.to_numeric(r["env_params.servo_kp"]) or 0.0),
            "network_class": r["net_params.network_class"],
            "state": r["state"],
            "complete": s is not None and s.step.max() >= READ_AT,
            "steps_trained": float(s.step.max()) if s is not None else np.nan,
            "budget": int(r["config.ppo.total_steps"]),
            "reward_n": n,
            "git_commit": r["git_commit"],
            "created_at": r["created_at"],
        })
    return (pd.DataFrame(rows)
            .sort_values([*FACTORS, "seed_pair", "wandb_id"],
                         ascending=[False, True, False, False, True, True, True])
            .reset_index(drop=True))


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

    df = build(runs, Store())
    n_cells = df[df.complete].condition.nunique()
    text = (df[["condition", "seed_pair", "reward_480M",
                *[f"steps_to_{lv}" for lv in LEVELS], "state", "steps_trained",
                "wandb_id"]].to_string()
            + f"\n\n{len(df)} runs, {int(df.complete.sum())} complete, {n_cells} of "
              f"{len(CONDITIONS)} possible cells with a complete run")
    print(text)
    report = (comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
              + "\n\n--- DESIGN AXES: expected to vary ---\n"
              + comparability_report(runs, invariant_cols=DESIGN_AXES,
                                     group_col="condition")
              + "\n" + text + "\n")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
