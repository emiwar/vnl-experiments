"""Does PPO under delay need a privileged critic? WalkerWalk, torque control.

Every delay run in this track until 2026-10 used an **asymmetric** actor-critic: the actor
sees the observation ``delay_k`` steps late (plus, optionally, the queue of actions taken
since), while the critic sees the *current* observation. That is the standard "privileged
critic" trick, and it was a design default, not a measurement. ``a37ce64`` added
``net.privileged_critic=false`` to ``DelayedMLP``: the Delay and the efference queue then
sit in front of the whole ``PPOAdapter``, so the critic's input is exactly the actor MLP's
input -- delayed observation, plus the efference queue when ``efference_length > 0``
(``vnl_experiments/delays/make_delayed_networks.py``).

The question is how much the privilege buys, and whether the answer depends on the
efference copy -- the one input that makes a delayed observation (close to) a sufficient
statistic for the present state.

Arms
----
``{critic}_{arm}``, with critic in ``privileged`` / ``delayed`` and arm in

* ``efference``     -- ``efference_length == delay_k > 0``
* ``no_efference``  -- ``efference_length == 0``, ``delay_k > 0``
* ``undelayed``     -- ``delay_k == 0``. Not a point on either sweep: here the two critics
  receive *identical information* (the fresh observation, no queue), so
  ``delayed_undelayed`` vs ``privileged_undelayed`` is a pure control for the change of
  module layout. It should be indistinguishable.

The budget, and why longer runs are allowed in
----------------------------------------------
The delayed-critic runs are 480 M steps at two seeds: i42/p1234 (2026-10-09) and i43/p43
(added 2026-10-12). The privileged arm at i42/p1234 and 480 M has holes -- no-copy runs at delays 2, 3 and 5, and
efference runs at 2, 3, 12 and 25 were never launched -- but the same configuration was run
for 1 G steps at seeds 45 / 46 / 47 / 51 / 52, 2 G and 4 G at a few more. nnx-ppo has no
learning-rate or entropy schedule, so a 1 G run at its 480 M-th step *is* a 480 M run up to
that point (same optimiser state trajectory, same everything); the only difference is the
eval cadence, handled by regridding below. So every privileged run with
``total_steps >= 480 M`` is read **at 480 M**, and ``budget_native`` records what it was
launched for.

That makes the privileged arm a multi-seed reference cloud and the delayed arm two seeds.
``seed_pair == "i42/p1234"`` marks the subset that is *matched* (no privileged run uses PPO
seed 43, so i43/p43 has no matched partner) -- per
``project-seed-shared-init-across-delays``, runs sharing a seed share 84-100 % of their
initial weights and the whole PPO random stream, so a matched pair differs almost only in
the manipulation. That is the cleanest contrast available and also n = 1 per delay; the
report reads the matched pairs first and the cloud second, never one alone.

Readouts
--------
From the eval series (``eval/episode_reward/mean``, unscaled, 1000-step episodes):

* ``reward_480`` -- mean of the eval points in (430 M, 480 M], on a common 4.8 M grid
  (README §6: never the last point).
* ``steps_to_900`` -- first step whose 24 M trailing mean on the grid reaches 900,
  censored at 480 M. Same definition as ``../efference-copy-across-tasks/``.

From the training-side series (every PPO iteration), for the mechanism question:

* ``critic_r2_*`` -- ``losses/critic_R^2``: 1 - MSE / Var(GAE target). The targets are in
  the same (x10 scaled) units for both critics, but they are *not* the same quantity: each
  critic's GAE target is built from its own value estimates. Read it as "how much of the
  return variance this critic accounts for", not as a like-for-like accuracy score.
* ``critic_loss_*`` -- 0.5 * MSE in scaled-reward units.
* ``clip_frac_*`` -- PPO clipping fraction: how far each update moved the policy.
* ``train_reward_*`` -- ``rollout_batch/reward/mean`` (x10 scaled, per step): the
  stochastic policy's reward on the training env.

each as a 50 M window at two places: ``*_early`` (0, 50 M] and ``*_480`` (430, 480 M].
``losses/advantages`` is **not** used: it is logged after per-batch normalisation, so its
mean is 0 and its std 1 by construction.

Selection gates no run name reveals
-----------------------------------
* ``DelayedMLP`` with default widths (actor [256]x4, critic [256]x5), ``min_std`` 0.001,
  torque control (``servo_kp`` absent or 0).
* Post-``971ab99`` only: excludes the eval-wrapper era (``d168093``) and, implicitly, the
  pre-Hydra runs, which have no ``net_params.*`` (track README).
* **Not a distillation run.** Distilled students are ``DelayedMLP`` runs too, with
  ``privileged_critic = True`` recorded because the default was never overridden; they are
  identified by ``distill.teacher``.
* ``privileged_critic`` is read with *absent* == True: every run before ``a37ce64``.
* De-duplicated on (params, both seeds), earliest launched -- as the sibling folders do.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-delayed-critic/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-delayed-critic/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-delayed-critic/extract.py --check

Artifacts (two ``history`` specs -- eval and training series are logged at different
cadences, and one multi-key fetch would keep only their intersection, README §2)::

    python -m vnl_experiments.artifacts ensure --kind history \\
        --runs analysis/dm_control_suite/walker-delayed-critic/runs.csv \\
        --set project='"emiwar-team/nnx-ppo-delays"'
    python -m vnl_experiments.artifacts ensure --kind history \\
        --runs analysis/dm_control_suite/walker-delayed-critic/runs.csv \\
        --set project='"emiwar-team/nnx-ppo-delays"' --set keys='<TRAIN_KEYS as JSON>'   # -> hist2000-a9ab7503

Writes ``data.csv`` (one row per run), ``curves.csv`` (eval series on the common grid, to
480 M) and ``train_curves.csv`` (training-side series, to 480 M). ``plot.py`` reads only
those.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: Eval series -- the same spec id as every sibling WalkerWalk folder.
HISTORY_SPEC_ID = "hist2000-8b97281e"
#: Training-side series. Keys logged every PPO iteration, so one aligned fetch is safe.
TRAIN_KEYS = [
    "losses/critic/mean",
    "losses/critic_R^2/mean",
    "losses/clipping_fraction/mean",
    "losses/actor/mean",
    "rollout_batch/reward/mean",
    "loglikelihood/mean",
]
TRAIN_SPEC_ID = "hist2000-a9ab7503"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}", f"history:{TRAIN_SPEC_ID}"]

TASK = "WalkerWalk"
NETWORK = "DelayedMLP"
MIN_STD = 0.001
ACTOR_SIZES = "[256, 256, 256, 256]"
CRITIC_SIZES = "[256, 256, 256, 256, 256]"
PRE_EVAL_FIX_COMMIT = "d168093"

BUDGET = 480_000_000
CTRL_DT_MS = 25
MATCHED_SEED = "i42/p1234"

GRID_STEPS = 4_800_000
WINDOW_STEPS = 50_000_000
SMOOTH_WINDOW_STEPS = 24_000_000
THRESHOLD = 900

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")

CRITICS = ("privileged", "delayed")
ARMS = ("efference", "no_efference", "undelayed")

#: One run per (params, both seeds). The budget is deliberately *not* in the key: read at
#: 480 M, a 480 M run and a 1 G run of the same configuration and seeds are the same draw
#: (no schedules), repeated. Two such pairs exist and agree to within ~1 reward
#: (957.3 / 958.4 and 976.5 / 977.7 -- the GPU-nondeterminism floor, quoted in report.md).
DEDUPE_KEY = [
    "net_params.delay_k",
    "net_params.efference_length",
    "privileged",
    "seed",
    "config.seed",
]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _privileged(df: pd.DataFrame) -> pd.Series:
    """``privileged_critic`` with *absent* read as True (every run before ``a37ce64``)."""
    return df["net_params.privileged_critic"].map(
        lambda v: True if pd.isna(v) else bool(v)).astype(bool)


def _base(df: pd.DataFrame) -> pd.Series:
    distill = (df["distill.teacher"].notna() if "distill.teacher" in df.columns
               else pd.Series(False, index=df.index))
    return (
        df["env"].eq(TASK)
        & df["net_params.network_class"].eq(NETWORK)
        & df["net_params.min_std"].eq(MIN_STD)
        & df["net_params.actor_hidden_sizes"].eq(ACTOR_SIZES)
        & df["net_params.critic_hidden_sizes"].eq(CRITIC_SIZES)
        & df["env_params.servo_kp"].fillna(0.0).eq(0.0)
        & df["net_params.delay_k"].notna()
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
        & df["config.ppo.total_steps"].ge(BUDGET)
        & ~distill
    )


def _cohort(df: pd.DataFrame) -> pd.Series:
    """Base gates, then one run per (params, both seeds): the earliest launched."""
    base = _base(df)
    pool = df[base].assign(privileged=_privileged(df[base]))
    keep = (pool.sort_values(["created_at", "wandb_id"])
                .groupby(DEDUPE_KEY, dropna=False)["wandb_id"].first())
    return base & df["wandb_id"].isin(set(keep))


def _cell(critic: str, arm: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        mask = _cohort(df) & _privileged(df).eq(critic == "privileged")
        delay = df["net_params.delay_k"]
        eff = df["net_params.efference_length"]
        if arm == "undelayed":
            return mask & delay.eq(0)
        mask &= delay.gt(0)
        return mask & (eff.eq(delay) if arm == "efference" else eff.eq(0))
    return selector


CONDITIONS = {f"{c}_{a}": _cell(c, a) for c in CRITICS for a in ARMS}

INVARIANTS = [
    "env",
    "net_params.network_class",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.normalize_obs",
    "net_params.activation",
    "net_params.initializer_scale",
    "net_params.std_scale",
    "config.ppo.n_envs",
    "config.ppo.rollout_length",
    "config.ppo.learning_rate",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "config.ppo.clip_range",
    "config.ppo.discounting_factor",
    "config.ppo.gae_lambda",
    "config.ppo.gradient_clipping",
    "config.ppo.normalize_advantages",
    "config.ppo.critic_loss_weight",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "env_params.sim_dt",
    "env_params.impl",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "repos.vnl_playground.commit",
]

DESIGN_AXES = [
    "net_params.privileged_critic",
    "net_params.delay_k",
    "net_params.efference_length",
    "config.ppo.total_steps",
    "summary._step",
    "config.eval.every_steps",
    "seed",
    "config.seed",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_experiments.dirty",
    "repos.nnx_ppo.dirty",
    "gpu",
    "state",
]


# ---------------------------------------------------------------------------
# Curve readouts
# ---------------------------------------------------------------------------

def load_history(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    if entry is None:
        raise SystemExit(f"missing history:{spec_id} for {wandb_id} -- run the `ensure` "
                         f"commands in this file's docstring (README §3: runs are never "
                         f"silently dropped).")
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None
    return None if frame.empty else frame


def series_of(hist: pd.DataFrame | None, keys) -> pd.DataFrame | None:
    if hist is None:
        return None
    key = next((k for k in keys if k in hist.columns), None)
    if key is None:
        return None
    out = hist.dropna(subset=[key])[["_step", key]]
    return out.rename(columns={"_step": "step", key: "value"}).sort_values("step")


def on_common_grid(series: pd.DataFrame, upper: int = BUDGET) -> pd.DataFrame:
    """Nearest eval sample to each multiple of GRID_STEPS, up to ``upper``.

    Nearest single sample, not a bin mean, so every grid point is one eval of the same
    number of episodes whichever cadence (6e5 or 4.8e6) the run used.
    """
    steps = series["step"].to_numpy()
    grid = np.arange(GRID_STEPS, min(steps.max(), upper + GRID_STEPS // 2) + 1, GRID_STEPS)
    if len(grid) == 0:
        return series.iloc[:0]
    idx = np.abs(steps[None, :] - grid[:, None]).argmin(axis=1)
    keep = series.iloc[np.unique(idx)].copy()
    keep = keep[keep["step"] <= upper + GRID_STEPS // 2]
    return keep.sort_values("step").reset_index(drop=True)


def window_mean(series: pd.DataFrame | None, lo: int, hi: int) -> tuple[float | None, int]:
    if series is None:
        return None, 0
    sub = series[(series["step"] > lo) & (series["step"] <= hi)]
    return (float(sub["value"].mean()), len(sub)) if len(sub) else (None, 0)


def trailing_mean(steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    lo = np.searchsorted(steps, steps - SMOOTH_WINDOW_STEPS, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.arange(1, len(steps) + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)
    return np.where(steps >= SMOOTH_WINDOW_STEPS, out, np.nan)


def steps_to(grid: pd.DataFrame, level: float) -> int | None:
    steps = grid["step"].to_numpy()
    smooth = trailing_mean(steps, grid["value"].to_numpy())
    hit = np.where((smooth >= level) & (steps <= BUDGET))[0]
    return int(steps[hit[0]]) if len(hit) else None


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _split(condition: str) -> tuple[str, str]:
    critic, arm = condition.split("_", 1)
    assert critic in CRITICS and arm in ARMS, condition
    return critic, arm


TRAIN_SHORT = {
    "losses/critic/mean": "critic_loss",
    "losses/critic_R^2/mean": "critic_r2",
    "losses/clipping_fraction/mean": "clip_frac",
    "losses/actor/mean": "actor_loss",
    "rollout_batch/reward/mean": "train_reward",
    "loglikelihood/mean": "loglik",
}


def build_row(run: pd.Series, eval_grid: pd.DataFrame | None,
              train: pd.DataFrame | None) -> dict:
    critic, arm = _split(run["condition"])
    seed_pair = f"i{int(run['seed'])}/p{int(run['config.seed'])}"
    actual = run.get("summary._step")
    row = {
        "condition": run["condition"],
        "critic": critic,
        "arm": arm,
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "state": run["state"],
        "git_commit": run["git_commit"],
        "nnx_ppo_commit": run.get("repos.nnx_ppo.commit"),
        "gpu": run["gpu"],
        "delay": int(run["net_params.delay_k"]),
        "delay_ms": int(run["net_params.delay_k"]) * CTRL_DT_MS,
        "efference_length": int(run["net_params.efference_length"]),
        "seed": int(run["seed"]),
        "ppo_seed": int(run["config.seed"]),
        "seed_pair": seed_pair,
        "matched_seed": seed_pair == MATCHED_SEED,
        "budget_native": int(run["config.ppo.total_steps"]),
        "actual_step": actual,
        "eval_every_steps": run.get("config.eval.every_steps"),
        "reward_source": "dmc_eval",
    }
    if eval_grid is None or eval_grid.empty:
        row.update(reward_480=None, reward_480_n=0, reward_early=None,
                   steps_to_900=None, curve_max_step=None)
    else:
        r480, n480 = window_mean(eval_grid, BUDGET - WINDOW_STEPS, BUDGET)
        early, _ = window_mean(eval_grid, 0, WINDOW_STEPS)
        row.update(reward_480=r480, reward_480_n=n480, reward_early=early,
                   steps_to_900=steps_to(eval_grid, THRESHOLD),
                   curve_max_step=int(eval_grid["step"].max()))
    for key, short in TRAIN_SHORT.items():
        s = series_of(train, [key])
        row[f"{short}_early"] = window_mean(s, 0, WINDOW_STEPS)[0]
        row[f"{short}_480"] = window_mean(s, BUDGET - WINDOW_STEPS, BUDGET)[0]
    row["usable"] = bool(pd.notna(actual) and actual >= BUDGET
                         and row["reward_480"] is not None
                         and row["reward_480_n"] >= 5)
    return row


def main() -> None:
    args = pipeline.parse_args(__doc__)
    project = args.project or PROJECT

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    rows, curves, tcurves = [], [], []
    for _, run in runs.iterrows():
        series = series_of(load_history(store, run["wandb_id"], HISTORY_SPEC_ID),
                           REWARD_KEYS)
        grid = None if series is None else on_common_grid(series)
        train = load_history(store, run["wandb_id"], TRAIN_SPEC_ID)
        rows.append(build_row(run, grid, train))
        if grid is not None:
            g = grid[grid["step"] <= BUDGET + GRID_STEPS // 2]
            curves.append(pd.DataFrame({"wandb_id": run["wandb_id"],
                                        "condition": run["condition"],
                                        "step": g["step"].astype(int),
                                        "reward": g["value"]}))
        if train is not None:
            t = train[train["_step"] <= BUDGET]
            keep = {"_step": "step", **{k: v for k, v in TRAIN_SHORT.items()
                                        if k in t.columns}}
            t = t[list(keep)].rename(columns=keep)
            t.insert(0, "condition", run["condition"])
            t.insert(0, "wandb_id", run["wandb_id"])
            tcurves.append(t)

    df = pd.DataFrame(rows).sort_values(["critic", "arm", "delay", "seed_pair"],
                                        ignore_index=True)
    curves_df = pd.concat(curves, ignore_index=True).sort_values(
        ["condition", "wandb_id", "step"], ignore_index=True)
    tcurves_df = pd.concat(tcurves, ignore_index=True).sort_values(
        ["condition", "wandb_id", "step"], ignore_index=True)
    # The 2000-sample cap is per *run*, so a 4 G run is sampled ~8x more coarsely than a
    # 480 M one over the same first 480 M. Recorded, not corrected: window means are still
    # unbiased, only noisier.
    df = df.merge(tcurves_df.groupby("wandb_id").size().rename("train_points"),
                  left_on="wandb_id", right_index=True, how="left")

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    design = comparability_report(runs, invariant_cols=DESIGN_AXES, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(
            report + "\n\n# Design axes (expected to vary)\n\n" + design)
    print(report)

    ok = all([
        pipeline.write_csv(df, HERE / "data.csv", check=args.check),
        pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check),
        pipeline.write_csv(tcurves_df, HERE / "train_curves.csv", check=args.check),
    ])
    unusable = df[~df["usable"]]
    if len(unusable):
        print("\nNot usable (kept in data.csv, flagged):")
        print(unusable[["condition", "wandb_name", "state", "actual_step",
                        "reward_480_n"]].to_string(index=False))
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
