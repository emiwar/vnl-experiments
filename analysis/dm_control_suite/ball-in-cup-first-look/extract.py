"""First look at BallInCup: no efference copy vs efference copy vs explicit forward model.

BallInCup has been in the registry since the legacy era but has never been written about,
and until 2026-09-29 had no no-copy arm (it is listed as excluded in
``efference-copy-across-tasks``). This folder is the first pass over the 83 runs that
now exist, all at one commit, one budget (480 M) and one delay grid (0, 2, 3, 5, 7, 10,
12, 15, 20, 25 control steps of 20 ms). Since the 2026-10-05 refresh all three arms share
the same three seed settings.

The five conditions
-------------------
* ``undelayed`` -- ``DelayedMLP`` at ``delay_k = 0``. With no delay there is no queue to
  copy, so the no-copy and copy arms are the same network here; these runs are a shared
  reference, drawn at x = 0, not counted in any arm.
* ``undelayed_fm`` -- ``FlatForwardModel`` at ``delay_k = 0``. The predictor has nothing to
  predict (its target is its own input), so this is the control for "the forward model is
  simply a better network on this task", separate from what it does under delay.
* ``no_efference`` -- ``DelayedMLP``, ``efference_length = 0``: the actor sees only the
  delayed observation.
* ``efference`` -- ``DelayedMLP``, ``efference_length == delay_k``: the actor also sees the
  queue of actions still in flight.
* ``flat_forward_model`` -- ``FlatForwardModel``, ``efference_length == delay_k``: as
  ``efference``, plus a 4x256 predictor (delayed obs, action queue) -> current obs trained
  by its own loss (``fm_loss_weight = 1``, ``detach_prediction = True``), whose output the
  actor sees.

Why the readouts are what they are
----------------------------------
BallInCup's eval reward is **bimodal at the level of the policy**: the eval point is a mean
over 256 episodes, and across all 5 500 points in this cohort 1 773 are below 50 and
3 655 above 800, with ~70 in between. A policy either catches the ball or it does not.
So the end-of-training mean reward of a *cell* is mostly a count of how many of its runs
happened to be in the caught state at the end, and a mean of {0, 950, 950} describes no
run. The primary readouts are therefore binary/time-to-event ones, with mean reward kept
alongside:

* ``solved`` -- an eval point >= ``SOLVE_LEVEL`` (500). Because the gap is empty, the
  choice of 500 is nearly immaterial; ``steps_to_solve_800`` is kept as the sensitivity
  check. A single point suffices -- it is already a 256-episode mean, and the bimodality
  means there is no near-threshold noise to smooth (unlike ReacherHard's 900 crossing).
* ``steps_to_solve`` -- first solved eval point; null (censored) if never.
* ``steps_to_stable`` -- first point that starts ``STABLE_POINTS`` consecutive solved
  points (24 M steps). Distinguishes "found it" from "found it and kept it for a while".
* ``n_collapses`` -- solved -> unsolved transitions; ``n_lasting_collapses`` counts only
  those that stay unsolved for >= 2 points (one-point blips recover within 4.8 M steps
  and are a different phenomenon from forgetting).
* ``frac_solved_after_first`` -- fraction of eval points from the first solve onward that
  are solved: stability conditional on having found the solution.
* ``never_lost`` -- solved, and no lasting collapse afterwards.
* ``solved_at_end`` -- a majority of the points in the last 50 M steps are solved.
* ``reward_final`` (last-50 M mean) and ``reward_mean_all`` (mean over the whole run,
  i.e. normalised area under the curve) -- the "mean reward" readouts.

Traps this folder gates on
--------------------------
* **Legacy runs.** Seven pre-Hydra BallInCup runs (2026-06-01, 240 M budget, ``766e004``)
  are on the wrong side of both the schema split and the ``971ab99`` eval fix. Excluded
  by requiring ``net_params.delay_k`` and ``created_at >= EVAL_FIX_MIN_DATE``.
* **Seeds are a pair** (track README). ``seed_init/seed_ppo`` is the label; the first
  batch ran at (42, 1234). Every arm now has (42, 1234), (43, 43) and (44, 44), so the
  arms are paired cell by cell (same init, same PPO stream, same delay);
  ``summary.txt`` reports that paired contrast and a per-seed breakdown.
* **Delays at one seed are near-replicates, not independent runs.** ``nnx.Rngs(seed)``
  hands layers keys in creation order, so at a fixed seed every layer whose shape does
  not depend on ``delay_k`` gets bit-identical weights at every delay -- 100 % of the
  parameters for ``no_efference``, 96-99 % for the other two arms (only the first actor
  / predictor layer, widened by the action queue, differs) -- and the PPO stream is the
  same too. Across arms at one seed, 84-99 % of the initial weights are shared. So the
  design is effectively 3 initialisations x 3 architectures, with delay as a
  perturbation, and a test that treats the 25-27 runs per arm as independent is
  overconfident. ``seed_clustered`` reports the honest version: a permutation that
  reassigns arm labels within each seed, whole delay sweep at a time (6^3 = 216
  relabellings; the smallest attainable two-sided p is ~0.02).
* **Dirty working copy.** ``repos.vnl_experiments.dirty`` is True for every ``efference``
  run and mixed elsewhere, with no ``diff.patch`` stored. ``summary.txt`` compares dirty
  and clean runs within the arms that have both, which is the only check available.
* **Crashed run.** ``efference`` delay 2 at (42, 1234) crashed with no logged steps and is
  not selected; delay 20 at (42, 1234) was never launched in that arm. So ``efference`` is
  25 runs, not 27.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/ball-in-cup-first-look/extract.py
    ../.venv/bin/python analysis/dm_control_suite/ball-in-cup-first-look/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/ball-in-cup-first-look/extract.py --check

Writes ``data.csv`` (one row per run), ``curves.csv`` (eval series), ``summary.txt``
(pooled rates, tests, the paired contrast) and ``comparability.txt``.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: Produced with `artifacts ensure --kind history --runs <this>/runs.csv
#: --set project='"emiwar-team/nnx-ppo-delays"'`. Asserted in main().
HISTORY_SPEC_ID = "hist2000-8b97281e"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}"]

TASK = "BallInCup"
MIN_STD = 0.001
EVAL_FIX_MIN_DATE = "2026-09-10"

#: An eval point at or above this is "caught". The distribution has an empty gap from
#: ~50 to ~800, so anything in it gives the same answer; 800 is kept as a check.
SOLVE_LEVEL = 500
SOLVE_LEVEL_STRICT = 800

#: Consecutive solved points (at the 4.8 M cadence: 24 M steps) that count as stable.
STABLE_POINTS = 5

#: A collapse lasting at least this many points is "lasting" rather than a blip.
LASTING_POINTS = 2

FINAL_WINDOW_STEPS = 50_000_000

ARMS = ("no_efference", "efference", "flat_forward_model")
REFERENCES = ("undelayed", "undelayed_fm")
CONDITION_ORDER = REFERENCES + ARMS
SEEDS = ("i42/p1234", "i43/p43", "i44/p44")


def _base(df: pd.DataFrame) -> pd.Series:
    reached_total = df["summary._step"].ge(df["config.ppo.total_steps"])
    return (
        df["env"].eq(TASK)
        # Excludes the pre-Hydra era outright: those have no `net_params.*` at all.
        & df["net_params.delay_k"].notna()
        & df["net_params.min_std"].eq(MIN_STD)
        & df["created_at"].ge(EVAL_FIX_MIN_DATE)
        & (df["state"].eq("finished") | reached_total)
    )


def _arm(arm: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        mask = _base(df)
        delay = df["net_params.delay_k"]
        eff = df["net_params.efference_length"]
        cls = df["net_params.network_class"]
        if arm == "undelayed":
            return mask & delay.eq(0) & cls.eq("DelayedMLP")
        if arm == "undelayed_fm":
            return mask & delay.eq(0) & cls.eq("FlatForwardModel")
        mask &= delay.gt(0)
        if arm == "no_efference":
            return mask & cls.eq("DelayedMLP") & eff.eq(0)
        if arm == "efference":
            return mask & cls.eq("DelayedMLP") & eff.eq(delay)
        return mask & cls.eq("FlatForwardModel") & eff.eq(delay)
    return selector


CONDITIONS = {arm: _arm(arm) for arm in CONDITION_ORDER}

INVARIANTS = [
    "env",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "repos.nnx_ppo.dirty",
    "repos.vnl_playground.dirty",
    "repos.vnl_experiments.dirty",
    "config.ppo.n_envs",
    "config.ppo.total_steps",
    "config.ppo.learning_rate",
    "config.ppo.rollout_length",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "config.ppo.discounting_factor",
    "summary._step",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "net_params.min_std",
    "net_params.entropy_weight",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "net_params.normalize_obs",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "config.eval.every_steps",
    "os",
    "cuda_version",
    "python",
]

REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")
LIFESPAN_KEYS = ("eval/lifespan/mean", "lifespan_mean")


# --------------------------------------------------------------------------------------
# history -> series
# --------------------------------------------------------------------------------------


def history_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def series_of(hist: pd.DataFrame | None, keys) -> pd.DataFrame | None:
    if hist is None:
        return None
    key = next((k for k in keys if k in hist.columns), None)
    if key is None:
        return None
    out = hist.dropna(subset=[key])[["_step", key]]
    out = out.rename(columns={"_step": "step", key: "value"}).sort_values("step")
    # A requeued run's curve can repeat a step across attempts; average the duplicates.
    return out.groupby("step", as_index=False)["value"].mean()


# --------------------------------------------------------------------------------------
# per-run readouts
# --------------------------------------------------------------------------------------


def first_step(steps: np.ndarray, mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return None if hits.size == 0 else int(steps[hits[0]])


def first_stable_step(steps: np.ndarray, solved: np.ndarray,
                      k: int = STABLE_POINTS) -> int | None:
    """First step that begins a run of `k` consecutive solved points."""
    run = 0
    for i, s in enumerate(solved):
        run = run + 1 if s else 0
        if run == k:
            return int(steps[i - k + 1])
    return None


def collapses(solved: np.ndarray, min_len: int = 1) -> int:
    """Solved -> unsolved transitions whose unsolved stretch lasts >= `min_len` points.
    A stretch cut off by the end of training counts if it has reached `min_len`."""
    count, i, n = 0, 1, len(solved)
    while i < n:
        if solved[i - 1] and not solved[i]:
            j = i
            while j < n and not solved[j]:
                j += 1
            if j - i >= min_len:
                count += 1
            i = j
        else:
            i += 1
    return count


def seed_label(run: pd.Series) -> str:
    return f"i{int(run['seed'])}/p{int(run['config.seed'])}"


def build_row(run: pd.Series, reward: pd.DataFrame | None,
              lifespan: pd.DataFrame | None) -> dict:
    resumed = run.get("requeue.resumed_from_step")
    row = {
        "condition": run["condition"],
        "task": run["env"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "gpu": run["gpu"],
        "vnlx_dirty": bool(run.get("repos.vnl_experiments.dirty")),
        "delay": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        "seed": seed_label(run),
        "seed_init": int(run["seed"]),
        "seed_ppo": int(run["config.seed"]),
        "ctrl_dt": float(run["env_params.ctrl_dt"]),
        "actual_step": int(run["summary._step"]),
        "n_restarts": int(run.get("requeue.restart_count") or 0),
        "resumed_from_step": None if pd.isna(resumed) else int(resumed),
        "reward_source": "dmc_eval",
        "reward_last_point": pipeline.first_present(
            run, "summary.eval/episode_reward/mean", "summary.episode_reward/mean"),
    }
    if reward is None or reward.empty:
        row["curve_points"] = 0
        return row

    steps = reward["step"].to_numpy()
    values = reward["value"].to_numpy()
    solved = values >= SOLVE_LEVEL
    window = steps >= steps.max() - FINAL_WINDOW_STEPS
    first = first_step(steps, solved)
    after = solved[np.flatnonzero(solved)[0]:] if first is not None else np.array([])

    row.update(
        curve_points=int(len(values)),
        curve_max_step=int(steps.max()),
        reward_final=float(values[window].mean()),
        reward_final_n=int(window.sum()),
        reward_mean_all=float(values.mean()),
        reward_max=float(values.max()),
        solved_ever=bool(solved.any()),
        steps_to_solve=first,
        steps_to_solve_800=first_step(steps, values >= SOLVE_LEVEL_STRICT),
        steps_to_stable=first_stable_step(steps, solved),
        frac_time_solved=float(solved.mean()),
        frac_solved_after_first=float(after.mean()) if after.size else None,
        n_collapses=collapses(solved, 1),
        n_lasting_collapses=collapses(solved, LASTING_POINTS),
        never_lost=bool(solved.any() and collapses(solved, LASTING_POINTS) == 0),
        frac_solved_final=float(solved[window].mean()),
        solved_at_end=bool(solved[window].mean() >= 0.5),
        # Sensitivity check for the threshold: points strictly inside the empty gap.
        n_points_in_gap=int(((values >= 50) & (values < 800)).sum()),
    )
    if lifespan is not None and not lifespan.empty:
        row["lifespan_min"] = float(lifespan["value"].min())
    return row


def build_curves(run: pd.Series, reward: pd.DataFrame | None) -> list[dict]:
    if reward is None:
        return []
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "delay": int(run["net_params.delay_k"]), "seed": seed_label(run),
             "step": int(r.step), "reward_mean": float(r.value)}
            for r in reward.itertuples()]


# --------------------------------------------------------------------------------------
# cohort summaries (written to summary.txt; plot.py recomputes nothing statistical)
# --------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _censored(s: pd.Series) -> np.ndarray:
    """Steps-to-event with never-reached as +inf, so a median/rank test counts failures
    as slow rather than dropping them."""
    return s.astype(float).fillna(np.inf).to_numpy()


def arm_table(df: pd.DataFrame, title: str) -> list[str]:
    lines = [title, ""]
    lines.append(f"{'condition':<20}{'n':>3}  {'ever solved':>18}  {'never lost it':>18}"
                 f"  {'solved at end':>18}"
                 f"  {'med steps->solve':>17}  {'med ->stable':>13}  {'reward_final':>13}"
                 f"  {'reward_mean_all':>15}  {'lasting collapses/run':>21}"
                 f"  {'frac solved|found':>17}")
    for cond in CONDITION_ORDER:
        g = df[df.condition == cond]
        if g.empty:
            continue
        n = len(g)
        k_ever, k_end = int(g.solved_ever.sum()), int(g.solved_at_end.sum())
        k_kept = int(g.never_lost.sum())
        lo, hi = wilson(k_ever, n)
        lo1, hi1 = wilson(k_kept, n)
        lo2, hi2 = wilson(k_end, n)
        med = np.median(_censored(g.steps_to_solve)) / 1e6
        meds = np.median(_censored(g.steps_to_stable)) / 1e6
        fmt = lambda x: "  >480 M" if np.isinf(x) else f"{x:7.1f} M"
        lines.append(
            f"{cond:<20}{n:>3}  {k_ever:>2}/{n:<2} [{lo:.2f},{hi:.2f}]  "
            f"{k_kept:>2}/{n:<2} [{lo1:.2f},{hi1:.2f}]  "
            f"{k_end:>2}/{n:<2} [{lo2:.2f},{hi2:.2f}]  {fmt(med):>17}  {fmt(meds):>13}"
            f"  {g.reward_final.mean():>8.0f} ±{g.reward_final.std(ddof=1):>4.0f}"
            f"  {g.reward_mean_all.mean():>10.0f} ±{g.reward_mean_all.std(ddof=1):>3.0f}"
            f"  {g.n_lasting_collapses.mean():>21.2f}"
            f"  {g.frac_solved_after_first.mean():>17.2f}")
    lines.append("")
    lines.append("[a,b] = 95 % Wilson interval. Medians count never-solved runs as > budget.")
    lines.append("'never lost it' = solved, and no collapse below the level lasting >= 2 evals.")
    lines.append("reward_* are mean ± sd over runs (dmc_eval, unscaled).")
    return lines


def pairwise_tests(df: pd.DataFrame, title: str) -> list[str]:
    lines = [title, ""]
    for a, b in [("no_efference", "efference"), ("efference", "flat_forward_model"),
                 ("no_efference", "flat_forward_model")]:
        ga, gb = df[df.condition == a], df[df.condition == b]
        if ga.empty or gb.empty:
            continue
        lines.append(f"{a} vs {b}  (n = {len(ga)} vs {len(gb)})")
        for col in ("solved_ever", "never_lost", "solved_at_end"):
            table = [[int(ga[col].sum()), int((~ga[col]).sum())],
                     [int(gb[col].sum()), int((~gb[col]).sum())]]
            p = stats.fisher_exact(table).pvalue
            lines.append(f"  {col:<24} Fisher exact            p = {p:.3g}"
                         f"   ({table[0][0]}/{len(ga)} vs {table[1][0]}/{len(gb)})")
        for col in ("steps_to_solve", "steps_to_stable"):
            u = stats.mannwhitneyu(_censored(ga[col]), _censored(gb[col]),
                                   alternative="two-sided")
            lines.append(f"  {col:<24} Mann-Whitney (cens=inf)  p = {u.pvalue:.3g}")
        for col in ("reward_final", "reward_mean_all", "frac_solved_after_first"):
            xa, xb = ga[col].dropna(), gb[col].dropna()
            u = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            lines.append(f"  {col:<24} Mann-Whitney             p = {u.pvalue:.3g}"
                         f"   ({xa.mean():.2f} vs {xb.mean():.2f})")
        lines.append("")
    lines.append("Uncorrected, and runs are treated as independent -- WHICH THEY ARE NOT:")
    lines.append("delays at one seed share 96-100 % of their initial weights. These p-values")
    lines.append("are kept for reference only; see the seed-clustered section for the honest test.")
    return lines


def per_delay_table(df: pd.DataFrame) -> list[str]:
    lines = ["Per delay: steps to first solve (M), '-' = never; '*' = not solved at end", ""]
    delays = sorted(df.delay.unique())
    lines.append(f"{'condition':<20}" + "".join(f"{d:>22}" for d in delays))
    for cond in CONDITION_ORDER:
        g = df[df.condition == cond]
        if g.empty:
            continue
        cells = []
        for d in delays:
            sub = g[g.delay == d].sort_values("seed")
            cells.append(",".join(
                ("-" if pd.isna(r.steps_to_solve) else f"{r.steps_to_solve / 1e6:.0f}")
                + ("" if r.solved_at_end else "*") for r in sub.itertuples()) or ".")
        lines.append(f"{cond:<20}" + "".join(f"{c:>22}" for c in cells))
    lines.append("")
    lines.append("Within a cell the order is by seed label: " + ", ".join(SEEDS) + ".")
    return lines


def paired_contrast(df: pd.DataFrame) -> list[str]:
    """Every (seed, delay) cell that all three arms have: same network init, same PPO
    stream, same delay, so the arms differ only in architecture. A sign test over cells
    on which arm caught the ball first (ties -- both on the same eval, or both never --
    are dropped)."""
    sub = df[df.condition.isin(ARMS)]
    wide = sub.pivot_table(index=["seed", "delay"], columns="condition",
                           values="steps_to_solve", aggfunc="first", dropna=False)
    counts = sub.groupby(["seed", "delay"]).condition.nunique()
    complete = counts[counts == len(ARMS)].index
    wide = wide.loc[complete]
    lines = [f"Paired contrast: {len(wide)} (seed, delay) cells present in all three arms",
             "", f"{'seed':>10} {'delay':>5}  " + "".join(f"{a:>20}" for a in ARMS)]
    for (seed, d), row in wide.iterrows():
        lines.append(f"{seed:>10} {d:>5}  " + "".join(
            f"{'never' if pd.isna(row[a]) else f'{row[a] / 1e6:.0f} M':>20}" for a in ARMS))
    lines.append("")
    for a, b in [("no_efference", "efference"), ("efference", "flat_forward_model"),
                 ("no_efference", "flat_forward_model")]:
        xa, xb = _censored(wide[a]), _censored(wide[b])
        first = int((xa < xb).sum()); second = int((xa > xb).sum())
        p = stats.binomtest(first, first + second).pvalue if first + second else 1.0
        ratio = np.median(xb[np.isfinite(xa) & np.isfinite(xb)]
                          / xa[np.isfinite(xa) & np.isfinite(xb)])
        lines.append(f"  {a} solves first in {first}/{first + second} untied cells vs {b}"
                     f"  (sign test p = {p:.3g}; median {b}/{a} step ratio where both"
                     f" solve = {ratio:.1f}x)")
    return lines


def dirty_check(df: pd.DataFrame) -> list[str]:
    """Within each arm, dirty vs clean working copy. Only the arms that have both can say
    anything; `efference` is all dirty."""
    lines = ["Dirty vs clean vnl_experiments working copy, within condition", "",
             f"{'condition':<20}{'dirty':>7}{'n':>4}{'ever':>8}{'at end':>8}"
             f"{'med steps->solve':>18}{'reward_final':>14}"]
    for cond in CONDITION_ORDER:
        for dirty, g in df[df.condition == cond].groupby("vnlx_dirty"):
            med = np.median(_censored(g.steps_to_solve)) / 1e6
            lines.append(f"{cond:<20}{str(dirty):>7}{len(g):>4}"
                         f"{int(g.solved_ever.sum()):>5}/{len(g):<2}"
                         f"{int(g.solved_at_end.sum()):>5}/{len(g):<2}"
                         f"{'>480 M' if np.isinf(med) else f'{med:.1f} M':>18}"
                         f"{g.reward_final.mean():>14.0f}")
    for cond in ("no_efference", "flat_forward_model"):
        g = df[df.condition == cond]
        if g.vnlx_dirty.nunique() == 2:
            u = stats.mannwhitneyu(_censored(g[g.vnlx_dirty].steps_to_solve),
                                   _censored(g[~g.vnlx_dirty].steps_to_solve),
                                   alternative="two-sided")
            lines.append(f"  {cond}: steps_to_solve dirty vs clean, Mann-Whitney "
                         f"p = {u.pvalue:.3g}")
    return lines


def _log_steps(s: pd.Series) -> pd.Series:
    """log10 steps-to-event, never-reached placed at twice the budget (rank-preserving)."""
    return np.log10(s.astype(float).fillna(2 * 480_000_000))


def seed_clustered(df: pd.DataFrame) -> list[str]:
    """The readouts at the level of the initialisation, which is the actual unit of
    replication here (see the docstring). Three parts: the arm x seed table, how much of
    each arm's variance seed explains, and a seed-block permutation test."""
    sub = df[df.condition.isin(ARMS)].assign(ls=lambda d: _log_steps(d.steps_to_solve))
    lines = ["Seed-clustered view (the unit of replication is the initialisation)", "",
             "Median steps to first solve (M) per arm x seed, over that seed's delays:", ""]
    med = (sub.groupby(["seed", "condition"]).steps_to_solve
           .apply(lambda x: np.median(_censored(x)) / 1e6).unstack()[list(ARMS)])
    lines += [f"{'seed':>10}" + "".join(f"{a:>20}" for a in ARMS)]
    lines += [f"{seed:>10}" + "".join(
        f"{'>480' if np.isinf(v) else f'{v:.1f}':>20}" for v in row)
        for seed, row in med.iterrows()]
    for col in ("never_lost", "solved_at_end"):
        frac = sub.groupby(["seed", "condition"])[col].mean().unstack()[list(ARMS)]
        lines += ["", f"Fraction {col} per arm x seed:", ""]
        lines += [f"{seed:>10}" + "".join(f"{v:>20.2f}" for v in row)
                  for seed, row in frac.iterrows()]

    lines += ["", "Share of each arm's variance in log10(steps to solve) explained by seed:"]
    for arm in ARMS:
        g = sub[sub.condition == arm]
        groups = [x.ls.to_numpy() for _, x in g.groupby("seed")]
        ss_between = sum(len(x) * (x.mean() - g.ls.mean()) ** 2 for x in groups)
        eta2 = ss_between / ((g.ls - g.ls.mean()) ** 2).sum()
        lines.append(f"  {arm:<20} eta^2 = {eta2:.2f}   Kruskal-Wallis p = "
                     f"{stats.kruskal(*groups).pvalue:.3g}")

    counts = sub.groupby(["seed", "delay"]).condition.nunique()
    cells = counts[counts == len(ARMS)].index
    lines += ["", f"Seed-block permutation tests over the {len(cells)} complete cells "
                  f"(arm labels permuted within each seed, delay sweep kept together):"]
    for col, label in (("ls", "slower to first solve"), ("never_lost", "never lost it"),
                       ("solved_at_end", "solved at end")):
        wide = (sub.set_index(["seed", "delay"]).loc[cells].reset_index()
                .pivot_table(index=["seed", "delay"], columns="condition", values=col,
                             aggfunc="first").astype(float))
        seeds = wide.index.get_level_values(0).unique()
        for a, b in (("no_efference", "efference"), ("flat_forward_model", "efference"),
                     ("no_efference", "flat_forward_model")):
            statistic = (lambda w: np.sign(w[b] - w[a]).sum()) if col == "ls" else \
                        (lambda w: (w[a] - w[b]).sum())
            observed = statistic(wide)
            null = []
            for perm in itertools.product(itertools.permutations(ARMS), repeat=len(seeds)):
                w = wide.copy()
                for seed, order in zip(seeds, perm):
                    w.loc[seed, list(ARMS)] = wide.loc[seed, list(order)].to_numpy()
                null.append(statistic(w))
            null = np.abs(np.array(null))
            p = float((null >= abs(observed) - 1e-9).mean())
            what = (f"{b} {label} than {a} in net {observed:+.0f} cells" if col == "ls"
                    else f"{label}: {a} - {b} = {observed:+.0f} cells")
            lines.append(f"  {what:<62} p = {p:.3f}")
    lines += ["", "Compare with the pooled tests above, which treat runs as independent and",
              "are therefore far too small: the pooled time-to-solve p for efference vs",
              "no_efference is reported there as < 0.01."]
    return lines


def resume_check(df: pd.DataFrame, curves: pd.DataFrame) -> list[str]:
    """Did resuming a run that was still stuck at 0 change anything? A light-checkpoint
    resume keeps weights and optimizer state but redraws every env state and zeroes the
    action queues, so it is a perturbation of exactly the kind a stuck run might need."""
    lines = ["Runs resumed (requeue) while not yet ever solved", ""]
    hits = []
    for r in df[df.resumed_from_step.fillna(0) > 0].itertuples():
        first = r.steps_to_solve if pd.notna(r.steps_to_solve) else np.inf
        if first <= r.resumed_from_step:
            continue
        gap = (first - r.resumed_from_step) / 1e6
        hits.append(gap <= 40)
        lines.append(f"  {r.condition:<20} {r.seed:>10} d{r.delay:<3} resumed at "
                     f"{r.resumed_from_step / 1e6:5.0f} M -> first solve "
                     f"{'never' if np.isinf(gap) else f'{gap:.0f} M later'}")
    base_hits = base_n = 0
    for r in df[df.resumed_from_step.isna()].itertuples():
        first = r.steps_to_solve if pd.notna(r.steps_to_solve) else np.inf
        for at in (200e6, 250e6, 300e6):
            if first > at:
                base_n += 1
                base_hits += first <= at + 40e6
    lines += ["", f"  solved within 40 M of the resume: {sum(hits)}/{len(hits)}",
              f"  base rate, never-resumed runs still unsolved at 200/250/300 M and solving "
              f"within the next 40 M: {base_hits}/{base_n}"]
    return lines


def threshold_check(df: pd.DataFrame) -> list[str]:
    d = df.dropna(subset=["steps_to_solve"])
    same = (d.steps_to_solve == d.steps_to_solve_800).mean()
    return [
        "Threshold sensitivity",
        "",
        f"eval points strictly in [50, 800): {int(df.n_points_in_gap.sum())} of "
        f"{int(df.curve_points.sum())}",
        f"runs whose first point >= 800 is the same point as first >= 500: "
        f"{same * 100:.0f} % of {len(d)} solved runs",
    ]


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    producer = get_producer("history")
    spec_id = producer.spec_id(producer.spec(project=PROJECT))
    if spec_id != HISTORY_SPEC_ID:
        raise SystemExit(
            f"history spec_id has drifted: got {spec_id}, expected {HISTORY_SPEC_ID}. "
            f"Update HISTORY_SPEC_ID deliberately and say so in report.md.")

    rows, curves = [], []
    for _, run in runs.iterrows():
        hist = history_of(store, run["wandb_id"], spec_id)
        reward = series_of(hist, REWARD_KEYS)
        rows.append(build_row(run, reward, series_of(hist, LIFESPAN_KEYS)))
        curves.extend(build_curves(run, reward))

    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], CONDITION_ORDER, ordered=True)
    df = df.sort_values(["condition", "delay", "seed", "wandb_id"], ignore_index=True)
    df["condition"] = df["condition"].astype(str)
    curves_df = pd.DataFrame(
        curves, columns=["wandb_id", "condition", "delay", "seed", "step", "reward_mean"])
    curves_df = curves_df.sort_values(["condition", "delay", "seed", "step"],
                                      ignore_index=True)

    missing = int((df["curve_points"] == 0).sum())
    if missing:
        raise SystemExit(
            f"*** {missing}/{len(df)} runs have no history artifact. Run:\n"
            f"    python -m vnl_experiments.artifacts ensure --kind history \\\n"
            f"        --runs analysis/dm_control_suite/{HERE.name}/runs.csv \\\n"
            f"        --set project='\"{PROJECT}\"'")
    stale = df[df.curve_max_step < df.actual_step - FINAL_WINDOW_STEPS]
    if not stale.empty:
        print(f"\n*** {len(stale)} history artifact(s) stop short of the run's final "
              f"step; re-produce with `--override`:\n"
              f"{stale[['wandb_id', 'curve_max_step', 'actual_step']].to_string(index=False)}\n")

    per_seed = []
    for seed in SEEDS:
        per_seed += arm_table(df[df.seed == seed], f"Seed {seed} only") + [""]
    summary = "\n".join(
        arm_table(df, "All runs, pooled over delays > 0 (undelayed* = delay 0 references)")
        + ["", "=" * 100, ""]
        + per_seed
        + ["=" * 100, ""]
        + pairwise_tests(df, "Pairwise tests, all runs")
        + ["", "=" * 100, ""]
        + per_delay_table(df)
        + ["", "=" * 100, ""]
        + paired_contrast(df)
        + ["", "=" * 100, ""]
        + seed_clustered(df)
        + ["", "=" * 100, ""]
        + dirty_check(df)
        + ["", "=" * 100, ""]
        + resume_check(df, curves_df)
        + ["", "=" * 100, ""]
        + threshold_check(df)) + "\n"

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report)
        (HERE / "summary.txt").write_text(summary)
    print(report)
    print(summary)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
