"""Does fixed motor noise make the efference copy matter on CartpoleSwingup?

The hypothesis (see ``vnl_experiments/delays/motor_noise.py``): on dm_control the copy
barely helps a delayed policy because the tasks are deterministic, so the actions in
flight are predictable from the delayed observation and carry no extra information.
Fixed Gaussian noise on the executed action breaks that, and *where* it is added decides
whether the queue sees it:

* ``noise_pre``  -- ``net.motor_noise_site=pre_efference``: the queue records the noisy
  action, so the in-flight actions are informative.
* ``noise_post`` -- ``post_efference``: the queue records the intended action, the body
  executes the noisy one. Unobserved motor noise.
* ``noise_no_copy`` -- ``efference_length = 0`` with noise. Kept as a selector for
  parity with ``../walker-motor-noise/``; **no such run exists** on CartpoleSwingup as of
  2026-10-28, so the copy's advantage under noise cannot be computed here.

Plus the noise-free ``copy`` / ``no_copy`` arms. Split out of ``motor-noise-first-look``
(now ``../walker-motor-noise/``) on 2026-10-28.

Cohort
------
* ``DelayedMLP``, privileged critic, ``min_std`` 0.001, torque control (``servo_kp`` absent
  or 0), no distillation, post-``d168093`` (the eval-wrapper fix).
* CartpoleSwingup. Every run has a 4.8e8 budget and is read at :data:`READ_AT`.
* No de-duplication: same-seed replicates are kept and averaged *within* a seed pair
  downstream (``plot_seeds`` does this), and :func:`determinism` reports them: they
  are the run-to-run floor (mostly cross-GPU) that a paired difference has to clear.

Readouts
--------
* ``reward``: mean of the eval points in ``(READ_AT - 50e6, READ_AT]`` (README §6). Eval
  episodes keep the motor noise on (it is part of the world), so a noisy run's reward is
  measured *under its own noise*.
* ``reward_median`` / ``dip_frac``: median of the eval points over the last
  :data:`ROBUST_WINDOW` steps, and the fraction of them below :data:`DIP_BELOW`. A
  secondary, crash-robust readout; see :data:`ROBUST_WINDOW`.
* ``steps_to_{600,800}``: first eval step at or above that reward; NaN if never.
* ``policy_sigma``: the policy's own exploration width in actuator units, from the logged
  ``loglikelihood/mean``. The sampler scores the tanh-squashed action, so per dimension
  ``E[log p] = -0.5 log(2 pi e) - log sigma_a`` with ``sigma_a`` the *action-space* width
  (pre-tanh sigma times the tanh slope), hence
  ``sigma_a ~= exp(-(ll / d) - 0.5 log(2 pi e))``, a geometric mean over dimensions.
  This is the yardstick for "was the noise big enough": during *training* the sampler's
  draw is already noise the queue records, i.e. already pre-efference noise of size
  ``policy_sigma``. The motor noise is only new information for the copy where it is not
  small against that. At eval the sampler is deterministic and only motor noise remains.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/cartpole-motor-noise/extract.py
    ../.venv/bin/python analysis/dm_control_suite/cartpole-motor-noise/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/cartpole-motor-noise/extract.py --check
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
#: Training-side curves for the policy-width yardstick. Same producer, explicit keys:
#:
#:     python -m vnl_experiments.artifacts ensure --kind history \
#:         --runs analysis/dm_control_suite/cartpole-motor-noise/runs.csv \
#:         --set project='"emiwar-team/nnx-ppo-delays"' --set keys="$TRAIN_KEYS"
TRAIN_KEYS = ["loglikelihood/mean", "rollout_batch/action/std",
              "rollout_batch/reward/mean"]
TRAIN_SPEC_ID = "hist2000-9cd5ea00"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}", f"history:{TRAIN_SPEC_ID}"]

NETWORK = "DelayedMLP"
MIN_STD = 0.001
PRE_EVAL_FIX_COMMIT = "d168093"

READ_AT = 480_000_000
WINDOW = 50_000_000
#: Earlier window for `late_gain`, the per-run check for README §6's budget trap.
EARLY_AT = READ_AT - 2 * WINDOW
#: Secondary, crash-robust readout: CartpoleSwingup's eval reward drops to ~200 and
#: recovers within a few eval points throughout training, clean runs included, so the
#: 5e7-window mean mostly measures whether a crash fell inside it. The median over the
#: last half of training (~48 eval points) is what the policy does when it is not
#: crashing; `dip_frac` is how often it crashes.
ROBUST_WINDOW = 240_000_000
DIP_BELOW = 600.0
#: Time to solve: the first eval step at which the reward reaches each level. NaN when
#: the run never does (every run here has the same 4.8e8 horizon).
SOLVE_LEVELS = (600, 800)

#: task -> (control step in ms, action dimension). Asserted against the runs in `main`
#: where the index records them.
TASKS = {"CartpoleSwingup": (10, 1)}

ARMS = ("noise_pre", "noise_post", "noise_no_copy", "copy", "no_copy")

GAUSS_ENTROPY_CONST = 0.5 * np.log(2 * np.pi * np.e)


def _std(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["net_params.motor_noise_std"], errors="coerce").fillna(0.0)


def _base(df: pd.DataFrame) -> pd.Series:
    return (
        df["net_params.network_class"].eq(NETWORK)
        & df["net_params.min_std"].eq(MIN_STD)
        & df["net_params.delay_k"].notna()
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
        & df["env_params.servo_kp"].fillna(0.0).eq(0.0)
        & df["net_params.privileged_critic"].map(lambda v: v is not False)
        & df["distill.teacher"].isna()
        & df["config.ppo.total_steps"].ge(READ_AT)
    )


def _cell(task: str, arm: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        mask = _base(df) & df["env"].eq(task)
        std, site = _std(df), df["net_params.motor_noise_site"]
        eff, delay = df["net_params.efference_length"], df["net_params.delay_k"]
        if arm == "noise_pre":
            return mask & std.gt(0) & site.eq("pre_efference") & eff.eq(delay)
        if arm == "noise_post":
            return mask & std.gt(0) & site.eq("post_efference") & eff.eq(delay)
        if arm == "noise_no_copy":
            # No queue, so the site is inert; either launch value belongs here.
            return mask & std.gt(0) & eff.eq(0) & delay.gt(0)
        mask &= std.eq(0)
        if arm == "copy":
            return mask & eff.eq(delay)
        return mask & eff.eq(0) & delay.gt(0)
    return selector


CONDITIONS = {f"{t}_{a}": _cell(t, a) for t in TASKS for a in ARMS}

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
    "repos.vnl_playground.commit",
]

DESIGN_AXES = [
    "env",
    "net_params.delay_k",
    "net_params.motor_noise_std",
    "net_params.motor_noise_site",
    "config.ppo.total_steps",
    "config.eval.every_steps",
    "state",
    "git_commit",
    "repos.nnx_ppo.commit",
    "repos.vnl_experiments.dirty",
    "gpu",
    "seed",
    "config.seed",
]


def _frame(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    if entry is None:
        raise KeyError(wandb_id)
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None
    return None if frame.empty else frame


def _series(frame: pd.DataFrame | None, key: str) -> pd.DataFrame | None:
    if frame is None or key not in frame.columns:
        return None
    out = frame.dropna(subset=[key])[["_step", key]]
    return out.rename(columns={"_step": "step", key: "value"}).sort_values("step")


def _window(series: pd.DataFrame | None, upper: int) -> tuple[float, int]:
    if series is None:
        return np.nan, 0
    sub = series[(series["step"] > upper - WINDOW) & (series["step"] <= upper)]
    return (float(sub["value"].mean()), len(sub)) if len(sub) else (np.nan, 0)


def build_row(run: pd.Series, evals: pd.DataFrame | None,
              train: pd.DataFrame | None) -> dict:
    task = next(t for t in TASKS if run["condition"].startswith(t + "_"))
    arm = run["condition"][len(task) + 1:]
    ctrl_ms, action_dim = TASKS[task]
    reward_series = _series(evals, "eval/episode_reward/mean")
    reward, n = _window(reward_series, READ_AT)
    early, _ = _window(reward_series, EARLY_AT)
    late = (reward_series[(reward_series.step > READ_AT - ROBUST_WINDOW)
                          & (reward_series.step <= READ_AT)]["value"]
            if reward_series is not None else pd.Series(dtype=float))
    ll, ll_n = _window(_series(train, "loglikelihood/mean"), READ_AT)
    act_std, _ = _window(_series(train, "rollout_batch/action/std"), READ_AT)
    # Absent (every run before 2026-10-27) is 0. Not `x or 0.0`: NaN is truthy.
    std = float(np.nan_to_num(pd.to_numeric(run.get("net_params.motor_noise_std"),
                                            errors="coerce")))
    return {
        "condition": run["condition"],
        "task": task,
        "arm": arm,
        "site": run.get("net_params.motor_noise_site") if std > 0 else "none",
        "noise_std": std,
        "delay_k": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        "ctrl_dt_ms": ctrl_ms,
        "seed_pair": f"i{int(run['seed'])}/p{int(run['config.seed'])}",
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "state": run["state"],
        "created_at": run["created_at"],
        "budget": int(run["config.ppo.total_steps"]),
        "actual_step": run.get("summary._step"),
        "git_commit": str(run["git_commit"])[:7],
        "dirty": run.get("repos.vnl_experiments.dirty"),
        "gpu": run["gpu"],
        "reward_source": "dmc_eval",
        "reward": reward,
        "reward_n": n,
        "late_gain": reward - early if n else np.nan,
        "reward_median": float(late.median()) if len(late) else np.nan,
        "dip_frac": float((late < DIP_BELOW).mean()) if len(late) else np.nan,
        **{f"steps_to_{level}": (
            float(reward_series.step[reward_series.value >= level].iloc[0])
            if reward_series is not None and (reward_series.value >= level).any()
            else np.nan) for level in SOLVE_LEVELS},
        "reward_last": (float(evals["eval/episode_reward/mean"].dropna().iloc[-1])
                        if evals is not None and "eval/episode_reward/mean" in evals
                        else np.nan),
        "loglik": ll,
        "loglik_n": ll_n,
        "policy_sigma": (float(np.exp(-ll / action_dim - GAUSS_ENTROPY_CONST))
                         if ll_n else np.nan),
        "batch_action_std": act_std,
    }


def determinism(df: pd.DataFrame) -> str:
    """Same seed pair, same task, delay, std, site: run more than once."""
    key = ["task", "delay_k", "efference_length", "noise_std", "site", "seed_pair"]
    # At std 0 the site is inert by construction, so group on it as "none" -- that
    # is exactly the claim under test.
    dup = df[df.duplicated(key, keep=False)].sort_values(key)
    lines = ["\nSame-config, same-seed replicates (reward at READ_AT; site column is "
             "the *launched* site for std 0)"]
    for k, g in dup.groupby(key):
        sites = df.loc[g.index].wandb_name.str.contains("mnpre").map(
            {True: "pre", False: ""})
        lines.append(f"  {k}: " + ", ".join(
            f"{r.wandb_id} {r.gpu.replace('NVIDIA ', '')[:10]} {r.reward:.2f}"
            for r in g.itertuples()) + f"  |diff| {g.reward.max() - g.reward.min():.2f}")
    return "\n".join(lines) + "\n"


def summary_tables(df: pd.DataFrame) -> str:
    lines = []
    use = df[df.reward_n > 0]
    per_seed = (use.groupby(["task", "arm", "delay_k", "noise_std", "seed_pair"])
                [["reward", "reward_median", "dip_frac", "policy_sigma"]].mean()
                .reset_index())
    for task, g in per_seed.groupby("task"):
        lines.append(f"\n{task}: reward at {READ_AT:.1e} (mean over seed pairs; n seeds; "
                     "range) | median and dip fraction over the last "
                     f"{ROBUST_WINDOW:.1e} | policy_sigma")
        cells = (g.groupby(["delay_k", "arm", "noise_std"])
                 .agg(reward=("reward", "mean"), n=("reward", "size"),
                      lo=("reward", "min"), hi=("reward", "max"),
                      median=("reward_median", "mean"),
                      dips=("dip_frac", "mean"),
                      sigma=("policy_sigma", "mean"))
                 .reset_index())
        for r in cells.itertuples():
            lines.append(f"  d{r.delay_k:>2} {r.arm:<10} std {r.noise_std:<4g} "
                         f"{r.reward:7.1f}  n={r.n}  [{r.lo:6.1f}, {r.hi:6.1f}]  "
                         f"median {r.median:6.1f}  dips {r.dips:.2f}  "
                         f"sigma_a {r.sigma:.3f}")
    return "\n".join(lines) + "\n"


def paired_site(df: pd.DataFrame) -> str:
    """pre minus post at the same task, delay, std and seed pair."""
    use = df[(df.reward_n > 0) & df.arm.isin(["noise_pre", "noise_post"])]
    piv = use.pivot_table(index=["task", "delay_k", "noise_std", "seed_pair"],
                          columns="arm", values="reward", aggfunc="mean").dropna()
    piv["pre_minus_post"] = piv["noise_pre"] - piv["noise_post"]
    return ("\nPaired pre - post (same task, delay, std, seed pair):\n"
            + piv.round(1).to_string() + "\n")


def attrition(df: pd.DataFrame) -> str:
    bad = df[df.reward_n == 0]
    return ("\nRuns with no readout at READ_AT:\n"
            + (bad[["wandb_id", "wandb_name", "state", "actual_step"]].to_string(
                index=False) if len(bad) else "  none") + "\n")


def main() -> None:
    args = pipeline.parse_args(__doc__)
    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    for task, (ms, _) in TASKS.items():
        got = runs.loc[runs["env"].eq(task), "env_params.ctrl_dt"].dropna().unique()
        if len(got) and sorted(got) != [ms / 1000]:
            raise SystemExit(f"{task}: TASKS says {ms} ms, runs carry {sorted(got)}")

    producer = get_producer("history")
    for want, spec in ((HISTORY_SPEC_ID, producer.spec(project=PROJECT)),
                       (TRAIN_SPEC_ID, producer.spec(project=PROJECT, keys=TRAIN_KEYS))):
        got = producer.spec_id(spec)
        if got != want:
            raise SystemExit(f"history spec id drifted: {got} != {want}")

    store = Store()
    rows, curves, absent = [], [], []
    for _, run in runs.iterrows():
        try:
            evals = _frame(store, run["wandb_id"], HISTORY_SPEC_ID)
            train = _frame(store, run["wandb_id"], TRAIN_SPEC_ID)
        except KeyError:
            absent.append(run["wandb_id"])
            continue
        row = build_row(run, evals, train)
        rows.append(row)
        series = _series(evals, "eval/episode_reward/mean")
        if series is not None:
            for s in series[series.step <= READ_AT].itertuples():
                curves.append({k: row[k] for k in
                               ("wandb_id", "task", "arm", "noise_std", "delay_k",
                                "seed_pair")} | {"step": int(s.step), "reward": s.value})
    if absent:
        raise SystemExit(
            f"{len(absent)} runs lack a history artifact; produce both specs:\n"
            f"  python -m vnl_experiments.artifacts ensure --kind history --runs "
            f"analysis/dm_control_suite/{HERE.name}/runs.csv "
            f"--set project='\"{PROJECT}\"' [--set keys='{TRAIN_KEYS}']")

    df = (pd.DataFrame(rows)
          .sort_values(["task", "arm", "delay_k", "noise_std", "seed_pair", "wandb_id"])
          .reset_index(drop=True))
    curves_df = (pd.DataFrame(curves)
                 .sort_values(["task", "arm", "delay_k", "noise_std", "seed_pair",
                               "wandb_id", "step"]).reset_index(drop=True))

    blocks = [summary_tables(df), paired_site(df), determinism(df), attrition(df)]
    for b in blocks:
        print(b)
    report = (comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
              + "\n\n--- DESIGN AXES: expected to vary ---\n"
              + comparability_report(runs, invariant_cols=DESIGN_AXES,
                                     group_col="condition")
              + "\n" + "\n".join(blocks))
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
