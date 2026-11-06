"""Does the noise's correlation time matter for delayed WalkerWalk, and is there a band?

``net.motor_noise_tau`` (2026-11-03) makes the motor noise an AR(1) process with
correlation time tau in control steps, at the same stationary std (``tau = 0`` is the iid
noise of ``../walker-motor-noise/``). The hypothesis it was added for:

* iid noise is largely low-pass filtered by the body;
* noise much slower than the delay is corrected by the delayed feedback anyway;
* so noise with tau around the delay is where delay hurts most, and where only a copy
  that *sees* the noise (``pre_efference``) can help.

Arms (as in ``../walker-motor-noise/``; the noisy ones at any tau):

* ``noise_pre`` / ``noise_post`` -- copy on, the queue sees / does not see the noise;
* ``noise_no_copy`` -- ``efference_length = 0`` with noise;
* ``copy`` / ``no_copy`` -- noise-free.

Noisy runs at std 0.2 and 0.5 only (the two levels the tau sweep uses). iid runs are kept
as the tau = 0 reference on every axis. WalkerWalk only.

Two networks, as a ``network`` column: ``DelayedMLP`` (conditions ``WalkerWalk_<arm>``) and,
from 2026-11-05, ``FlatForwardModel`` (``WalkerWalk_fm_<arm>``), whose efference queue is
the predictor's action buffer: pre-efference noise tells the predictor the executed
actions, post-efference noise leaves it to predict from the intended ones.

Cohort gates are those of ``../walker-motor-noise/`` (privileged critic where the key
exists, ``min_std`` 0.001, torque, no distillation, post-``d168093``, budget >= 4.8e8).

Readouts
--------
* ``reward``: mean of the eval points in ``(READ_AT - 50e6, READ_AT]``. Eval keeps the
  motor noise on, so a noisy run is measured under its own noise.
* ``reward_1b``: the same window ending at :data:`LONG_AT`, for runs that reach it (the
  second seed pair and the d0 tau 12 / 15 runs were trained to 1e9). A complement only;
  ``reward`` stays the comparison readout, so 1e9 runs pool with 4.8e8 ones.
* ``steps_to_{500,800,900,950}``: first eval step at or above that reward, **searched only
  up to READ_AT**, so every run has the same 4.8e8 horizon (the longer runs are censored
  there too). NaN = not reached by READ_AT.
* ``policy_sigma``: the policy's own action-space width, from ``loglikelihood/mean``
  (see ``../walker-motor-noise/extract.py``).
* ``reg_loss``: ``losses/regularization/mean`` over the readout window. For
  ``FlatForwardModel`` that is the predictor's L2 loss (``fm_loss_weight`` 1) plus the
  sampler's entropy term; the entropy term alone is what ``DelayedMLP`` reports (about
  -0.01 noise-free, +0.02 with noise). ``fm_pred_mse`` itself is not logged.
* ``batch_action_std``: std of the *executed* actions in the rollout batch (the env
  action, after the motor noise). Pools envs and time, so it sees the stationary noise
  variance whatever tau is -- a check that tau changes the timescale and not the size.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/walker-correlated-noise/extract.py
    ../.venv/bin/python analysis/dm_control_suite/walker-correlated-noise/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/walker-correlated-noise/extract.py --check
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: The ids every sibling folder reads (see ``../walker-motor-noise/extract.py``).
HISTORY_SPEC_ID = "hist2000-8b97281e"
TRAIN_KEYS = ["loglikelihood/mean", "rollout_batch/action/std",
              "rollout_batch/reward/mean"]
TRAIN_SPEC_ID = "hist2000-9cd5ea00"
#: The regularization loss, which for FlatForwardModel carries the predictor's L2 loss.
REG_KEYS = ["losses/regularization/mean"]
REG_SPEC_ID = "hist2000-14366f48"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}", f"history:{TRAIN_SPEC_ID}",
            f"history:{REG_SPEC_ID}"]

#: network_class -> condition-name prefix. DelayedMLP keeps the original, unprefixed
#: names so its runs.csv rows are unchanged.
NETWORKS = {"DelayedMLP": "", "FlatForwardModel": "fm_"}
MIN_STD = 0.001
PRE_EVAL_FIX_COMMIT = "d168093"

READ_AT = 480_000_000
WINDOW = 50_000_000
LONG_AT = 1_000_000_000
SOLVE_LEVELS = (500, 800, 900, 950)
#: The noise levels the tau sweep was run at.
NOISE_STDS = (0.2, 0.5)

TASK = "WalkerWalk"
CTRL_MS, ACTION_DIM = 25, 6

ARMS = ("noise_pre", "noise_post", "noise_no_copy", "copy", "no_copy")

GAUSS_ENTROPY_CONST = 0.5 * np.log(2 * np.pi * np.e)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column; absent (runs before the knob existed) is 0."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _base(df: pd.DataFrame, network: str) -> pd.Series:
    return (
        df["env"].eq(TASK)
        & df["net_params.network_class"].eq(network)
        & df["net_params.min_std"].eq(MIN_STD)
        & df["net_params.delay_k"].notna()
        & ~df["git_commit"].str.startswith(PRE_EVAL_FIX_COMMIT, na=False)
        & df["env_params.servo_kp"].fillna(0.0).eq(0.0)
        & df["net_params.privileged_critic"].map(lambda v: v is not False)
        & df["distill.teacher"].isna()
        & df["config.ppo.total_steps"].ge(READ_AT)
    )


def _cell(network: str, arm: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        mask = _base(df, network)
        std = _num(df, "net_params.motor_noise_std")
        site = df["net_params.motor_noise_site"]
        eff, delay = df["net_params.efference_length"], df["net_params.delay_k"]
        noisy = mask & std.isin(NOISE_STDS)
        if arm == "noise_pre":
            return noisy & site.eq("pre_efference") & eff.eq(delay)
        if arm == "noise_post":
            return noisy & site.eq("post_efference") & eff.eq(delay)
        if arm == "noise_no_copy":
            # No queue, so the site is inert; either launch value belongs here.
            return noisy & eff.eq(0) & delay.gt(0)
        clean = mask & std.eq(0)
        if arm == "copy":
            return clean & eff.eq(delay)
        return clean & eff.eq(0) & delay.gt(0)
    return selector


CONDITIONS = {f"{TASK}_{prefix}{a}": _cell(net, a)
              for net, prefix in NETWORKS.items() for a in ARMS}

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
    "net_params.network_class",
    "net_params.delay_k",
    "net_params.predictor_hidden_sizes",
    "net_params.fm_loss_weight",
    "net_params.detach_prediction",
    "net_params.motor_noise_std",
    "net_params.motor_noise_site",
    "net_params.motor_noise_tau",
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
              train: pd.DataFrame | None, reg: pd.DataFrame | None) -> dict:
    network = run["net_params.network_class"]
    arm = run["condition"][len(TASK) + 1 + len(NETWORKS[network]):]
    reward_series = _series(evals, "eval/episode_reward/mean")
    reward, n = _window(reward_series, READ_AT)
    long, long_n = _window(reward_series, LONG_AT)
    ll, ll_n = _window(_series(train, "loglikelihood/mean"), READ_AT)
    act_std, _ = _window(_series(train, "rollout_batch/action/std"), READ_AT)
    reg_loss, _ = _window(_series(reg, "losses/regularization/mean"), READ_AT)
    horizon = (reward_series[reward_series.step <= READ_AT]
               if reward_series is not None else None)
    solve = {}
    for level in SOLVE_LEVELS:
        hit = horizon[horizon.value >= level] if horizon is not None else pd.DataFrame()
        solve[f"steps_to_{level}"] = float(hit.step.iloc[0]) if len(hit) else np.nan
    std = float(_num(run.to_frame().T, "net_params.motor_noise_std").iloc[0])
    tau = float(_num(run.to_frame().T, "net_params.motor_noise_tau").iloc[0])
    return {
        "condition": run["condition"],
        "network": network,
        "arm": arm,
        "site": run.get("net_params.motor_noise_site") if std > 0 else "none",
        "noise_std": std,
        "noise_tau": tau if std > 0 else np.nan,
        "delay_k": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
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
        "reward_1b": long,
        "reward_1b_n": long_n,
        "last_eval_step": (float(horizon.step.iloc[-1])
                           if horizon is not None and len(horizon) else np.nan),
        **solve,
        "loglik": ll,
        "policy_sigma": (float(np.exp(-ll / ACTION_DIM - GAUSS_ENTROPY_CONST))
                         if ll_n else np.nan),
        "batch_action_std": act_std,
        "reg_loss": reg_loss,
    }


def summary_table(df: pd.DataFrame) -> str:
    use = df[df.reward_n > 0].assign(tau=df.noise_tau.fillna(-1))
    cells = (use.groupby(["network", "delay_k", "arm", "noise_std", "tau"])
             .agg(reward=("reward", "mean"), n=("reward", "size"),
                  lo=("reward", "min"), hi=("reward", "max"),
                  t900=("steps_to_900", "median"),
                  act_std=("batch_action_std", "mean"),
                  sigma=("policy_sigma", "mean"))
             .reset_index())
    lines = [f"\nReward at {READ_AT:.1e} (mean over runs; n; range) | median steps to 900 "
             "| executed-action std | policy sigma   (tau -1 = noise-free)"]
    for r in cells.itertuples():
        lines.append(f"  {r.network[:6]:<6} d{r.delay_k:>2} {r.arm:<13} std {r.noise_std:<3g} tau {r.tau:<4g}"
                     f" {r.reward:6.1f} n={r.n:<2} [{r.lo:6.1f}, {r.hi:6.1f}]  "
                     f"t900 {r.t900 / 1e6:6.0f}M  act {r.act_std:.3f}  "
                     f"sigma {r.sigma:.3f}")
    return "\n".join(lines) + "\n"


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

    got = runs["env_params.ctrl_dt"].dropna().unique()
    if len(got) and sorted(got) != [CTRL_MS / 1000]:
        raise SystemExit(f"{TASK}: expected {CTRL_MS} ms, runs carry {sorted(got)}")

    producer = get_producer("history")
    for want, spec in ((HISTORY_SPEC_ID, producer.spec(project=PROJECT)),
                       (TRAIN_SPEC_ID, producer.spec(project=PROJECT, keys=TRAIN_KEYS)),
                       (REG_SPEC_ID, producer.spec(project=PROJECT, keys=REG_KEYS))):
        got = producer.spec_id(spec)
        if got != want:
            raise SystemExit(f"history spec id drifted: {got} != {want}")

    store = Store()
    rows, curves, absent = [], [], []
    for _, run in runs.iterrows():
        try:
            evals = _frame(store, run["wandb_id"], HISTORY_SPEC_ID)
            train = _frame(store, run["wandb_id"], TRAIN_SPEC_ID)
            reg = _frame(store, run["wandb_id"], REG_SPEC_ID)
        except KeyError:
            absent.append(run["wandb_id"])
            continue
        row = build_row(run, evals, train, reg)
        rows.append(row)
        series = _series(evals, "eval/episode_reward/mean")
        # Curves only for the noisy runs, out to LONG_AT; the clean baselines' curves
        # are in ../walker-motor-noise/.
        if series is not None and row["noise_std"] > 0:
            for s in series[series.step <= LONG_AT].itertuples():
                curves.append({k: row[k] for k in
                               ("wandb_id", "network", "arm", "noise_std", "noise_tau",
                                "delay_k",
                                "seed_pair")} | {"step": int(s.step), "reward": s.value})
    if absent:
        raise SystemExit(
            f"{len(absent)} runs lack a history artifact; produce both specs:\n"
            f"  python -m vnl_experiments.artifacts ensure --kind history --runs "
            f"analysis/dm_control_suite/{HERE.name}/runs.csv "
            f"--set project='\"{PROJECT}\"' [--set keys='{TRAIN_KEYS}' | "
            f"--set keys='{REG_KEYS}']")

    key = ["network", "arm", "delay_k", "noise_std", "noise_tau", "seed_pair", "wandb_id"]
    df = pd.DataFrame(rows).sort_values(key).reset_index(drop=True)
    curves_df = (pd.DataFrame(curves).sort_values(key + ["step"])
                 .reset_index(drop=True))

    blocks = [summary_table(df), attrition(df)]
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
