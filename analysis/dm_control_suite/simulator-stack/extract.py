"""Does the 2026-11-09 environment update change what dm_control runs learn, and how fast?

The cluster venv rebuild of 2026-11-09 changed, for every dm_control run:

* **MuJoCo 3.6.0 -> 4.0.0** (warp-lang 1.12.0 -> 1.19.0, jax 0.10.1 -> 0.12.0.1), and
* **the physics backend, MJX-JAX -> MuJoCo Warp.** ``conf/env/dmc/*.yaml`` never set
  ``impl``, so it is playground's default, and playground 0.1.0 -> 0.2.0 flipped that
  default from ``"jax"`` to ``"warp"``. Nothing in the run name or tags says so;
  ``env_params.impl`` does.

This is a before/after comparison of the whole update, so the two changes are not
separated:

==================== ===========================================================
``mjx_jax_mujoco36`` before: dm_control runs 2026-09-09 .. 2026-11-06
``warp_mujoco40``    after: every run since 2026-11-09
==================== ===========================================================

Two HopperHop runs of 2026-09-15 used Warp on the old stack (an early test of the
backend). They are neither condition, and are excluded.

The condition is inferred from ``env_params.impl`` plus which buffer-size key the config
carries (playground 0.1.0 wrote ``nconmax``, 0.2.0 writes ``naconmax``), and then
*verified* per run against the package versions in the run's WandB
``requirements.txt`` (``stack_versions.csv``). A mismatch aborts.

Two questions, two cohorts:

**Reward (does the update change what is learned?).** Anchored on the post-update runs:
for every finished ``warp_mujoco40`` run (not WalkerWalkTerrain, which has no old-stack counterpart) take
its full config -- every ``config.*`` / ``env_params.*`` / ``net_params.*`` field -- and
select every run of *any* stack with that config, at any seed pair. Before comparing,
fields are normalised for schema drift so an absent key is not mistaken for a difference:
``motor_noise_*`` absent = 0 / post, ``privileged_critic`` absent = True (it was always
privileged before ``a37ce64``), ``servo_*`` absent = no servo, ``nconmax`` = ``naconmax``,
and cartpole's ``vision_config.*`` (renamed in 0.2.0, inert with ``vision = False``) is
dropped.
``total_steps`` and the eval / video cadences are ignored: nnx-ppo has no schedule, so a
1 G run read at 480 M is a 480 M run, and the cadences do not touch training
(``../README.md``). Within a cell, the same seed pair on two stacks shares its network
init exactly; the other seed pairs give the spread a simulator effect has to beat.

**Speed.** The same anchors, widened over delay, efference length, motor noise and critic
(none of which changes the cost of a step materially), and compared only within one GPU
model (``../../README.md`` §6). Read as the median per-iteration gauge from the ``timing``
artifact after dropping the first 10 % of iterations (compilation).

Reward is the mean ``eval/episode_reward/mean`` over (430 M, 480 M] -- the window rule
from ``../../README.md`` §6 -- from the ``history`` artifact; lifespan likewise, as a
check that every run sits on the bare-eval side of ``971ab99``.

Run it
------
    ../.venv/bin/python analysis/dm_control_suite/simulator-stack/extract.py
    ../.venv/bin/python analysis/dm_control_suite/simulator-stack/extract.py --sync --refresh
    ../.venv/bin/python analysis/dm_control_suite/simulator-stack/extract.py --check

``--refresh`` re-runs the selectors, so new post-update reruns pull in their own
pre-update twins automatically.

Outputs: ``data.csv`` (one row per run), ``curves.csv`` (reward-cell eval curves),
``cells.csv`` (each post-update run against its cell), ``repeats.csv`` (old-stack
same-seed reruns: the noise floor for a same-seed difference), ``speed.csv``. It also fetches ``requirements.txt`` for any selected run
not yet in ``stack_versions.csv`` (network); frozen mode never touches the network.
"""

from __future__ import annotations

import functools
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-delays"

#: The ids every sibling dm_control folder reads (see ``../walker-motor-noise/``).
HISTORY_SPEC_ID = "hist2000-8b97281e"
TIMING_SPEC_ID = "timing20000-3089bf7d"
REQUIRES = ["index", f"history:{HISTORY_SPEC_ID}", f"timing:{TIMING_SPEC_ID}"]

#: Commit of the eval-wrapper fix (``../README.md``). Every run here must descend from it.
EVAL_FIX_COMMIT = "971ab99"

READ_AT = 480_000_000
WINDOW = 50_000_000
#: Fraction of iterations dropped from the front of a timing series (XLA compilation).
WARMUP_FRACTION = 0.10

#: Schema additions: the value an absent key stands for.
ABSENT_MEANS = {
    "net_params.motor_noise_std": 0.0,
    "net_params.motor_noise_tau": 0,
    "net_params.motor_noise_site": "post_efference",
    "net_params.privileged_critic": True,
    "env_params.servo_kp": 0.0,
    "env_params.servo_damping_ratio": 1.0,
    "env_params.servo_center": "qpos0",
    "env_params.servo_unlimited_half_range": 0.0,
}
#: Never part of a configuration key: the stack itself, the budget, and cadences.
NOT_CONFIG = {
    "env_params.impl", "env_params.nconmax", "env_params.naconmax",
    "config.ppo.total_steps", "config.eval.every_steps", "config.video.every_steps",
    "config.video.render_kwargs.camera",
}
#: Cartpole's ``vision_config`` block: inert unless ``env_params.vision`` (never set here,
#: checked in ``_usable``), and its keys were renamed wholesale in playground 0.2.0.
INERT_PREFIX = "env_params.vision_config."
#: Additionally ignored when grouping for speed.
NOT_COST = {
    "seed", "config.seed", "delay_k", "efference_length", "net_params.delay_k",
    "net_params.efference_length", "net_params.motor_noise_std",
    "net_params.motor_noise_tau", "net_params.motor_noise_site",
    "net_params.privileged_critic",
}
CONFIG_PREFIXES = ("config", "env_params", "net_params", "servo", "distill")
CONFIG_TOPLEVEL = ("env", "seed", "delay_k", "efference_length", "network_class")

STACKS = {
    "mjx_jax_mujoco36": {"impl": "jax", "playground_key": "nconmax", "mujoco": "3.6.0"},
    "warp_mujoco40": {"impl": "warp", "playground_key": "naconmax", "mujoco": "4.0.0"},
}
VERSION_PACKAGES = ("mujoco", "mujoco-mjx", "warp-lang", "jax", "playground", "brax",
                    "flax", "wandb")


# --------------------------------------------------------------------------------------
# Configuration keys
# --------------------------------------------------------------------------------------

def _norm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return v


def _config_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns
                  if (c.split(".")[0] in CONFIG_PREFIXES or c in CONFIG_TOPLEVEL)
                  and c not in NOT_CONFIG and not c.startswith(INERT_PREFIX))


def _key(df: pd.DataFrame, drop: set[str] = frozenset()) -> pd.Series:
    """A hashable key per run over every config field not in ``drop``, schema-normalised."""
    x = df.copy()
    for col, value in ABSENT_MEANS.items():
        if col in x:
            x[col] = x[col].where(x[col].notna(), value)
    cols = [c for c in _config_columns(x) if c not in drop]
    return pd.Series([tuple(_norm(v) for v in row) for row in x[cols].itertuples(index=False)],
                     index=df.index)


def _short(key: tuple) -> str:
    return hashlib.sha1(repr(key).encode()).hexdigest()[:8]


def _stack(df: pd.DataFrame) -> pd.Series:
    """Condition from the config alone; verified against requirements.txt later."""
    impl = df["env_params.impl"]
    has_naconmax = df.get("env_params.naconmax", pd.Series(np.nan, index=df.index)).notna()
    out = pd.Series(None, index=df.index, dtype=object)
    out[(impl == "jax") & ~has_naconmax] = "mjx_jax_mujoco36"
    # impl == "warp" without naconmax: the 2026-09-15 HopperHop Warp test on the old
    # stack. Not part of the before/after question; left unlabelled, so excluded.
    out[(impl == "warp") & has_naconmax] = "warp_mujoco40"
    return out


@functools.cache
def _descends_from(commit: str, ancestor: str) -> bool:
    import subprocess

    return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, commit],
                          cwd=HERE, capture_output=True).returncode == 0


def _completed(df: pd.DataFrame) -> pd.Series:
    return df["state"].eq("finished") | (df["summary._step"].fillna(0) >= READ_AT)


def _usable(df: pd.DataFrame) -> pd.Series:
    """Current schema, a known stack, not the terrain task, trained to READ_AT, and on
    the bare-eval side of the eval-wrapper fix."""
    vision = df.get("env_params.vision", pd.Series(False, index=df.index))
    post_fix = df["git_commit"].map(
        lambda c: isinstance(c, str) and _descends_from(c, EVAL_FIX_COMMIT))
    return (df["env_params.impl"].notna() & _stack(df).notna()
            & ~df["env"].eq("WalkerWalkTerrain") & ~vision.eq(True) & _completed(df)
            & post_fix)


def _anchors(df: pd.DataFrame) -> pd.DataFrame:
    return df[_usable(df) & _stack(df).eq("warp_mujoco40") & df["state"].eq("finished")]


def _selector(stack: str):
    def selector(df: pd.DataFrame) -> pd.Series:
        anchors = _anchors(df)
        cost_key = _key(df, NOT_COST)
        return (_usable(df) & _stack(df).eq(stack)
                & cost_key.isin(set(cost_key[anchors.index])))
    return selector


CONDITIONS = {stack: _selector(stack) for stack in STACKS}

INVARIANTS = [
    "env",
    "network_class",
    "config.ppo.n_envs",
    "config.ppo.rollout_length",
    "config.ppo.learning_rate",
    "config.ppo.n_epochs",
    "config.ppo.n_minibatches",
    "net_params.actor_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "env_params.reward_scale",
    "env_params.episode_length",
    "env_params.ctrl_dt",
    "env_params.sim_dt",
    "config.eval.n_envs",
    "config.eval.max_episode_length",
    "repos.nnx_ppo.commit",
    "repos.vnl_playground.commit",
    "repos.nnx_ppo.dirty",
    "repos.vnl_playground.dirty",
    "repos.vnl_experiments.dirty",
    "git_commit",
    "python",
    "cuda_version",
    "env_params.impl",
]


# --------------------------------------------------------------------------------------
# Package versions (the verification of the condition label)
# --------------------------------------------------------------------------------------

def _fetch_versions(wandb_id: str) -> dict:
    import tempfile
    import wandb

    api = wandb.Api(timeout=60)
    run = api.run(f"{PROJECT}/{wandb_id}")
    with tempfile.TemporaryDirectory() as tmp:
        path = run.file("requirements.txt").download(root=tmp, replace=True).name
        text = Path(path).read_text()
    found = {"wandb_id": wandb_id}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)$", line.strip())
        if m and m.group(1).lower().replace("_", "-") in VERSION_PACKAGES:
            found[m.group(1).lower().replace("_", "-")] = m.group(2)
    return found


def stack_versions(runs: pd.DataFrame, *, fetch: bool) -> pd.DataFrame:
    path = HERE / "stack_versions.csv"
    have = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=["wandb_id"])
    missing = sorted(set(runs["wandb_id"]) - set(have["wandb_id"]))
    if missing and not fetch:
        raise SystemExit(f"{len(missing)} runs are not in stack_versions.csv; run with "
                         f"--refresh to fetch their requirements.txt: {missing[:5]}...")
    if missing:
        print(f"fetching requirements.txt for {len(missing)} runs ...")
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(_fetch_versions, missing))
        have = pd.concat([have, pd.DataFrame(got)], ignore_index=True)
        have = have[have["wandb_id"].isin(runs["wandb_id"])].sort_values("wandb_id")
        have.to_csv(path, index=False)
    return have.set_index("wandb_id")


def verify_stacks(runs: pd.DataFrame, versions: pd.DataFrame) -> None:
    bad = []
    for _, run in runs.iterrows():
        want = STACKS[run["condition"]]["mujoco"]
        got = versions.loc[run["wandb_id"]].get("mujoco")
        if got != want:
            bad.append(f"{run['wandb_id']} ({run['condition']}): mujoco {got}")
    if bad:
        raise SystemExit("condition label disagrees with requirements.txt:\n  "
                         + "\n  ".join(bad))


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------

def _read(store: Store, kind: str, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup(kind, wandb_id, spec_id)
    if entry is None:
        raise KeyError(wandb_id)
    try:
        frame = pd.read_csv(store.root / entry.path)
    except pd.errors.EmptyDataError:
        return None
    return None if frame.empty else frame


def _window(frame: pd.DataFrame | None, key: str) -> tuple[float, int]:
    if frame is None or key not in frame:
        return np.nan, 0
    s = frame.dropna(subset=[key])
    s = s[(s["_step"] > READ_AT - WINDOW) & (s["_step"] <= READ_AT)][key]
    return (float(s.mean()), len(s)) if len(s) else (np.nan, 0)


def _median_after_warmup(frame: pd.DataFrame | None, key: str) -> float:
    if frame is None or key not in frame:
        return np.nan
    s = frame.dropna(subset=[key])
    s = s[s["_step"] <= READ_AT]
    s = s.iloc[int(len(s) * WARMUP_FRACTION):]
    return float(s[key].median()) if len(s) else np.nan


# --------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------

def build_row(run: pd.Series, hist: pd.DataFrame | None, timing: pd.DataFrame | None,
              versions: pd.Series) -> dict:
    reward, n_reward = _window(hist, "eval/episode_reward/mean")
    lifespan, _ = _window(hist, "eval/lifespan/mean")
    return {
        "condition": run["condition"],
        "task": run["env"],
        "network": run["network_class"],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "created_at": run["created_at"],
        "state": run["state"],
        "git_commit": run["git_commit"],
        "gpu": str(run["gpu"]).replace("NVIDIA ", ""),
        "delay_k": int(run["net_params.delay_k"]),
        "efference_length": int(run["net_params.efference_length"]),
        "seed_pair": f"i{int(run['seed'])}/p{int(run['config.seed'])}",
        "config_id": run["config_id"],
        "twin_id": run["twin_id"],
        "cost_id": run["cost_id"],
        "in_reward_cell": bool(run["in_reward_cell"]),
        "total_steps": run["config.ppo.total_steps"],
        "actual_step": run["summary._step"],
        "eval_every_steps": run["config.eval.every_steps"],
        "reward_source": "dmc_eval",
        "reward_480M": reward,
        "n_eval_points": n_reward,
        "lifespan_480M": lifespan,
        "train_sps": _median_after_warmup(timing, "train_sps"),
        "eval_sps": _median_after_warmup(timing, "eval_sps"),
        **{f"pkg_{p}": versions.get(p) for p in VERSION_PACKAGES},
    }


def curve_rows(row: dict, hist: pd.DataFrame | None) -> list[dict]:
    if hist is None or not row["in_reward_cell"]:
        return []
    s = hist.dropna(subset=["eval/episode_reward/mean"])
    s = s[s["_step"] <= READ_AT]
    return [{"wandb_id": row["wandb_id"], "condition": row["condition"],
             "config_id": row["config_id"], "seed_pair": row["seed_pair"],
             "step": int(r["_step"]), "reward": float(r["eval/episode_reward/mean"])}
            for _, r in s.iterrows()]


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per reward cell: the post-update runs against the pre-update seed spread."""
    out = []
    for cid, g in df[df["in_reward_cell"]].groupby("config_id"):
        old = g[g["condition"] == "mjx_jax_mujoco36"]
        for _, w in g[g["condition"] != "mjx_jax_mujoco36"].iterrows():
            twin = old[old["seed_pair"] == w["seed_pair"]]
            others = old[old["seed_pair"] != w["seed_pair"]]["reward_480M"]
            out.append({
                "config_id": cid, "task": w["task"], "delay_k": w["delay_k"],
                "condition": w["condition"], "wandb_id": w["wandb_id"],
                "seed_pair": w["seed_pair"], "reward_480M": w["reward_480M"],
                "twin_wandb_id": ",".join(twin["wandb_id"]) if len(twin) else None,
                "twin_n": len(twin),
                "twin_reward_480M": twin["reward_480M"].mean() if len(twin) else np.nan,
                "twin_min": twin["reward_480M"].min() if len(twin) else np.nan,
                "twin_max": twin["reward_480M"].max() if len(twin) else np.nan,
                "old_other_seeds_n": len(others),
                "old_other_seeds_mean": others.mean(),
                "old_other_seeds_sd": others.std(ddof=1) if len(others) > 1 else np.nan,
                "old_other_seeds_min": others.min(),
                "old_other_seeds_max": others.max(),
            })
    t = pd.DataFrame(out)
    if len(t):
        t["minus_twin"] = t["reward_480M"] - t["twin_reward_480M"]
        t["z_vs_other_seeds"] = ((t["reward_480M"] - t["old_other_seeds_mean"])
                                 / t["old_other_seeds_sd"])
    return t


def repeat_table(df: pd.DataFrame) -> pd.DataFrame:
    """Old-stack reruns of one exact configuration *and* seed pair (same ``twin_id``).

    The noise floor for a same-seed before/after difference: the same network init on the
    same stack, differing only in GPU / node / XLA nondeterminism. Any run in the
    selection counts, not only the reward cells.
    """
    old = df[df["condition"] == "mjx_jax_mujoco36"]
    rep = old.groupby("twin_id").filter(lambda g: len(g) >= 2)
    t = rep.groupby("twin_id").agg(
        task=("task", "first"), delay_k=("delay_k", "first"),
        seed_pair=("seed_pair", "first"), in_reward_cell=("in_reward_cell", "first"),
        n=("wandb_id", "size"), wandb_ids=("wandb_id", ",".join),
        gpus=("gpu", lambda g: ",".join(g)),
        reward_min=("reward_480M", "min"), reward_max=("reward_480M", "max"))
    t["reward_range"] = t["reward_max"] - t["reward_min"]
    return t.reset_index().sort_values(["task", "delay_k", "twin_id"], ignore_index=True)


def speed_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["task", "gpu", "condition"])
    t = g.agg(n=("wandb_id", "size"), train_sps=("train_sps", "median"),
              eval_sps=("eval_sps", "median"),
              train_sps_min=("train_sps", "min"), train_sps_max=("train_sps", "max"),
              eval_sps_min=("eval_sps", "min"), eval_sps_max=("eval_sps", "max"))
    t = t.reset_index()
    base = t[t["condition"] == "mjx_jax_mujoco36"].set_index(["task", "gpu"])
    for col in ("train_sps", "eval_sps"):
        t[f"{col}_vs_jax"] = [r[col] / base.loc[(r["task"], r["gpu"]), col]
                              if (r["task"], r["gpu"]) in base.index else np.nan
                              for _, r in t.iterrows()]
    return t


# --------------------------------------------------------------------------------------

def main() -> None:
    args = pipeline.parse_args(__doc__)
    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project or PROJECT)

    # Three keys over the schema-normalised config: the exact configuration including
    # both seeds (same twin_id = same draw), the configuration without seeds (a reward
    # cell), and the configuration without anything that does not change step cost.
    keyed = runs.copy()
    keyed["twin_id"] = _key(runs).map(_short)
    keyed["config_id"] = _key(runs, {"seed", "config.seed"}).map(_short)
    keyed["cost_id"] = _key(runs, NOT_COST).map(_short)
    warp_cells = set(keyed.loc[keyed["condition"] != "mjx_jax_mujoco36", "config_id"])
    keyed["in_reward_cell"] = keyed["config_id"].isin(warp_cells)

    bad_commit = sorted({c for c in keyed["git_commit"].dropna().unique()
                         if not _descends_from(c, EVAL_FIX_COMMIT)})
    if bad_commit:
        raise SystemExit(f"commits not descended from the eval-wrapper fix "
                         f"{EVAL_FIX_COMMIT} (or unknown locally -- git fetch): {bad_commit}")

    pipeline.write_coverage(runs, REQUIRES, HERE)

    for kind, want in (("history", HISTORY_SPEC_ID), ("timing", TIMING_SPEC_ID)):
        producer = get_producer(kind)
        got = producer.spec_id(producer.spec(project=PROJECT))
        if got != want:
            raise SystemExit(f"{kind} spec id drifted: {got} != {want}")

    versions = stack_versions(keyed, fetch=args.refresh)
    verify_stacks(keyed, versions)

    store = Store()
    rows, curves, absent = [], [], []
    for _, run in keyed.iterrows():
        try:
            hist = _read(store, "history", run["wandb_id"], HISTORY_SPEC_ID)
            timing = _read(store, "timing", run["wandb_id"], TIMING_SPEC_ID)
        except KeyError:
            absent.append(run["wandb_id"])
            continue
        row = build_row(run, hist, timing, versions.loc[run["wandb_id"]])
        rows.append(row)
        curves.extend(curve_rows(row, hist))
    if absent:
        rel = f"analysis/dm_control_suite/{HERE.name}/runs.csv"
        raise SystemExit(
            f"{len(absent)} runs lack an artifact; produce them with\n"
            f"  python -m vnl_experiments.artifacts ensure --kind history --runs {rel} "
            f"--set project='\"{PROJECT}\"'\n"
            f"  python -m vnl_experiments.artifacts ensure --kind timing --runs {rel} "
            f"--set project='\"{PROJECT}\"'")

    df = pd.DataFrame(rows).sort_values(["task", "condition", "delay_k", "seed_pair",
                                         "wandb_id"], ignore_index=True)
    off_era = df[df["in_reward_cell"] & (df["lifespan_480M"] < 999)]
    if len(off_era):
        raise SystemExit(f"lifespan < 1000 in the readout window -- wrapped eval? "
                         f"{off_era['wandb_id'].tolist()}")

    cells = cell_table(df)
    repeats = repeat_table(df)
    speed = speed_table(df)
    pd.set_option("display.width", 200)
    print("\n== reward cells ==\n" + cells.to_string(index=False, float_format="%.1f"))
    print("\n== old-stack same-seed repeats ==\n"
          + repeats.drop(columns=["wandb_ids", "gpus"]).to_string(index=False,
                                                                  float_format="%.1f"))
    print("\n== speed (median per-iteration gauge, per GPU) ==\n"
          + speed.to_string(index=False, float_format="%.3g"))

    reward_runs = runs[runs["wandb_id"].isin(df.loc[df["in_reward_cell"], "wandb_id"])]
    report = ("# Reward cohort (in_reward_cell)\n"
              + comparability_report(reward_runs, invariant_cols=INVARIANTS,
                                     group_col="condition")
              + "\n\n# Speed cohort (all runs)\n"
              + comparability_report(runs, invariant_cols=INVARIANTS,
                                     group_col="condition"))
    if not args.check:
        (HERE / "comparability.txt").write_text(report)

    ok = all([
        pipeline.write_csv(df, HERE / "data.csv", check=args.check),
        pipeline.write_csv(pd.DataFrame(curves), HERE / "curves.csv", check=args.check),
        pipeline.write_csv(cells, HERE / "cells.csv", check=args.check),
        pipeline.write_csv(repeats, HERE / "repeats.csv", check=args.check),
        pipeline.write_csv(speed, HERE / "speed.csv", check=args.check),
    ])
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
