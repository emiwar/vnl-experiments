"""Does an explicitly-trained forward model represent the current state better than one the policy gradient shapes on its own?

Both arms are the same network, ``RodentForwardModel``: a predictor that maps [delayed
proprioception + efference copy] to a prediction ``p-hat``, and a decoder that acts on
[task latent + p-hat]. They differ only in how the predictor is trained:

* **explicit** (``fm_loss_weight=1``, ``detach_prediction=True``) -- the predictor is shaped
  *only* by a self-supervised L2 against the true current proprioception, and the policy
  gradient cannot reach it.
* **implicit** (``fm_loss_weight=0``, ``detach_prediction=False``) -- no L2 at all; the
  predictor is an ordinary policy layer trained by the policy gradient.

Identical architecture means identical layer trees, so a single depth axis compares them
layer by layer. We linearly decode the current (un-delayed) proprioception, and the part of
it the delayed input cannot supply (``delta``), from every layer -- see
:mod:`vnl_experiments.probes.linear_decoding`. Measured at two budgets (600 M and 2 G steps)
and four delays (0/10/20/50), on the held-out ``old_eval`` clips.

This re-asks the question of ``analysis/implicit-forward-model/`` on the current setting --
new walker XML and ``reference_root`` targets -- and narrows it to the same-architecture
contrast, so what varies is the *loss*, not the wiring. Note what that costs: there is no
efference-only (no-predictor) arm here, so this cannot answer the original
"does an architecture with no forward model build one anyway?".

All artifacts are the post-fix (v2) generation. Every earlier offline eval and activation
recording of these runs was made on the *wrong body* -- ``parse_env_config`` replaced the
run's ``rodent_no_tail_collisions.xml`` with the local default ``rodent.xml`` -- so the v1
spec ids must never be mixed in here. See the walker-XML entry in ``analysis/README.md``.

Run it
------
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-probe/extract.py            # frozen
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-probe/extract.py --select-only
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-probe/extract.py --sync --refresh
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-probe/extract.py --redecode --jobs 4
    ../.venv/bin/python analysis/explicit-vs-implicit-fm-probe/extract.py --check

Frozen is the default. Decoding one recording takes minutes, so results are cached per run
and a frozen rebuild reads the cache -- or, on a fresh clone with no artifacts at all, the
committed ``data.csv`` itself. ``--redecode`` forces the fits from the HDF5s and is the real
reproduction test. This is the only script here that reads the index or the store;
``plot.py`` reads the CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import get_producer
from vnl_experiments.artifacts.store import Store
from vnl_experiments.probes import linear_decoding as ld
from vnl_experiments.probes import pathways
from vnl_experiments.wandb_utils import comparability_report, index, pipeline

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".decode-cache"
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

# --------------------------------------------------------------------------------------
# Artifact identity
# --------------------------------------------------------------------------------------

#: v2 = post walker-XML fix. Pinning these is what keeps wrong-body files out by
#: construction; `assert_spec_ids` makes a future producer bump a loud failure here.
ACT_SPEC_ID = "act-old_eval-fa8b8144"
EVAL_SPEC_ID = "eval3ds-347333e3"
HISTORY_SPEC_ID = "hist2000-fc46b078"      # the 11-key spec: carries fm_pred_mse
REWARD_SPEC_ID = "hist2000-09fea177"       # the 4-key spec: reward/lifespan/throughput

#: Key ORDER is load-bearing: `normalise_spec` sorts dict keys but preserves list order, so
#: a reordering silently mints a different spec_id. This exact list hashes to fc46b078.
HISTORY_KEYS = [
    "eval/episode_reward/mean", "eval/lifespan/mean",
    "eval/net/3/action/1/fm_pred_mse/mean", "eval/net/3/action/1/fm_pred_mse/std",
    "net/3/action/1/fm_pred_mse/p25", "net/3/action/1/fm_pred_mse/p50",
    "net/3/action/1/fm_pred_mse/p75",
    "eval/net/3/action/1/decoder/5/sigma/mean",
    "eval/net/3/action/0/task_obs/6/kl_divergence/mean",
    "eval/env/terminations/any/mean", "eval/env/joint_l2_error/mean",
]

REQUIRES = ["index", f"activations:{ACT_SPEC_ID}", f"eval:{EVAL_SPEC_ID}",
            f"history:{HISTORY_SPEC_ID}", f"history:{REWARD_SPEC_ID}"]

#: The body every artifact here must have been produced on.
EXPECTED_XML = "rodent_no_tail_collisions.xml"


def assert_spec_ids() -> None:
    """Fail loudly if a producer's spec_id has drifted from the committed constants."""
    checks = [
        ("activations", get_producer("activations").spec(), ACT_SPEC_ID),
        ("eval", get_producer("eval").spec(), EVAL_SPEC_ID),
        ("history", get_producer("history").spec(keys=HISTORY_KEYS), HISTORY_SPEC_ID),
        ("history", get_producer("history").spec(), REWARD_SPEC_ID),
    ]
    drift = []
    for kind, spec, expected in checks:
        got = get_producer(kind).spec_id(spec)
        if got != expected:
            drift.append(f"{kind}: got {got}, expected {expected}")
    if drift:
        raise SystemExit(
            "spec_ids have drifted from the committed constants:\n  "
            + "\n  ".join(drift)
            + "\nThe artifacts this analysis reads were made by a different producer "
              "VERSION. Update the constants deliberately, re-produce, and say so in "
              "report.md -- do not silently repoint at a different generation of data.")


# --------------------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------------------

NEW_XML = "rodent_no_tail_collisions.xml"
DELAYS = (0, 10, 20, 50)
#: total_steps -> the `summary._step` a completed run of that budget reaches.
BUDGETS = {600_000_000: 600_064_000, 2_000_000_000: 2_000_076_800}

#: `failed` is admitted *because* the step gate is the real inclusion criterion. All four
#: explicit 600 M runs are `failed`: they completed 600 064 000 steps and died afterwards in
#: the post-training inline eval, which is why they alone lack `final_eval/*`. A run that
#: died *during* training cannot reach the step count, so these two gates together admit the
#: former and exclude the latter. Precedent: collision-model-xml, xml-ceiling-vs-convergence.
ACCEPTED_STATES = ("finished", "failed")

STD_ARCH = {
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.fm_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
}


def _cell(arm: str, budget: int):
    def selector(df: pd.DataFrame) -> pd.Series:
        is_fm = df["tags"].fillna("").str.split(",").apply(lambda t: "ForwardModel" in t)
        if arm == "explicit":
            # `ne(False)`, not `fillna(True)`: unset means the constructor default (True).
            knobs = (df["fm_loss_weight"] == 1) & df["detach_prediction"].ne(False)
        else:
            knobs = (df["fm_loss_weight"] == 0) & (df["detach_prediction"] == False)  # noqa: E712
        mask = (
            (df["env"] == "AbsoluteImitation")
            & (df["seed"] == 42)
            # The discriminator against the d02b854a min_std=0.25 sweep, which otherwise
            # matches these delays exactly.
            & (df["net_params.min_std"] == 0.1)
            & (df["net_params.latent_size"] == 32)
            & (df["config.ppo.total_steps"] == budget)
            & (df["summary._step"] == BUDGETS[budget])
            & df["state"].isin(ACCEPTED_STATES)
            & is_fm & knobs
            # env_params, never the training script at the run's commit (README §6).
            & df["env_params.walker_xml_path"].astype(str).str.contains(NEW_XML)
            & (df["env_params.body_target_frame"] == "reference_root")
            & (df["env_params.torque_actuators"] == True)  # noqa: E712
            & (df["delay_k"] == df["efference_length"])
            & df["delay_k"].isin(DELAYS)
        )
        for column, value in STD_ARCH.items():
            mask &= df[column] == value
        return mask
    return selector


CONDITIONS = {
    "expfm_600m": _cell("explicit", 600_000_000),
    "pgfm_600m": _cell("implicit", 600_000_000),
    "expfm_2g": _cell("explicit", 2_000_000_000),
    "pgfm_2g": _cell("implicit", 2_000_000_000),
}
ARM_OF = {"expfm_600m": "explicit", "expfm_2g": "explicit",
          "pgfm_600m": "implicit", "pgfm_2g": "implicit"}
NETWORK_OF = {"expfm_600m": "forward_model", "expfm_2g": "forward_model",
              "pgfm_600m": "pg_forward_model", "pgfm_2g": "pg_forward_model"}
BUDGET_OF = {"expfm_600m": 600_000_000, "pgfm_600m": 600_000_000,
             "expfm_2g": 2_000_000_000, "pgfm_2g": 2_000_000_000}
LABEL_OF = {600_000_000: "600M", 2_000_000_000: "2G"}

#: Every figure assumes one run per (condition, delay). The exception is deliberate:
#: cgs8q5gj and kwk401pl are the same config, seed and commit at implicit/600M/delay-10, and
#: keeping both gives this cohort its only run-to-run noise estimate -- everything else is a
#: single seed. Dropping one would need a tie-break rule that silently discards a run.
EXPECTED_GRID = {(condition, delay): 1 for condition in CONDITIONS for delay in DELAYS}
EXPECTED_GRID[("pgfm_600m", 10)] = 2


def assert_grid(runs: pd.DataFrame) -> None:
    counts = {k: int(v) for k, v in
              runs.groupby(["condition", "delay_k"]).size().items()}
    if counts != EXPECTED_GRID:
        added = {k: v for k, v in counts.items() if EXPECTED_GRID.get(k) != v}
        missing = {k: v for k, v in EXPECTED_GRID.items() if counts.get(k) != v}
        raise SystemExit(
            f"cohort grid changed.\n  now:      {added}\n  expected: {missing}\n"
            f"Every figure here assumes one run per (condition, delay) except the "
            f"documented pgfm_600m/delay-10 pair. Update EXPECTED_GRID deliberately and "
            f"say why in report.md.")


# --------------------------------------------------------------------------------------
# Comparability
# --------------------------------------------------------------------------------------

INVARIANTS = [
    "env", "seed",
    "net_params.latent_size", "net_params.kl_weight", "net_params.min_std",
    "net_params.latent_min_std", "net_params.std_scale", "net_params.normalize_obs",
    "net_params.enc_hidden_sizes", "net_params.dec_hidden_sizes",
    "net_params.fm_hidden_sizes", "net_params.critic_hidden_sizes",
    "env_params.clip_length", "env_params.clip_set", "env_params.ctrl_dt",
    "env_params.sim_dt", "env_params.solver", "env_params.iterations",
    "env_params.ls_iterations", "env_params.njmax", "env_params.naconmax",
    "env_params.rescale_factor", "env_params.mujoco_impl",
    "env_params.walker_xml_path", "env_params.torque_actuators",
    "env_params.body_target_frame",
    "config.ppo.n_envs", "config.ppo.learning_rate", "config.ppo.rollout_length",
    "config.ppo.n_epochs", "config.ppo.n_minibatches", "config.ppo.clip_range",
    "config.ppo.discounting_factor", "config.ppo.gae_lambda",
    "config.ppo.total_steps", "summary._step", "git_commit", "gpu",
]

# --------------------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------------------

CURVE_COLUMNS = {
    "eval/episode_reward/mean": "reward_mean",
    "eval/lifespan/mean": "lifespan_mean",
    "eval/net/3/action/1/fm_pred_mse/mean": "fm_mse_eval",
    "net/3/action/1/fm_pred_mse/p25": "fm_mse_train_p25",
    "net/3/action/1/fm_pred_mse/p50": "fm_mse_train_p50",
    "net/3/action/1/fm_pred_mse/p75": "fm_mse_train_p75",
    "eval/net/3/action/1/decoder/5/sigma/mean": "action_sigma",
    "eval/net/3/action/0/task_obs/6/kl_divergence/mean": "encoder_kl",
    "eval/env/joint_l2_error/mean": "joint_l2_error",
    "eval/env/terminations/any/mean": "termination_rate",
}

#: Eval runs every ~10 M steps and a single point moves 1-2 % on GPU-nondeterministic
#: physics, so a point value is not a measurement; the mean of the five in (X-50M, X] is.
WINDOW = 50_000_000


def read_history(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    if entry is None:
        return None
    frame = pd.read_csv(store.root / entry.path)
    return frame.rename(columns={"_step": "step", **CURVE_COLUMNS}).sort_values("step")


def window_mean(curve: pd.DataFrame | None, column: str, step: int,
                window: int = WINDOW) -> float | None:
    """Mean of the points in ``(step - window, step]``; None if fewer than three.

    A one- or two-point "average" would reintroduce the noise this exists to remove.
    """
    if curve is None or column not in curve:
        return None
    sub = curve[(curve["step"] > step - window) & (curve["step"] <= step)].dropna(
        subset=[column])
    return float(sub[column].mean()) if len(sub) >= 3 else None


# --------------------------------------------------------------------------------------
# Decoding, with a cache
# --------------------------------------------------------------------------------------

DECODE_SEED = 0


def decode_key() -> str:
    """Hash of everything that affects the fits, so a settings change invalidates the cache."""
    payload = json.dumps({"version": ld.DECODE_VERSION,
                          "probe_set": pathways.PROBE_SET_VERSION,
                          "seed": DECODE_SEED,
                          "test_frac": ld.DEFAULT_TEST_FRAC,
                          "val_frac": ld.DEFAULT_VAL_FRAC,
                          "max_samples": ld.DEFAULT_MAX_SAMPLES,
                          "lambdas": list(ld.LAMBDA_GRID)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def cache_path(wandb_id: str) -> Path:
    return CACHE / wandb_id / f"{ACT_SPEC_ID}__{decode_key()}.csv"


def assert_artifact_body(entry, run: pd.Series, kind: str) -> None:
    """The artifact must say it used the body this run trained on.

    Pre-fix artifacts carry no stamp at all; that is the signal, so absence is an error here
    rather than something to shrug at.
    """
    stamp = (entry.resolved or {}).get("walker_xml_path")
    if stamp is None:
        raise SystemExit(
            f"{kind} artifact for {run['wandb_id']} has no resolved.walker_xml_path, which "
            f"means it predates the 2026-08-18 walker-XML fix and was produced on the wrong "
            f"body. Re-produce it (see analysis/README.md).")
    if stamp != EXPECTED_XML:
        raise SystemExit(f"{kind} artifact for {run['wandb_id']} was produced on {stamp}, "
                         f"expected {EXPECTED_XML}.")


def decode_run(args: tuple) -> tuple[str, list[dict]]:
    """Decode one recording. Top-level so ProcessPoolExecutor can pickle it."""
    wandb_id, path, meta = args
    return wandb_id, ld.decode_file(path, meta=meta, seed=DECODE_SEED)


def probe_rows(runs: pd.DataFrame, store: Store, *, redecode: bool,
               jobs: int, only: set[str] | None) -> pd.DataFrame:
    """One row per (run, probe, target), from the cache, the committed CSV, or the HDF5s."""
    committed = None
    data_path = HERE / "data.csv"
    if data_path.exists():
        committed = pd.read_csv(data_path)
        stale = (committed.get("decode_version", -1) != ld.DECODE_VERSION) | \
                (committed.get("probe_set_version", -1) != pathways.PROBE_SET_VERSION)
        committed = committed[~stale]

    rows: list[dict] = []
    todo: list[tuple] = []
    for _, run in runs.iterrows():
        wandb_id = run["wandb_id"]
        if only and wandb_id not in only:
            continue
        meta = {"condition": run["condition"], "arm": ARM_OF[run["condition"]],
                "budget": BUDGET_OF[run["condition"]],
                "budget_label": LABEL_OF[BUDGET_OF[run["condition"]]],
                "wandb_id": wandb_id, "wandb_name": run["wandb_name"]}

        cached = cache_path(wandb_id)
        if not redecode and cached.exists():
            rows.extend(pd.read_csv(cached).to_dict("records"))
            print(f"  {wandb_id}: cache")
            continue
        if not redecode and committed is not None and \
                (committed["wandb_id"] == wandb_id).any():
            sub = committed[committed["wandb_id"] == wandb_id]
            rows.extend(sub.to_dict("records"))
            cached.parent.mkdir(parents=True, exist_ok=True)
            sub.to_csv(cached, index=False)          # warm the cache from the snapshot
            print(f"  {wandb_id}: data.csv")
            continue

        entry = store.lookup("activations", wandb_id, ACT_SPEC_ID)
        if entry is None:
            print(f"  {wandb_id}: NO ACTIVATIONS ({ACT_SPEC_ID}) - see coverage.txt")
            continue
        assert_artifact_body(entry, run, "activations")
        todo.append((wandb_id, store.root / entry.path, meta))

    def store_result(wandb_id: str, run_rows: list[dict], done: int) -> None:
        """Cache each recording as it finishes, not at the end of the batch.

        Collecting the whole batch first meant a 40-minute run showed no progress and lost
        everything on interruption. One CSV per recording as it lands is resumable.
        """
        cache_path(wandb_id).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(run_rows).to_csv(cache_path(wandb_id), index=False)
        rows.extend(run_rows)
        print(f"  [{done}/{len(todo)}] {wandb_id}: decoded ({len(run_rows)} rows)",
              flush=True)

    if todo:
        # These fits are BLAS-bound and numpy is already multi-threaded, so process-level
        # parallelism oversubscribes: 4 workers on a 32-core box drove the load average to
        # 110 and finished nothing in 40 minutes. jobs=1 gives BLAS every core with no
        # contention and is the right default; jobs>1 only pays off if the BLAS thread count
        # is capped to cores/jobs in the environment.
        blas_threads = os.environ.get("OMP_NUM_THREADS")
        if jobs > 1 and blas_threads is None:
            print(f"  WARNING: --jobs {jobs} with OMP_NUM_THREADS unset oversubscribes "
                  f"({os.cpu_count()} cores x {jobs} workers of BLAS threads). Either use "
                  f"--jobs 1, or set OMP_NUM_THREADS={max(1, (os.cpu_count() or 8)//jobs)}.",
                  flush=True)
        print(f"  decoding {len(todo)} recording(s), {jobs} worker(s), "
              f"OMP_NUM_THREADS={blas_threads or 'unset'} ...", flush=True)

        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(decode_run, item): item[0] for item in todo}
                for done, future in enumerate(as_completed(futures), start=1):
                    wandb_id, run_rows = future.result()
                    store_result(wandb_id, run_rows, done)
        else:
            for done, item in enumerate(todo, start=1):
                wandb_id, run_rows = decode_run(item)
                store_result(wandb_id, run_rows, done)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Per-run reward table
# --------------------------------------------------------------------------------------

def reward_row(run: pd.Series, store: Store) -> dict:
    """One row of data_reward.csv: the reward every figure annotates from."""
    condition = run["condition"]
    wandb_id = run["wandb_id"]
    budget = BUDGET_OF[condition]
    step = BUDGETS[budget]

    reward_curve = read_history(store, wandb_id, REWARD_SPEC_ID)
    fm_curve = read_history(store, wandb_id, HISTORY_SPEC_ID)
    eval_entry = store.lookup("eval", wandb_id, EVAL_SPEC_ID)
    act_entry = store.lookup("activations", wandb_id, ACT_SPEC_ID)

    row = {
        "condition": condition, "arm": ARM_OF[condition],
        "network": NETWORK_OF[condition], "budget": budget,
        "budget_label": LABEL_OF[budget], "delay_k": int(run["delay_k"]),
        "wandb_id": wandb_id, "wandb_name": run["wandb_name"],
        "state": run["state"], "git_commit": str(run["git_commit"])[:8],
        "gpu": run.get("gpu"), "created_at": run.get("created_at"),
        "actual_step": run.get("summary._step"), "runtime_s": run.get("runtime_s"),
        "have_activations": act_entry is not None,
        "have_eval": eval_entry is not None,
        "have_history": fm_curve is not None,
        # The headline: correct body, complete for every run, noise-robust.
        "reward_window": window_mean(reward_curve, "reward_mean", step),
        "lifespan_window": window_mean(reward_curve, "lifespan_mean", step),
        # Different measurement (in-memory weights vs newest checkpoint on disk); kept for
        # the record and NEVER mixed with the offline numbers in one figure.
        "inline_old_eval_reward": pipeline.first_present(
            run, "summary.final_eval/old_eval/episode_reward/mean"),
    }

    if eval_entry is not None:
        assert_artifact_body(eval_entry, run, "eval")
        record = json.loads((store.root / eval_entry.path).read_text())
        datasets = record.get("datasets", {})
        for name in ("train", "old_eval", "new_eval"):
            data = datasets.get(name)
            if not data:
                continue
            row[f"reward_{name}"] = data["episode_reward"]["mean"]
            if name == "old_eval":
                row["reward_old_eval_std"] = data["episode_reward"]["std"]
                row["survived_old_eval"] = data["termination_rate"]["survived"]
                row["lifespan_old_eval"] = data["lifespan_steps"]["mean"]
                row["reward_per_step_old_eval"] = (
                    data["episode_reward"]["mean"] / data["lifespan_steps"]["mean"]
                    if data["lifespan_steps"]["mean"] else None)
            # The path embeds the layer index, so match by suffix.
            fm = next((v for k, v in data.get("net_metrics", {}).items()
                       if k.endswith("fm_pred_mse")), None)
            row[f"fm_mse_{name}"] = fm
        if row.get("reward_train"):
            row["generalization_ratio"] = row.get("reward_old_eval") / row["reward_train"]

    if fm_curve is not None:
        row["fm_mse_curve_final"] = window_mean(fm_curve, "fm_mse_eval", step)
        row["fm_mse_curve_at_600M"] = window_mean(fm_curve, "fm_mse_eval", 600_064_000)
        row["reward_curve_at_600M"] = window_mean(fm_curve, "reward_mean", 600_064_000)
        for column in ("action_sigma", "encoder_kl", "joint_l2_error", "termination_rate"):
            row[f"{column}_final"] = window_mean(fm_curve, column, step)

    if act_entry is not None:
        with ld.Recording(store.root / act_entry.path) as rec:
            row.update(ld.valid_stats(rec.dones, rec.delay_k))
            row["recorded_step"] = int(rec.attrs["step"])
            row["n_clips"] = int(rec.attrs["n_clips"])
    return row


def curve_rows(runs: pd.DataFrame, store: Store) -> pd.DataFrame:
    frames = []
    for _, run in runs.iterrows():
        curve = read_history(store, run["wandb_id"], HISTORY_SPEC_ID)
        if curve is None:
            continue
        keep = ["step"] + [c for c in CURVE_COLUMNS.values() if c in curve]
        frame = curve[keep].copy()
        frame.insert(0, "delay_k", int(run["delay_k"]))
        frame.insert(0, "arm", ARM_OF[run["condition"]])
        frame.insert(0, "condition", run["condition"])
        frame.insert(0, "budget_label", LABEL_OF[BUDGET_OF[run["condition"]]])
        frame.insert(0, "wandb_id", run["wandb_id"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------

def sanity_checks(data: pd.DataFrame, reward: pd.DataFrame) -> str:
    """Checks that must hold, or nothing downstream means anything. Written to checks.txt."""
    lines = ["explicit-vs-implicit-fm-probe -- sanity checks", "=" * 46, ""]

    def r2(probe: str, target: str = "proprio") -> pd.Series:
        return data[(data["probe"] == probe) & (data["target"] == target)]["test_r2"]

    ceiling = r2("input::current_proprio")
    leak = r2("layer::2/proprioception")
    lines.append(f"ceiling  input::current_proprio     min {ceiling.min():.4f}  "
                 f"(expect ~1.00; below that the target or mask is wrong)")
    lines.append(f"leak     layer::2/proprioception    min {leak.min():.4f}  "
                 f"(pre-Delay Normalizer: expect ~1.00)")

    # The network's own delay buffer must match a k-shift of the target.
    lines.append("")
    lines.append("delay buffer vs k-shifted floor (expect equal to ~3 dp):")
    for delay, group in data[data["delay_k"] > 0].groupby("delay_k"):
        buf = group[(group["probe"] == "layer::3/action/1/delay") &
                    (group["target"] == "proprio")]["test_r2"]
        floor = group[(group["probe"] == "input::delayed_proprio") &
                      (group["target"] == "proprio")]["test_r2"]
        if len(buf) and len(floor):
            lines.append(f"  delay {int(delay):>3}: buffer {buf.mean():.4f}  "
                         f"floor {floor.mean():.4f}  |diff| {abs(buf.mean()-floor.mean()):.4f}")

    # Encoder leakage control.
    enc = data[(data["pathway"] == "encoder") & (data["target"] == "proprio")]["test_r2"]
    lines += ["", f"encoder pathway, decoding current proprio: "
                  f"{enc.min():.3f} - {enc.max():.3f} "
                  f"(expect low; high means the reference target leaks the current pose)"]

    # Degenerate cells.
    degen = data[data["target_degenerate"]]
    lines.append(f"degenerate rows (delta at delay 0): {len(degen)}, "
                 f"all NaN: {bool(degen['test_r2'].isna().all())}")

    # Row bookkeeping.
    lines += ["", "rows per run (expect 68, or 66 at delay 0 -- no delay leaf):"]
    for (wandb_id, delay), group in data.groupby(["wandb_id", "delay_k"]):
        flag = "" if len(group) == (66 if delay == 0 else 68) else "   *** UNEXPECTED ***"
        lines.append(f"  {wandb_id} delay {int(delay):>3}: {len(group)}{flag}")

    strays = pathways.unclassified(data["probe"].unique())
    lines += ["", f"unclassified probes: {strays if strays else 'none'}"]

    # Arms must separate on the forward-model error itself.
    lines.append("")
    if "fm_mse_old_eval" in reward:
        for arm, group in reward.groupby("arm"):
            values = group["fm_mse_old_eval"].dropna()
            if len(values):
                lines.append(f"fm_mse_old_eval, {arm:<9}: "
                             f"{values.min():.4f} - {values.max():.4f}")

    # Cross-artifact consistency: the offline held-out FM error against the endpoint of the
    # in-training curve. These are *related but not identical* measurements -- different clip
    # population (held-out old_eval vs the in-training eval) and frame-0 latching -- so the
    # expectation is "same order, same ranking", not equality. Recorded here rather than drawn
    # on fm_prediction_curves.png, which compares series *within* one measurement.
    if {"fm_mse_curve_final", "fm_mse_old_eval"} <= set(reward.columns):
        both = reward.dropna(subset=["fm_mse_curve_final", "fm_mse_old_eval"])
        ratio = both["fm_mse_old_eval"] / both["fm_mse_curve_final"]
        lines += ["", "offline / in-training FM error ratio (expect ~1, delay 0 excepted "
                      "where both are ~1e-4 and the ratio is unstable):",
                  f"  delay > 0: {ratio[both.delay_k > 0].min():.2f} - "
                  f"{ratio[both.delay_k > 0].max():.2f}  (n={int((both.delay_k > 0).sum())})",
                  f"  delay = 0: {ratio[both.delay_k == 0].min():.2f} - "
                  f"{ratio[both.delay_k == 0].max():.2f}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    # This folder's own flags are consumed first; the rest go to the shared gate, which
    # errors on anything it does not recognise.
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--select-only", action="store_true",
                       help="write runs.csv + coverage.txt and stop, before the decode")
    extra.add_argument("--redecode", action="store_true",
                       help="ignore the cache and refit from the HDF5s")
    extra.add_argument("--only", help="comma-separated wandb_ids to decode")
    extra.add_argument("--jobs", type=int, default=1,
                       help="decode this many recordings in parallel (~1 GB each)")
    args, rest = extra.parse_known_args()
    parser_args = pipeline.parse_args(__doc__, rest)

    assert_spec_ids()
    store = Store()

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=parser_args.refresh,
                                      sync=parser_args.sync, project=parser_args.project)
    runs = runs.merge(index.load(project=parser_args.project),
                      on=["wandb_id"], how="left", suffixes=("", "_idx"))
    assert_grid(runs)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not parser_args.check:
        (HERE / "comparability.txt").write_text(report)
    print(report)

    if args.select_only:
        print("--select-only: wrote runs.csv and coverage.txt; stopping before the decode.")
        return

    reward = pd.DataFrame([reward_row(run, store) for _, run in runs.iterrows()])
    reward = reward.sort_values(["budget", "arm", "delay_k"], ignore_index=True)

    data = probe_rows(runs, store, redecode=args.redecode, jobs=args.jobs,
                      only=set(args.only.split(",")) if args.only else None)
    if len(data):
        data = data.sort_values(["budget", "arm", "delay_k", "wandb_id", "pathway",
                                "stage_index", "probe", "target"], ignore_index=True)

    curves = curve_rows(runs, store)

    if not parser_args.check and len(data):
        (HERE / "checks.txt").write_text(sanity_checks(data, reward))

    ok = pipeline.write_csv(reward, HERE / "data_reward.csv", check=parser_args.check)
    ok &= pipeline.write_csv(data, HERE / "data.csv", check=parser_args.check)
    ok &= pipeline.write_csv(curves, HERE / "curves.csv", check=parser_args.check)

    print(f"\n{len(runs)} runs, {len(data)} probe rows, {len(curves)} curve rows")
    if len(reward):
        pivot = reward.pivot_table(index="delay_k", columns=["budget_label", "arm"],
                                  values="reward_window")
        print("\nreward (trailing-50M window):")
        print(pivot.to_string(float_format=lambda v: f"{v:.0f}"))
    if parser_args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
