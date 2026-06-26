"""Extract train/eval generalization metrics for the standard EncDec runs.

Run from the repo root::

    ../.venv/bin/python analysis/train-eval-generalization/extract.py

This analysis joins THREE sources (extract.py is the only stage that touches them):

  1. The committed analysis CSVs (``analysis/*/data.csv``) — the authoritative,
     already-vetted set of runs and their ``condition`` labels.
  2. WandB (``emiwar-team/nnx-ppo-rodent-delays``) — independent re-verification of the
     comparability invariants (git commit, architecture, steps, clip_length, ...).
  3. The local ``eval_results/*.json`` produced by ``vnl_experiments.delays.eval_runs``,
     which re-evaluated each checkpoint on THREE datasets: ``train`` (80% split),
     ``old_eval`` (held-out 20% of the same data, same 250-step clips) and ``new_eval``
     (32 fresh 1500-step clips).

Output is a **long** ``data.csv``: one row per (run, dataset). Because the new-eval clips
are 1500 steps vs 250 for train/old, cumulative ``episode_reward`` is NOT comparable across
datasets — so we also emit length-normalised metrics (``reward_per_step``, ``survival_frac``)
and the per-step error / per-episode termination breakdowns, which ARE comparable.

We focus on the standard with-efference EncDec decoder (``efference``), plus the
larger/deeper decoder variants (for the does-size-change-overfitting question) and the
no-efference floor. Forward-model and truncated-buffer runs are out of scope here.
"""

import csv
import glob
import json
import math
from pathlib import Path

from vnl_experiments.wandb_utils import (
    comparability_report,
    fetch_runs,
    git_commit_summary,
    records_to_df,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EVAL_DIR = REPO_ROOT / "eval_results"
ANALYSIS_GLOB = str(REPO_ROOT / "analysis" / "*" / "data.csv")

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRE_TAGS = ["TrainEvalSplit"]

# Conditions kept for this analysis (all RodentEncDecDelays, standard with-efference family).
INCLUDE_CONDITIONS = {"efference", "no_efference", "efference_larger", "efference_deeper"}

NEW_EVAL_CLIP_LENGTH = 1500  # fixed length of the new eval clips (eval_runs.py)
EXPECTED_TRAIN_CLIP_LENGTH = 250

DATASETS = ["train", "old_eval", "new_eval"]
TERMINATIONS = ["root_too_far", "root_too_rotated", "pose_error", "nan_termination"]
ERRORS = {
    "root_pos_distance": "err_root_pos_distance",
    "root_angular_error": "err_root_angular_error",
    "joint_l2_error": "err_joint_l2_error",
    "joint_vel_l2_error": "err_joint_vel_l2_error",
    "body_errors/total": "err_body_total",
    "body_errors/end_eff_total": "err_body_end_eff",
}

# Invariants that must be single-valued WITHIN a condition. dec_hidden_sizes is the
# deliberate variable across conditions (the network-size axis), so it is excluded.
# checkpoint_step (the actual trained step restored for eval) is the authoritative
# "same training length" check, so n_envs/total_steps are not re-listed here.
INVARIANTS = [
    "git_commit", "body_target_frame", "latent_size", "kl_weight",
    "enc_hidden_sizes", "critic_hidden_sizes", "checkpoint_step", "clip_length_train",
]


def load_condition_map() -> dict[str, dict]:
    """wandb_id -> {condition, delay_k, efference_length} from the committed CSVs."""
    out: dict[str, dict] = {}
    for c in sorted(glob.glob(ANALYSIS_GLOB)):
        for row in csv.DictReader(open(c)):
            wid = row.get("wandb_id")
            if not wid or wid in out:
                continue
            cond = row.get("condition")
            if cond not in INCLUDE_CONDITIONS:
                continue
            out[wid] = {
                "condition": cond,
                "delay_k": int(float(row["delay_k"])) if row.get("delay_k") not in (None, "") else None,
                "efference_length": int(float(row["efference_length"]))
                if row.get("efference_length") not in (None, "") else None,
            }
    return out


def wandb_invariants(run) -> dict:
    """Pull the comparability invariants straight from the WandB config (independent check)."""
    c = run.config
    net = c.get("net_params", {}) or {}
    env_params = c.get("env_params", {}) or {}
    git = ((run.metadata or {}).get("git", {}) or {}).get("commit")
    return {
        "git_commit": (git or "")[:8],
        "body_target_frame": net.get("body_target_frame"),
        "latent_size": net.get("latent_size"),
        "kl_weight": net.get("kl_weight"),
        "enc_hidden_sizes": tuple(net.get("enc_hidden_sizes") or []),
        "dec_hidden_sizes": tuple(net.get("dec_hidden_sizes") or []),
        "critic_hidden_sizes": tuple(net.get("critic_hidden_sizes") or []),
        "clip_length_train": env_params.get("clip_length"),  # mocap frames @ mocap_hz
        "ctrl_dt": env_params.get("ctrl_dt"),
        "mocap_hz": env_params.get("mocap_hz"),
    }


def rollout_steps_for(frames: int, ctrl_dt: float, mocap_hz: int) -> int:
    """Env steps the eval scan runs to traverse a clip of ``frames`` mocap frames.

    Mirrors ``eval_runs.steps_for``: each env step advances the reference by
    ``ctrl_dt * mocap_hz`` frames, so a full clip needs ``ceil(frames / that) + 2`` steps
    (e.g. 250 frames @ ctrl_dt=0.01, mocap_hz=50 -> 502 steps; 1500 -> 3002).
    """
    return int(math.ceil(frames / (ctrl_dt * mocap_hz))) + 2


def eval_rows(wid: str, ej: dict, base: dict) -> list[dict]:
    """One row per dataset, merging provenance/invariants (base) with eval metrics."""
    rows = []
    pc = ej.get("param_counts", {}) or {}
    for ds in DATASETS:
        d = ej["datasets"].get(ds)
        if d is None:
            continue
        clip_frames = NEW_EVAL_CLIP_LENGTH if ds == "new_eval" else base["clip_length_train"]
        rollout_steps = rollout_steps_for(clip_frames, base["ctrl_dt"], base["mocap_hz"])
        rew = d["episode_reward"]["mean"]
        life = d["lifespan_steps"]["mean"]
        term = d.get("termination_rate", {}) or {}
        errs = d.get("errors", {}) or {}
        row = dict(base)
        row.update({
            "wandb_id": wid,
            "wandb_name": ej.get("wandb_name"),
            "total_params": pc.get("total"),
            "decoder_params": pc.get("decoder"),
            "dataset": ds,
            "n_clips": d.get("n_clips"),
            "clip_frames": clip_frames,        # clip length in mocap frames
            "rollout_steps": rollout_steps,    # env steps the scan ran (survival denom)
            "episode_reward_mean": rew,
            "episode_reward_std": d["episode_reward"]["std"],
            "lifespan_steps": life,
            "survival_frac": (life / rollout_steps) if rollout_steps else None,
            "reward_per_step": (rew / life) if life else None,
            "survived": term.get("survived"),
        })
        for t in TERMINATIONS:
            row[f"term_{t}"] = term.get(t)
        for k, col in ERRORS.items():
            row[col] = errs.get(k, {}).get("mean") if k in errs else None
        rows.append(row)
    return rows


def main() -> None:
    cond_map = load_condition_map()
    print(f"{len(cond_map)} runs of interest from committed CSVs "
          f"(conditions: {sorted(INCLUDE_CONDITIONS)})")

    # Independent WandB re-verification of invariants for exactly these runs.
    runs = {r.id: r for r in fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)}
    print(f"Fetched {len(runs)} finished TrainEvalSplit runs from WandB")

    records = []
    missing_eval, missing_wandb = [], []
    for wid, info in cond_map.items():
        ej_path = EVAL_DIR / f"{wid}.json"
        if not ej_path.exists():
            missing_eval.append(wid)
            continue
        if wid not in runs:
            missing_wandb.append(wid)
            continue
        ej = json.loads(ej_path.read_text())
        base = dict(info)
        base.update(wandb_invariants(runs[wid]))
        base["checkpoint_step"] = ej.get("step")
        records.extend(eval_rows(wid, ej, base))

    df = records_to_df(records)
    df = df.sort_values(["condition", "delay_k", "dataset"]).reset_index(drop=True)

    # Verify clip lengths / rollout horizons came out as expected (calibration sanity check).
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        fr = sorted(sub["clip_frames"].dropna().unique())
        rs = sorted(sub["rollout_steps"].dropna().unique())
        mx = sub["lifespan_steps"].max()
        print(f"  {ds:9s} frames={fr} rollout_steps={rs} max_lifespan={mx:.0f}")

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))
    if missing_eval:
        print(f"\n{len(missing_eval)} runs had no eval_results JSON (skipped): {missing_eval}")
    if missing_wandb:
        print(f"{len(missing_wandb)} runs missing from WandB fetch (skipped): {missing_wandb}")

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    n_runs = df["wandb_id"].nunique()
    print(f"\nWrote {len(df)} rows ({n_runs} runs × {len(DATASETS)} datasets) "
          f"to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
