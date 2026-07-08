"""Offline batch-eval companion to extract.py: position vs torque on the three eval datasets.

Run from the repo root::

    ../.venv/bin/python analysis/position-vs-torque-control/extract_eval.py

extract.py answers the two questions on the *training-clip* WandB reward. This script re-uses
the SAME run set (read from the committed ``data.csv``) but joins the offline
``eval_results/*.json`` metrics, so the position-vs-torque and forward-model conclusions can be
checked on held-out / longer clips with length-fair metrics.

Three sources (this is the only stage that touches them):
  1. committed ``data.csv`` in this folder — the authoritative run set + condition labels
     (``control_mode`` / ``network`` / ``delay_k``), already comparability-checked by extract.py;
  2. WandB — independent re-verification of the invariants (read from ``env_params``, not the
     inert ``net_params`` copy — see analysis/README.md "body_target_frame bug");
  3. local ``eval_results/eval_results/*.json`` — the three-dataset metrics.

Output is a **long** ``data_eval.csv`` (one row per run × dataset). Cumulative
``episode_reward`` is NOT comparable across datasets (new_eval clips are 30 s / 3002 steps vs
5 s / 502), so we emit length-fair metrics (``reward_per_step``, per-second ``hazard_rate``)
plus per-episode termination and per-step error breakdowns, which ARE comparable.
"""

import csv
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
EVAL_DIR = REPO_ROOT / "eval_results" / "eval_results"

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRE_TAGS = ["TrainEvalSplit"]

NEW_EVAL_CLIP_LENGTH = 1500       # fixed length of the new eval clips (eval_runs.py)
DATASETS = ["train", "old_eval", "new_eval"]
TERMINATIONS = ["root_too_far", "root_too_rotated", "pose_error", "nan_termination"]
ERRORS = {
    "root_pos_distance": "err_root_pos_distance",
    "root_angular_error": "err_root_angular_error",
    "joint_l2_error": "err_joint_l2_error",
    "body_errors/total": "err_body_total",
    "body_errors/end_eff_total": "err_body_end_eff",
}

# Invariants that must be single-valued WITHIN a condition. control_mode/network/delay_k and
# (by design) git_commit are the experimental / varying axes and are excluded; body_target_frame
# IS included -- it must stay current_root everywhere.
INVARIANTS = [
    "body_target_frame", "latent_size", "kl_weight", "enc_hidden_sizes",
    "dec_hidden_sizes", "critic_hidden_sizes", "checkpoint_step", "clip_length_train",
]


def load_condition_map() -> dict:
    """wandb_id -> {condition, control_mode, network, delay_k} from this folder's data.csv."""
    out = {}
    for row in csv.DictReader(open(HERE / "data.csv")):
        wid = row.get("wandb_id")
        if not wid or wid in out:
            continue
        out[wid] = {
            "condition": row["condition"],
            "control_mode": row["control_mode"],
            "network": row["network"],
            "delay_k": int(float(row["delay_k"])) if row.get("delay_k") not in (None, "") else None,
        }
    return out


def wandb_invariants(run) -> dict:
    """Pull comparability invariants from the WandB config (env_params is authoritative)."""
    c = run.config
    net = c.get("net_params", {}) or {}
    env = c.get("env_params", {}) or {}
    git = ((run.metadata or {}).get("git", {}) or {}).get("commit")
    return {
        "git_commit": (git or "")[:8],
        "torque_actuators": env.get("torque_actuators"),
        "body_target_frame": env.get("body_target_frame"),
        "latent_size": net.get("latent_size"),
        "kl_weight": net.get("kl_weight"),
        "enc_hidden_sizes": tuple(net.get("enc_hidden_sizes") or []),
        "dec_hidden_sizes": tuple(net.get("dec_hidden_sizes") or []),
        "critic_hidden_sizes": tuple(net.get("critic_hidden_sizes") or []),
        "clip_length_train": env.get("clip_length"),
        "ctrl_dt": env.get("ctrl_dt"),
        "mocap_hz": env.get("mocap_hz"),
    }


def rollout_steps_for(frames, ctrl_dt, mocap_hz) -> int:
    """Env steps the eval scan runs to traverse a clip (mirrors eval_runs.steps_for)."""
    return int(math.ceil(frames / (ctrl_dt * mocap_hz))) + 2


def eval_rows(wid, ej, base) -> list:
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
        net_m = d.get("net_metrics", {}) or {}
        row = dict(base)
        row.update({
            "wandb_id": wid,
            "wandb_name": ej.get("wandb_name"),
            "network_class": ej.get("network_class"),
            "fm_loss_weight": ej.get("fm_loss_weight"),
            "fm_pred_mse": net_m.get("3/action/1/fm_pred_mse"),
            "dataset": ds,
            "n_clips": d.get("n_clips"),
            "clip_frames": clip_frames,
            "rollout_steps": rollout_steps,
            "episode_reward_mean": rew,
            "episode_reward_std": d["episode_reward"]["std"],
            "lifespan_steps": life,
            "survival_frac": (life / rollout_steps) if rollout_steps else None,
            "reward_per_step": (rew / life) if life else None,
            "survived": term.get("survived"),
            # Per-second failure hazard, excluding end-of-clip truncations: constant-hazard MLE
            # = (1 - survived) / total-alive-time. Length-independent, so it compares fairly
            # across the 5 s and 30 s clips (survived counts only failure terminations).
            "hazard_rate": (
                (1.0 - term["survived"]) / (life * base["ctrl_dt"])
                if term.get("survived") is not None and life else None
            ),
        })
        for t in TERMINATIONS:
            row[f"term_{t}"] = term.get(t)
        for k, col in ERRORS.items():
            row[col] = errs.get(k, {}).get("mean") if k in errs else None
        rows.append(row)
    return rows


def main() -> None:
    cond_map = load_condition_map()
    print(f"{len(cond_map)} runs from data.csv")

    runs = {r.id: r for r in fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)}
    print(f"Fetched {len(runs)} finished TrainEvalSplit runs from WandB")

    records, missing_eval, missing_wandb = [], [], []
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
    df = df.sort_values(["control_mode", "network", "delay_k", "dataset"]).reset_index(drop=True)

    # Calibration sanity check: clip lengths / rollout horizons per dataset.
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
        print(f"\n{len(missing_eval)} runs had no eval JSON (skipped): {missing_eval}")
    if missing_wandb:
        print(f"{len(missing_wandb)} runs missing from WandB (skipped): {missing_wandb}")

    (HERE / "data_eval.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability_eval.txt").write_text(report + "\n")
    n_runs = df["wandb_id"].nunique()
    print(f"\nWrote {len(df)} rows ({n_runs} runs × {len(DATASETS)} datasets) to {HERE / 'data_eval.csv'}")


if __name__ == "__main__":
    main()
