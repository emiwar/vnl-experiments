"""Forward model vs regular encoder-decoder, multi-seed, on the batch eval sets.

Run from the repo root::

    ../.venv/bin/python analysis/forward-model-vs-encdec-seeds/extract.py

Question: does the explicit **forward model** beat the **regular with-efference
encoder-decoder**, and is the difference "just better across the board" or a
qualitatively different reward/failure profile? Unlike the earlier single-seed
``forward-model-new-eval`` question, here every (condition, delay) cell has **>=3
training seeds** (42/43/44), so we can show per-seed spread and a seed-mean.

Two conditions only (the "no efference copy" floor is deliberately dropped):
  - ``forward_model`` — RodentForwardModel, canonical ``fm_loss_weight == 1``,
    efference matched (``efference_length == delay_k``).
  - ``efference``     — the regular RodentEncDecDelays with-efference decoder,
    efference matched (``efference_length == delay_k``).

Data source: the batch-eval JSONs under ``eval_results/eval_results/`` (the current,
most complete set of re-evaluations; 3 datasets each: ``train`` / ``old_eval`` /
``new_eval``). Conditions are derived directly from each JSON's ``network_class`` /
``fm_loss_weight`` / ``efference_length`` rather than from other analyses' committed
CSVs, because this question includes freshly-added runs not present in those CSVs.
WandB is queried only to independently re-verify the comparability invariants
(``seed``, git, frame, network sizes).

Output is a **long** ``data.csv`` (one row per run x dataset). Cumulative
``episode_reward`` is comparable across conditions only *within* a dataset (the new
clips are 30 s / 3002-step vs 5 s / 502-step), so we also emit length-fair metrics
(``reward_per_step``, ``hazard_rate``), per-step tracking errors, per-reason
termination rates, and per-term rewards (for the reward/failure-profile question).
"""

import csv  # noqa: F401  (kept for parity with sibling extracts; not required)
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
# The reorganised eval outputs: eval_results/eval_results/ is the current, most
# complete re-evaluation set (eval_results/old_eval_results/ is the previous one).
EVAL_DIR = REPO_ROOT / "eval_results" / "eval_results"

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

CANONICAL_FM_LOSS_WEIGHT = 1.0
# The regular encoder-decoder uses the standard 4x512 decoder. The bigger/deeper
# decoder variants (from forward-model-vs-bigger-decoder) are ALSO efference-matched,
# so they must be excluded here to keep the network identical except for the FM head.
STANDARD_DEC_HIDDEN = (512, 512, 512, 512)
NEW_EVAL_CLIP_LENGTH = 1500          # new eval clip length in mocap frames (eval_runs.py)
EXPECTED_TRAIN_CLIP_LENGTH = 250
DATASETS = ["train", "old_eval", "new_eval"]

TERMINATIONS = ["root_too_far", "root_too_rotated", "pose_error", "nan_termination"]

# Per-step tracking errors (raw units: metres / degrees / rad-ish). Converted to mm
# in plot.py for the distance errors.
ERRORS = {
    "root_pos_distance": "err_root_pos_m",       # metres
    "root_angular_error": "err_root_angular_deg",  # degrees
    "joint_l2_error": "err_joint_l2",
    "joint_vel_l2_error": "err_joint_vel_l2",
    "body_errors/total": "err_body_total_m",     # metres (overall body tracking error)
    "body_errors/end_eff_total": "err_body_end_eff_m",  # metres
}

# Reward terms that are actually active (bodies_pos / joints_vel are logged as 0).
REWARD_TERMS = [
    "root_pos", "root_quat", "joints", "end_eff", "torso_z_range",
    "control_cost", "control_diff_cost", "energy_cost",
]

# Invariants that must hold WITHIN a condition. network_class / dec-head differ across
# conditions by design (that IS the axis), so they are not required to match overall.
INVARIANTS = [
    "git_commit", "body_target_frame", "latent_size", "kl_weight",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
    "checkpoint_step", "clip_length_train",
]


def classify(ej: dict) -> str | None:
    """Assign an eval JSON to a condition, or None to drop it."""
    cls = ej.get("network_class")
    delay = ej.get("delay_k")
    eff = ej.get("efference_length")
    if eff != delay:  # keep only efference-matched runs (drops eff-length / no-eff sweeps)
        return None
    if cls == "RodentForwardModel" and ej.get("fm_loss_weight") == CANONICAL_FM_LOSS_WEIGHT:
        return "forward_model"
    if cls == "RodentEncDecDelays":
        return "efference"
    return None


def wandb_invariants(run) -> dict:
    """Comparability invariants + seed, read straight from the WandB config."""
    c = run.config
    net = c.get("net_params", {}) or {}
    env = c.get("env_params", {}) or {}
    git = ((run.metadata or {}).get("git", {}) or {}).get("commit")
    return {
        "seed": c.get("seed"),
        "detach_prediction": c.get("detach_prediction"),
        "git_commit": (git or "")[:8],
        # Authoritative frame lives on env_params, NOT net_params (see analysis/README.md).
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


def rollout_steps_for(frames: int, ctrl_dt: float, mocap_hz: int) -> int:
    """Env steps the eval scan runs for a clip of ``frames`` mocap frames (see eval_runs)."""
    return int(math.ceil(frames / (ctrl_dt * mocap_hz))) + 2


def eval_rows(wid: str, ej: dict, base: dict) -> list[dict]:
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
        rterms = d.get("reward_terms", {}) or {}
        survived = term.get("survived")
        row = dict(base)
        row.update({
            "wandb_id": wid,
            "wandb_name": ej.get("wandb_name"),
            "wandb_project": PROJECT,
            "network_class": ej.get("network_class"),
            "fm_loss_weight": ej.get("fm_loss_weight"),
            "total_params": pc.get("total"),
            "decoder_params": pc.get("decoder"),
            "fm_pred_mse": net_m.get("3/action/1/fm_pred_mse"),
            "dataset": ds,
            "n_clips": d.get("n_clips"),
            "clip_frames": clip_frames,
            "rollout_steps": rollout_steps,
            "episode_reward_mean": rew,
            "episode_reward_std": d["episode_reward"]["std"],
            "lifespan_steps": life,
            "lifespan_s": d.get("lifespan_s", {}).get("mean") if d.get("lifespan_s") else None,
            "survival_frac": (life / rollout_steps) if rollout_steps else None,
            "reward_per_step": (rew / life) if life else None,
            "survived": survived,
            # Constant-hazard MLE: failures per unit alive-time, end-of-clip truncations
            # censored (survived counts only failure terminations). Length-invariant.
            "hazard_rate": (
                (1.0 - survived) / (life * base["ctrl_dt"])
                if survived is not None and life else None
            ),
        })
        for t in TERMINATIONS:
            row[f"term_{t}"] = term.get(t)
        for k, col in ERRORS.items():
            row[col] = errs.get(k, {}).get("mean") if k in errs else None
        # Per-term reward, both cumulative (raw) and per alive-step (composition-fair).
        for t in REWARD_TERMS:
            v = rterms.get(t, {}).get("mean") if t in rterms else None
            row[f"rt_{t}"] = v
            row[f"rtps_{t}"] = (v / life) if (v is not None and life) else None
        rows.append(row)
    return rows


def main() -> None:
    ej_paths = sorted(glob.glob(str(EVAL_DIR / "*.json")))
    print(f"{len(ej_paths)} eval JSONs in {EVAL_DIR}")

    keep: dict[str, dict] = {}
    for p in ej_paths:
        ej = json.loads(Path(p).read_text())
        cond = classify(ej)
        if cond is None:
            continue
        keep[ej["wandb_id"]] = {"cond": cond, "ej": ej}
    conds = {}
    for v in keep.values():
        conds[v["cond"]] = conds.get(v["cond"], 0) + 1
    print(f"{len(keep)} runs kept: {conds}")

    # Independent WandB re-verification of invariants + seed for exactly these runs.
    runs = {r.id: r for r in fetch_runs(PROJECT, finished_only=False)}
    print(f"Fetched {len(runs)} runs from WandB for invariant cross-check")

    records = []
    missing_wandb = []
    dropped_decoder = []
    for wid, v in keep.items():
        if wid not in runs:
            missing_wandb.append(wid)
            continue
        ej = v["ej"]
        base = {"condition": v["cond"], "delay_k": ej.get("delay_k"),
                "efference_length": ej.get("efference_length")}
        base.update(wandb_invariants(runs[wid]))
        # Regular encoder-decoder must use the standard decoder; drop the bigger/deeper
        # decoder variants that are efference-matched but belong to another question.
        if v["cond"] == "efference" and base["dec_hidden_sizes"] != STANDARD_DEC_HIDDEN:
            dropped_decoder.append(wid)
            continue
        base["checkpoint_step"] = ej.get("step")
        records.extend(eval_rows(wid, ej, base))

    df = records_to_df(records)
    df = df.sort_values(["condition", "delay_k", "seed", "dataset"]).reset_index(drop=True)

    # Calibration sanity: clip lengths / rollout horizons per dataset.
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        fr = sorted(sub["clip_frames"].dropna().unique())
        rs = sorted(sub["rollout_steps"].dropna().unique())
        mx = sub["lifespan_steps"].max()
        print(f"  {ds:9s} frames={fr} rollout_steps={rs} max_lifespan={mx:.0f}")

    # Seed coverage per condition (the whole point of this question).
    print("\nSeed coverage (runs per condition x seed):")
    for cond, g in df[df.dataset == "old_eval"].groupby("condition"):
        print(f"  {cond:14s}: {g['seed'].value_counts().sort_index().to_dict()}")

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))
    if missing_wandb:
        print(f"\n{len(missing_wandb)} runs missing from WandB fetch (skipped): {missing_wandb}")
    if dropped_decoder:
        print(f"{len(dropped_decoder)} efference runs dropped (non-standard decoder): {dropped_decoder}")

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    n_runs = df["wandb_id"].nunique()
    print(f"\nWrote {len(df)} rows ({n_runs} runs x {len(DATASETS)} datasets) to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
