"""Imitation-target representation: absolute vs relative target, reference- vs current-root frame.

Run from the repo root::

    ../.venv/bin/python analysis/imitation-target-representation/extract.py

Two questions, answered with a single 6-run cohort (git ``f315e336``, "Comparing different
reference representations"), scored on the three batch-eval datasets (``train`` / ``old_eval`` /
``new_eval``) from ``vnl_experiments/delays/eval_runs.py``:

  1. **Absolute vs relative target.** The baseline ``Imitation`` env builds the joint/body
     imitation target *relative* to the agent's current state (it subtracts current joint angles
     and body positions). ``AbsoluteImitation`` makes those targets absolute. How much does going
     absolute cost?
  2. **Reference-root vs current-root frame.** Within ``AbsoluteImitation``, ``body_target_frame``
     expresses the absolute body target either in the *reference* root frame (``reference_root``)
     or in the agent's *current* / simulation root frame (``current_root``). Does referencing the
     reference root rather than the simulation root hurt?

Three conditions (each at delay 0 and delay 10, efference_length == delay):
  - ``relative``            — base ``Imitation`` env (proprio-relative target). [the two runs the
                              user flagged: config "env" wrongly says AbsoluteImitation, but the
                              eval correctly used ``Imitation`` — confirmed by the JSON env_class.]
  - ``absolute_current``    — ``AbsoluteImitation``, body_target_frame=current_root.
  - ``absolute_reference``  — ``AbsoluteImitation``, body_target_frame=reference_root
                              (this is the standard training config used everywhere else).

Condition is derived from the authoritative ``env_class`` recorded in each eval JSON (NOT the
WandB "env" config string, which is wrong for the two relative runs) plus ``body_target_frame``.

It joins TWO sources (extract.py is the only stage that touches them):
  1. WandB ``emiwar-team/nnx-ppo-rodent-delays`` — comparability invariants + body_target_frame;
  2. local ``eval_results/*.json`` — authoritative env_class + the three-dataset metrics.

Output is a **long** ``data.csv`` (one row per run × dataset). Cumulative ``episode_reward`` is
comparable *within* a dataset (fixed clip length) but NOT across datasets, so we also emit
length-fair metrics (``reward_per_step``, ``survival_frac``, ``hazard_rate``).
"""

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

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
# The reference-representation cohort is exactly the runs at this commit.
COHORT_GIT = "f315e336"

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

# Invariants that must be single-valued WITHIN a condition. The experimental axes
# (env_class, body_target_frame, delay_k, efference_length) are deliberately excluded.
INVARIANTS = [
    "git_commit", "latent_size", "kl_weight",
    "enc_hidden_sizes", "dec_hidden_sizes", "critic_hidden_sizes",
    "checkpoint_step", "clip_length_train",
]


def condition_for(env_class: str, body_target_frame) -> str | None:
    """Map (authoritative eval env_class, WandB body_target_frame) -> condition label."""
    if env_class == "Imitation":
        return "relative"
    if env_class == "AbsoluteImitation":
        if body_target_frame == "reference_root":
            return "absolute_reference"
        if body_target_frame == "current_root":
            return "absolute_current"
    return None


def wandb_invariants(run) -> dict:
    """Pull comparability invariants + the body_target_frame axis from the WandB config."""
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
        "config_env_string": c.get("env"),  # the (wrong-for-relative) logged env name
    }


def rollout_steps_for(frames: int, ctrl_dt: float, mocap_hz: int) -> int:
    """Env steps the eval scan runs to traverse a clip of ``frames`` mocap frames.

    Mirrors ``eval_runs.steps_for``: each env step advances the reference by
    ``ctrl_dt * mocap_hz`` frames, so a full clip needs ``ceil(frames / that) + 2`` steps
    (250 frames @ ctrl_dt=0.01, mocap_hz=50 -> 502 steps; 1500 -> 3002).
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
            "network_class": ej.get("network_class"),
            "env_class": ej.get("env_class"),       # authoritative env actually evaluated
            "delay_k": ej.get("delay_k"),
            "efference_length": ej.get("efference_length"),
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
            # Per-second failure hazard, excl. end-of-clip truncations: constant-hazard MLE
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
    runs = {r.id: r for r in fetch_runs(PROJECT, finished_only=True)}
    print(f"Fetched {len(runs)} finished runs from WandB")

    records = []
    cohort = []
    for wid, run in runs.items():
        inv = wandb_invariants(run)
        if inv["git_commit"] != COHORT_GIT:
            continue
        ej_path = EVAL_DIR / f"{wid}.json"
        if not ej_path.exists():
            print(f"  skip {wid}: cohort run but no eval_results JSON (e.g. crashed run)")
            continue
        ej = json.loads(ej_path.read_text())
        cond = condition_for(ej.get("env_class"), inv["body_target_frame"])
        if cond is None:
            print(f"  skip {wid}: could not assign condition "
                  f"(env_class={ej.get('env_class')!r}, btf={inv['body_target_frame']!r})")
            continue
        base = dict(inv)
        base["condition"] = cond
        base["checkpoint_step"] = ej.get("step")
        records.extend(eval_rows(wid, ej, base))
        cohort.append((wid, cond, ej.get("delay_k"), ej.get("env_class"),
                       inv["body_target_frame"], inv["config_env_string"]))

    df = records_to_df(records)
    df = df.sort_values(["condition", "delay_k", "dataset"]).reset_index(drop=True)

    print(f"\nCohort ({len(cohort)} runs): wid / condition / delay / env_class / "
          f"body_target_frame / config_env_string")
    for wid, cond, dly, envc, btf, cfg_env in sorted(cohort, key=lambda x: (x[1], x[2])):
        flag = "  <-- config_env WRONG" if cfg_env != envc else ""
        print(f"  {wid}  {cond:19s} delay={dly:<3} {envc:18s} btf={str(btf):14s} "
              f"config_env={cfg_env}{flag}")

    # Verify clip lengths / rollout horizons came out as expected (calibration sanity check).
    print()
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        fr = sorted(sub["clip_frames"].dropna().unique())
        rs = sorted(sub["rollout_steps"].dropna().unique())
        mx = sub["lifespan_steps"].max()
        print(f"  {ds:9s} frames={fr} rollout_steps={rs} max_lifespan={mx:.0f}")

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    n_runs = df["wandb_id"].nunique()
    print(f"\nWrote {len(df)} rows ({n_runs} runs × {len(DATASETS)} datasets) "
          f"to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
