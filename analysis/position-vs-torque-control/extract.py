"""Position vs torque control: is imitation easier, and does the forward model still win?

Run from the repo root::

    ../.venv/bin/python analysis/position-vs-torque-control/extract.py

This is the ONLY script that talks to WandB. It fetches the relevant runs, assigns a
``condition``, runs the programmatic comparability check, and writes ``data.csv`` +
``comparability.txt``.

Control mode is set by the environment flag ``torque_actuators`` (modular_rodent):
``True`` converts the motors to torque-mode actuators (the project default), ``False`` keeps
them as MuJoCo position (PD) actuators. It is logged authoritatively under
``env_params.torque_actuators``.

Two questions, a 2x2 (control-mode x network) design over an efference-matched delay sweep
(``efference_length == delay_k``), seed 42, standard architecture, frame held at ``current_root``:

  Q1. Is position control easier / more delay-robust than torque control?
  Q2. Is an explicit forward model still advantageous under position control?

Conditions and their (pinned) commits -- control mode is confounded with the code commit, so
each condition is pinned to a single commit and the training-relevant diffs were checked by
hand (see report.md "Comparability"):
  - pos_efference        position (torque_actuators=False), EncDec        git 891cd0d3
  - pos_forward_model    position (torque_actuators=False), ForwardModel  git 891cd0d3
  - torque_efference     torque   (torque_actuators=True),  EncDec        git 1cd5838f
  - torque_forward_model torque   (torque_actuators=True),  ForwardModel  git 54643764

The position runs record HEAD=891cd0d3 but were launched with local edits enabling position
control (env_params.torque_actuators=False) and keeping current_root; 891cd0d3 differs from
909e774d (the reference-root cohort, already certified equal to the baselines modulo frame)
only by eval_runs.txt. See report.md.

Reward metric: ``torque_actuators`` changes the actuators, not the imitation reward shaping;
the only mode-sensitive reward terms (energy_cost, control_cost, control_diff_cost) are ~3%
of the total (dominated by control-mode-agnostic tracking terms), so eval reward is a fair
overall-performance measure across modes (see report.md caveat). The "new logging api"
renamed ``episode_reward/mean`` -> ``eval/episode_reward/mean``; we coalesce the two keys.
"""

from pathlib import Path

from vnl_experiments.wandb_utils import (
    comparability_report,
    fetch_runs,
    git_commit_summary,
    records_to_df,
)

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRE_TAGS = ["TrainEvalSplit"]

# Every condition includes ALL of its comparable efference-matched runs (no fixed delay grid),
# up to MAX_DELAY. The efference sweeps are dense (position: the 891cd0d3 coarse sweep plus the
# b18513ae fine-delay fill-in; torque: the fine 1cd5838f/54643764 sweeps); the position FM
# sweep only exists at the coarse delays. FM-advantage curves are paired per control mode on
# whatever delays both networks share (pandas index alignment in plot.py).
MAX_DELAY = 100
STD_ARCH = {
    "enc_hidden_sizes": [512] * 4,
    "dec_hidden_sizes": [512] * 4,
    "critic_hidden_sizes": [1024, 1024],
}
# Position control is confounded with commit; both position-efference commits are the SAME
# training code (891cd0d3 -> b18513ae changes only analysis artifacts + eval_runs.txt).
POS_EFF_GITS = {"891cd0d3", "b18513ae"}  # coarse cohort + fine-delay fill-in
POS_FM_GIT = "891cd0d3"    # position forward-model cohort (coarse delays only)
CUR_EFF_GIT = "1cd5838f"   # canonical torque efference sweep
CUR_FM_GIT = "54643764"    # canonical torque forward-model sweep

# Metric keys, old-logging first then new-logging alias (coalesced left-to-right).
REWARD_MEAN_KEYS = ["episode_reward/mean", "eval/episode_reward/mean"]
REWARD_STD_KEYS = ["episode_reward/std", "eval/episode_reward/std"]
LIFESPAN_KEYS = ["lifespan_mean", "eval/lifespan/mean"]
FM_MSE_KEYS = ["eval/net/3/action/1/fm_pred_mse/mean"]

# Invariants that must be single-valued WITHIN a condition. control_mode/torque_actuators,
# network, delay_k and (by design) git_commit are the experimental / varying axes and are
# excluded; body_target_frame IS included -- it must stay current_root everywhere.
INVARIANTS = [
    "body_target_frame", "latent_size", "kl_weight", "enc_hidden_sizes",
    "dec_hidden_sizes", "critic_hidden_sizes", "seed", "clip_length", "actual_step",
]


def git8(run) -> str:
    return (((run.metadata or {}).get("git", {}) or {}).get("commit", "") or "")[:8]


def std_arch(net: dict) -> bool:
    return all(list(net.get(k, [])) == v for k, v in STD_ARCH.items())


def coalesce(summary, keys):
    for k in keys:
        v = summary.get(k)
        if v is not None:
            return v
    return None


def as_bool(v):
    return v if isinstance(v, bool) else (str(v) == "True")


def condition_of(run):
    """Map a run to (condition, control_mode, network) or None to exclude it."""
    c = run.config
    net = c.get("net_params", {}) or {}
    env = c.get("env_params", {}) or {}
    tags = set(run.tags)
    frame = env.get("body_target_frame")
    ta = env.get("torque_actuators")
    delay = c.get("delay_k")
    eff = c.get("efference_length")

    # Common gates: current_root frame held constant, efference-matched, seed 42, std arch,
    # delay within the swept range.
    if frame != "current_root" or eff != delay or c.get("seed") != 42 or not std_arch(net):
        return None
    if delay is None or not (0 <= delay <= MAX_DELAY):
        return None

    is_fm = "ForwardModel" in tags
    is_enc = "EncDec" in tags and not is_fm
    # Canonical explicit forward model only: trained L2 loss, detached prediction.
    fm_ok = (is_fm and c.get("fm_loss_weight") == 1
             and c.get("detach_prediction") in (None, True) and "nodetach" not in tags)
    g = git8(run)

    if as_bool(ta) is False:                       # position control
        if fm_ok and g == POS_FM_GIT:
            return ("pos_forward_model", "position", "forward_model")
        if is_enc and g in POS_EFF_GITS:
            return ("pos_efference", "position", "efference")
    elif as_bool(ta) is True:                       # torque control (baselines)
        if is_enc and g == CUR_EFF_GIT:
            return ("torque_efference", "torque", "efference")
        if fm_ok and g == CUR_FM_GIT:
            return ("torque_forward_model", "torque", "forward_model")
    return None


def record_of(run, cond, control_mode, network) -> dict:
    c = run.config
    net = c.get("net_params", {}) or {}
    env = c.get("env_params", {}) or {}
    s = run.summary
    return {
        # provenance
        "wandb_id": run.id,
        "wandb_name": run.name,
        "wandb_project": PROJECT,
        "git_commit": git8(run),
        "tags": ",".join(sorted(run.tags)),
        # experimental axes
        "condition": cond,
        "control_mode": control_mode,
        "network": network,
        "delay_k": c.get("delay_k"),
        "efference_length": c.get("efference_length"),
        # authoritative control mode + fm knobs (sanity)
        "torque_actuators": env.get("torque_actuators"),
        "fm_loss_weight": c.get("fm_loss_weight"),
        "detach_prediction": c.get("detach_prediction"),
        # invariants
        "body_target_frame": env.get("body_target_frame"),
        "seed": c.get("seed"),
        "latent_size": net.get("latent_size"),
        "kl_weight": net.get("kl_weight"),
        "enc_hidden_sizes": tuple(net.get("enc_hidden_sizes") or []),
        "dec_hidden_sizes": tuple(net.get("dec_hidden_sizes") or []),
        "critic_hidden_sizes": tuple(net.get("critic_hidden_sizes") or []),
        "clip_length": env.get("clip_length"),
        "actual_step": s.get("_step"),
        # metrics (coalesced across the old/new logging key names)
        "reward_mean": coalesce(s, REWARD_MEAN_KEYS),
        "reward_std": coalesce(s, REWARD_STD_KEYS),
        "lifespan_mean": coalesce(s, LIFESPAN_KEYS),
        "fm_pred_mse": coalesce(s, FM_MSE_KEYS),
    }


def main() -> None:
    runs = fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)
    print(f"Fetched {len(runs)} finished runs with tags {REQUIRE_TAGS}")

    records = []
    for r in runs:
        hit = condition_of(r)
        if hit is None:
            continue
        cond, control_mode, network = hit
        records.append(record_of(r, cond, control_mode, network))

    df = records_to_df(records)
    df = df.sort_values(["control_mode", "network", "delay_k", "wandb_id"]).reset_index(drop=True)

    print(f"\nCohort ({len(df)} rows): condition / delay / control / btf / git")
    for _, row in df.iterrows():
        print(f"  {row['wandb_id']}  {row['condition']:22s} delay={row['delay_k']:<4} "
              f"torque_act={str(row['torque_actuators']):5s} btf={row['body_target_frame']:12s} "
              f"git={row['git_commit']}")

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")


if __name__ == "__main__":
    main()
