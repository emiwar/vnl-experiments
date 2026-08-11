"""New (almost-full-collision) vs old (sparse-collision) rodent walker XML.

Run from the repo root::

    ../.venv/bin/python analysis/collision-model-xml/extract.py

This is the ONLY script that talks to WandB. It fetches the relevant runs, assigns a
``condition``, runs the programmatic comparability check, and writes ``data.csv``,
``curves.csv`` and ``comparability.txt``.

The question
------------
A cohort of runs was trained with ``env_params.walker_xml_path`` pointing at
``rodent_no_tail_collisions.xml`` ("new" XML: every body except the tail carries a
collision primitive) instead of the previous default ``rodent.xml`` ("old" XML: only a
handful of collision geoms -- feet, lower limbs, skull). We ask (1) does performance
change, (2) is the change the same across network / control conditions, (3) what frame
was actually used, (4) does convergence speed change, (5) how much slower is it.

Cohorts
-------
Every included run: ``env == AbsoluteImitation``, ``seed == 42``, standard architecture
(enc [512]x4, dec [512]x4, critic [1024,1024], latent 32, kl 1e-3), efference-matched
(``efference_length == delay_k``), finished, ``actual_step == 600,064,000``, clip 250.

The XML axis is paired with a **matched** baseline inside each network/control cell, so
each pair differs only in the XML (plus the code commit, checked by hand -- see report):

  cell "efference"          torque control, efference EncDec
    old_efference           rodent.xml            git 1cd5838f  (canonical eff sweep)
    new_efference           no_tail_collisions    git 201d6e11  (2026-07-09)
    old_efference_refroot   rodent.xml            git 909e774d  (robustness baseline:
                            same cell but reference_root frame + the *new* logging api,
                            so it controls for the episode_reward key rename)

  cell "forward_model"      POSITION control, explicit FM (fm_loss_weight=1, detached)
    old_forward_model       rodent.xml            git 891cd0d3  (2026-07-06)
    new_forward_model       no_tail_collisions    git 201d6e11  (2026-07-09)

  cell "pg_forward_model"   policy-gradient FM (fm_loss_weight=0, no detach)
    old_pg_forward_model    rodent.xml, TORQUE    git d4bd4dc0/d33e5bcf
    new_pg_forward_model    no_tail_collisions, POSITION  git 0560d402 (2026-07-19)
    NOTE: this pair is *doubly* confounded (XML *and* actuator mode). It is extracted
    for completeness and plotted separately, never as evidence about the XML alone.

Frame
-----
The authoritative frame is ``env_params.body_target_frame`` (the ``net_params`` copy is
inert -- see analysis/README.md). **Every new-XML run logs ``current_root``**, and the
post-fix training script no longer writes the key into ``net_params`` at all, so there is
no conflicting label. The frame is therefore matched (``current_root``) on both sides of
each primary pair; ``old_efference_refroot`` is the one deliberate exception.

Speed
-----
``throughput/train_sps`` (full PPO iteration: rollout + gradient) and
``throughput/eval_sps`` (rollout only -- the cleaner probe of physics cost) are recorded
both as the final summary value and as a median over the run's history. Runs were
scheduled on a mix of **A100-SXM4-80GB and H200** nodes, which changes throughput by
~1.6x, so ``gpu`` is extracted and every speed comparison in plot.py is made within one
GPU model at a matched delay.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils import (
    comparability_report,
    fetch_runs,
    git_commit_summary,
    records_to_df,
)

HERE = Path(__file__).resolve().parent

PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRE_TAGS = ["TrainEvalSplit"]

OLD_XML = "rodent.xml"
NEW_XML = "rodent_no_tail_collisions.xml"

STD_ARCH = {
    "enc_hidden_sizes": [512] * 4,
    "dec_hidden_sizes": [512] * 4,
    "critic_hidden_sizes": [1024, 1024],
}
EXPECTED_STEP = 600_064_000

# Metric keys: old-logging first, then the new-logging alias (coalesced left-to-right).
REWARD_MEAN_KEYS = ["episode_reward/mean", "eval/episode_reward/mean"]
REWARD_STD_KEYS = ["episode_reward/std", "eval/episode_reward/std"]
LIFESPAN_KEYS = ["lifespan_mean", "eval/lifespan/mean"]

# Invariants that must be single-valued WITHIN a condition. The experimental axes
# (xml, network, control_mode, body_target_frame, delay_k) and git_commit vary by design.
INVARIANTS = [
    "env", "seed", "latent_size", "kl_weight", "enc_hidden_sizes", "dec_hidden_sizes",
    "critic_hidden_sizes", "clip_length", "ctrl_dt", "sim_dt", "solver", "iterations",
    "ls_iterations", "njmax", "naconmax", "rescale_factor", "mujoco_impl",
    "n_envs", "total_steps", "actual_step",
    "walker_xml", "torque_actuators", "body_target_frame", "git_commit",
]

# Per-condition (xml, torque_actuators, network, fm_loss_weight, detach_prediction,
# body_target_frame, allowed creation dates).
CONDITIONS = {
    "old_efference":         (OLD_XML, True,  "efference",         None, None,  "current_root",   ("2026-06-11", "2026-06-12")),
    "new_efference":         (NEW_XML, True,  "efference",         None, None,  "current_root",   ("2026-07-09",)),
    "old_efference_refroot": (OLD_XML, True,  "efference",         None, None,  "reference_root", ("2026-07-06",)),
    "old_forward_model":     (OLD_XML, False, "forward_model",     1,    True,  "current_root",   None),
    "new_forward_model":     (NEW_XML, False, "forward_model",     1,    True,  "current_root",   None),
    "old_pg_forward_model":  (OLD_XML, True,  "pg_forward_model",  0,    False, "current_root",   None),
    "new_pg_forward_model":  (NEW_XML, False, "pg_forward_model",  0,    False, "current_root",   None),
}


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


def condition_of(run) -> str | None:
    """Map a run to a condition label, or None to exclude it."""
    c = run.config
    net = c.get("net_params", {}) or {}
    env = c.get("env_params", {}) or {}
    tags = set(run.tags)

    if c.get("env") != "AbsoluteImitation" or c.get("seed") != 42:
        return None
    if not std_arch(net) or c.get("efference_length") != c.get("delay_k"):
        return None
    if run.summary.get("_step") != EXPECTED_STEP:
        return None

    is_fm = "ForwardModel" in tags
    network = "efference" if (not is_fm and "EncDec" in tags) else None
    if is_fm:
        # Canonical explicit FM (trained, detached) vs the policy-gradient FM (loss 0,
        # gradient flows through the predictor).
        if c.get("fm_loss_weight") == 1 and c.get("detach_prediction") in (None, True):
            network = "forward_model"
        elif c.get("fm_loss_weight") == 0 and c.get("detach_prediction") is False:
            network = "pg_forward_model"
    if network is None:
        return None

    key = (
        str(env.get("walker_xml_path")).split("/")[-1],
        env.get("torque_actuators"),
        network,
        c.get("fm_loss_weight"),
        c.get("detach_prediction"),
        env.get("body_target_frame"),
    )
    for name, (*spec, dates) in CONDITIONS.items():
        if tuple(spec) == key and (dates is None or run.created_at[:10] in dates):
            return name
    return None


def record_of(run, cond: str) -> dict:
    c = run.config
    net = c.get("net_params", {}) or {}
    env = c.get("env_params", {}) or {}
    ppo = (c.get("config") or {}).get("ppo", {}) or {}
    s = run.summary
    md = run.metadata or {}
    xml = str(env.get("walker_xml_path")).split("/")[-1]

    hist = run.history(
        keys=["throughput/train_sps", "throughput/eval_sps"], samples=400, pandas=True
    )
    def med(col):
        if col not in hist or hist[col].dropna().empty:
            return None
        # Drop the first 10% of samples: the first iterations include XLA compilation.
        v = hist[col].dropna().to_numpy()
        return float(np.median(v[max(1, len(v) // 10):]))

    return {
        # provenance
        "wandb_id": run.id,
        "wandb_name": run.name,
        "wandb_project": PROJECT,
        "state": run.state,
        "git_commit": git8(run),
        "tags": ",".join(sorted(run.tags)),
        "notes": (run.notes or "").strip(),
        "created_at": run.created_at,
        # experimental axes
        "condition": cond,
        "xml": "new" if xml == NEW_XML else "old",
        "walker_xml": xml,
        "network": CONDITIONS[cond][2],
        "control_mode": "torque" if env.get("torque_actuators") else "position",
        "delay_k": c.get("delay_k"),
        "efference_length": c.get("efference_length"),
        # authoritative frame + fm knobs (sanity)
        "body_target_frame": env.get("body_target_frame"),
        "net_params_body_target_frame": net.get("body_target_frame"),
        "torque_actuators": env.get("torque_actuators"),
        "fm_loss_weight": c.get("fm_loss_weight"),
        "detach_prediction": c.get("detach_prediction"),
        # invariants
        "env": c.get("env"),
        "seed": c.get("seed"),
        "latent_size": net.get("latent_size"),
        "kl_weight": net.get("kl_weight"),
        "enc_hidden_sizes": tuple(net.get("enc_hidden_sizes") or []),
        "dec_hidden_sizes": tuple(net.get("dec_hidden_sizes") or []),
        "critic_hidden_sizes": tuple(net.get("critic_hidden_sizes") or []),
        "clip_length": env.get("clip_length"),
        "ctrl_dt": env.get("ctrl_dt"),
        "sim_dt": env.get("sim_dt"),
        "solver": env.get("solver"),
        "iterations": env.get("iterations"),
        "ls_iterations": env.get("ls_iterations"),
        "njmax": env.get("njmax"),
        "naconmax": env.get("naconmax"),
        "rescale_factor": env.get("rescale_factor"),
        "mujoco_impl": env.get("mujoco_impl"),
        "n_envs": ppo.get("n_envs"),
        "total_steps": ppo.get("total_steps"),
        "actual_step": s.get("_step"),
        # hardware (speed comparisons are only valid within one GPU model)
        "gpu": md.get("gpu"),
        "host": md.get("host"),
        # metrics
        "reward_mean": coalesce(s, REWARD_MEAN_KEYS),
        "reward_std": coalesce(s, REWARD_STD_KEYS),
        "lifespan_mean": coalesce(s, LIFESPAN_KEYS),
        "train_sps_final": s.get("throughput/train_sps"),
        "eval_sps_final": s.get("throughput/eval_sps"),
        "train_sps_median": med("throughput/train_sps"),
        "eval_sps_median": med("throughput/eval_sps"),
        "runtime_s": s.get("_runtime"),
    }


def learning_curve(run, cond: str, delay: int) -> list[dict]:
    """The run's eval-reward series (eval runs every 10M steps -> ~60 points).

    Uses the *sampled* history endpoint rather than ``scan_history``: the runs log
    ~7 300 iterations x ~50 keys, and streaming all of that for 80 runs takes hours,
    whereas the eval series itself is only ~60 points. ``samples`` is set well above
    the true number of eval points so no sampling actually happens.
    """
    key = "eval/episode_reward/mean" if any(
        k.startswith("eval/episode_reward") for k in run.summary.keys()
    ) else "episode_reward/mean"
    hist = run.history(keys=[key], samples=500, pandas=True)
    if key not in hist:
        return []
    hist = hist.dropna(subset=[key])
    return [
        {
            "wandb_id": run.id,
            "condition": cond,
            "delay_k": delay,
            "step": int(row["_step"]),
            "reward_mean": float(row[key]),
        }
        for _, row in hist.iterrows()
    ]


def main() -> None:
    runs = fetch_runs(PROJECT, finished_only=True, tags=REQUIRE_TAGS)
    print(f"Fetched {len(runs)} finished runs with tags {REQUIRE_TAGS}")

    records, curves = [], []
    for r in runs:
        cond = condition_of(r)
        if cond is None:
            continue
        records.append(record_of(r, cond))
        curves.extend(learning_curve(r, cond, r.config.get("delay_k")))

    df = records_to_df(records)
    df = df.sort_values(["network", "xml", "delay_k", "wandb_id"]).reset_index(drop=True)
    curves_df = pd.DataFrame(curves).sort_values(["condition", "delay_k", "step"])

    print(f"\nCohort ({len(df)} rows):")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:24s} n={len(sub):2d} delays={sorted(sub['delay_k'])}")
        print(f"  {'':24s} xml={sub['walker_xml'].unique()} "
              f"torque={sub['torque_actuators'].unique()} "
              f"frame={sub['body_target_frame'].unique()} "
              f"git={sorted(sub['git_commit'].unique())} "
              f"gpu={sorted(set(sub['gpu'].dropna()))}")

    # The frame question (Q3): assert what the new-XML runs *actually* used.
    print("\nbody_target_frame by xml (authoritative env_params value):")
    print(pd.crosstab(df["xml"], df["body_target_frame"]))
    print("net_params.body_target_frame (inert copy) values:",
          df["net_params_body_target_frame"].unique())

    report = comparability_report(df, invariant_cols=INVARIANTS, group_col="condition")
    print("\n" + report)
    print("\ngit commits:", git_commit_summary(df))

    (HERE / "data.csv").write_text(df.to_csv(index=False))
    (HERE / "curves.csv").write_text(curves_df.to_csv(index=False))
    (HERE / "comparability.txt").write_text(report + "\n")
    print(f"\nWrote {len(df)} rows to {HERE / 'data.csv'}")
    print(f"Wrote {len(curves_df)} rows to {HERE / 'curves.csv'}")


if __name__ == "__main__":
    main()
