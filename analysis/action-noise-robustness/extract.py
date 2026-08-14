"""Is the explicit forward model more sensitive to motor noise than the enc-dec?

The question
------------
The explicit forward model replaces the decoder's proprioceptive input with the
predictor's output — there is no skip path around it
(``vnl_experiments/delays/forward_model.py``). So the architecture's advantage over the
enc-dec baseline may rest on execution being clean: if the body does not do what the
efference copy says it did, the prediction is wrong by construction and the policy has no
unfiltered channel to fall back on.

To test that, the same checkpoints are re-evaluated with a **fixed** Gaussian
perturbation added to the executed action. Fixed rather than the policy's own learned
``std``: the learned std is state-dependent and differs per run and architecture, so
reusing it would confound "robust to perturbation" with "has a wide learned
distribution". Every condition gets the identical perturbation.

Where the noise enters, and why it matters
------------------------------------------
The noise is added *after* the network has produced its action and *outside*
``EfferenceCopy``, so the efference queue holds the **intended** action while the body
executes the perturbed one. This is unobserved motor noise: the predictor's input is the
command, its target is the consequence of the command *plus* noise, and the discrepancy
is irreducible. The noise does reach the policy eventually and indirectly, through the
env's ``prev_action`` / ``actuator_ctrl`` proprioception channels, delayed by ``delay_k``.

Both arms use ``EfferenceCopy`` and both get a clean queue, so the comparison is
symmetric. sigma is in post-tanh action units, i.e. a fraction of the actuator
half-range (actions in ``[-1, 1]`` map to +-max force with torque actuators), clipped
back into range; the effect is therefore the same size wherever ``mu`` sits.

The second question: does wider training exploration buy robustness?
--------------------------------------------------------------------
The 2026-08-13 tranche trained with ``min_std = 0.25`` rather than 0.1, i.e. with 2.5x the
exploration noise. If robustness to *evaluation* noise is partly just "was trained under
noise", those runs should degrade more slowly. They exist for the explicit and the
policy-gradient forward model at delays 0/5/10/20/50 — **not** for the enc-dec, so this
axis is internal to the forward model and cannot be crossed with the architecture
comparison above.

Producing the artifacts
-----------------------
One eval artifact per (run, sigma), each covering all three datasets::

    python -m vnl_experiments.artifacts plan --kind eval \\
        --runs analysis/action-noise-robustness/runs.csv \\
        --set action_noise=0.05 --out todo_n05.txt
    # cluster:
    sbatch slurm_eval.sh todo_n05.txt eval --set action_noise=0.05
    python -m vnl_experiments.artifacts pull --kind eval \\
        --runs analysis/action-noise-robustness/runs.csv

Run it
------
    ../.venv/bin/python analysis/action-noise-robustness/extract.py           # frozen
    ../.venv/bin/python analysis/action-noise-robustness/extract.py --refresh
    ../.venv/bin/python analysis/action-noise-robustness/extract.py --check

Output is a **long** ``data.csv``: one row per (run, sigma, dataset). Cells whose artifact
is missing are emitted with ``have_artifact = False`` and null metrics rather than
dropped, so a partial sweep can never read as a complete one. Cumulative
``episode_reward`` is comparable only *within* a dataset (``new_eval`` clips are 30 s /
3002 steps vs 5 s / 502), so ``reward_per_step`` and ``hazard_rate`` are also emitted.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: All three splits: the 80 % train split, the held-out 20 % (``old_eval``), and the fixed
#: 32-clip 30 s set (``new_eval``). Keeping ``train`` is what makes it possible to ask
#: whether noise sensitivity is a generalisation phenomenon or a control one.
DATASETS = ("train", "old_eval", "new_eval")

#: Noise levels, in post-tanh action units. 0.0 is included as a *measured* baseline so
#: the zero point comes from the same code path, spec and batch as the noisy points
#: rather than from the pre-existing ``eval3ds-66aaff5b`` records.
#:
#: A smoke test on two ``min_std = 0.1`` checkpoints lost ~90 % of lifespan at
#: sigma = 0.1 already (318 -> 30 steps for the explicit arm, 227 -> 46 for the enc-dec),
#: so 0.02 and 0.05 are where the graded difference between the arms should live. 0.25 is
#: kept anyway: that saturation argument was measured on ``min_std = 0.1`` policies, and
#: whether the ``min_std = 0.25`` tranche still has a floor left at sigma = 0.25 is
#: exactly the secondary question. Expect the ``min_std = 0.1`` arms to be on the floor
#: there — that is the contrast, not a wasted cell.
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.1, 0.25)

#: Committed ``sigma -> spec_id``. The ids are derived from the producer at import time so
#: they cannot drift out of sync with the spec, and asserted against these constants so a
#: ``VERSION`` bump or a spec change is a loud failure rather than a silent switch to
#: different data. Regenerate by running this module and pasting the printed mapping.
EXPECTED_SPEC_IDS = {
    0.0: "eval3ds-n00-04ceda93",
    0.02: "eval3ds-n02-5bcf9203",
    0.05: "eval3ds-n05-ead26b7d",
    0.1: "eval3ds-n10-2d7c9136",
    0.25: "eval3ds-n25-9443d2fe",
}


def spec_for(sigma: float) -> dict:
    return get_producer("eval").spec(datasets=list(DATASETS), action_noise=sigma)


def spec_ids() -> dict[float, str]:
    producer = get_producer("eval")
    out = {s: producer.spec_id(spec_for(s)) for s in NOISE_LEVELS}
    drift = {s: (sid, EXPECTED_SPEC_IDS.get(s)) for s, sid in out.items()
             if EXPECTED_SPEC_IDS.get(s) != sid}
    if drift:
        raise SystemExit(
            "eval spec_ids have drifted from the committed constants: "
            + ", ".join(f"sigma={s}: got {got}, expected {want}"
                        for s, (got, want) in drift.items())
            + "\nThe artifacts this analysis reads were made by a different eval spec or "
              "producer VERSION. Update EXPECTED_SPEC_IDS deliberately, and re-produce.")
    return out


REQUIRES = ["index"] + [f"eval:{EXPECTED_SPEC_IDS[s]}" for s in NOISE_LEVELS]


# --------------------------------------------------------------------------------------
# cohort definition
# --------------------------------------------------------------------------------------

NEW_XML = "rodent_no_tail_collisions.xml"
EXPECTED_STEP = 600_064_000

#: Terminal states that are acceptable *given* that ``summary._step`` reached
#: ``EXPECTED_STEP``, following ``analysis/collision-model-xml``. Completing all 600 M
#: steps is the real inclusion criterion; the WandB state only says how the process
#: exited. The 2026-08-11 ``ef060b73`` tranche — which is both of the primary conditions
#: here — is marked ``failed`` because it died in the *post-training* evaluation: it has
#: every training metric, normal runtimes, and no ``final_eval/*`` keys. A run that died
#: during training cannot reach ``EXPECTED_STEP``, so this pair of conditions admits the
#: former and excludes the latter. The three ``crashed`` forward-model runs at
#: ``25732c42`` have ``_step = NaN`` and are excluded by that gate, not by the state list.
ACCEPTED_STATES = ("finished", "failed")

#: Shared by every run in the cohort. ``min_std`` is deliberately *not* here — it is the
#: second experimental axis (see the module docstring).
COHORT = {
    "env": "AbsoluteImitation",
    "seed": 42,
    "env_params.torque_actuators": True,
    "env_params.body_target_frame": "reference_root",
    "net_params.latent_size": 32,
    "net_params.kl_weight": 0.001,
    "net_params.std_scale": 1,
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
    "config.ppo.total_steps": 600_000_000,
}


def _tagged(df: pd.DataFrame, tag: str) -> pd.Series:
    return df["tags"].fillna("").str.split(",").apply(lambda t: tag in t)


def _base(df: pd.DataFrame) -> pd.Series:
    mask = (
        (df["summary._step"] == EXPECTED_STEP)
        & df["state"].isin(ACCEPTED_STATES)
        & (df["delay_k"] == df["efference_length"])
        & df["env_params.walker_xml_path"].astype(str).str.contains(NEW_XML)
    )
    for column, value in COHORT.items():
        mask &= df[column] == value
    return mask


def _network(df: pd.DataFrame, kind: str) -> pd.Series:
    is_fm = _tagged(df, "ForwardModel")
    if kind == "encdec":
        return _tagged(df, "EncDec") & ~is_fm
    if kind == "explicit_fm":
        # `fillna(True)`: unset means the constructor default, which is True.
        return (is_fm & (df["fm_loss_weight"] == 1)
                & df["detach_prediction"].fillna(True).astype(bool))
    if kind == "pg_fm":
        return is_fm & (df["fm_loss_weight"] == 0) & (df["detach_prediction"] == False)  # noqa: E712
    raise ValueError(kind)


def _cell(network: str, min_std: float):
    def selector(df: pd.DataFrame) -> pd.Series:
        return (_base(df) & _network(df, network)
                & (df["net_params.min_std"] == min_std))

    return selector


#: Two primary conditions (the matched 23-delay ``ef060b73`` sweep, one run per delay per
#: arm) plus the two ``min_std = 0.25`` conditions at delays 0/5/10/20/50. There is no
#: enc-dec run at ``min_std = 0.25`` anywhere in the project, so the exploration-width
#: axis exists only inside the forward model.
CONDITIONS = {
    "expfm": _cell("explicit_fm", 0.1),
    "encdec": _cell("encdec", 0.1),
    "expfm_std25": _cell("explicit_fm", 0.25),
    "pgfm_std25": _cell("pg_fm", 0.25),
}

ARM_OF = {"expfm": "explicit", "encdec": "encdec",
          "expfm_std25": "explicit", "pgfm_std25": "implicit"}
MIN_STD_OF = {"expfm": 0.1, "encdec": 0.1, "expfm_std25": 0.25, "pgfm_std25": 0.25}

#: ``min_std`` and ``git_commit`` vary by design (the two ``std25`` conditions are a
#: separate tranche at ``d02b854a``); everything else must be single-valued.
INVARIANTS = [
    "env", "seed", "net_params.latent_size", "net_params.kl_weight",
    "net_params.latent_min_std", "net_params.std_scale",
    "net_params.enc_hidden_sizes", "net_params.dec_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "env_params.clip_length", "env_params.ctrl_dt", "env_params.sim_dt",
    "env_params.solver", "env_params.iterations", "env_params.ls_iterations",
    "env_params.njmax", "env_params.naconmax", "env_params.rescale_factor",
    "env_params.mujoco_impl", "env_params.walker_xml_path",
    "env_params.torque_actuators", "env_params.body_target_frame",
    "config.ppo.n_envs", "config.ppo.learning_rate", "config.ppo.rollout_length",
    "config.ppo.total_steps", "summary._step",
    "net_params.min_std", "git_commit",
]

_TERMINATION_REASONS = ("root_too_far", "root_too_rotated", "pose_error",
                        "nan_termination")


# --------------------------------------------------------------------------------------
# reading the eval artifacts
# --------------------------------------------------------------------------------------


def read_eval(store: Store, wandb_id: str, spec_id: str) -> dict | None:
    entry = store.lookup("eval", wandb_id, spec_id)
    if entry is None:
        return None
    record = json.loads((store.root / entry.path).read_text())
    record["_checkpoint_step"] = (entry.resolved or {}).get("checkpoint_step")
    return record


def _fm_pred_mse(net_metrics: dict) -> float | None:
    """The predictor's L2 error against the true current proprioception. The metric path
    embeds the layer index, so match on the leaf name. Logged by both forward-model arms
    (in the ``pg_fm`` arm the target is never used in the loss, so it is the "would-be"
    prediction error); absent for the enc-dec."""
    hits = [v for k, v in net_metrics.items() if k.endswith("fm_pred_mse")]
    return float(hits[0]) if hits else None


def build_row(run: pd.Series, sigma: float, spec_id: str, dataset: str,
              record: dict | None) -> dict:
    condition = run["condition"]
    row = {
        "condition": condition,
        "arm": ARM_OF[condition],
        "min_std": MIN_STD_OF[condition],
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "state": run["state"],
        "delay_k": int(run["delay_k"]),
        "git_commit": (run.get("git_commit") or "")[:8],
        "action_noise": sigma,
        "spec_id": spec_id,
        "dataset": dataset,
        "have_artifact": False,
    }
    if record is None or dataset not in record.get("datasets", {}):
        return row

    data = record["datasets"][dataset]
    lifespan = data["lifespan_steps"]["mean"]
    survived = data["termination_rate"]["survived"]
    row.update({
        "have_artifact": True,
        "checkpoint_step": record.get("_checkpoint_step") or record.get("step"),
        # The record's own copy of the measurement setting; must agree with the spec.
        "record_action_noise": record.get("action_noise"),
        "n_clips": data["n_clips"],
        "n_steps": data["n_steps"],
        "episode_reward": data["episode_reward"]["mean"],
        "episode_reward_std": data["episode_reward"]["std"],
        "lifespan_steps": lifespan,
        "lifespan_s": data["lifespan_s"]["mean"],
        "survived": survived,
        # Length-fair metrics (README §6): reward per step is comparable where a
        # cumulative total is not, and the hazard rate turns "fraction that died" into a
        # per-step rate that does not depend on how long the clip is.
        "reward_per_step": (data["episode_reward"]["mean"] / lifespan
                            if lifespan else np.nan),
        "hazard_rate": ((1.0 - survived) / lifespan
                        if survived is not None and lifespan else np.nan),
        "fm_pred_mse": _fm_pred_mse(data["net_metrics"]),
    })
    for reason in _TERMINATION_REASONS:
        row[f"term_{reason}"] = data["termination_rate"].get(reason)
    for key in ("joint_l2_error", "root_pos_distance", "body_errors/total"):
        if key in data["errors"]:
            row[key.replace("/", "_")] = data["errors"][key]["mean"]
    return row


def main() -> None:
    args = pipeline.parse_args(__doc__)

    ids = spec_ids()
    print("eval spec ids (all three datasets):")
    for sigma, spec_id in ids.items():
        print(f"  sigma={sigma:<5} {spec_id}")

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)
    print(f"\nSweep size: {len(runs)} runs x {len(NOISE_LEVELS)} sigma = "
          f"{len(runs) * len(NOISE_LEVELS)} eval artifacts")

    store = Store()
    rows = []
    for _, run in runs.iterrows():
        for sigma, spec_id in ids.items():
            record = read_eval(store, run["wandb_id"], spec_id)
            for dataset in DATASETS:
                rows.append(build_row(run, sigma, spec_id, dataset, record))

    df = pd.DataFrame(rows).sort_values(
        ["condition", "delay_k", "wandb_id", "action_noise", "dataset"],
        ignore_index=True)

    present = df[df["have_artifact"]]
    if len(present):
        # A mismatch means an artifact was filed under the wrong spec_id, and every
        # figure built on it would be mislabelled.
        bad = present[~np.isclose(present["record_action_noise"].astype(float),
                                  present["action_noise"].astype(float))]
        if len(bad):
            raise SystemExit(f"action_noise mismatch between spec and record:\n{bad}")

    report = comparability_report(runs, invariant_cols=INVARIANTS,
                                  group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report + "\n")
    print(report)

    print("\nCohort:")
    for condition, sub in runs.groupby("condition"):
        delays = sorted(sub["delay_k"]) if "delay_k" in sub else []
        print(f"  {condition:12s} n={len(sub):3d}  arm={ARM_OF[condition]:8s} "
              f"min_std={MIN_STD_OF[condition]}  states={sorted(set(sub['state']))}"
              + (f"  delays={delays}" if delays else ""))

    print(f"\nCoverage: {len(present)}/{len(df)} (run, sigma, dataset) cells")
    if len(present):
        for dataset in DATASETS:
            sub = present[present["dataset"] == dataset]
            if sub.empty:
                continue
            print(f"\nreward_per_step on {dataset}:")
            print(sub.pivot_table(index="action_noise", columns="condition",
                                  values="reward_per_step").round(3).to_string())

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
