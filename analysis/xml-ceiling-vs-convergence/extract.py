"""Is the new-XML reward deficit a lower ceiling, or just a slower climb?

The question
------------
[`collision-model-xml/`](../collision-model-xml/) found that the new (almost-full-
collision) walker XML costs up to -20 % reward under torque control in a band of
observation delays (~25-60), and closed with a hedge: at delay 30 the new arm was still
climbing hard at 600 M steps (unfinished training), while at delays 50-60 it had
flattened (a real plateau). That split was inferred from the *remaining slope* at the end
of a 600 M-step run -- an extrapolation.

On 2026-08-13 eight runs of the going-forward configuration (**new XML +
`reference_root` + torque**) were trained for **2 G steps**, 3.3x the standard budget, at
delays 0/10/20/50 in two networks. They turn the extrapolation into a measurement:

* if the 600 M deficit *closes* when the new arm is given more steps, it was
  **convergence speed**;
* if the new arm *flattens below* the baseline, it is a **ceiling**.

Design
------
Three tiers at the same four delays, so every comparison is delay- and network-matched:

===========================  =========================================================
``*_new_2g``                 the 2 G-step runs, new XML + ``reference_root``
``*_new_600m``               the same configuration at the standard 600 M budget
``*_old_600m``               the old-XML + ``current_root`` baseline it is compared to
===========================  =========================================================

in the explicit forward model (``expfm``: ``fm_loss_weight = 1``, prediction detached)
and the policy-gradient forward model (``pgfm``: ``fm_loss_weight = 0``, not detached).
``expfm_oldref_600m`` adds the frame-matched baseline (old XML + ``reference_root``) as a
control, since the primary contrast moves XML and frame together.

The ``*_new_600m`` tier does double duty. It is the step-matched arm of the contrast, and
-- because a 2 G run and a 600 M run of the same configuration are the *same process* up
to the stopping point (constant learning rate; ``total_steps`` only bounds the training
loop, see ``nnx_ppo/algorithms/ppo.py``) -- comparing a 2 G run's reward *at 600 M* with
its 600 M twin measures nothing but run-to-run noise. That gives this cohort something
every previous single-seed question lacked: an empirical noise floor.

What this design cannot do
--------------------------
There is **no 2 G old-XML run**. "Ceiling" therefore means "flat, and below the
baseline's 600 M value", not "below the baseline's asymptote" -- the baselines are
themselves still gaining a few percent per 100 M at 600 M. It is a one-sided test: it can
show the new XML fails to catch up, and it can show it catches up, but it cannot compare
converged optima. Also, the 2 G runs sample delays 0/10/20/50, so only **delay 50** lies
inside the 25-60 deficit band, and there is no 2 G EncDec run -- the network with the
largest deficit.

Data sources
------------
Configs and summaries from the committed run index; reward/lifespan curves from
``history`` artifacts (200 sampled points for the 2 G runs, 60 for the 600 M ones -- eval
runs every 10 M steps either way); offline batch evaluation from ``eval`` artifacts, now
**29/29** (see ``coverage.txt``). Nothing here touches the WandB API.

The eval spec id is the **v2** one (``eval3ds-347333e3``), minted by the 2026-08-18
walker-XML fix. Before that fix every offline eval of a new-XML run was silently
re-simulated on ``rodent.xml``, which cost the new arm up to 42 % of its reward and would
have manufactured exactly the ceiling this question is testing for. ``assert_artifact_body``
below refuses any eval artifact that does not stamp the body its run trained on, so this
analysis cannot read the pre-fix generation even by accident.

Run it
------
    ../.venv/bin/python analysis/xml-ceiling-vs-convergence/extract.py           # frozen
    ../.venv/bin/python analysis/xml-ceiling-vs-convergence/extract.py --refresh
    ../.venv/bin/python analysis/xml-ceiling-vs-convergence/extract.py --check
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"

#: The v2 eval spec: ``EvalProducer.VERSION = 2``, post walker-XML fix. Pinned rather than
#: resolved from the producer so that a future bump is a loud failure in
#: ``assert_spec_ids`` rather than a silent swap to a different generation of data.
EVAL_SPEC_ID = "eval3ds-347333e3"
EVAL_DATASETS = ("train", "old_eval", "new_eval")

REQUIRES = ["index", "history", f"eval:{EVAL_SPEC_ID}"]

NEW_XML = "rodent_no_tail_collisions.xml"

#: The two training budgets, and the ``summary._step`` each actually reaches. Steps come
#: in whole rollouts, so the run overshoots ``total_steps`` slightly and lands on a fixed
#: value; requiring it is how a crashed run is excluded (see ACCEPTED_STATES).
BUDGETS = {600_000_000: 600_064_000, 2_000_000_000: 2_000_076_800}

#: The delays the 2 G cohort covers. Everything is restricted to these so that every
#: condition is the same four cells and no contrast is carried by an unmatched delay.
DELAYS = (0, 10, 20, 50)

STD_MIN_STD = 0.1
STD_ARCH = {
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
}

REWARD_MEAN_KEYS = ("summary.eval/episode_reward/mean", "summary.episode_reward/mean")
LIFESPAN_KEYS = ("summary.eval/lifespan/mean", "summary.lifespan_mean")
#: The logging API was renamed mid-project; the June baselines still use the old names.
CURVE_REWARD_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")
CURVE_LIFESPAN_KEYS = ("eval/lifespan/mean", "lifespan_mean")

#: Averaging window for every "reward at step X" number below. Eval runs every 10 M
#: steps and a single eval point moves ~1-2 % on GPU-nondeterministic physics, so a point
#: value at one step is not a measurement; the mean of the five points in
#: ``(X - 50 M, X]`` is.
WINDOW = 50_000_000

#: Where the curves are read off. 600 M is the step-matched comparison point, 2 G the
#: end of the long runs.
MILESTONES = (200_000_000, 400_000_000, 600_000_000, 1_000_000_000,
              1_500_000_000, 2_000_000_000)


# --------------------------------------------------------------------------------------
# condition selectors
# --------------------------------------------------------------------------------------


#: Terminal states acceptable *given* that ``summary._step`` reached the budget's
#: expected value. The 2026-08-11 new-XML explicit-FM cohort (``ef060b73``) is marked
#: ``failed`` because it died in the post-training evaluation, after completing every
#: training step. A run that died *during* training cannot reach the expected step, so
#: this pair of conditions admits the former and excludes the latter.
ACCEPTED_STATES = ("finished", "failed")


def _is_new_xml(df: pd.DataFrame) -> pd.Series:
    return df["env_params.walker_xml_path"].astype(str).str.contains(NEW_XML)


def _network(df: pd.DataFrame, kind: str) -> pd.Series:
    """Network identity. No single config field separates the explicit forward model
    from the policy-gradient one, so both forward-model knobs are needed."""
    is_fm = df["tags"].fillna("").str.split(",").apply(lambda t: "ForwardModel" in t)
    if kind == "expfm":
        # `ne(False)` rather than `fillna(True)`: an unset detach_prediction means the
        # constructor default, which is True.
        return is_fm & (df["fm_loss_weight"] == 1) & df["detach_prediction"].ne(False)
    if kind == "pgfm":
        return is_fm & (df["fm_loss_weight"] == 0) & (df["detach_prediction"] == False)  # noqa: E712
    raise ValueError(kind)


def _cell(network: str, *, budget: int, new_xml: bool, frame: str,
          commits: tuple[str, ...] | None = None):
    def selector(df: pd.DataFrame) -> pd.Series:
        mask = (
            (df["env"] == "AbsoluteImitation")
            & (df["seed"] == 42)
            & (df["net_params.min_std"] == STD_MIN_STD)
            & (df["delay_k"] == df["efference_length"])
            & pipeline.full_decoder_inputs_mask(df)
            & df["delay_k"].isin(DELAYS)
            & (df["env_params.torque_actuators"] == True)  # noqa: E712
            & (df["config.ppo.total_steps"] == budget)
            & (df["summary._step"] == BUDGETS[budget])
            & _network(df, network)
            & (_is_new_xml(df) == new_xml)
            & (df["env_params.body_target_frame"] == frame)
            & df["tags"].fillna("").str.split(",").apply(lambda t: "TrainEvalSplit" in t)
            & df["state"].isin(ACCEPTED_STATES)
        )
        for column, value in STD_ARCH.items():
            mask &= df[column] == value
        if commits is not None:
            mask &= df["git_commit"].str[:8].isin(commits)
        return mask

    return selector


CONDITIONS = {
    # --- the long runs -----------------------------------------------------------------
    "expfm_new_2g": _cell("expfm", budget=2_000_000_000, new_xml=True,
                          frame="reference_root"),
    "pgfm_new_2g": _cell("pgfm", budget=2_000_000_000, new_xml=True,
                         frame="reference_root"),

    # --- the same configuration at the standard budget ---------------------------------
    # Step-matched arm of the contrast, and the replication partner of the 2 G runs.
    "expfm_new_600m": _cell("expfm", budget=600_000_000, new_xml=True,
                            frame="reference_root"),
    "pgfm_new_600m": _cell("pgfm", budget=600_000_000, new_xml=True,
                           frame="reference_root"),

    # --- the baselines -----------------------------------------------------------------
    # Pinned to the same launch tranches collision-model-xml used, so the numbers here
    # and there refer to the same runs.
    "expfm_old_600m": _cell("expfm", budget=600_000_000, new_xml=False,
                            frame="current_root", commits=("54643764",)),
    "pgfm_old_600m": _cell("pgfm", budget=600_000_000, new_xml=False,
                           frame="current_root", commits=("d4bd4dc0",)),
    # Frame-matched control: the primary contrast moves XML and frame together, this
    # cell holds the frame at reference_root on the old XML.
    "expfm_oldref_600m": _cell("expfm", budget=600_000_000, new_xml=False,
                               frame="reference_root", commits=("909e774d",)),
}

#: ``(baseline, changed, label)``. Read as "changed vs baseline".
PAIRS = {
    # The question: does 3.3x the budget close the 600 M deficit?
    "budget_expfm": ("expfm_old_600m", "expfm_new_2g", "Explicit FM: new @2G vs old @600M"),
    "budget_pgfm": ("pgfm_old_600m", "pgfm_new_2g", "PG-FM: new @2G vs old @600M"),
    # The deficit as collision-model-xml measured it, restricted to these four delays.
    "matched_expfm": ("expfm_old_600m", "expfm_new_600m", "Explicit FM: new vs old @600M"),
    "matched_pgfm": ("pgfm_old_600m", "pgfm_new_600m", "PG-FM: new vs old @600M"),
    # Same config, two runs: the noise floor.
    "replicate_expfm": ("expfm_new_600m", "expfm_new_2g", "Explicit FM: 2G run @600M vs its 600M twin"),
    "replicate_pgfm": ("pgfm_new_600m", "pgfm_new_2g", "PG-FM: 2G run @600M vs its 600M twin"),
    # Frame-matched baseline control.
    "frame_control": ("expfm_old_600m", "expfm_oldref_600m",
                      "Explicit FM: reference_root vs current_root, old XML"),
}

INVARIANTS = [
    "env", "seed", "net_params.latent_size", "net_params.kl_weight",
    "net_params.min_std", "net_params.latent_min_std", "net_params.std_scale",
    "net_params.enc_hidden_sizes", "net_params.dec_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "env_params.clip_length", "env_params.ctrl_dt", "env_params.sim_dt",
    "env_params.solver", "env_params.iterations", "env_params.ls_iterations",
    "env_params.njmax", "env_params.naconmax", "env_params.rescale_factor",
    "env_params.mujoco_impl",
    "config.ppo.n_envs", "config.ppo.learning_rate", "config.ppo.rollout_length",
    "config.ppo.n_epochs", "config.ppo.n_minibatches", "config.ppo.clip_range",
    "config.ppo.discounting_factor", "config.ppo.gae_lambda",
    "config.ppo.total_steps", "summary._step",
    "env_params.walker_xml_path", "env_params.torque_actuators",
    "env_params.body_target_frame", "git_commit",
]

NETWORK_OF = {"expfm": "forward_model", "pgfm": "pg_forward_model"}


# --------------------------------------------------------------------------------------
# curve reductions
# --------------------------------------------------------------------------------------


def history_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def series(hist: pd.DataFrame | None, keys: tuple[str, ...]) -> pd.DataFrame | None:
    """The first of ``keys`` the run actually logged, as a tidy ``step``/``value`` frame."""
    if hist is None:
        return None
    key = next((k for k in keys if k in hist.columns), None)
    if key is None:
        return None
    out = hist.dropna(subset=[key])[["_step", key]]
    return out.rename(columns={"_step": "step", key: "value"}).sort_values("step")


def window_mean(curve: pd.DataFrame | None, step: int,
                window: int = WINDOW) -> float | None:
    """Mean of the eval points in ``(step - window, step]``.

    ``None`` if the run never got there, or if fewer than three points land in the
    window -- a one- or two-point "average" would reintroduce the noise this is for.
    """
    if curve is None:
        return None
    sub = curve[(curve["step"] > step - window) & (curve["step"] <= step)]
    return float(sub["value"].mean()) if len(sub) >= 3 else None


def smoothed(curve: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Trailing-window mean at every eval step, for threshold crossings."""
    points = int(round(window / 10_000_000))  # eval every 10 M steps
    out = curve.copy()
    out["value"] = out["value"].rolling(points, min_periods=1).mean()
    return out


def first_step_at_least(curve: pd.DataFrame | None, target: float) -> float | None:
    """First step whose smoothed reward reaches ``target``; ``None`` if never."""
    if curve is None or not np.isfinite(target):
        return None
    hit = smoothed(curve)
    hit = hit[hit["value"] >= target]
    return None if hit.empty else float(hit["step"].iloc[0])


def curve_stats(curve: pd.DataFrame | None) -> dict:
    """Endpoint, peak and remaining-slope summary of one training curve."""
    if curve is None or curve.empty:
        return {}
    last = int(curve["step"].iloc[-1])
    final = window_mean(curve, last)
    sm = smoothed(curve)
    peak_at = int(sm["value"].idxmax())
    stats = {
        "final_step": last,
        "reward_final": final,
        "reward_peak": float(sm["value"].max()),
        "peak_step": int(sm.loc[peak_at, "step"]),
    }
    for label, back in (("100M", 100_000_000), ("500M", 500_000_000)):
        earlier = window_mean(curve, last - back)
        stats[f"gain_last_{label}_pct"] = (
            None if earlier in (None, 0) or final is None
            else 100 * (final / earlier - 1))
    # How much of the final level was already reached at the standard budget.
    if final:
        for frac in (0.95, 0.99):
            step = first_step_at_least(curve, frac * final)
            stats[f"steps_to_{int(frac * 100)}pct_of_final"] = step
    return stats


def build_row(run: pd.Series, hist: pd.DataFrame | None) -> dict:
    condition = run["condition"]
    reward = series(hist, CURVE_REWARD_KEYS)
    lifespan = series(hist, CURVE_LIFESPAN_KEYS)
    row = {
        # provenance
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "wandb_project": PROJECT,
        "state": run["state"],
        "git_commit": (run["git_commit"] or "")[:8],
        "created_at": run["created_at"],
        "tags": run["tags"],
        # experimental axes
        "condition": condition,
        "network": NETWORK_OF[condition.split("_")[0]],
        "xml": "new" if _is_new_xml(pd.DataFrame([run])).iat[0] else "old",
        "body_target_frame": run["env_params.body_target_frame"],
        "budget": int(run["config.ppo.total_steps"]),
        "delay_k": int(run["delay_k"]),
        # invariants worth eyeballing in the CSV itself
        "min_std": run["net_params.min_std"],
        "learning_rate": run["config.ppo.learning_rate"],
        "n_envs": run["config.ppo.n_envs"],
        "actual_step": run["summary._step"],
        "gpu": run["gpu"],
        "runtime_s": run.get("summary._runtime"),
        # end-of-training summary (eval_env == train_env: eval on the training clips)
        "summary_reward": pipeline.first_present(run, *REWARD_MEAN_KEYS),
        "summary_lifespan": pipeline.first_present(run, *LIFESPAN_KEYS),
    }
    row.update(curve_stats(reward))
    row["lifespan_final"] = (None if lifespan is None
                             else window_mean(lifespan, int(lifespan["step"].iloc[-1])))
    for milestone in MILESTONES:
        row[f"reward_at_{milestone // 1_000_000}M"] = window_mean(reward, milestone)
    row["lifespan_at_600M"] = window_mean(lifespan, 600_000_000)
    # Inline end-of-training eval; present only on runs from 2026-08-10 onward, and a
    # *different* measurement from the batch eval artifacts (in-memory weights vs the
    # newest checkpoint on disk). Never mix the two in one figure.
    for dataset in EVAL_DATASETS:
        row[f"inline_{dataset}_reward"] = run.get(
            f"summary.final_eval/{dataset}/episode_reward/mean")
        row[f"inline_{dataset}_survived"] = run.get(
            f"summary.final_eval/{dataset}/termination_rate/survived")
    return row


def build_curves(run: pd.Series, hist: pd.DataFrame | None) -> list[dict]:
    reward = series(hist, CURVE_REWARD_KEYS)
    if reward is None:
        return []
    lifespan = series(hist, CURVE_LIFESPAN_KEYS)
    life_by_step = ({} if lifespan is None
                    else dict(zip(lifespan["step"], lifespan["value"])))
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "delay_k": int(run["delay_k"]), "budget": int(run["config.ppo.total_steps"]),
             "step": int(r["step"]), "reward_mean": float(r["value"]),
             "lifespan_mean": life_by_step.get(r["step"])}
            for _, r in reward.iterrows()]


def assert_spec_ids() -> None:
    """Fail loudly if the eval producer's spec_id has drifted from the pinned constant."""
    got = get_producer("eval").spec_id(get_producer("eval").spec())
    if got != EVAL_SPEC_ID:
        raise SystemExit(
            f"eval spec_id has drifted: got {got}, expected {EVAL_SPEC_ID}.\n"
            f"The artifacts this analysis reads were made by a different eval spec or "
            f"producer VERSION. Update EVAL_SPEC_ID deliberately, re-produce, and say so "
            f"in report.md -- do not silently repoint at a different generation of data.")


def assert_artifact_body(entry, run: pd.Series) -> None:
    """The eval must say it simulated the body this run trained on.

    This cohort deliberately spans two bodies, so the check is per-run rather than against
    one expected name. Pre-fix artifacts carry no stamp at all; absence is therefore an
    error here, not something to shrug at -- an unstamped file predates the 2026-08-18
    walker-XML fix and was produced on ``rodent.xml`` whatever the run trained on.
    """
    stamp = (entry.resolved or {}).get("walker_xml_path")
    trained = Path(str(run["env_params.walker_xml_path"])).name
    if stamp is None:
        raise SystemExit(
            f"eval artifact for {run['wandb_id']} has no resolved.walker_xml_path, so it "
            f"predates the 2026-08-18 walker-XML fix and was simulated on the wrong body. "
            f"Re-produce it (see analysis/README.md §6).")
    if stamp != trained:
        raise SystemExit(f"eval artifact for {run['wandb_id']} was produced on {stamp}, "
                         f"but the run trained on {trained}.")


def build_eval_rows(store: Store, run: pd.Series) -> list[dict]:
    """One row per (run, dataset) from the batch offline evaluation, if present."""
    entry = store.lookup("eval", run["wandb_id"], EVAL_SPEC_ID)
    if entry is None:
        return []
    assert_artifact_body(entry, run)
    record = json.loads((store.root / entry.path).read_text())
    rows = []
    for dataset in EVAL_DATASETS:
        block = record.get("datasets", {}).get(dataset)
        if not block:
            continue
        lifespan = block["lifespan_steps"]["mean"]
        rows.append({
            "wandb_id": run["wandb_id"],
            "condition": run["condition"],
            "network": NETWORK_OF[run["condition"].split("_")[0]],
            "xml": "new" if _is_new_xml(pd.DataFrame([run])).iat[0] else "old",
            "budget": int(run["config.ppo.total_steps"]),
            "delay_k": int(run["delay_k"]),
            "dataset": dataset,
            "checkpoint_step": entry.resolved.get("checkpoint_step"),
            # The body the eval actually simulated -- the field whose absence *was* the
            # 2026-08-18 bug. Carried into the CSV so the fix is auditable from the data.
            "walker_xml": entry.resolved.get("walker_xml_path"),
            "n_clips": block["n_clips"],
            "episode_reward": block["episode_reward"]["mean"],
            "lifespan_steps": lifespan,
            # Raw reward is comparable within a dataset only (clip lengths differ 6x).
            "reward_per_step": block["episode_reward"]["mean"] / lifespan,
            "survived": block["termination_rate"]["survived"],
            "hazard_rate": ((1 - block["termination_rate"]["survived"])
                            / (lifespan * float(run["env_params.ctrl_dt"]))),
        })
    return rows


def main() -> None:
    args = pipeline.parse_args(__doc__)
    assert_spec_ids()

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    producer = get_producer("history")
    history_spec_id = producer.spec_id(producer.spec())

    rows, curves, evals = [], [], []
    for _, run in runs.iterrows():
        hist = history_of(store, run["wandb_id"], history_spec_id)
        rows.append(build_row(run, hist))
        curves.extend(build_curves(run, hist))
        evals.extend(build_eval_rows(store, run))

    df = pd.DataFrame(rows).sort_values(["condition", "delay_k", "wandb_id"],
                                        ignore_index=True)
    curves_df = pd.DataFrame(curves).sort_values(["condition", "delay_k", "step"],
                                                 ignore_index=True)
    eval_df = pd.DataFrame(evals)
    if not eval_df.empty:
        eval_df = eval_df.sort_values(["condition", "dataset", "delay_k"],
                                      ignore_index=True)

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report + "\n")

    print("\nCohort:")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:20s} n={len(sub):d}  {sub['network'].iat[0]:17s} "
              f"xml={sub['xml'].iat[0]:3s} {sub['body_target_frame'].iat[0]:14s} "
              f"budget={sub['budget'].iat[0] // 1_000_000:>4d}M "
              f"delays={sorted(sub['delay_k'])} git={sorted(sub['git_commit'].unique())}")

    missing = df[df["reward_final"].isna()]
    if not missing.empty:
        print(f"\n*** {len(missing)} run(s) have no reward curve: "
              f"{missing['wandb_id'].tolist()}")

    print("\nReward at 600 M (window mean) and at the end of training:")
    print(df.pivot_table(index="delay_k", columns="condition",
                         values="reward_at_600M").round(0).to_string())

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    ok &= pipeline.write_csv(eval_df, HERE / "data_eval.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
