"""New (almost-full-collision) walker XML vs the old (very sparse) one.

Five questions, one dataset:

1. how does the new XML perform relative to the old one?
2. is the difference uniform across networks/conditions, or worse somewhere?
3. how does ``body_target_frame`` come into it?
4. is convergence slower?
5. how much slower is the new XML to simulate?

Cohorts
-------
The new-XML runs are confounded with actuator mode -- the new EncDec runs used torque
control, the new forward-model runs used position control -- so the analysis is built
from two *internally matched* pairs plus one explicitly-flagged doubly-confounded pair:

===========================  ================================================
``old/new_efference``        EncDec + efference copy, torque, matched
``old/new_forward_model``    explicit FM (loss 1, detached), position, matched
``old/new_pg_forward_model`` policy-gradient FM -- **old = torque, new =
                             position**, shown only to explain the apparent
                             difference away
``old_efference_refroot``    the primary old-XML baseline for the EncDec pair:
                             909e774d, three days before the new cohort, after
                             the nnx-ppo 0.3.0-dev pin and the logging rename
===========================  ================================================

The canonical ``old_efference`` sweep (1cd5838f) predates both of those changes, so it
is kept as a second, independent baseline rather than the primary one. They agree to
within 8 reward.

On the frame (question 3)
-------------------------
Every new-XML run in this selection logs ``env_params.body_target_frame =
current_root``, although the training script at their recorded commits (``201d6e11``,
``0560d402``) reads ``reference_root``: the cluster working copy had been edited, a
state only committed later as ``456fbd7``. WandB stored no ``diff.patch`` for these
runs, so ``env_params`` is the only record of what actually ran -- which is why this
script filters on ``env_params.*`` and never on the training script at the run's commit.
(``ef060b7``, 2026-08-11, has since set ``reference_root`` in both training scripts, so
runs launched after that date do use it.)

Data sources
------------
Run configs and summaries come from the committed run index; the reward curves and the
throughput series come from the ``history`` artifacts in the store. Neither touches the
WandB API, so this script runs offline in a couple of seconds.

Run it
------
    ../.venv/bin/python analysis/collision-model-xml/extract.py           # frozen
    ../.venv/bin/python analysis/collision-model-xml/extract.py --refresh
    ../.venv/bin/python analysis/collision-model-xml/extract.py --check
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vnl_experiments.artifacts import Store, get_producer
from vnl_experiments.wandb_utils import comparability_report, pipeline

HERE = Path(__file__).resolve().parent
PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
REQUIRES = ["index", "history"]

OLD_XML = "rodent.xml"
NEW_XML = "rodent_no_tail_collisions.xml"

STD_ARCH = {
    "net_params.enc_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.dec_hidden_sizes": "[512, 512, 512, 512]",
    "net_params.critic_hidden_sizes": "[1024, 1024]",
}
EXPECTED_STEP = 600_064_000

REWARD_MEAN_KEYS = ("summary.eval/episode_reward/mean", "summary.episode_reward/mean")
REWARD_STD_KEYS = ("summary.eval/episode_reward/std", "summary.episode_reward/std")
LIFESPAN_KEYS = ("summary.eval/lifespan/mean", "summary.lifespan_mean")
CURVE_KEYS = ("eval/episode_reward/mean", "episode_reward/mean")


# --------------------------------------------------------------------------------------
# condition selectors
# --------------------------------------------------------------------------------------


def _xml_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.rsplit("/", n=1).str[-1]


def _base(df: pd.DataFrame) -> pd.Series:
    """What all seven cells share: standard architecture, the standard 600 M-step
    budget, seed 42, and efference length tied to the delay."""
    mask = (
        (df["env"] == "AbsoluteImitation")
        & (df["seed"] == 42)
        & (df["summary._step"] == EXPECTED_STEP)
        & (df["delay_k"] == df["efference_length"])
    )
    for column, value in STD_ARCH.items():
        mask &= df[column] == value
    return mask


def _cell(xml: str, torque: bool, network: str, frame: str = "current_root",
          dates: tuple[str, ...] | None = None):
    """A selector for one (xml, control mode, network, frame) cell.

    ``network`` is derived from the run's tags plus the forward-model knobs rather than
    read from a config field, because no single field separates the explicit forward
    model (``fm_loss_weight=1``, detached) from the policy-gradient one
    (``fm_loss_weight=0``, gradient flowing through the predictor).
    """

    def selector(df: pd.DataFrame) -> pd.Series:
        tags = df["tags"].fillna("").str.split(",")
        is_fm = tags.apply(lambda t: "ForwardModel" in t)
        is_encdec = tags.apply(lambda t: "EncDec" in t)

        if network == "efference":
            net_mask = is_encdec & ~is_fm
        elif network == "forward_model":
            net_mask = (is_fm & (df["fm_loss_weight"] == 1)
                        & df["detach_prediction"].fillna(True).astype(bool))
        elif network == "pg_forward_model":
            net_mask = (is_fm & (df["fm_loss_weight"] == 0)
                        & (df["detach_prediction"] == False))  # noqa: E712
        else:
            raise ValueError(network)

        mask = (
            _base(df)
            & net_mask
            & (_xml_name(df["env_params.walker_xml_path"]) == xml)
            & (df["env_params.torque_actuators"] == torque)
            & (df["env_params.body_target_frame"] == frame)
        )
        if dates is not None:
            mask &= df["created_at"].str[:10].isin(dates)
        return mask

    return selector


CONDITIONS = {
    "old_efference":         _cell(OLD_XML, True,  "efference",
                                   dates=("2026-06-11", "2026-06-12")),
    "new_efference":         _cell(NEW_XML, True,  "efference", dates=("2026-07-09",)),
    "old_efference_refroot": _cell(OLD_XML, True,  "efference", frame="reference_root",
                                   dates=("2026-07-06",)),
    "old_forward_model":     _cell(OLD_XML, False, "forward_model"),
    "new_forward_model":     _cell(NEW_XML, False, "forward_model"),
    "old_pg_forward_model":  _cell(OLD_XML, True,  "pg_forward_model"),
    "new_pg_forward_model":  _cell(NEW_XML, False, "pg_forward_model"),
}

# The experimental axes (xml, network, control mode, frame, delay) and git_commit vary by
# design; everything else must be single-valued within a condition.
INVARIANTS = [
    "env", "seed", "net_params.latent_size", "net_params.kl_weight",
    "net_params.enc_hidden_sizes", "net_params.dec_hidden_sizes",
    "net_params.critic_hidden_sizes",
    "env_params.clip_length", "env_params.ctrl_dt", "env_params.sim_dt",
    "env_params.solver", "env_params.iterations", "env_params.ls_iterations",
    "env_params.njmax", "env_params.naconmax", "env_params.rescale_factor",
    "env_params.mujoco_impl",
    "config.ppo.n_envs", "config.ppo.total_steps", "summary._step",
    "env_params.walker_xml_path", "env_params.torque_actuators",
    "env_params.body_target_frame", "git_commit",
]


# --------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------


def _network_of(condition: str) -> str:
    return condition.split("_", 1)[1].replace("efference_refroot", "efference")


def history_of(store: Store, wandb_id: str, spec_id: str) -> pd.DataFrame | None:
    entry = store.lookup("history", wandb_id, spec_id)
    return None if entry is None else pd.read_csv(store.root / entry.path)


def median_sps(hist: pd.DataFrame | None, column: str) -> float | None:
    """Median throughput, dropping the first 10 % of samples (XLA compilation)."""
    if hist is None or column not in hist or hist[column].dropna().empty:
        return None
    values = hist[column].dropna().to_numpy()
    return float(np.median(values[max(1, len(values) // 10):]))


def build_row(run: pd.Series, hist: pd.DataFrame | None) -> dict:
    xml = str(run["env_params.walker_xml_path"]).rsplit("/", 1)[-1]
    notes = run["notes"]
    return {
        # provenance
        "wandb_id": run["wandb_id"],
        "wandb_name": run["wandb_name"],
        "wandb_project": PROJECT,
        "state": run["state"],
        "git_commit": (run["git_commit"] or "")[:8],
        "tags": run["tags"],
        "notes": notes.strip() if isinstance(notes, str) else "",
        "created_at": run["created_at"],
        # experimental axes
        "condition": run["condition"],
        "xml": "new" if xml == NEW_XML else "old",
        "walker_xml": xml,
        "network": _network_of(run["condition"]),
        "control_mode": "torque" if run["env_params.torque_actuators"] else "position",
        "delay_k": run["delay_k"],
        "efference_length": run["efference_length"],
        # authoritative frame + fm knobs (sanity)
        "body_target_frame": run["env_params.body_target_frame"],
        "net_params_body_target_frame": run.get("net_params.body_target_frame"),
        "torque_actuators": run["env_params.torque_actuators"],
        "fm_loss_weight": run.get("fm_loss_weight"),
        "detach_prediction": run.get("detach_prediction"),
        # invariants
        "env": run["env"],
        "seed": run["seed"],
        "latent_size": run["net_params.latent_size"],
        "kl_weight": run["net_params.kl_weight"],
        "enc_hidden_sizes": run["net_params.enc_hidden_sizes"],
        "dec_hidden_sizes": run["net_params.dec_hidden_sizes"],
        "critic_hidden_sizes": run["net_params.critic_hidden_sizes"],
        "clip_length": run["env_params.clip_length"],
        "ctrl_dt": run["env_params.ctrl_dt"],
        "sim_dt": run["env_params.sim_dt"],
        "solver": run["env_params.solver"],
        "iterations": run["env_params.iterations"],
        "ls_iterations": run["env_params.ls_iterations"],
        "njmax": run["env_params.njmax"],
        "naconmax": run["env_params.naconmax"],
        "rescale_factor": run["env_params.rescale_factor"],
        "mujoco_impl": run["env_params.mujoco_impl"],
        "n_envs": run["config.ppo.n_envs"],
        "total_steps": run["config.ppo.total_steps"],
        "actual_step": run["summary._step"],
        # hardware -- speed comparisons are only valid within one GPU model
        "gpu": run["gpu"],
        "host": run["host"],
        # metrics
        "reward_mean": pipeline.first_present(run, *REWARD_MEAN_KEYS),
        "reward_std": pipeline.first_present(run, *REWARD_STD_KEYS),
        "lifespan_mean": pipeline.first_present(run, *LIFESPAN_KEYS),
        "train_sps_final": run.get("summary.throughput/train_sps"),
        "eval_sps_final": run.get("summary.throughput/eval_sps"),
        "train_sps_median": median_sps(hist, "throughput/train_sps"),
        "eval_sps_median": median_sps(hist, "throughput/eval_sps"),
        "runtime_s": run.get("summary._runtime"),
    }


def build_curve(run: pd.Series, hist: pd.DataFrame | None) -> list[dict]:
    """The run's eval-reward series (evaluated every 10 M steps -> ~60 points)."""
    if hist is None:
        return []
    key = next((k for k in CURVE_KEYS if k in hist.columns), None)
    if key is None:
        return []
    return [{"wandb_id": run["wandb_id"], "condition": run["condition"],
             "delay_k": run["delay_k"], "step": int(row["_step"]),
             "reward_mean": float(row[key])}
            for _, row in hist.dropna(subset=[key]).iterrows()]


def main() -> None:
    args = pipeline.parse_args(__doc__)

    runs = pipeline.resolve_selection(HERE, CONDITIONS, refresh=args.refresh,
                                      sync=args.sync, project=args.project)
    pipeline.write_coverage(runs, REQUIRES, HERE)

    store = Store()
    producer = get_producer("history")
    spec_id = producer.spec_id(producer.spec())

    rows, curves = [], []
    for _, run in runs.iterrows():
        hist = history_of(store, run["wandb_id"], spec_id)
        rows.append(build_row(run, hist))
        curves.extend(build_curve(run, hist))

    df = pd.DataFrame(rows).sort_values(["network", "xml", "delay_k", "wandb_id"],
                                        ignore_index=True)
    curves_df = pd.DataFrame(curves).sort_values(["condition", "delay_k", "step"],
                                                 ignore_index=True)

    report = comparability_report(runs, invariant_cols=INVARIANTS, group_col="condition")
    if not args.check:
        (HERE / "comparability.txt").write_text(report + "\n")

    print("\nCohort:")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:24s} n={len(sub):2d} delays={sorted(sub['delay_k'])}")
        print(f"  {'':24s} xml={sub['walker_xml'].unique()} "
              f"torque={sub['torque_actuators'].unique()} "
              f"frame={sub['body_target_frame'].unique()} "
              f"git={sorted(sub['git_commit'].unique())} "
              f"gpu={sorted(set(sub['gpu'].dropna()))}")

    # Question 3 in one table: which frame each XML was actually trained with.
    print("\nbody_target_frame by xml (authoritative env_params value):")
    print(pd.crosstab(df["xml"], df["body_target_frame"]).to_string())
    print("net_params.body_target_frame (inert copy):",
          df["net_params_body_target_frame"].unique())

    ok = pipeline.write_csv(df, HERE / "data.csv", check=args.check)
    ok &= pipeline.write_csv(curves_df, HERE / "curves.csv", check=args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
