"""Is anything eligible being left out -- and are the gaps in the sweep real?

Two claims in report.md are claims about runs that are *absent*, which no figure can show:

1. **The holes are real.** Torque has no ``efference_length`` 0 or 1 run and position has no
   usable eff-100 run, and that is because those configurations were never launched -- not
   because a filter here dropped them. Follow-up #1 is written on the strength of this.
2. **Loosening the stack gates changes nothing.** The comparability rules were relaxed on
   the owner's word that no compatibility-breaking change landed in nnx-ppo or
   vnl-playground, so old runs may join provided their parameters match and they are not
   tagged ``BUG``. That relaxation admits **zero** additional runs, and saying so is only
   worth anything if it is re-checked whenever the index is synced.

So this script goes the other way round from ``extract.py``: instead of selecting, it takes
every run in the project that carries the ``noproprio`` or ``nointent`` tag, and every
parameter-matched candidate for the reference conditions, and reports which cohort each
landed in or the named reason it did not. It also re-tests each relaxable gate one at a
time, so "relaxing this would admit N runs" is a measurement.

It additionally asserts the equivalence ``sound_training_mask`` relies on: that the ``BUG``
tag and ``pipeline.UNREGULARIZED_COMMITS`` still pick out exactly the same runs. If a future
run is tagged ``BUG`` for some *other* reason, that assertion fires and the union in
``extract.py`` stops being a free lunch.

    ../.venv/bin/python analysis/efference-copy-vs-proprioception/selection_audit.py
    ../.venv/bin/python analysis/efference-copy-vs-proprioception/selection_audit.py --check

Writes ``selection_audit.txt``. Reads the run index (not the artifact store), so it is an
extract-side script; it feeds report.md, not a figure.
"""

import argparse
import difflib
import importlib.util
from pathlib import Path

import pandas as pd

from vnl_experiments.wandb_utils import index, pipeline

HERE = Path(__file__).resolve().parent
OUT = HERE / "selection_audit.txt"

#: Import extract.py by path: the folder name contains hyphens, so it is not a module name.
_spec = importlib.util.spec_from_file_location("_extract", HERE / "extract.py")
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)

def _gates(df: pd.DataFrame) -> dict:
    """Every clause of ``extract._standard``, named, so one can be dropped at a time."""
    ppo = pd.Series(True, index=df.index)
    for key, value in extract.PPO_PARAMS.items():
        ppo &= df[f"config.ppo.{key}"] == value
    return {
        "env is AbsoluteImitation": df["env"] == "AbsoluteImitation",
        "walker XML": df["env_params.walker_xml_path"] == extract.NEW_XML,
        "body_target_frame": df["env_params.body_target_frame"] == "reference_root",
        "net_params.network_class present and RodentEncDecDelays":
            df["net_params.network_class"] == "RodentEncDecDelays",
        "network sizes": ((df["net_params.enc_hidden_sizes"] == "[512, 512, 512, 512]")
                          & (df["net_params.dec_hidden_sizes"] == "[512, 512, 512, 512]")
                          & (df["net_params.critic_hidden_sizes"] == "[1024, 1024]")
                          & (df["net_params.latent_size"] == 32)),
        "regularisation hyperparameters":
            ((df["net_params.kl_weight"] == 0.001)
             & (df["net_params.entropy_weight"] == 0.01)
             & (df["net_params.min_std"] == 0.1)),
        "PPO hyperparameters": ppo,
        "seed == 42": df["seed"] == 42,
        "not tagged BUG / not an unregularised commit":
            extract.sound_training_mask(df),
        "held-out eval (created >= 2026-08-20)":
            pd.to_datetime(df["created_at"], utc=True) >= extract.HELD_OUT_EVAL_SINCE,
    }


def _standard_without(df: pd.DataFrame, skip: str | None = None) -> pd.Series:
    """The cohort gate with one named clause removed -- the counterfactual selection."""
    mask = pd.Series(True, index=df.index)
    for name, clause in _gates(df).items():
        if name != skip:
            mask &= clause
    return mask


def _reasons(df: pd.DataFrame) -> dict:
    """Named, ordered exclusion reasons; the first that matches is the one reported."""
    created = pd.to_datetime(df["created_at"], utc=True)
    return {
        "tagged BUG / unregularised commit": ~extract.sound_training_mask(df),
        "different walker XML": df["env_params.walker_xml_path"] != extract.NEW_XML,
        "different body_target_frame": df["env_params.body_target_frame"] != "reference_root",
        "different architecture": df["net_params.network_class"] != "RodentEncDecDelays",
        "different network sizes": ~(
            (df["net_params.enc_hidden_sizes"] == "[512, 512, 512, 512]")
            & (df["net_params.dec_hidden_sizes"] == "[512, 512, 512, 512]")
            & (df["net_params.critic_hidden_sizes"] == "[1024, 1024]")
            & (df["net_params.latent_size"] == 32)),
        "different regularisation": ~((df["net_params.kl_weight"] == 0.001)
                                      & (df["net_params.entropy_weight"] == 0.01)
                                      & (df["net_params.min_std"] == 0.1)),
        "different PPO settings": ~pd.concat(
            [df[f"config.ppo.{k}"] == v for k, v in extract.PPO_PARAMS.items()],
            axis=1).all(axis=1),
        "different seed": df["seed"] != 42,
        "eval/* is the train split (pre-2026-08-20)": created < extract.HELD_OUT_EVAL_SINCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    df = index.load()
    selected = pd.read_csv(HERE / "runs.csv", dtype={"wandb_id": str})
    in_cohort = dict(zip(selected["wandb_id"], selected["condition"]))
    reasons = _reasons(df)
    failures = []

    lines = ["Who is eligible, who is in, and what a looser gate would add", ""]

    # -- 1. the BUG / unregularised-commit equivalence -----------------------------------
    bug = df["tags"].apply(extract.has_tag, tag="BUG")
    unreg = df["git_commit"].isin(pipeline.UNREGULARIZED_COMMITS)
    lines.append("1. Does the BUG tag still mean exactly 'unregularised commit'?")
    lines.append(f"   tagged BUG: {int(bug.sum())}   unregularised commit: "
                 f"{int(unreg.sum())}   both: {int((bug & unreg).sum())}")
    lines.append(f"   tagged but not unregularised: {int((bug & ~unreg).sum())}   "
                 f"unregularised but not tagged: {int((unreg & ~bug).sum())}")
    if (bug ^ unreg).any():
        lines.append("   -> THEY HAVE DIVERGED. `sound_training_mask` takes the union, so "
                     "the cohort is still")
        lines.append("      correct, but report.md's claim that the two rules coincide is "
                     "now stale.")
        failures.append("BUG tag and UNREGULARIZED_COMMITS no longer coincide")
    else:
        lines.append("   -> identical sets, so gating on the tag and gating on the commit "
                     "are the same rule.")
    lines.append("")

    # -- 2. every ablated run in the project ---------------------------------------------
    ablated = df[df["tags"].apply(extract.has_tag, tag="noproprio")
                 | df["tags"].apply(extract.has_tag, tag="nointent")]
    lines.append(f"2. Every run in the project tagged noproprio or nointent "
                 f"({len(ablated)}), and where it went")
    counted = {"in cohort": 0}
    for _, run in ablated.sort_values("created_at").iterrows():
        wid = run["wandb_id"]
        if wid in in_cohort:
            verdict = f"-> {in_cohort[wid]}"
            counted["in cohort"] += 1
        else:
            reason = next((name for name, mask in reasons.items() if mask.loc[run.name]),
                          "UNEXPLAINED")
            verdict = f"EXCLUDED: {reason}"
            counted[reason] = counted.get(reason, 0) + 1
            if reason == "UNEXPLAINED":
                failures.append(f"{wid} is excluded for no reason this script knows")
        lines.append(f"   {wid:9s} {str(run['wandb_name'])[:48]:48s} "
                     f"{run['created_at'][:10]}  {verdict}")
    lines.append("   " + "; ".join(f"{k}: {v}" for k, v in counted.items()))
    lines.append("")

    # -- 3. what dropping each single gate would add ---------------------------------------
    base = _standard_without(df)
    lines.append("3. If ONE gate were dropped and the rest kept, who would newly qualify?")
    lines.append(f"   ({int(base.sum())} runs pass every gate; the six CONDITIONS then "
                 f"select {len(in_cohort)} of them.")
    lines.append("    The rest are full-proprioception runs at some nonzero delay or "
                 "efference length --")
    lines.append("    parameter-matched, but they answer the delay question, not this "
                 "one.)")
    width = max(len(n) for n in _gates(df))
    for name in _gates(df):
        entrants = df[_standard_without(df, skip=name) & ~base]
        ablated_in = entrants[
            entrants["net_params.dec_use_proprioception"] == False]  # noqa: E712
        lines.append(f"   drop {name:<{width}s} -> +{len(entrants):3d} run(s), "
                     f"{len(ablated_in):2d} of them ablated")
    lines.append("")
    lines.append("   Only the ablated column can change the swept arms, and it is zero for "
                 "every gate. The")
    lines.append("   non-ablated entrants would land in the reference conditions, whose "
                 "spread is already")
    lines.append("   under 0.7 %, and most of them cannot be used anyway: dropping the "
                 "date gate admits")
    lines.append("   runs whose eval/* is the train split, and dropping the "
                 "network_class gate admits")
    lines.append("   RodentForwardModel runs, which never recorded that field either.")
    lines.append("")

    # -- 4. the holes ----------------------------------------------------------------------
    lines.append("4. Which efference lengths exist at all, per mode "
                 "(cohort membership, before usability)")
    swept = df[df["wandb_id"].isin(
        [w for w, c in in_cohort.items() if c.endswith("noproprio")])]
    data = pd.read_csv(HERE / "data.csv", dtype={"wandb_id": str})
    for mode, torque in (("position", False), ("torque", True)):
        arm = swept[swept["env_params.torque_actuators"] == torque]
        launched = sorted(int(x) for x in arm["net_params.efference_length"].dropna().unique())
        lines.append(f"   {mode:9s} launched:      {launched}")
        for budget in ("600M", "400M"):
            ids = set(data.loc[data[f"usable_{budget}"], "wandb_id"])
            usable = sorted({int(r["net_params.efference_length"])
                             for _, r in arm.iterrows() if r["wandb_id"] in ids})
            lines.append(f"   {mode:9s} usable @{budget}: {usable}"
                         + (f"   missing: {sorted(set(launched) - set(usable))}"
                            if set(launched) - set(usable) else ""))
    lines.append("")
    lines.append("   Both arms are complete at the primary 600 M readout except position "
                 "efference 3 and")
    lines.append("   torque efference 50 -- interior points, not anchors. Position 3 was "
                 "launched once and")
    lines.append("   crashed at 415 M (its 400 M value is in data.csv and brackets "
                 "correctly between 2 and")
    lines.append("   5); torque 50 was launched once and died before its first eval. The "
                 "torque 0 and 1")
    lines.append("   anchors, absent when this folder was first written, arrived in the "
                 "2026-09-03 relaunch")
    lines.append("   and are what let the torque arm be read as a recovery curve rather "
                 "than a flat line.")
    lines.append("")
    lines.append("FAILURES: " + (", ".join(failures) if failures else "none"))

    text = "\n".join(lines) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("\n".join(difflib.unified_diff(
                (OUT.read_text() if OUT.exists() else "").splitlines(),
                text.splitlines(), fromfile=f"committed/{OUT.name}",
                tofile=f"rebuilt/{OUT.name}", lineterm="")))
            raise SystemExit(1)
        print(f"CHECK: {OUT.name} unchanged")
    else:
        OUT.write_text(text)
        print(text)

    if failures:
        raise SystemExit("the selection audit found something unexplained; see above")


if __name__ == "__main__":
    main()
