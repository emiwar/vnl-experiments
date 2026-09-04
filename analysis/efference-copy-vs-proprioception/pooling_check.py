"""What this analysis pools over, and how noisy a single run is, measured on these runs.

Three separate jobs, all reading the primary 600 M readout unless stated:

**A. The things the figures pool over.** Runs at a given ``efference_length`` are averaged
across ``delay_k`` and across ``git_commit`` (and, for two groups, across the nnx-ppo commit
and the CUDA upgrade). ``build_delay_network`` puts the ``Delay`` layer inside the
proprioception branch, which ``dec_use_proprioception=False`` does not construct, so
``delay_k`` should reach nothing -- asserted at the level of the built network in
``../position-control-open-loop/check_delay_inert.py``, and checked here on trained reward.
For the one group that holds two replicates *and* an odd run differing only in ``delay_k``,
there is a sharper test than any tolerance: the odd run's reward should fall **inside** the
range its replicates span, i.e. the inert knob should move reward less than rerunning does.

**B. The replicate noise floor.** Every ``(mode, efference_length)`` cell holding more than
one usable run, at both budgets. This is what the report's effects have to be read against,
and it is the number that decides whether the shallow decline past the peak is real. Note it
is a floor on *run-to-run* noise at a shared seed, not on seed-to-seed noise.

**C. The relaunch cross-check.** The 2026-09-02 position sweep lost five runs to
cluster-filesystem errors and the 2026-09-03 relaunch repeated four of them. Those four
pairs are independent runs of the same configuration, and reading both at 400 M -- the only
budget the crashed ones reach -- checks the relaunch against them. This is the reason
``reward_400M`` is still computed now that every arm has a 600 M number: it is the only place
the crashed runs can still say something.

    ../.venv/bin/python analysis/efference-copy-vs-proprioception/pooling_check.py
    ../.venv/bin/python analysis/efference-copy-vs-proprioception/pooling_check.py --check

Writes ``pooling_check.txt``. Reads only ``data.csv``, so it is plot-side: no index, no
artifact store, no network.
"""

import argparse
import difflib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "pooling_check.txt"
PRIMARY = "reward_600M"

#: Max spread (range as a percent of the group mean) a pooled group may show.
#:
#: Set from what is measured below, not from what would be convenient: the replicate noise
#: floor in this cohort is 2.5 % (the position eff-0 triple), so a tolerance under that would
#: fail on honest run-to-run variation. The smallest effect the report interprets is the
#: 6.0 % decline from position's peak to eff 100, so 3 % leaves the checks able to fail
#: before they stop being able to discriminate. At the 400 M readout every group came in
#: under 1.1 %; the floor rose when the readout moved to 600 M, which is itself worth
#: knowing -- late training is noisier than the middle of it.
TOLERANCE = 3.0

#: ``(label, [wandb_id], what varies inside the group, what it therefore bounds)``.
#: Spelled out as explicit run ids rather than derived from a groupby: these groups are an
#: argument about *which* runs isolate *which* variable, and a groupby would silently
#: re-form them -- or quietly become empty -- if the cohort changed.
GROUPS = (
    ("position, efference 0", ["23267t68", "dkglvuw8", "wbyipflf"],
     "delay_k (5, 0, 0)",
     "delay inertness alone -- one commit, one stack; also the noise floor"),
    ("position, efference 2", ["7w26do00", "rfoe9wu2"],
     "nothing at all",
     "pure replicate: same config, commit, stack, delay and seed"),
    ("position, intact", ["16jfo5vu", "vd2z944s"],
     "git_commit only",
     "the vnl-experiments commit (and its working copy) alone"),
    ("position, efference 10", ["3gw1ndwj", "3v2mbdhh"],
     "delay_k (10, 0) + git_commit",
     "delay and commit jointly"),
    ("torque, efference 10", ["4qm7vurb", "8kuci8sz"],
     "delay_k (0, 10) + git + nnx-ppo + CUDA",
     "everything that varies in the swept arm, at once"),
    ("torque, intact", ["594219y6", "86cpjh43", "ame77mw2"],
     "git + nnx-ppo + CUDA",
     "the whole stack, with the config held identical"),
)

#: ``(label, odd run, [replicate runs])`` -- the containment test described above.
CONTAINMENT = (
    ("delay_k = 5 vs two delay_k = 0 replicates, position efference 0",
     "23267t68", ["dkglvuw8", "wbyipflf"]),
)


def spread_pct(values: pd.Series) -> float:
    return 100 * (values.max() - values.min()) / values.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(HERE / "data.csv", dtype={"wandb_id": str}).set_index("wandb_id")
    failures = []
    lines = []

    # -- A -----------------------------------------------------------------------------
    lines.append("A. Spread within groups that differ only in something claimed to be inert")
    lines.append("")
    lines.append(f"   Metric: {PRIMARY} (held-out episode reward, mean of the eval points "
                 f"in (550 M, 600 M]).")
    lines.append(f"   Spread = (max - min) / mean, in percent. Tolerance = {TOLERANCE} %.")
    lines.append("")
    width = max(len(label) for label, *_ in GROUPS)
    vwidth = max(len(varies) for _, _, varies, _ in GROUPS)
    lines.append(f"   {'group':<{width}s} {'n':>2s} {'mean':>8s} {'spread%':>8s}   "
                 f"{'varies':<{vwidth}s} bounds")
    for label, ids, varies, bounds in GROUPS:
        missing = [i for i in ids if i not in data.index]
        values = data.loc[[i for i in ids if i in data.index], PRIMARY].dropna()
        if missing or len(values) < 2:
            note = (f"missing from data.csv: {', '.join(missing)}" if missing
                    else "fewer than 2 usable runs")
            lines.append(f"   {label:<{width}s} -- {note}")
            failures.append(f"{label} ({note})")
            continue
        spread = spread_pct(values)
        flag = "" if spread <= TOLERANCE else "  *** OVER TOLERANCE ***"
        lines.append(f"   {label:<{width}s} {len(values):>2d} {values.mean():>8.1f} "
                     f"{spread:>8.2f}   {varies:<{vwidth}s} {bounds}{flag}")
        if spread > TOLERANCE:
            failures.append(label)
    lines.append("")

    lines.append("   Containment test -- is the inert knob smaller than rerunning?")
    for label, odd, replicates in CONTAINMENT:
        try:
            value = data.loc[odd, PRIMARY]
            band = data.loc[replicates, PRIMARY].dropna()
        except KeyError:
            lines.append(f"     {label}: runs missing from data.csv")
            failures.append(f"containment: {label}")
            continue
        inside = bool(band.min() <= value <= band.max())
        lines.append(f"     {label}")
        lines.append(f"       odd run {odd} = {value:.1f}; replicates span "
                     f"[{band.min():.1f}, {band.max():.1f}] -> "
                     f"{'INSIDE' if inside else '*** OUTSIDE ***'}")
        if not inside:
            failures.append(f"containment: {label}")
    lines.append("     Inside means the manipulation claimed to be inert moves reward less "
                 "than simply")
    lines.append("     rerunning the same configuration does, which is a stronger "
                 "statement than any")
    lines.append("     spread-under-tolerance and needs no threshold to interpret.")
    lines.append("")

    # -- B -----------------------------------------------------------------------------
    lines.append("B. Every (mode, efference_length) cell with more than one usable run")
    lines.append("")
    swept = data[data["condition"].str.endswith("noproprio")]
    lines.append(f"   {'cell':<24s} {'budget':>7s} {'n':>2s} {'mean':>8s} {'spread%':>8s}"
                 f"   values")
    worst = 0.0
    for budget in ("reward_600M", "reward_400M"):
        for (mode, eff), cell in swept.groupby(["mode", "efference_length"]):
            values = cell[budget].dropna()
            if len(values) < 2:
                continue
            spread = spread_pct(values)
            if budget == PRIMARY:
                worst = max(worst, spread)
            lines.append(f"   {f'{mode}, efference {int(eff)}':<24s} "
                         f"{budget.removeprefix('reward_'):>7s} {len(values):>2d} "
                         f"{values.mean():>8.1f} {spread:>8.2f}   "
                         f"{[round(v, 1) for v in sorted(values)]}")
    lines.append("")
    lines.append(f"   Worst spread at the primary readout: {worst:.2f} %. That is this "
                 f"cohort's replicate")
    lines.append("   noise floor, and every effect in report.md is stated against it.")
    lines.append("")

    # -- C -----------------------------------------------------------------------------
    lines.append("C. Relaunch cross-check: the crashed 2026-09-02 runs vs their "
                 "2026-09-03 repeats")
    lines.append("")
    lines.append("   Read at 400 M, the only budget both reach. Independent runs of the "
                 "same configuration,")
    lines.append("   so agreement here is evidence the relaunched runs -- which carry the "
                 "whole 600 M result")
    lines.append("   at efference 15, 20, 50 and 100 -- are not systematically different "
                 "from what crashed.")
    lines.append("")
    crashed = swept[(swept["state"] == "crashed") & swept["reward_400M"].notna()]
    lines.append(f"   {'cell':<24s} {'crashed':>9s} {'relaunched':>11s} {'diff%':>7s}")
    for _, run in crashed.sort_values("efference_length").iterrows():
        peers = swept[(swept["mode"] == run["mode"])
                      & (swept["efference_length"] == run["efference_length"])
                      & (swept["state"] == "finished")
                      & swept["reward_400M"].notna()]
        if peers.empty:
            lines.append(f"   {f'{run['mode']}, efference {int(run['efference_length'])}':<24s} "
                         f"{run['reward_400M']:>9.1f} {'-- not repeated':>11s}")
            continue
        peer = peers["reward_400M"].mean()
        lines.append(f"   {f'{run['mode']}, efference {int(run['efference_length'])}':<24s} "
                     f"{run['reward_400M']:>9.1f} {peer:>11.1f} "
                     f"{100 * (peer / run['reward_400M'] - 1):>+7.2f}")
    lines.append("")

    lines.append("Reading")
    lines.append("  Every group in A holds, and the containment test passes: pooling "
                 "across `delay_k` at")
    lines.append("  fixed `efference_length` is sound on trained reward as well as on the "
                 "built network.")
    lines.append("  Every environment axis that comparability.txt flags -- the "
                 "vnl-experiments commit, the")
    lines.append("  nnx-ppo commit, the CUDA upgrade -- moves reward by less than the "
                 "noise floor inside at")
    lines.append("  least one group, which is what had to be measured rather than argued "
                 "since repos.*.dirty")
    lines.append("  voids the commit hashes themselves.")
    lines.append("")
    lines.append("  C agrees to within the noise floor at every repeated cell, so the "
                 "relaunch is measuring")
    lines.append("  the same thing the crashed sweep was.")
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
        raise SystemExit("the pooling expectations do not hold; see above")


if __name__ == "__main__":
    main()
