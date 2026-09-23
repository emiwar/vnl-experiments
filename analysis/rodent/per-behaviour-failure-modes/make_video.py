"""The 2x2 comparison video for per-behaviour-failure-modes, and its stills.

    ../../../../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/make_video.py
    ... --check     # verify the committed text outputs, render nothing

A third kind of script in this folder, beside `extract.py` (artifacts -> CSVs) and
`plot.py` (CSVs -> figures): it reads the ``video`` artifacts and the session asset, and
writes one gitignored mp4 plus three committed files -- ``video_captions.csv``,
``video_audit.txt`` and ``figures/video_contact_sheet.png``. The mp4 stays out of git by
the store's rule (``analysis/README.md`` SS3, "Videos in a report"), so the contact sheet is
what makes the report readable without it.

The four tiles are the reference footage and three policies whose ``old_eval`` means sit
within 1.5 % of each other. That is the whole point of putting them side by side: a
difference visible here is one the matched mean does not carry. What it is *not* is
evidence -- four of 32 clips, one rollout each, and `audit_lifetimes` measures how far
that one rollout can sit from the ones the CSVs were built on (on clip 0, 20 s).

Tiling, trimming, captions and death marks all live in
``vnl_experiments/video_editing/make_collage.py`` under the preset named in
:data:`PRESET`; this script owns the *choices* -- which runs, which labels, which guards --
and calls that. Three guards run before anything is rendered:

  * **the reference clips are the clips the labels describe** -- each artifact's
    ``reference_qpos`` is compared bit-for-bit against ``eval_clips_32x30s.h5``. This is
    what licenses drawing a MotionMapper caption over the footage at all; a caption joined
    to the wrong clips would be invisible rather than wrong-looking.
  * **the video and the numbers are the same weights** -- the video artifact's
    ``resolved`` block must agree with the per-clip ``eval`` artifact's on checkpoint step,
    walker XML, arena, env class and network class, and neither may come from the laptop.
  * **the actuator is the one the run trained with** -- `extract.assert_artifact_actuator`
    re-run on the three tiles, because the corruption it exists for (see its docstring)
    would silently change what the animal in the video is.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# `extract` is this folder's, hence the sys.path line above; the guards are shared with
# it rather than restated, so a boundary that moves there moves here too.
from extract import EVAL_PC_SPEC_ID, assert_artifact_actuator  # noqa: E402

import cv2  # noqa: E402
from vnl_experiments.artifacts import Store  # noqa: E402
from vnl_experiments.video_editing import make_collage as mc  # noqa: E402
from vnl_experiments.wandb_utils.style import (  # noqa: E402
    apply_style,
    behaviour_label,
    provenance,
)

#: The preset in `make_collage.PRESETS` this script drives. The grid there names the runs
#: `select_tiles` re-derives below; they are asserted equal rather than duplicated, so a
#: preset edited without re-running this script fails loudly.
PRESET = "noproprio_vs_delay10"

#: Conditions in tile order (row-major after the reference cell).
TILE_CONDITIONS = ("pos_intact", "pos_noproprio_eff2", "torque_delay10")

#: The dataset the videos are rendered on, and the one the selection rule reads.
DATASET = "new_eval"

#: The session asset the eval clips were cut from; `assert_reference_clips` compares the
#: renders' stored reference against it.
EVAL_CLIPS_H5 = (HERE.parents[2] / "assets" / "art" / "2020_12_22_1"
                 / "eval_clips_32x30s.h5")

#: A caption segment must last at least this many frames (0.5 s at 50 fps). The raw
#: per-frame labels flicker -- clip 2 alone has 62 runs of them in 30 s -- and a caption
#: that changes every few frames is unreadable. See `caption_segments` for what the
#: smoothing does and does not licence.
CAPTION_MIN_FRAMES = 25

#: Seconds after the first tile in a clip dies to grab that clip's still: long enough for
#: the failure to be legible, short enough that the survivors are still tracking.
STILL_AFTER_DEATH_S = 1.0

CAPTIONS = HERE / "video_captions.csv"
AUDIT = HERE / "video_audit.txt"
CONTACT_SHEET = HERE / "figures" / "video_contact_sheet.png"


# --------------------------------------------------------------------------- tiles

def select_tiles(data: pd.DataFrame, runs: pd.DataFrame,
                 have_video: set[str]) -> tuple[dict[str, str], list[str]]:
    """One run per tile condition, by a rule fixed before the videos were watched.

    **Among the runs of the condition that have a video artifact, take the one whose
    ``new_eval`` episode reward is closest to the condition's mean over all its runs;
    break ties by the order in ``runs.csv``.** The mean is over every run, including ones
    with no video, so a missing render cannot move the target it is judged against.

    `new_eval` is the criterion because it is the dataset these videos are rendered on;
    the conditions were matched on `old_eval`, and on that statistic all three sit within
    1.5 % of each other either way.

    Returns the picks and the lines explaining them, including every run passed over --
    with n = 2 the rule is a near-tie and with n = 1 it is no choice at all, and both
    facts belong in the audit rather than in a sentence nobody can check.
    """
    order = {rid: i for i, rid in enumerate(runs["wandb_id"])}
    picks, notes = {}, []
    for condition in TILE_CONDITIONS:
        rows = data[(data.condition == condition) & (data.dataset == DATASET)]
        if rows.empty:
            raise SystemExit(f"{condition}: no {DATASET} rows in data.csv")
        target = float(rows.reward.mean())
        reward = dict(zip(rows.wandb_id, rows.reward.astype(float)))
        candidates = [r for r in rows.wandb_id if r in have_video]
        if not candidates:
            raise SystemExit(
                f"{condition}: none of {sorted(reward)} has a {mc.VIDEO_SPEC_V3} video")
        pick = min(candidates, key=lambda r: (abs(reward[r] - target), order[r]))
        picks[condition] = pick
        notes.append(f"{condition}: condition mean {DATASET} reward {target:.0f} over "
                     f"{len(reward)} run(s); picked {pick} ({reward[pick]:.0f}, "
                     f"{reward[pick] - target:+.0f})")
        for rid in rows.wandb_id:
            if rid == pick:
                continue
            why = ("no video artifact" if rid not in have_video
                   else f"{abs(reward[rid] - target):.0f} from the mean vs "
                        f"{abs(reward[pick] - target):.0f}")
            notes.append(f"    passed over {rid} ({reward[rid]:.0f}): {why}")
    return picks, notes


def preset_runs() -> list[str]:
    """The run ids the preset's grid actually points at, in row-major order."""
    ids = []
    for row in mc.PRESETS[PRESET]["grid"]:
        for cell in row:
            if cell is None:
                continue
            path = Path(cell[0])
            if path.parent.parent.name == "video":  # artifacts/video/<id>/<spec>.mp4
                ids.append(path.parent.name)
    return ids


# --------------------------------------------------------------------------- guards

def assert_reference_clips(store: Store, run_ids: list[str]) -> str:
    """Assert every render's stored reference *is* ``new_eval`` clips 0..n-1, exactly.

    The behaviour caption is joined to the footage by clip index and frame number and by
    nothing else, so this is the join's only licence. It is also the cheapest possible
    check: the producer writes the reference pose it fed the env into the ``.h5`` beside
    the mp4, and the asset it came from is right here, so the two can be compared
    bit-for-bit rather than within a tolerance.
    """
    with h5py.File(EVAL_CLIPS_H5, "r") as f:
        asset = f["qpos"][:]
        per_clip = int(f.attrs["n_frames_per_clip"])
    worst = []
    for rid in run_ids:
        entry = store.lookup("video", rid, mc.VIDEO_SPEC_V3)
        with h5py.File((store.root / entry.path).with_suffix(".h5"), "r") as g:
            reference = g["reference_qpos"][:]
        n_clips = len(reference) // per_clip
        delta = float(np.abs(reference - asset[:len(reference)]).max())
        if delta != 0.0:
            raise SystemExit(
                f"{rid}: the render's reference_qpos differs from "
                f"{EVAL_CLIPS_H5.name} clips 0..{n_clips - 1} by {delta:g}. The "
                f"behaviour captions would be joined to the wrong frames.")
        worst.append(n_clips)
    if len(set(worst)) != 1:
        raise SystemExit(f"tiles render different clip counts: {worst}")
    return (f"reference clips: all {len(run_ids)} renders are {EVAL_CLIPS_H5.name} "
            f"clips 0..{worst[0] - 1}, bit-identical (max |delta| 0)")


def assert_same_weights(store: Store, run_ids: list[str], eval_spec: str) -> str:
    """Assert each video was rendered from the same checkpoint the CSVs were built from.

    Compares the two artifacts' ``resolved`` blocks -- the same-checkpoint fields only,
    since the specs deliberately differ elsewhere (4 clips vs 3 datasets). Both artifacts
    read the *same* config.json, so this cannot catch a corrupt one; that is what
    `assert_artifact_actuator` is for, and it is why both run.
    """
    shared = ("checkpoint_step", "walker_xml_path", "arena_xml_path", "env_class",
              "network_class")
    lines = []
    for rid in run_ids:
        video = store.lookup("video", rid, mc.VIDEO_SPEC_V3)
        per_clip = store.lookup("eval", rid, eval_spec)
        if video is None or per_clip is None:
            raise SystemExit(f"{rid}: missing a video or per-clip eval artifact")
        for key in shared:
            a, b = (video.resolved or {}).get(key), (per_clip.resolved or {}).get(key)
            if a != b:
                raise SystemExit(f"{rid}: video {key}={a!r} but per-clip eval {key}={b!r}")
        for entry, kind in ((video, "video"), (per_clip, "eval")):
            gpu = str((entry.producer or {}).get("gpu", ""))
            if "RTX 4060" in gpu:
                raise SystemExit(f"{rid}: {kind} artifact was produced on {gpu}, where "
                                 f"the rollout is not reproducible")
        lines.append(f"    {rid}: step {video.resolved['checkpoint_step']}, "
                     f"{video.resolved['walker_xml_path']}, "
                     f"{(video.producer or {}).get('gpu', '?')}")
    return ("same weights as the CSVs: video and per-clip eval agree on "
            + ", ".join(shared) + f" for all {len(run_ids)} tiles\n"
            + "\n".join(lines))


# --------------------------------------------------------------------------- captions

def caption_segments(labels: np.ndarray, coarse: dict[str, str], names: np.ndarray,
                     clips: range) -> pd.DataFrame:
    """Coarse behaviour as readable caption segments, one row per stretch.

    Smoothed twice, and **for legibility only**: a centred mode filter of
    :data:`CAPTION_MIN_FRAMES`, then any surviving stretch shorter than that is absorbed
    into whichever neighbour is longer. Nothing quantitative may be read off
    this file -- every number in ``report.md`` comes from the raw per-frame labels in
    ``behaviour_labels_frames.npz``, and this is a subtitle track.

    Absorbing into the *longer* neighbour rather than the preceding one keeps the
    smoothing from drifting labels systematically later in time.
    """
    half = CAPTION_MIN_FRAMES // 2
    rows = []
    for clip in clips:
        raw = np.array([coarse[n] for n in names[labels[clip] - 1]])
        # Centred mode filter: the most common label in a +-half window.
        smooth = np.empty_like(raw)
        for i in range(len(raw)):
            window = raw[max(0, i - half): i + half + 1]
            values, counts = np.unique(window, return_counts=True)
            smooth[i] = values[counts.argmax()]
        # Run-length encode, then absorb runs shorter than half the filter width.
        runs = []
        for i, value in enumerate(smooth):
            if runs and runs[-1][2] == value:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1, value])
        while len(runs) > 1:
            short = [i for i, r in enumerate(runs)
                     if r[1] - r[0] < CAPTION_MIN_FRAMES]
            if not short:
                break
            i = short[0]
            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None
            keep_left = right is None or (
                left is not None and (left[1] - left[0]) >= (right[1] - right[0]))
            if keep_left:
                left[1] = runs[i][1]
            else:
                right[0] = runs[i][0]
            runs.pop(i)
            merged = [runs[0]]
            for run in runs[1:]:
                if run[2] == merged[-1][2]:
                    merged[-1][1] = run[1]
                else:
                    merged.append(run)
            runs = merged
        for start, end, value in runs:
            rows.append({"clip": clip, "start_frame": start, "end_frame": end,
                         "coarse": value, "text": behaviour_label(value)})
    return pd.DataFrame(rows)


def captions_text(frame: pd.DataFrame) -> str:
    """The caption file's exact contents, so ``--check`` can compare without writing."""
    header = (
        "# Caption track for the per-behaviour-failure-modes collage, drawn on the\n"
        "# reference tile by video_editing/make_collage.py. Half-open, clip-local frame\n"
        "# ranges at 50 fps. Written by make_video.py from behaviour_labels_frames.npz\n"
        f"# after a {CAPTION_MIN_FRAMES}-frame mode filter -- a DISPLAY smoothing. Every\n"
        "# number in report.md uses the raw per-frame labels, never this file.\n")
    return header + frame.to_csv(index=False)


# --------------------------------------------------------------------------- audit

def audit_lifetimes(store: Store, picks: dict[str, str], clips: pd.DataFrame,
                    failures: pd.DataFrame, captions: pd.DataFrame,
                    n_clips: int) -> tuple[str, dict[int, float]]:
    """Compare each tile's rendered episode against the two the CSVs were built from.

    Three independent GPU passes over the same weights and the same clip exist here: the
    per-clip ``eval`` artifact (``clips.csv``), the ``trace`` artifact
    (``failure_context.csv``) and this render. They are *not* guaranteed to agree --
    MuJoCo Warp is nondeterministic across passes at the ~1 % level on aggregates, and a
    30 s clip whose outcome hinges on one manoeuvre can land either side of it. Measuring
    that here is the difference between a video that illustrates the analysis and a video
    that is quietly a fourth, unreported experiment.

    Also returns, per clip, the first death among the tiles -- the moment the contact
    sheet grabs its still from.
    """
    lines = ["per-tile, per-clip episode end (s), one column per independent pass:",
             "",
             f"{'condition':<20}{'run':<10}{'clip':>5}{'eval':>8}{'trace':>8}"
             f"{'video':>8}   behaviour at the traced end"]
    first_death: dict[int, float] = {}
    spread = []
    for condition, rid in picks.items():
        entry = store.lookup("video", rid, mc.VIDEO_SPEC_V3)
        stats = json.loads((store.root / entry.path).with_suffix(".stats.json")
                           .read_text())
        horizon = stats["clip_length"] / stats["fps"]
        for clip in range(n_clips):
            video = float(stats["per_clip"][clip]["time_alive_s"])
            row = clips[(clips.wandb_id == rid) & (clips.dataset == DATASET)
                        & (clips.clip_index == clip)]
            ev = float(row.lifespan_s.iloc[0]) if len(row) else float("nan")
            fail = failures[(failures.wandb_id == rid) & (failures.clip_index == clip)]
            tr = float(fail.term_s.iloc[0]) if len(fail) else float("nan")
            where = (str(fail.behaviour_window_modal.iloc[0]) if len(fail)
                     else "survived the clip")
            if mc._fell(stats["per_clip"][clip], horizon):
                first_death[clip] = min(first_death.get(clip, 1e9), video)
            cell = (lambda x: f"{x:>8.2f}" if x == x else f"{'-':>8}")
            lines.append(f"{condition:<20}{rid:<10}{clip:>5}{cell(ev)}{cell(tr)}"
                         f"{cell(video)}   {where}")
            passes = [x for x in (ev, tr, video) if x == x]
            spread.append((max(passes) - min(passes), condition, clip))
    spread.sort(reverse=True)
    worst = spread[0]
    lines += [
        "",
        f"Worst disagreement between passes: {worst[0]:.1f} s ({worst[1]}, clip "
        f"{worst[2]}). The video is therefore an illustration of the conditions and not",
        "a measurement of a clip: read the CSVs for any number, including this one.",
        "",
        "Caption track (the reference animal's coarse behaviour, smoothed for display):",
    ]
    for clip in range(n_clips):
        segments = captions[captions["clip"] == clip]
        lines.append(f"  clip {clip}: " + "  ".join(
            f"{r.text} {r.start_frame / 50:.1f}-{r.end_frame / 50:.1f}s"
            for r in segments.itertuples()))
    return "\n".join(lines), first_death


# --------------------------------------------------------------------------- stills

def contact_sheet(video: Path, timeline: list, first_death: dict[int, float],
                  captions: pd.DataFrame, path: Path) -> str:
    """A committed still per clip, so ``report.md`` stands without the gitignored mp4.

    Each still is grabbed :data:`STILL_AFTER_DEATH_S` after the first tile in that clip
    dies -- a rule, not a choice of the nicest frame -- and located through the
    ``timeline`` the collage returns, because trimming means output frame != source frame.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 50.0

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    picked = []
    for ax, (clip, out_start, count) in zip(axes.ravel(), timeline):
        within = round((first_death.get(clip, count / fps / 2) + STILL_AFTER_DEATH_S)
                       * fps)
        within = int(min(max(within, 0), count - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, out_start + within)
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"cannot read frame {out_start + within} of {video}")
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        segment = captions[(captions["clip"] == clip) & (captions.start_frame <= within)
                           & (captions.end_frame > within)]
        behaviour = segment.text.iloc[0] if len(segment) else "unscored"
        ax.set_title(f"Clip {clip}, t = {within / fps:.1f} s  ({behaviour.lower()})",
                     fontsize=10)
        picked.append((clip, within / fps))
    capture.release()
    fig.tight_layout()
    stamp = provenance(fig, HERE, CAPTIONS)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return ("contact sheet: " + path.name + " at "
            + ", ".join(f"clip {c} t={t:.1f}s" for c, t in picked)
            + f"\n    (not in figures/manifest.json, which plot.py owns; provenance: "
              f"{stamp})")


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="re-run the guards, verify video_captions.csv, render "
                             "nothing. video_audit.txt and the contact sheet are "
                             "rewritten only by a full render, which regenerates them")
    parser.add_argument("--no-render", action="store_true",
                        help="run the guards, refresh video_captions.csv and print the "
                             "audit, but render nothing")
    args = parser.parse_args()

    store = Store()
    data = pd.read_csv(HERE / "data.csv")
    runs = pd.read_csv(HERE / "runs.csv")
    clips = pd.read_csv(HERE / "clips.csv")
    failures = pd.read_csv(HERE / "failure_context.csv")
    with np.load(HERE / "behaviour_labels_frames.npz", allow_pickle=False) as z:
        labels, names = z["labels"], z["names"].astype(str)
        coarse = dict(zip(names, z["coarse"].astype(str)))

    have_video = {rid for rid in runs.wandb_id
                  if store.have("video", rid, mc.VIDEO_SPEC_V3)}
    picks, notes = select_tiles(data, runs, have_video)
    chosen = [picks[c] for c in TILE_CONDITIONS]
    if sorted(preset_runs()) != sorted(chosen):
        raise SystemExit(
            f"make_collage preset {PRESET!r} renders {sorted(preset_runs())} but the "
            f"selection rule picks {sorted(chosen)}; update the preset's grid.")

    report = ["=== tile selection ===", *notes, "", "=== guards ==="]
    report.append(assert_reference_clips(store, chosen))
    report.append(assert_same_weights(store, chosen, EVAL_PC_SPEC_ID))
    tiles = clips[clips.wandb_id.isin(chosen)]
    report.append(assert_artifact_actuator(tiles, label="video tiles (via per-clip eval)"))

    # How many clips the renders cover comes from the artifacts themselves, never from
    # a constant here: a preset pointing at a different video spec would otherwise
    # caption clips that were never rendered.
    spec_clips = {int(store.lookup("video", rid, mc.VIDEO_SPEC_V3).spec["n_clips"])
                  for rid in chosen}
    if len(spec_clips) != 1:
        raise SystemExit(f"tiles disagree on n_clips: {sorted(spec_clips)}")
    n_clips = spec_clips.pop()
    with h5py.File(EVAL_CLIPS_H5, "r") as f:
        per_clip_frames = int(f.attrs["n_frames_per_clip"])
    if per_clip_frames != mc.CLIP_FRAMES:
        raise SystemExit(f"asset has {per_clip_frames}-frame clips, collage assumes "
                         f"{mc.CLIP_FRAMES}")

    captions = caption_segments(labels, coarse, names, range(n_clips))
    text = captions_text(captions)
    if args.check:
        same = CAPTIONS.exists() and CAPTIONS.read_text() == text
        print(f"CHECK: video_captions.csv {'unchanged' if same else 'CHANGED'} "
              f"({len(captions)} rows)")
        if not same:
            raise SystemExit(1)
    else:
        CAPTIONS.write_text(text)

    audit, first_death = audit_lifetimes(store, picks, clips, failures, captions, n_clips)
    report += ["", "=== episodes ===", audit]

    preset = mc.PRESETS[PRESET]
    output = Path(preset["output"])
    if args.check or args.no_render:
        print("\n".join(report))
        print(f"\n(not rendered; {output} {'exists' if output.exists() else 'missing'})")
        return

    grid = preset["grid"]
    for row in grid:
        for cell in row:
            if cell is not None and not Path(cell[0]).exists():
                raise SystemExit(f"missing input video: {cell[0]}")
    order_stats = mc._cell_stats(grid)
    plan = mc.build_plan(grid, order_stats, mc.TAIL_S, preset["sort_clips"], mc.FPS)
    overlay = mc.Overlay(grid, mark_deaths=True,
                         captions=mc.read_captions(CAPTIONS),
                         caption_cell=preset["caption_cell"], fps=mc.FPS)
    timeline = mc.make_collage(grid, output, mc.TILE_W, mc.TILE_H, mc.FPS, plan, overlay)

    report += ["", "=== output ===",
               f"{output} ({sum(n for _, _, n in timeline)} clip frames + "
               f"{mc.FADE_FRAMES * (len(timeline) - 1)} fade frames)",
               *(f"    clip {c}: output frames {s}..{s + n} ({n / mc.FPS:.1f} s)"
                 for c, s, n in timeline)]
    report.append(contact_sheet(output, timeline, first_death, captions, CONTACT_SHEET))
    AUDIT.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nWrote {AUDIT.name}, {CAPTIONS.name}, figures/{CONTACT_SHEET.name}")


if __name__ == "__main__":
    main()
