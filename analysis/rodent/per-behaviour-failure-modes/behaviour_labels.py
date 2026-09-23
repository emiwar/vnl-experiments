#!/usr/bin/env python3
"""Attach behaviour labels to both eval clip sets, and prove the attachment is sound.

This folder asks which *behaviours* a policy fails at, so every number downstream rests on
a label being attached to the right frame. There are two independent labellings, with
different taxonomies, different clip sets and different granularity -- which is what makes
their agreement evidence rather than restatement:

**The 5 s clips (``train`` / ``old_eval``): one label per clip, six classes.**
``reference_clips.h5`` carries ``config['model']['snips_order']`` -- 842 snippet basenames
like ``RGroom_4`` -- whose alphabetic prefix is the behaviour. ``ReferenceClips.split()``
propagates ``clip_names`` to the test half *in clip-axis order*, and the eval rolls out
``clip_ids = jp.arange(n_clips)`` over exactly that object, so clip *i* of ``old_eval``
has label ``split()[1].clip_names[i]``.

**The 30 s clips (``new_eval``): one label per frame, 100 MotionMapper clusters.**
``assets/art/2020_12_22_1/2020_12_22_1.h5`` holds ``behavior/motion_mapper``, a
360 000-frame cluster-id vector over the whole session with a ``names`` attribute mapping
id to a human-readable behaviour. ``eval_clips_32x30s.h5`` stores the
``clip_start_frame`` / ``clip_end_frame`` of each of its 32 clips into that same session,
so the labels slice straight in.

Two properties of that second join are load-bearing and are *asserted* here rather than
assumed, because both would fail silently:

1. **Frame alignment is checked on ``pose/keypoints``, never on ``pose/qpos``.** The
   session file is a different STAC fit from the one the eval clips were cut from
   (``art_2020_12_22_1_new_STAC_from_Charles.h5``): their ``qpos`` differ by up to 1.7,
   while their raw keypoints -- the same DANNCE output, in mm rather than m and in a
   different channel order -- agree to float32 rounding. So the keypoints are the common
   timebase and the only safe thing to align on.
2. **``names`` is indexed 1-based**: label *k* means ``names[k - 1]``. This is forced
   structurally (id 0 never occurs, id 100 does, and ``len(names) == 100``), but a
   structural argument would not survive a re-export that changed the convention. So it is
   also pinned kinematically: under the 1-based reading mean speed rises monotonically
   ProneStill < ProneSlow < Amble < Walk < WalkFast and snout height separates Rear* from
   Prone*, and under the 0-based reading it does not.

Outputs, all committed:

``behaviour_labels_frames.npz``   ``labels`` (32, 1500) uint8 cluster ids, ``names`` (100,)
                                  and ``coarse`` (100,) -- the coarse group of each id, so
                                  the grouping travels with the labels instead of being
                                  restated by every consumer
``behaviour_labels_clips.csv``    one row per (dataset, clip): label, coarse group, provenance
``behaviour_labels.txt``          the verdict of every assertion, plus file hashes

Run it
------
    ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/behaviour_labels.py
    ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/behaviour_labels.py --verify
    ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/behaviour_labels.py --check

``--verify`` adds the two expensive cross-checks: it loads the real ``ReferenceClips`` and
asserts its ``split()[1].clip_names`` matches the lazily-derived labels element-wise, and it
re-runs the keypoint alignment over the whole 360 000 frames rather than three windows.
``--check`` rebuilds and diffs against the committed outputs, exiting non-zero on drift; it
**implies ``--verify``**, because the committed ``behaviour_labels.txt`` records the full
verdict and a drift check that skipped half the assertions would compare the wrong text
against it. The committed outputs are therefore always the ``--verify`` ones; plain runs are
for iterating and take ~2 s against ~16 s.
"""

import argparse
import difflib
import hashlib
import io
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

ASSET_DIR = REPO_ROOT / "assets" / "art" / "2020_12_22_1"
#: Per-frame MotionMapper clusters + ephys for the whole session.
SESSION_H5 = ASSET_DIR / "2020_12_22_1.h5"
#: The re-STAC the 30 s eval clips were cut from -- the frame index `clip_start_frame` is into.
BASE_STAC_H5 = ASSET_DIR / "art_2020_12_22_1_new_STAC_from_Charles.h5"
#: The 30 s eval set itself (`new_eval`).
NEW_EVAL_H5 = ASSET_DIR / "eval_clips_32x30s.h5"

MOCAP_HZ = 50

#: Frames per 5 s clip in the train/old_eval set, and the split `ReferenceClips.split()` does.
CLIP_LENGTH_5S = 250
TRAIN_RATIO = 0.8
SPLIT_SEED = 0

#: `ReferenceClips._extract_clip_names`'s pattern, applied to the snippet basename.
SNIPPET_RE = re.compile(r"([A-Za-z]+)_\d+(\.p)?")

# ---------------------------------------------------------------------------
# The coarse grouping
# ---------------------------------------------------------------------------
# The two labellings use different vocabularies, so any figure that compares them needs a
# shared axis. This is that axis, defined once. The order is sedentary -> dynamic, so the
# hypothesis under test ("near-ceiling when still, worse when moving") reads as a slope
# rather than as a permutation the reader has to hold in their head.
COARSE_ORDER = ("groom", "still", "rear", "locomote")

#: MotionMapper behaviour name -> coarse group. `(check)` and `TrackingError` are not
#: behaviours -- they mark frames the annotator flagged or the tracker lost -- so they get
#: their own group and are excluded from every per-behaviour statistic.
MOTION_MAPPER_COARSE = {
    "FaceGroom": "groom",
    "AmbleGroom": "groom",
    "ProneStill": "still",
    "ProneSlow": "still",
    "ProneSniff": "still",
    "Hunch": "still",
    "RearDown": "rear",
    "RearLow": "rear",
    "RearMid": "rear",
    "RearHigh": "rear",
    "RearSniff": "rear",
    "Amble": "locomote",
    "Walk": "locomote",
    "WalkFast": "locomote",
    "(check)": "exclude",
    "TrackingError": "exclude",
}

#: Snippet behaviour prefix -> the same coarse groups.
SNIPPET_COARSE = {
    "FaceGroom": "groom",
    "RGroom": "groom",
    "LGroom": "groom",
    "Rear": "rear",
    "Walk": "locomote",
    "FastWalk": "locomote",
}

# ---------------------------------------------------------------------------
# The kinematic check that pins the 1-based convention
# ---------------------------------------------------------------------------
#: Mean horizontal speed must rise along this chain. Under the 0-based reading it does not
#: (WalkFast comes out slower than ProneStill), which is what makes this discriminating.
SPEED_ORDER = ("ProneStill", "ProneSlow", "Amble", "Walk", "WalkFast")
#: Every Rear* behaviour must hold the snout higher than every Prone* behaviour.
REAR_NAMES = ("RearLow", "RearMid", "RearHigh")
PRONE_NAMES = ("ProneStill", "ProneSlow", "ProneSniff")
#: Frames used for the kinematic check. Enough to give every name in SPEED_ORDER a few
#: hundred frames; the full session is not needed to settle an ordering this coarse.
KINEMATIC_FRAMES = 120_000
#: Keypoint-alignment tolerance, in mm. The two files agree to float32 quantisation of
#: mm-scale values (~1.5e-5 mm measured), so this is four orders of magnitude of slack.
ALIGN_TOL_MM = 1e-3


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class Verdicts:
    """Collects assertion results so the report records what was checked, not just that
    nothing raised. A failure is recorded *and* raised: the .txt is evidence, not a log."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        self.lines.append(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures += 1
        return ok

    def note(self, text: str) -> None:
        self.lines.append(text)

    def raise_on_failure(self) -> None:
        if self.failures:
            raise SystemExit(
                f"{self.failures} assertion(s) failed; see the FAIL lines above. "
                f"The labels were NOT written."
            )


# ---------------------------------------------------------------------------
# The 30 s clips: per-frame MotionMapper labels
# ---------------------------------------------------------------------------

def check_keypoint_alignment(v: Verdicts, full: bool) -> None:
    """Assert the session file and the eval clips' source share a frame index.

    Compares ``pose/keypoints`` (mm, its own channel order) against the base re-STAC's
    ``kp_data`` (m, alphabetical order), reordered by name and scaled. ``qpos`` is
    deliberately not used: it is a different fit and differs by O(1).
    """
    with h5py.File(SESSION_H5, "r") as f, h5py.File(BASE_STAC_H5, "r") as g:
        new_names = [str(x) for x in f["pose/keypoints"].attrs["names"]]
        old_names = [s.decode() for s in g["kp_names"][()]]
        v.check(sorted(new_names) == sorted(old_names), "keypoint name sets match",
                f"{len(new_names)} keypoints")
        v.check(f["pose/keypoints"].shape[0] == g["qpos"].shape[0],
                "session and base re-STAC have the same frame count",
                f"{f['pose/keypoints'].shape[0]} frames")

        old_idx = {n: i for i, n in enumerate(old_names)}
        n_total = f["pose/keypoints"].shape[0]
        windows = ([(0, n_total)] if full
                   else [(0, 3000), (17_600, 3000), (n_total - 3000, 3000)])

        worst = 0.0
        for start, n in windows:
            # Read in slabs: the full-session mode would otherwise materialise ~500 MB
            # twice over.
            for lo in range(start, start + n, 20_000):
                hi = min(lo + 20_000, start + n)
                a = np.asarray(f["pose/keypoints"][lo:hi])              # (n, 3, 23) mm
                b = np.asarray(g["kp_data"][lo:hi]).reshape(hi - lo, len(old_names), 3)
                b = np.stack([b[:, old_idx[nm], :] for nm in new_names], axis=2) * 1000.0
                worst = max(worst, float(np.nanmax(np.abs(a - b))))

        scope = "all 360k frames" if full else "3 windows x 3000 frames"
        v.check(worst < ALIGN_TOL_MM,
                "keypoints agree between session file and eval-clip source",
                f"max |delta| = {worst:.3e} mm over {scope} (tol {ALIGN_TOL_MM:g})")


def check_label_indexing(v: Verdicts, labels: np.ndarray, names: np.ndarray) -> None:
    """Assert `names` is 1-based, structurally and then kinematically."""
    lo, hi = int(labels.min()), int(labels.max())
    v.check(lo >= 1, "cluster ids start at 1 (id 0 never occurs)", f"min id = {lo}")
    v.check(hi <= len(names), "cluster ids fit names[k-1]",
            f"max id = {hi}, len(names) = {len(names)}")
    # The discriminating structural fact: if ids were 0-based, id == len(names) could not
    # occur. That it does rules the 0-based reading out on its own.
    v.check(hi == len(names),
            "max id equals len(names), which only a 1-based index permits",
            f"{hi} == {len(names)}")

    with h5py.File(SESSION_H5, "r") as f:
        kp = f["pose/keypoints"]
        kp_names = [str(x) for x in kp.attrs["names"]]
        spine, snout = kp_names.index("SpineM"), kp_names.index("Snout")
        K = np.asarray(kp[:KINEMATIC_FRAMES])                  # (N, 3, 23) mm
        frame_labels = f["behavior/motion_mapper"][:KINEMATIC_FRAMES]

    snout_z = K[:, 2, snout]
    step = np.linalg.norm(np.diff(K[:, :2, spine], axis=0), axis=1)
    speed = np.r_[step, step[-1]] * MOCAP_HZ / 1000.0           # m/s

    for shift, tag in ((1, "1-based"), (0, "0-based")):
        nm = names[np.clip(frame_labels - shift, 0, len(names) - 1)]
        speeds = {b: float(np.nanmean(speed[nm == b])) for b in SPEED_ORDER}
        rear = [float(np.nanmean(snout_z[nm == b])) for b in REAR_NAMES]
        prone = [float(np.nanmean(snout_z[nm == b])) for b in PRONE_NAMES]
        monotone = all(speeds[a] < speeds[b]
                       for a, b in zip(SPEED_ORDER, SPEED_ORDER[1:]))
        reared = min(rear) > max(prone)
        chain = " < ".join(f"{b} {speeds[b]:.3f}" for b in SPEED_ORDER)
        if shift == 1:
            v.check(monotone, "1-based: speed rises ProneStill -> WalkFast", chain)
            v.check(reared, "1-based: every Rear* snout is above every Prone* snout",
                    f"min(Rear*) = {min(rear):.0f} mm > max(Prone*) = {max(prone):.0f} mm")
        else:
            # The negative control. If this ever passed, the check above would not be
            # evidence for anything.
            v.check(not (monotone and reared),
                    "0-based reading is kinematically incoherent, as required",
                    f"speed chain would be {chain}")


def motion_mapper_labels(v: Verdicts) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Slice the session's per-frame clusters into the 32 x 1500 `new_eval` layout."""
    with h5py.File(SESSION_H5, "r") as f:
        mm = f["behavior/motion_mapper"]
        names = np.array([str(x) for x in mm.attrs["names"]])
        frame_labels = mm[()]
    with h5py.File(NEW_EVAL_H5, "r") as e:
        start = e["clip_start_frame"][()]
        end = e["clip_end_frame"][()]
        src_label = [s.decode() for s in e["clip_src_label"][()]]
        n_frames = int(e.attrs["n_frames_per_clip"])
        n_clips = int(e.attrs["n_clips"])
        source_file = str(e.attrs["source_file"])

    v.check(source_file == BASE_STAC_H5.name,
            "eval clips declare the source file this script aligned against",
            f"{source_file}")
    v.check(bool(np.all(end - start == n_frames)),
            "every clip spans exactly n_frames_per_clip",
            f"{n_frames} frames x {n_clips} clips")
    v.check(int(end.max()) <= len(frame_labels),
            "every clip lies inside the labelled session",
            f"max end frame {int(end.max())} <= {len(frame_labels)}")

    check_label_indexing(v, frame_labels, names)

    labels = np.stack([frame_labels[a:b] for a, b in zip(start, end)]).astype(np.uint8)
    v.check(labels.shape == (n_clips, n_frames), "label block has the clip layout",
            f"{labels.shape}")

    unknown = sorted(set(names[np.unique(labels) - 1]) - set(MOTION_MAPPER_COARSE))
    v.check(not unknown, "every behaviour present has a coarse group",
            "unmapped: " + ", ".join(unknown) if unknown else "13 names, all mapped")

    # Per-clip rows: a 30 s clip is a behaviour *mixture*, so its "label" is the modal
    # behaviour plus the fraction of frames that agree with it. Downstream work uses the
    # per-frame array; these rows are for reading and for provenance.
    rows = []
    for i in range(n_clips):
        seg = names[labels[i].astype(int) - 1]
        vals, counts = np.unique(seg, return_counts=True)
        modal = str(vals[counts.argmax()])
        rows.append({
            "dataset": "new_eval",
            "clip_index": i,
            "behaviour": modal,
            "coarse": MOTION_MAPPER_COARSE[modal],
            "modal_frac": round(float(counts.max() / len(seg)), 6),
            "n_behaviours": int(len(vals)),
            "label_source": "motion_mapper",
            "source_clip_index": -1,
            "clip_start_frame": int(start[i]),
            "clip_end_frame": int(end[i]),
            "clip_src_label": src_label[i],
        })
    return labels, names, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The 5 s clips: one snippet label per clip
# ---------------------------------------------------------------------------

def snippet_labels(v: Verdicts, verify: bool) -> pd.DataFrame:
    """Reproduce `ReferenceClips.split()` and attach the snippet behaviour to each clip.

    Reads only the `config` YAML and `qpos.shape` from the 200 MB reference file, so
    nothing large is loaded and no jax import is triggered.
    """
    from vnl_playground.tasks.rodent import consts

    ref_path = Path(consts.IMITATION_REFERENCE_PATH)
    with h5py.File(ref_path, "r") as f:
        config = yaml.safe_load(f["config"][()])
        n_frames = f["qpos"].shape[0]

    snips = config["model"]["snips_order"]
    n_clips = n_frames // CLIP_LENGTH_5S
    # ReferenceClips never checks this, and a mismatch would misalign every label rather
    # than raise -- so it is the first thing asserted.
    v.check(len(snips) == n_clips,
            "snips_order has one entry per clip",
            f"{len(snips)} snippets, {n_frames} frames / {CLIP_LENGTH_5S} = {n_clips}")

    behaviours = []
    for name in snips:
        m = SNIPPET_RE.search(Path(str(name)).name)
        if m is None:
            raise SystemExit(f"snippet name {name!r} does not match {SNIPPET_RE.pattern}")
        behaviours.append(m.group(1))
    behaviours = np.array(behaviours)

    unknown = sorted(set(behaviours) - set(SNIPPET_COARSE))
    v.check(not unknown, "every snippet behaviour has a coarse group",
            "unmapped: " + ", ".join(unknown) if unknown
            else f"{len(set(behaviours))} classes, all mapped")

    n_train = int(n_clips * TRAIN_RATIO)
    order = np.random.RandomState(SPLIT_SEED).permutation(n_clips)
    splits = {"train": order[:n_train], "old_eval": order[n_train:]}

    if verify:
        # The independent record: build the real object and compare. This is what catches
        # `split()` or `_extract_clip_names` changing under us.
        from vnl_playground.tasks.reference_clips import ReferenceClips
        clips = ReferenceClips(str(ref_path), CLIP_LENGTH_5S)
        train_clips, test_clips = clips.split(train_ratio=TRAIN_RATIO, seed=SPLIT_SEED)
        for name, obj in (("train", train_clips), ("old_eval", test_clips)):
            theirs = np.array([str(x) for x in obj.clip_names])
            mine = behaviours[splits[name]]
            v.check(theirs.shape == mine.shape and bool(np.all(theirs == mine)),
                    f"ReferenceClips.split() {name} labels match the lazy derivation",
                    f"{len(mine)} clips, element-wise equal")
    else:
        v.note("  [SKIP] ReferenceClips.split() cross-check (pass --verify)")

    rows = []
    for name, base_idx in splits.items():
        for i, src in enumerate(base_idx):
            behaviour = str(behaviours[src])
            rows.append({
                "dataset": name,
                "clip_index": i,
                "behaviour": behaviour,
                "coarse": SNIPPET_COARSE[behaviour],
                "modal_frac": 1.0,        # one snippet, one behaviour
                "n_behaviours": 1,
                "label_source": "snips_order",
                "source_clip_index": int(src),
                "clip_start_frame": int(src) * CLIP_LENGTH_5S,
                "clip_end_frame": (int(src) + 1) * CLIP_LENGTH_5S,
                "clip_src_label": str(snips[src]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(v: Verdicts, clips: pd.DataFrame, labels: np.ndarray,
                 names: np.ndarray, verify: bool, full_align: bool) -> str:
    out = io.StringIO()
    w = out.write
    w("Behaviour labels for the rodent eval clip sets\n")
    w("=" * 78 + "\n\n")
    w("Written by behaviour_labels.py. Every line below is an assertion that ran; the\n")
    w("labels are only written when all of them pass.\n\n")
    w(f"Mode: {'--verify' if verify else 'default'}"
      f" (keypoint alignment over {'all frames' if full_align else '3 windows'})\n\n")

    w("Assertions\n----------\n")
    w("\n".join(v.lines) + "\n\n")

    w("Source files\n------------\n")
    for p in (SESSION_H5, BASE_STAC_H5, NEW_EVAL_H5):
        w(f"  {p.name:46s} {p.stat().st_size:>12d} B  sha256 {sha256(p)[:16]}\n")
    from vnl_playground.tasks.rodent import consts
    ref = Path(consts.IMITATION_REFERENCE_PATH)
    w(f"  {ref.name:46s} {ref.stat().st_size:>12d} B  sha256 {sha256(ref)[:16]}\n\n")

    w("new_eval: per-frame MotionMapper coverage\n")
    w("-" * 42 + "\n")
    seg = names[labels.astype(int).ravel() - 1]
    total = len(seg)
    w(f"  {total} frames = {total / MOCAP_HZ:.0f} s over {labels.shape[0]} clips; "
      f"{len(np.unique(labels))} of {len(names)} clusters occur\n\n")
    w(f"  {'behaviour':14s} {'coarse':9s} {'frames':>8s} {'seconds':>9s} {'%':>7s}\n")
    vals, counts = np.unique(seg, return_counts=True)
    for i in np.argsort(-counts):
        b = str(vals[i])
        w(f"  {b:14s} {MOTION_MAPPER_COARSE[b]:9s} {counts[i]:8d} "
          f"{counts[i] / MOCAP_HZ:9.1f} {100 * counts[i] / total:6.2f}%\n")
    w("\n  By coarse group:\n")
    for g in COARSE_ORDER:
        n = sum(c for b, c in zip(vals, counts) if MOTION_MAPPER_COARSE[str(b)] == g)
        w(f"    {g:9s} {n:8d} frames {n / MOCAP_HZ:9.1f} s {100 * n / total:6.2f}%\n")
    excl = sum(c for b, c in zip(vals, counts)
               if MOTION_MAPPER_COARSE[str(b)] == "exclude")
    w(f"    {'exclude':9s} {excl:8d} frames "
      f"{excl / MOCAP_HZ:9.1f} s {100 * excl / total:6.2f}%\n\n")

    w("5 s clips: one snippet label per clip\n")
    w("-" * 38 + "\n")
    for ds in ("train", "old_eval"):
        sub = clips[clips.dataset == ds]
        w(f"  {ds} ({len(sub)} clips): ")
        vc = sub.behaviour.value_counts()
        w(", ".join(f"{b} {n}" for b, n in vc.items()) + "\n")
        cc = sub.coarse.value_counts()
        w(f"    coarse: "
          + ", ".join(f"{g} {cc.get(g, 0)}" for g in COARSE_ORDER) + "\n")
    w("\nThe two taxonomies are not the same partition -- MotionMapper resolves rearing\n")
    w("into five clusters and stillness into four, while the snippets have no 'still'\n")
    w("class at all (every snippet is groom, rear or locomote). Compare them on the\n")
    w("coarse groups above, never name-to-name.\n")
    return out.getvalue()


def write_or_check(path: Path, text: str, check: bool) -> bool:
    if not check:
        path.write_text(text)
        return True
    if not path.exists():
        print(f"{path.name}: missing")
        return False
    old = path.read_text()
    if old == text:
        return True
    diff = list(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                     f"{path.name} (committed)", f"{path.name} (rebuilt)"))
    print("".join(diff[:40]))
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true",
                   help="add the ReferenceClips.split() cross-check and align over all "
                        "360k frames instead of three windows")
    p.add_argument("--check", action="store_true",
                   help="rebuild and diff against the committed outputs; non-zero on drift. "
                        "Implies --verify, since the committed report records the full "
                        "verdict.")
    args = p.parse_args()
    # The committed outputs are the --verify ones (see the module docstring), so a drift
    # check has to run the same assertions or it would diff against the wrong text.
    verify = args.verify or args.check

    for path in (SESSION_H5, BASE_STAC_H5, NEW_EVAL_H5):
        if not path.exists():
            raise SystemExit(f"missing asset: {path}")

    v = Verdicts()
    v.note("frame alignment (session file vs the eval clips' source re-STAC)")
    check_keypoint_alignment(v, full=verify)
    v.note("")
    v.note("new_eval: MotionMapper cluster labels")
    labels, names, mm_clips = motion_mapper_labels(v)
    v.note("")
    v.note("train / old_eval: snippet labels")
    snip_clips = snippet_labels(v, verify=verify)
    v.raise_on_failure()

    clips = pd.concat([snip_clips, mm_clips], ignore_index=True)
    clips = clips.sort_values(["dataset", "clip_index"], ignore_index=True)
    report = build_report(v, clips, labels, names, verify, verify)

    ok = True
    csv_path = HERE / "behaviour_labels_clips.csv"
    ok &= write_or_check(csv_path, clips.to_csv(index=False), args.check)
    ok &= write_or_check(HERE / "behaviour_labels.txt", report, args.check)

    npz_path = HERE / "behaviour_labels_frames.npz"
    if args.check:
        if not npz_path.exists():
            print(f"{npz_path.name}: missing")
            ok = False
        else:
            with np.load(npz_path, allow_pickle=False) as z:
                same = (np.array_equal(z["labels"], labels)
                        and np.array_equal(z["names"].astype(str), names.astype(str))
                        and "coarse" in z
                        and np.array_equal(
                            z["coarse"].astype(str),
                            np.array([MOTION_MAPPER_COARSE[n] for n in names])))
            if not same:
                print(f"{npz_path.name}: contents differ")
                ok = False
    else:
        np.savez_compressed(
            npz_path, labels=labels, names=names.astype("S16"),
            # Per *id*, not per name, so a consumer indexes it exactly as it indexes
            # `names` and never has to hold `MOTION_MAPPER_COARSE` itself.
            coarse=np.array([MOTION_MAPPER_COARSE[n] for n in names], dtype="S10"))

    print(report)
    if args.check:
        print("behaviour_labels: CLEAN" if ok else "behaviour_labels: DRIFT")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
