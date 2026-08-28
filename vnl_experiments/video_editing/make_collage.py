#!/usr/bin/env python3
"""Tile several videos into a labelled grid collage.

Reads a grid of videos frame-by-frame in lockstep, resizes each into a tile,
draws a label on it, lays them out in a grid, and streams the result to a
high-quality H.264 file (``HQVideoWriter``). Only one frame per input is held at
a time, so memory stays small.

The default ``GRID`` is the requested 2x2: the raw reference footage in the
top-left, then the ``_nodetach`` forward-model checkpoints (``_noreset`` free-run
videos) for delays 0 / 5 / 10 (= 0 / 50 / 100 ms at ctrl_dt 10 ms).

The ``*_newxml`` presets instead draw on the artifact store
(``artifacts/video/<wandb_id>/``), whose renders use the newer
``rodent_no_tail_collisions.xml`` body; there the delay ladder is 0 / 10 / 20
steps because the store has no delay-5 video.

Those presets also re-cut the timeline from the ``.stats.json`` sidecars
(``--trim`` / ``--sort-clips``), which needs no re-rendering:

  * **trim** -- a clip stops ``TAIL_S`` after its *last* tile dies, so the
    post-mortem flailing is kept (it shows what failure means) but bounded.
    A clip that anyone survives plays in full.
  * **sort** -- clips play easiest-first by mean lifetime or reward, so the
    comparison builds instead of jumping around.

Run:
    cd vnl-experiments && ../.venv/bin/python -m vnl_experiments.video_editing.make_collage
    ... --preset expfm_newxml
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from vnl_experiments.video_editing.hq_video import HQVideoWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval_videos"
ARTIFACT_VIDEO_DIR = REPO_ROOT / "artifacts" / "video"

# Default artifact video spec: see _artifact for what it pins down.
VIDEO_SPEC = "vid4c-67714d32"

# Raw Camera-4 footage for new-eval clips 0-3, built by extract_raw_eval_video.py.
# Same eval h5 (eval_clips_32x30s.h5) and frame count (6075) as both the legacy
# rollout_noreset.mp4 renders and the artifact-store videos, so it aligns with either.
REFERENCE = (EVAL_DIR / "raw_camera4_clips0-3.mp4", "Reference")


def _noreset(run_dir: str):
    """The free-run rollout video for a rendered checkpoint dir (legacy layout).

    These are all ``rodent.xml`` (the old collision model); the newer
    almost-full-collision renders live in the artifact store -- see ``_artifact``.
    """
    return EVAL_DIR / run_dir / "rollout_noreset.mp4"


def _artifact(wandb_id: str, spec: str = VIDEO_SPEC):
    """The artifact-store video for a run, keyed by wandb id rather than run dir.

    ``vid4c-67714d32`` is the producer-v2 spec: 4 new-eval clips, auto_reset off
    (so these are free-run, like ``rollout_noreset.mp4``), rendered on
    ``rodent_no_tail_collisions.xml``. The older ``vid4c-ea6ef8a8`` spec sits
    beside it but predates the walker XML being recorded in the metadata.
    """
    return ARTIFACT_VIDEO_DIR / wandb_id / f"{spec}.mp4"


# The six 2 B-step forward-model runs behind the *_newxml presets, pooled to give
# both arms a single clip ordering.
NEWXML_2G_IDS = ["dgv264dt", "itedqjyt", "3dfjggpw",   # expfm  delay 0 / 10 / 20
                 "nsfdsuk2", "1dbmh27r", "x1wjdvt9"]   # pgfm   delay 0 / 10 / 20

# Named 2x2 layouts: reference top-left, then delay 0 / 5 / 10 (= 0 / 50 / 100 ms).
# Each grid is a row-major list of (video_path, label) cells (None => black cell).
PRESETS = {
    "fm_nodetach": {
        "output": EVAL_DIR / "collage_2x2_nodetach_noreset.mp4",
        "grid": [
            [REFERENCE,
             (_noreset("RodentForwardModel_delay0_eff0_nodetach-20260630-083013"),
              "No delay")],
            [(_noreset("RodentForwardModel_delay5_eff5_nodetach-20260630-083050"),
              "50ms delay"),
             (_noreset("RodentForwardModel_delay10_eff10_nodetach-20260630-083050"),
              "100ms delay")],
        ],
    },
    "fm": {
        "output": EVAL_DIR / "collage_2x2_fm_noreset.mp4",
        "grid": [
            [REFERENCE,
             (_noreset("RodentForwardModel_delay0_eff0-20260619-032834"),
              "No delay")],
            [(_noreset("RodentForwardModel_delay5_eff5-20260619-033243"),
              "50ms delay"),
             (_noreset("RodentForwardModel_delay10_eff10-20260619-033822"),
              "100ms delay")],
        ],
    },
    # Imitation-target representation comparison at delay 0 (analysis:
    # imitation-target-representation). Same network, three env target frames.
    "target_repr": {
        "output": EVAL_DIR / "collage_2x2_target_repr_noreset.mp4",
        "grid": [
            [REFERENCE,
             (_noreset("RodentEncDec_delay0_eff0-20260624-070654"),  # o49pypx0
              "Relative target")],
            [(_noreset("RodentEncDec_delay0_eff0-20260624-070609"),  # cwuwoywj
              "Joints absolute; root relative"),
             (_noreset("RodentEncDec_delay0_eff0-20260624-070620"),  # 2f08y5is
              "Absolute target")],
        ],
    },
    "encdec": {
        "output": EVAL_DIR / "collage_2x2_encdec_noreset.mp4",
        "grid": [
            [REFERENCE,
             (_noreset("RodentEncDec_delay0_eff0-20260629-090548"), "No delay")],
            [(_noreset("RodentEncDec_delay5_eff5-20260623-085807"), "50ms delay"),
             (_noreset("RodentEncDec_delay10_eff10-20260629-094431"), "100ms delay")],
        ],
    },
    # --- almost-full collisions (rodent_no_tail_collisions.xml) --------------
    # Both arms order their clips by the pooled difficulty of all six runs
    # (NEWXML_2G_IDS), so the two videos stay comparable clip-for-clip; lifetime
    # and reward happen to agree on that ordering. Trimming, by contrast, is
    # per-video: each collage cuts on its own three tiles.
    # The 2 B-step forward-model runs, seed 42, train commit 25732c42, all
    # rendered by producer commit afbeea0. Delay 5 has no video in the store, so
    # the ladder is 0 / 10 / 20 steps = 0 / 100 / 200 ms at ctrl_dt 10 ms.
    "expfm_newxml": {
        "output": EVAL_DIR / "collage_2x2_expfm_2g_newxml.mp4",
        "trim": True,
        "sort_clips": "lifetime",
        "order_ids": NEWXML_2G_IDS,
        "grid": [
            [REFERENCE,
             (_artifact("dgv264dt"), "No delay")],           # delay0_eff0-20260813-073550
            [(_artifact("itedqjyt"), "100ms delay"),         # delay10_eff10-20260813-074327
             (_artifact("3dfjggpw"), "200ms delay")],        # delay20_eff20-20260813-074628
        ],
    },
    "pgfm_newxml": {
        "output": EVAL_DIR / "collage_2x2_pgfm_2g_newxml.mp4",
        "trim": True,
        "sort_clips": "lifetime",
        "order_ids": NEWXML_2G_IDS,
        "grid": [
            [REFERENCE,
             (_artifact("nsfdsuk2"), "No delay")],           # delay0_eff0_nodetach-20260813-064425
            [(_artifact("1dbmh27r"), "100ms delay"),         # delay10_eff10_nodetach-20260813-064420
             (_artifact("x1wjdvt9"), "200ms delay")],        # delay20_eff20_nodetach-20260813-070304
        ],
    },
}
DEFAULT_PRESET = "fm_nodetach"

# Each tile's size; the collage is (cols * TILE_W) x (rows * TILE_H).
TILE_W, TILE_H = 960, 600
FPS = 50

# Clip layout of every rendered rollout: N clips of CLIP_FRAMES, with FADE_FRAMES
# of logistic fade between consecutive clips (so 4 clips == 4*1500 + 3*25 = 6075).
# Must match vnl_experiments.delays.eval_videos.
CLIP_FRAMES = 1500
FADE_FRAMES = 25

# How long to keep rolling after the last tile dies, before fading to the next clip.
TAIL_S = 3.0

# A clip's `done` flag is `terminated OR truncated` (imitation.py:210), and
# truncation fires a few frames short of the clip end (_last_valid_frame). So a
# tile that "terminated" within this margin of the horizon actually *survived*
# the clip, and must not count towards "everyone has fallen".
TRUNC_MARGIN_S = 0.2


def fade_factor(t: int, n: int) -> float:
    """Logistic fade-out weight (1->0), identical to Imitation.render."""
    return 1.0 / (1.0 + math.exp(10.0 * (t / n - 0.5)))


def _load_stats(video: Path):
    """The ``.stats.json`` sidecar beside an artifact video, or None if absent.

    Legacy ``eval_videos/<run>/rollout*.mp4`` renders have no sidecar, so presets
    built from those cannot be trimmed or reordered.
    """
    sidecar = video.with_suffix(".stats.json")
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text())


def _cell_stats(grid):
    """Stats sidecars for every grid cell that has one (i.e. the simulated tiles)."""
    return [st for st in (_load_stats(cell[0]) for row in grid for cell in row
                          if cell is not None) if st is not None]


def _fell(per_clip: dict, horizon_s: float) -> bool:
    """Did this tile actually fall in this clip (as opposed to reaching the end)?"""
    return (bool(per_clip["terminated"])
            and per_clip["time_alive_s"] < horizon_s - TRUNC_MARGIN_S)


def build_plan(grid, order_stats, tail_s: float, sort_by: str, fps: int):
    """Plan the output as ``[(source_clip_idx, n_frames), ...]`` in play order.

    Trimming: a clip is cut ``tail_s`` after its *last* tile dies, but only once
    every tile has died -- while anyone is still up there is something worth
    watching. Clips nobody fails out of play in full.

    Ordering: clips are sorted easiest-first by the mean over ``order_stats`` of
    either per-clip lifetime or per-clip reward. ``order_stats`` is passed in
    separately from the grid so sibling collages (e.g. the two arms) can share one
    ordering and stay comparable clip-for-clip.
    """
    tile_stats = _cell_stats(grid)
    if not tile_stats:
        raise ValueError("no .stats.json sidecars found; cannot trim or reorder")

    n_clips = min(st["n_eval_clips"] for st in tile_stats)
    clip_frames = min(st["clip_length"] for st in tile_stats)
    horizon_s = clip_frames / fps
    tail = round(tail_s * fps)

    key = {"lifetime": "time_alive_s", "reward": "total_reward"}[sort_by]
    difficulty = {
        c: sum(st["per_clip"][c][key] for st in order_stats) / len(order_stats)
        for c in range(n_clips)
    }

    plan = []
    for c in sorted(range(n_clips), key=lambda c: -difficulty[c]):  # easiest first
        per_clip = [st["per_clip"][c] for st in tile_stats]
        if all(_fell(pc, horizon_s) for pc in per_clip):
            last_death = max(pc["time_alive_s"] for pc in per_clip)
            n = min(clip_frames, round(last_death * fps) + tail)
        else:
            n = clip_frames
        plan.append((c, n))
        print(f"  clip {c}: {sort_by} {difficulty[c]:8.2f} -> {n:>4} frames "
              f"({n / fps:5.1f}s){'' if n < clip_frames else '  [full]'}")
    return plan


def draw_label(tile: np.ndarray, text: str) -> None:
    """Draw a label with a translucent dark backing box (top-left, in place)."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 8
    x, y = 12, 12 + th
    box = tile[y - th - pad: y + base + pad, x - pad: x + tw + pad]
    if box.size:  # blend a dark rectangle behind the text for legibility
        box[:] = (0.35 * box).astype(np.uint8)
    cv2.putText(tile, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def open_caps(grid):
    """Open every VideoCapture; return (caps grid, min frame count)."""
    caps, counts = [], []
    for row in grid:
        crow = []
        for cell in row:
            if cell is None:
                crow.append(None)
                continue
            path, _ = cell
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open {path}")
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            crow.append(cap)
        caps.append(crow)
    return caps, min(counts)


def compose(grid, caps, tile_w, tile_h, out_w, out_h):
    """Read one frame from every cap and lay the labelled tiles out on a canvas."""
    black = np.zeros((tile_h, tile_w, 3), np.uint8)
    canvas = np.zeros((out_h, out_w, 3), np.uint8)
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            cap = caps[r][c]
            if cap is None:
                continue
            ok, frame = cap.read()
            if not ok:
                tile = black
            else:
                tile = cv2.resize(frame, (tile_w, tile_h),
                                  interpolation=cv2.INTER_AREA)
                draw_label(tile, cell[1])
            canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    return canvas


def make_collage(grid, output: Path, tile_w: int, tile_h: int, fps: int,
                 plan=None) -> None:
    rows, cols = len(grid), max(len(r) for r in grid)
    caps, n_frames = open_caps(grid)
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT))
              for row in caps for c in row if c is not None]
    if len(set(counts)) > 1:
        print(f"WARNING: input frame counts differ {sorted(set(counts))}; "
              f"using the shortest ({n_frames}).")

    out_w, out_h = cols * tile_w, rows * tile_h
    total = n_frames if plan is None else (
        sum(n for _, n in plan) + FADE_FRAMES * (len(plan) - 1))
    print(f"Collage {rows}x{cols} -> {out_w}x{out_h}, {total} frames @ {fps} fps")
    writer = HQVideoWriter(output, out_w, out_h, fps, pix_fmt="bgr24")

    if plan is None:
        # Stream the inputs straight through, fades and clip order as rendered.
        for _ in range(n_frames):
            writer.write(compose(grid, caps, tile_w, tile_h, out_w, out_h))
    else:
        # Play the planned clips in order, seeking each input to the clip start
        # (verified frame-exact on these files) and fading between segments.
        for seg, (clip_idx, count) in enumerate(plan):
            start = clip_idx * (CLIP_FRAMES + FADE_FRAMES)
            for row in caps:
                for cap in row:
                    if cap is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            canvas = None
            for _ in range(count):
                canvas = compose(grid, caps, tile_w, tile_h, out_w, out_h)
                writer.write(canvas)
            # Fade the composed canvas (labels included) between clips, matching
            # the per-tile fade the renderer bakes in at clip boundaries.
            if seg < len(plan) - 1 and canvas is not None:
                for t in range(FADE_FRAMES):
                    writer.write((canvas * fade_factor(t, FADE_FRAMES)).astype(np.uint8))

    writer.close()
    for row in caps:
        for cap in row:
            if cap is not None:
                cap.release()
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    p.add_argument("--output", type=Path, default=None,
                   help="override the preset's default output path")
    p.add_argument("--tile-w", type=int, default=TILE_W)
    p.add_argument("--tile-h", type=int, default=TILE_H)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--trim", dest="trim", action="store_true", default=None,
                   help="cut each clip TAIL_S after its last tile dies")
    p.add_argument("--no-trim", dest="trim", action="store_false",
                   help="play every clip in full, as rendered")
    p.add_argument("--tail-s", type=float, default=TAIL_S,
                   help="seconds to keep rolling after the last tile dies")
    p.add_argument("--sort-clips", choices=["none", "lifetime", "reward"],
                   default=None, help="order clips easiest-first by this metric")
    args = p.parse_args()

    preset = PRESETS[args.preset]
    grid = preset["grid"]
    output = args.output or preset["output"]
    trim = preset.get("trim", False) if args.trim is None else args.trim
    sort_by = args.sort_clips or preset.get("sort_clips", "none")

    for row in grid:
        for cell in row:
            if cell is not None and not cell[0].exists():
                raise FileNotFoundError(f"Missing input video: {cell[0]}")

    plan = None
    if trim or sort_by != "none":
        # Ordering may pool sibling runs (see NEWXML_2G_IDS); default to this
        # preset's own tiles.
        order_ids = preset.get("order_ids")
        order_stats = ([_load_stats(_artifact(i)) for i in order_ids] if order_ids
                       else _cell_stats(grid))
        if any(st is None for st in order_stats):
            raise FileNotFoundError("missing a .stats.json sidecar for clip ordering")
        print(f"Plan (trim={trim}, sort_clips={sort_by}, tail={args.tail_s}s):")
        plan = build_plan(grid, order_stats,
                          args.tail_s if trim else 0.0,
                          sort_by if sort_by != "none" else "lifetime", args.fps)
        if not trim:  # ordering only: keep every clip whole
            plan = [(c, CLIP_FRAMES) for c, _ in plan]

    make_collage(grid, output, args.tile_w, args.tile_h, args.fps, plan)


if __name__ == "__main__":
    main()
