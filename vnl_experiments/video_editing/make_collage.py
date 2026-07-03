#!/usr/bin/env python3
"""Tile several videos into a labelled grid collage.

Reads a grid of videos frame-by-frame in lockstep, resizes each into a tile,
draws a label on it, lays them out in a grid, and streams the result to a
high-quality H.264 file (``HQVideoWriter``). Only one frame per input is held at
a time, so memory stays small.

The default ``GRID`` is the requested 2x2: the raw reference footage in the
top-left, then the ``_nodetach`` forward-model checkpoints (``_noreset`` free-run
videos) for delays 0 / 5 / 10 (= 0 / 50 / 100 ms at ctrl_dt 10 ms).

Run:
    cd vnl-experiments && ../.venv/bin/python -m vnl_experiments.video_editing.make_collage
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from vnl_experiments.video_editing.hq_video import HQVideoWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval_videos"

REFERENCE = (EVAL_DIR / "raw_camera4_clips0-3.mp4", "Reference")


def _noreset(run_dir: str):
    """The free-run rollout video for a rendered checkpoint dir."""
    return EVAL_DIR / run_dir / "rollout_noreset.mp4"


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
}
DEFAULT_PRESET = "fm_nodetach"

# Each tile's size; the collage is (cols * TILE_W) x (rows * TILE_H).
TILE_W, TILE_H = 960, 600
FPS = 50


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


def make_collage(grid, output: Path, tile_w: int, tile_h: int, fps: int) -> None:
    rows, cols = len(grid), max(len(r) for r in grid)
    caps, n_frames = open_caps(grid)
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT))
              for row in caps for c in row if c is not None]
    if len(set(counts)) > 1:
        print(f"WARNING: input frame counts differ {sorted(set(counts))}; "
              f"using the shortest ({n_frames}).")

    out_w, out_h = cols * tile_w, rows * tile_h
    print(f"Collage {rows}x{cols} -> {out_w}x{out_h}, {n_frames} frames @ {fps} fps")
    writer = HQVideoWriter(output, out_w, out_h, fps, pix_fmt="bgr24")

    black = np.zeros((tile_h, tile_w, 3), np.uint8)
    for _ in range(n_frames):
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
        writer.write(canvas)

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
    args = p.parse_args()

    preset = PRESETS[args.preset]
    grid = preset["grid"]
    output = args.output or preset["output"]

    for row in grid:
        for cell in row:
            if cell is not None and not cell[0].exists():
                raise FileNotFoundError(f"Missing input video: {cell[0]}")

    make_collage(grid, output, args.tile_w, args.tile_h, args.fps)


if __name__ == "__main__":
    main()
