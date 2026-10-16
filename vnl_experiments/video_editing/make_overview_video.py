#!/usr/bin/env python3
"""A 1x3 overview video of the imitation pipeline, for audiences new to the project.

Columns, left to right: the raw Camera-4 footage, the STAC reference poses the policy
is asked to imitate (rendered by ``render_reference_video``), and the policy's own
physics rollout. All three are the same camera view of the same moment, so they line
up frame-for-frame.

Unlike ``make_collage`` this is a presentation piece rather than a comparison grid:

  * a static header names each column in plain language, with arrows between them
    reading as a pipeline, and a footer says in one sentence what is going on;
  * only clips the policy survives are shown (checked against ``.stats.json``);
  * the rat is small in the full 1920x1200 view, so every tile is cropped to the same
    ``WINDOW`` that pans to follow the *target* body (smoothed), keeping the three
    tiles pixel-aligned with each other.

Run:
    cd vnl-experiments && ../.venv/bin/python -m \
        vnl_experiments.video_editing.make_overview_video
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

from vnl_experiments.video_editing.hq_video import HQVideoWriter
from vnl_experiments.video_editing.make_collage import (
    CLIP_FRAMES, EVAL_DIR, FADE_FRAMES, FPS, REFERENCE, TRUNC_MARGIN_S, VIDEO_SPEC_V3,
    _artifact, _fell, _load_stats, fade_factor)

# The least specialised policy: plain MLP encoder-decoder, no sensory or motor delay,
# torque actuators, all inputs, no config overrides
# (RodentEncDec_delay0_eff0-job43288960). Best new_eval and old_eval reward of the five
# delay-0 enc-dec runs with a video in the store.
WANDB_ID = "594219y6"
ROLLOUT = _artifact(WANDB_ID, VIDEO_SPEC_V3)
TARGET = EVAL_DIR / f"reference_{WANDB_ID}_{VIDEO_SPEC_V3}.mp4"
OUTPUT = EVAL_DIR / f"overview_imitation_{WANDB_ID}.mp4"

# Rendered clips 0 and 2 are the two the policy keeps up with for the full 30 s.
CLIPS = [0, 2]

SRC_W, SRC_H = 1920, 1200
WINDOW_W, WINDOW_H = 960, 600     # source crop, 2x zoom; same 16:10 as the source
TILE_W, TILE_H = 640, 400
HEADER_H, FOOTER_H = 132, 64
TRACK_SIGMA_S = 0.6               # pan smoothing

COLUMNS = [
    ("1  Real rat", "Video of a rat moving freely in an arena"),
    ("2  Target pose", "3D body pose tracked from video (STAC)"),
    ("3  Neural-network control", "Physics-simulated body moved by a network"),
]
FOOTER = ("A neural network is trained by reinforcement learning to move a simulated "
          "rat body so that it copies the real rat's movements.")
# The colour of each column's number badge: green for the arena, then the amber and
# teal the two renders use (render_reference_video.TARGET_RGBA; the walker XML).
ACCENTS = [(90, 170, 90), (235, 153, 64), (0, 160, 170)]

BG = (18, 22, 30)
TEXT = (240, 240, 240)
SUBTEXT = (175, 182, 195)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(bold: bool, size: int):
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold
                                              else "DejaVuSans.ttf")), size)


def draw_chrome(width: int, height: int) -> np.ndarray:
    """The static header and footer, as an RGB canvas the tiles are pasted into."""
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    title_f, sub_f, foot_f, arrow_f = (_font(True, 30), _font(False, 20),
                                       _font(False, 21), _font(True, 40))
    for i, ((title, sub), accent) in enumerate(zip(COLUMNS, ACCENTS)):
        cx = i * TILE_W + TILE_W // 2
        d.text((cx, 50), title, font=title_f, fill=TEXT, anchor="ms")
        d.text((cx, 88), sub, font=sub_f, fill=SUBTEXT, anchor="ms")
        d.rectangle([i * TILE_W + 40, HEADER_H - 14, (i + 1) * TILE_W - 40,
                     HEADER_H - 9], fill=accent)
        if i:
            d.text((i * TILE_W, 50), "→", font=arrow_f, fill=SUBTEXT, anchor="ms")
    d.text((width // 2, height - FOOTER_H // 2), FOOTER, font=foot_f, fill=TEXT,
           anchor="mm")
    return np.asarray(img)


def track_target(path: Path, clip: int) -> np.ndarray:
    """Per-frame crop origin ``(x0, y0)`` following the amber target body.

    The body's pixel centroid is found by colour (the target render is the only amber
    thing in frame), gaps are held, then the path is Gaussian-smoothed so the pan reads
    as a slow camera move rather than a jitter.
    """
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip * (CLIP_FRAMES + FADE_FRAMES))
    centres, last = [], (SRC_W / 2, SRC_H / 2)
    for _ in range(CLIP_FRAMES):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{path} ended early in clip {clip}")
        b, r = frame[..., 0].astype(int), frame[..., 2].astype(int)
        ys, xs = np.nonzero((r > 150) & (r - b > 80))
        if len(xs):
            last = (xs.mean(), ys.mean())
        centres.append(last)
    cap.release()
    centres = gaussian_filter1d(np.array(centres), TRACK_SIGMA_S * FPS, axis=0,
                                mode="nearest")
    x0 = np.clip(centres[:, 0] - WINDOW_W / 2, 0, SRC_W - WINDOW_W)
    y0 = np.clip(centres[:, 1] - WINDOW_H / 2, 0, SRC_H - WINDOW_H)
    return np.round(np.stack([x0, y0], axis=1)).astype(int)


def check_survives(clips) -> None:
    """Refuse to show a clip the policy falls in -- this video is the success case."""
    stats = _load_stats(ROLLOUT)
    horizon_s = stats["clip_length"] / stats["fps"]
    for c in clips:
        pc = stats["per_clip"][c]
        if _fell(pc, horizon_s):
            raise ValueError(f"policy falls in clip {c} at {pc['time_alive_s']} s "
                             f"(margin {TRUNC_MARGIN_S} s)")


def make_overview(clips, output: Path) -> None:
    check_survives(clips)
    sources = [REFERENCE[0], TARGET, ROLLOUT]
    for p in sources:
        if not p.exists():
            raise FileNotFoundError(f"Missing input video: {p}")

    width, height = 3 * TILE_W, HEADER_H + TILE_H + FOOTER_H
    chrome = draw_chrome(width, height)
    caps = [cv2.VideoCapture(str(p)) for p in sources]
    print(f"Overview {width}x{height}, clips {clips}")

    with HQVideoWriter(output, width, height, FPS, pix_fmt="rgb24") as writer:
        canvas = chrome.copy()
        for seg, clip in enumerate(clips):
            origins = track_target(TARGET, clip)
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip * (CLIP_FRAMES + FADE_FRAMES))
            for t in range(CLIP_FRAMES):
                x0, y0 = origins[t]
                for i, cap in enumerate(caps):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"{sources[i]} ended early in clip {clip}")
                    crop = np.ascontiguousarray(
                        frame[y0:y0 + WINDOW_H, x0:x0 + WINDOW_W, ::-1])
                    canvas[HEADER_H:HEADER_H + TILE_H, i * TILE_W:(i + 1) * TILE_W] = (
                        cv2.resize(crop, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA))
                writer.write(canvas)
            # Fade only the video band between clips; the labels stay put.
            if seg < len(clips) - 1:
                band = canvas[HEADER_H:HEADER_H + TILE_H].copy()
                for t in range(FADE_FRAMES):
                    canvas[HEADER_H:HEADER_H + TILE_H] = (
                        band * fade_factor(t, FADE_FRAMES)).astype(np.uint8)
                    writer.write(canvas)
    for cap in caps:
        cap.release()
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", type=int, nargs="+", default=CLIPS)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--still", type=Path, default=None,
                   help="write only the header/footer chrome to this PNG and exit")
    args = p.parse_args()
    if args.still:
        cv2.imwrite(str(args.still), draw_chrome(3 * TILE_W, HEADER_H + TILE_H
                                                 + FOOTER_H)[..., ::-1])
        return
    make_overview(args.clips, args.output)


if __name__ == "__main__":
    main()
