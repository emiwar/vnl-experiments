#!/usr/bin/env python3
"""Extract the raw camera video matching the first N eval clips, frame-for-frame.

The rendered rollout videos (``eval_videos/{ckpt}/rollout.mp4``, produced by
``vnl_experiments.delays.eval_videos``) play the first ``N_CLIPS`` eval clips
back to back with a short logistic fade-to-black between consecutive clips. This
script builds the *raw camera-4 footage* counterpart so the two can be laid
side by side and stay aligned frame-for-frame.

Why this lines up 1:1:
  * The STAC source (``art_2020_12_22_1_new_STAC_from_Charles.h5``) has exactly one
    pose per camera frame — 360000 frames == the 360000-frame, 50 fps Camera-4
    video — so mocap/reference frame index == video frame index (no offset).
  * Each rendered clip is 1500 reference frames rendered 1:1 (3000 env steps at
    frame_skip 2, 0.5 ref-frames/step). Clip ``c`` occupies source/video frames
    ``[clip_start_frame, clip_end_frame)`` from the eval file's provenance.
  * Between clips the render appends ``FADE_FRAMES`` faded copies of the clip's
    last frame (logistic fade, ``imitation.py:587-593``); we apply the identical
    fade to the raw clip's last frame so both videos darken in lockstep.

Result: ``N_CLIPS * 1500 + (N_CLIPS - 1) * FADE_FRAMES`` frames at ``FPS``,
identical in count/fps to the rendered mp4 (verified at the end).

Run:
    cd vnl-experiments && ../.venv/bin/python -m vnl_experiments.video_editing.extract_raw_eval_video
"""

import argparse
import glob
import math
from pathlib import Path

import cv2
import h5py
import numpy as np

from vnl_experiments.video_editing.hq_video import HQVideoWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "assets" / "art" / "2020_12_22_1"

DEFAULT_EVAL_H5 = ASSET_DIR / "eval_clips_32x30s.h5"
DEFAULT_VIDEO = ASSET_DIR / "videos" / "Camera4" / "0.mp4"
DEFAULT_OUTPUT = REPO_ROOT / "eval_videos" / "raw_camera4_clips0-3.mp4"

# These MUST match vnl_experiments/delays/eval_videos.py (verified against a real
# rendered rollout.mp4 at the end of this script, which fails loudly on drift).
N_CLIPS = 4
FADE_FRAMES = 25
FPS = 50


def fade_factor(t: int, n: int) -> float:
    """Logistic fade-out weight (1->0), identical to Imitation.render."""
    return 1.0 / (1.0 + math.exp(10.0 * (t / n - 0.5)))


def read_clip_ranges(eval_h5: Path, n_clips: int):
    with h5py.File(eval_h5, "r") as f:
        starts = f["clip_start_frame"][:n_clips].astype(int)
        ends = f["clip_end_frame"][:n_clips].astype(int)
        labels = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in f["clip_src_label"][:n_clips]]
    return list(zip(starts.tolist(), ends.tolist(), labels))


def extract(video: Path, ranges, output: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source {video.name}: fps={src_fps}  frames={src_n}  {w}x{h}")
    if abs(src_fps - FPS) > 1e-6:
        print(f"  WARNING: source fps {src_fps} != expected {FPS}")

    # cv2 decodes BGR; feed ffmpeg bgr24 and let libx264 encode at high quality.
    writer = HQVideoWriter(output, w, h, FPS, pix_fmt="bgr24")
    n_written = 0
    for ci, (start, end, label) in enumerate(ranges):
        if end > src_n:
            raise ValueError(f"Clip {ci} ({label}) needs frame {end} > {src_n}")
        # Seek to the clip start, then read sequentially (frame-accurate).
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        last = None
        for _ in range(end - start):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Read failed in clip {ci} ({label})")
            writer.write(frame)
            last = frame
            n_written += 1
        print(f"  clip {ci} [{label}]: frames {start}-{end} ({end - start})")
        # Match the render: fade after every clip except the last.
        if ci < len(ranges) - 1:
            for t in range(FADE_FRAMES):
                writer.write((last * fade_factor(t, FADE_FRAMES)).astype(np.uint8))
                n_written += 1

    writer.close()
    cap.release()
    return n_written, w, h


def verify(output: Path, expected_frames: int) -> None:
    cap = cv2.VideoCapture(str(output))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    print(f"Output {output.name}: fps={fps}  frames={n}  {w}x{h}")
    assert abs(fps - FPS) < 1e-6, f"fps {fps} != {FPS}"
    assert n == expected_frames, f"frame count {n} != expected {expected_frames}"

    # Cross-check against the current-format rendered clips. (A stale
    # single-clip render from the old script may linger in eval_videos/ with a
    # different frame count; report it but don't fail on it.)
    rendered = sorted(glob.glob(str(REPO_ROOT / "eval_videos" / "*" / "rollout.mp4")))
    matches, outliers = [], []
    for r in rendered:
        rc = cv2.VideoCapture(r)
        rfps, rn = rc.get(cv2.CAP_PROP_FPS), int(rc.get(cv2.CAP_PROP_FRAME_COUNT))
        rc.release()
        rel = Path(r).relative_to(REPO_ROOT)
        (matches if (abs(rfps - fps) < 1e-6 and rn == n) else outliers).append(
            (rel, rfps, rn))
    for rel, rfps, rn in outliers:
        print(f"  (ignoring non-matching render {rel}: fps={rfps} frames={rn})")
    assert matches, (
        f"No rendered rollout.mp4 matched the raw video (fps={fps}, frames={n}); "
        f"found: {[(str(r), f, c) for r, f, c in outliers]}")
    print(f"  MATCH: {len(matches)}/{len(rendered)} rendered rollouts align "
          f"frame-for-frame ({n} frames @ {fps} fps).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-h5", type=Path, default=DEFAULT_EVAL_H5)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-clips", type=int, default=N_CLIPS)
    args = p.parse_args()

    ranges = read_clip_ranges(args.eval_h5, args.n_clips)
    expected = sum(e - s for s, e, _ in ranges) + (len(ranges) - 1) * FADE_FRAMES
    print(f"Extracting {len(ranges)} clips -> {expected} frames "
          f"({sum(e - s for s, e, _ in ranges)} clip + "
          f"{(len(ranges) - 1) * FADE_FRAMES} fade)")
    n_written, _, _ = extract(args.video, ranges, args.output)
    print(f"Wrote {n_written} frames to {args.output}")
    verify(args.output, expected)


if __name__ == "__main__":
    main()
