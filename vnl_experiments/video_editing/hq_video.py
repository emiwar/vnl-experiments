"""High-quality H.264 video writer that streams frames to ffmpeg.

OpenCV's ``cv2.VideoWriter`` on this machine only opens the low-quality ``mp4v``
(MPEG-4 Part 2) encoder — its H.264 path targets a missing hardware encoder and
fails. Standalone ffmpeg with ``libx264`` is available, so we pipe raw frames to
it and encode at a low CRF (near-lossless) instead.

Used by the raw-clip extractor (``video_editing/extract_raw_eval_video.py``).

Example:
    with HQVideoWriter(path, width, height, fps, pix_fmt="rgb24") as w:
        for frame in frames:            # HxWx3 uint8
            w.write(frame)
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np

# CRF 0 is lossless, ~18 is visually lossless, 23 is the x264 default. 16 keeps
# fine detail (fur, contact shadows) without an enormous file.
DEFAULT_CRF = 16
DEFAULT_PRESET = "slow"


class HQVideoWriter:
    """Stream ``uint8`` frames to an ffmpeg ``libx264`` process.

    Args:
        path: output .mp4 path.
        width, height: frame size in pixels.
        fps: frames per second.
        pix_fmt: pixel order of the frames passed to ``write`` — ``"rgb24"`` for
            MuJoCo renders, ``"bgr24"`` for frames read via OpenCV.
        crf, preset: libx264 quality knobs (lower CRF = higher quality).
    """

    def __init__(self, path, width: int, height: int, fps: float,
                 pix_fmt: str = "rgb24", crf: int = DEFAULT_CRF,
                 preset: str = DEFAULT_PRESET):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; cannot write HQ video.")
        self.path = Path(path)
        self.width, self.height = width, height
        self._n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", pix_fmt,
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            # yuv420p keeps the file broadly playable (QuickTime, browsers).
            "-pix_fmt", "yuv420p",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"frame {frame.shape[:2]} != ({self.height}, {self.width})")
        self._proc.stdin.write(
            np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self._n += 1

    def close(self) -> int:
        self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc} writing {self.path}")
        return self._n

    def __enter__(self) -> "HQVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        # On a clean exit finalize the file; on error tear the pipe down.
        if exc[0] is None:
            self.close()
        else:
            self._proc.stdin.close()
            self._proc.wait()
