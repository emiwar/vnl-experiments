"""2x2 video collage: what a given reward looks like on 0.3 m blocks, at delays 0 / 5 / 7 / 10.

    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/make_video.py

An illustration for the report, not a measurement: it gives an intuition for what the
raw endpoint rewards in the figures mean as behaviour. Each tile is one run's **last
eval video logged to WandB** (`media/videos/eval_video_<step>_*.mp4`: one 1000-step eval
episode, 640x480), so nothing is re-rendered and no checkpoint is needed. The videos are
downloaded to ``eval_videos/dm_control/`` (gitignored), named as ``make_collage._dmc``
expects.

Run choice, fixed before watching: blind random blocks (`terrain_blind`) at 0.3 m, on the
seed pair the sweeps were launched at (:data:`SWEEP_SEEDS`). Delay 10 at 0.3 m only exists
at i43/p43, so that tile uses it. Each tile's label carries **the run's final eval
reward**: the run's last ``eval/episode_reward/mean`` point, read from its ``history``
artifact (the 480 M runs' final eval is at 480.2 M, just after the last video at 479.97 M;
``curves.csv`` stops at 480 M, so it is not there).
That is a mean over eval episodes, while the video is a single different episode, so a
tile can look better or worse than its number. The numbers are written to
``video_audit.txt`` beside the report's own 430-480 M endpoint for the same run.

Tiling follows ``vnl_experiments/video_editing/make_collage.py`` (same output folder,
same ``_dmc`` naming, same dark-backed top-left label), but frames are read with
imageio and labels drawn with PIL, because that module needs OpenCV, which the current
``.venv`` does not have. Writing goes through the shared ``HQVideoWriter``.

Writes ``eval_videos/collage_2x2_walker_terrain_h0.3.mp4`` (gitignored),
``video_audit.txt`` and ``figures/video_contact_sheet.png`` (one frame per tile at
:data:`STILL_S`), so the report can stand without the mp4.
"""

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from vnl_experiments.artifacts import Store
from vnl_experiments.video_editing.hq_video import HQVideoWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import _series  # noqa: E402  (this folder's; same history spec)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
#: make_collage.EVAL_DIR / DMC_DIR, restated because that module needs cv2.
EVAL_DIR = REPO_ROOT / "eval_videos"
DMC_DIR = EVAL_DIR / "dm_control"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
PROJECT = "emiwar-team/nnx-ppo-delays"
TASK = "WalkerWalkTerrain"
CONDITION = "terrain_blind"
HEIGHT = 0.3
DELAYS = (0, 5, 7, 10)
SWEEP_SEEDS = "i42/p1234"
CTRL_MS = 25
READ_AT = 480_000_000
#: One control step per frame, so playing at 1 / ctrl_dt is real time.
FPS = round(1000 / CTRL_MS)
TILE = (640, 480)
STILL_S = 12.0

OUTPUT = EVAL_DIR / "collage_2x2_walker_terrain_h0.3.mp4"
AUDIT = HERE / "video_audit.txt"
CONTACT_SHEET = HERE / "figures" / "video_contact_sheet.png"


def select(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in DELAYS:
        cell = data[(data.condition == CONDITION) & (data.height == HEIGHT)
                    & (data.delay_k == d) & (data.budget == READ_AT)]
        pick = cell[cell.seed_pair == SWEEP_SEEDS]
        if pick.empty:
            pick = cell.sort_values("seed_pair")
        if pick.empty:
            raise SystemExit(f"no {CONDITION} run at {HEIGHT} m, delay {d}")
        rows.append(pick.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def last_video(wandb_id: str) -> tuple[int, Path]:
    """Download (once) the run's last eval video at or before READ_AT."""
    import wandb

    run = wandb.Api().run(f"{PROJECT}/{wandb_id}")
    videos = {int(f.name.split("_")[2]): f for f in run.files()
              if f.name.startswith("media/videos/eval_video_")}
    step = max(s for s in videos if s <= READ_AT)
    dest = DMC_DIR / f"{TASK}_{wandb_id}_step{step}.mp4"  # make_collage._dmc naming
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = Path(videos[step].download(root=str(dest.parent / ".dl"), replace=True).name)
        got.rename(dest)
    return step, dest


def final_eval(store: Store, wandb_id: str) -> tuple[int, float]:
    """The run's last eval point, from its history artifact."""
    s = _series(store, wandb_id)
    last = s.iloc[-1]
    return int(last.step), float(last.value)


def draw_label(tile: np.ndarray, text: str) -> np.ndarray:
    """Top-left label on a dark translucent box, as make_collage.draw_label."""
    img = Image.fromarray(tile)
    d = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype(str(FONT), 22)
    x0, y0, x1, y1 = d.textbbox((16, 12), text, font=font)
    d.rectangle((x0 - 8, y0 - 8, x1 + 8, y1 + 8), fill=(0, 0, 0, 165))
    d.text((16, 12), text, font=font, fill=(255, 255, 255))
    return np.asarray(img)


def compose(frames, labels) -> np.ndarray:
    tiles = [draw_label(f, lab) for f, lab in zip(frames, labels)]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def contact_sheet(videos, labels, path: Path) -> None:
    frames = [_frame_at(v, int(STILL_S * FPS)) for v in videos]
    path.parent.mkdir(exist_ok=True)
    iio.imwrite(path, compose(frames, labels))


def _frame_at(video: Path, index: int) -> np.ndarray:
    for i, f in enumerate(iio.imiter(video, plugin="FFMPEG")):
        if i == index:
            return f
    raise SystemExit(f"{video} has fewer than {index + 1} frames")


def _n_frames(video: Path) -> int:
    return sum(1 for _ in iio.imiter(video, plugin="FFMPEG"))


def make_collage(videos, labels, output: Path) -> int:
    readers = [iio.imiter(v, plugin="FFMPEG") for v in videos]
    w, h = TILE
    writer = HQVideoWriter(output, 2 * w, 2 * h, FPS, pix_fmt="rgb24")
    for frames in zip(*readers):  # stops at the shortest input
        if any(f.shape[:2] != (h, w) for f in frames):
            raise SystemExit(f"expected {w}x{h} frames, got {[f.shape for f in frames]}")
        writer.write(compose(frames, labels))
    n = writer.close()
    print(f"Wrote {output} ({n} frames @ {FPS} fps)")
    return n


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter
                            ).parse_args()
    data = pd.read_csv(HERE / "data.csv")
    store = Store()
    picks = select(data)

    videos, labels, audit = [], [], [
        f"# {OUTPUT.name}: blind random blocks, {HEIGHT} m; one WandB eval video per tile.",
        "# label reward = the run's last eval point (history artifact); endpoint = 430-480 M mean "
        "(data.csv, the report's number).",
        "delay_k,delay_ms,wandb_id,seed_pair,video_step,eval_step,final_eval_reward,"
        "endpoint_reward",
    ]
    for r in picks.itertuples():
        vstep, path = last_video(r.wandb_id)
        estep, reward = final_eval(store, r.wandb_id)
        videos.append(path)
        labels.append(f"Delay {r.delay_k} ({r.delay_k * CTRL_MS} ms)  |  "
                      f"eval reward {reward:.0f}")
        audit.append(f"{r.delay_k},{r.delay_k * CTRL_MS},{r.wandb_id},{r.seed_pair},"
                     f"{vstep},{estep},{reward:.1f},{r.reward:.1f}")

    frames = sorted({_n_frames(v) for v in videos})
    audit.append(f"# frames per input video: {frames}; played at {FPS} fps "
                 f"(1 / ctrl_dt, real time)")
    AUDIT.write_text("\n".join(audit) + "\n")
    print("\n".join(audit))

    contact_sheet(videos, labels, CONTACT_SHEET)
    make_collage(videos, labels, OUTPUT)


if __name__ == "__main__":
    main()
