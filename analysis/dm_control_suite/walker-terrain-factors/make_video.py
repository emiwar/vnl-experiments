"""2x2 video collage of four factor-grid runs at 0.3 m / delay 5 (``data.csv``).

    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-factors/make_video.py

An illustration, not a measurement. Each tile is the run's **last eval video logged to
WandB** at or before 480 M (``media/videos/eval_video_<step>_*.mp4``: one 1000-step eval
episode, 640x480), so nothing is re-rendered. Videos are downloaded to
``eval_videos/dm_control/`` (gitignored), named as ``make_collage._dmc`` expects.

Each label lists only the factors that are *on* for that run, with ``blind`` turned
around into "block sensor" so every item is an addition: efference copy, forward model,
servo, block sensor. The privileged critic is on for all four runs and is not listed (the
script stops if a run has it off); a run with none of them is labelled "baseline".

The second label line is the run's ``reward_480M`` from ``data.csv`` (mean eval reward
over 430-480 M), not a single eval point: ``sog4k5xi`` was trained to 2 G, so its final
eval would not be comparable. The video is one eval episode, so a tile can look better or
worse than its number.

Same tiling as ``walker-terrain-first-look/make_video.py`` and
``vnl_experiments/video_editing/make_collage.py`` (dark-backed top-left label), with
frames read by imageio and labels drawn by PIL because ``make_collage`` needs OpenCV,
which the current ``.venv`` does not have. Writing goes through ``HQVideoWriter``.

Writes ``eval_videos/collage_2x2_walker_terrain_factors_h0.3_d5.mp4`` (gitignored),
``video_audit.txt`` and ``figures/video_contact_sheet.png``.
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from vnl_experiments.video_editing.hq_video import HQVideoWriter

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
#: make_collage.EVAL_DIR / DMC_DIR, restated because that module needs cv2.
EVAL_DIR = REPO_ROOT / "eval_videos"
DMC_DIR = EVAL_DIR / "dm_control"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
PROJECT = "emiwar-team/nnx-ppo-delays"
TASK = "WalkerWalkTerrain"
#: Tile order: top-left, top-right, bottom-left, bottom-right.
RUNS = ("sog4k5xi", "u8xdfjid", "d7ytv3fx", "0xmqmajf")
DELAY = 5
CTRL_MS = 25
READ_AT = 480_000_000
#: One control step per frame, so playing at 1 / ctrl_dt is real time.
FPS = round(1000 / CTRL_MS)
TILE = (640, 480)
STILL_S = 12.0

OUTPUT = EVAL_DIR / "collage_2x2_walker_terrain_factors_h0.3_d5.mp4"
AUDIT = HERE / "video_audit.txt"
CONTACT_SHEET = HERE / "figures" / "video_contact_sheet.png"


def attributes(r: pd.Series) -> list[str]:
    if not r.privileged_critic:
        raise SystemExit(f"{r.wandb_id}: privileged critic off; the labels assume it is on")
    on = [("efference copy", r.efference), ("forward model", r.forward_model),
          ("servo", r.servo), ("block sensor", not r.blind)]
    return [name for name, flag in on if flag]


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


def _frame_at(video: Path, index: int) -> np.ndarray:
    for i, f in enumerate(iio.imiter(video, plugin="FFMPEG")):
        if i == index:
            return f
    raise SystemExit(f"{video} has fewer than {index + 1} frames")


def _n_frames(video: Path) -> int:
    return sum(1 for _ in iio.imiter(video, plugin="FFMPEG"))


def contact_sheet(videos, labels, path: Path) -> None:
    frames = [_frame_at(v, int(STILL_S * FPS)) for v in videos]
    path.parent.mkdir(exist_ok=True)
    iio.imwrite(path, compose(frames, labels))


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
    data = pd.read_csv(HERE / "data.csv").set_index("wandb_id", drop=False)
    missing = [w for w in RUNS if w not in data.index]
    if missing:
        raise SystemExit(f"not in data.csv: {missing}")

    videos, labels = [], []
    audit = [f"# {OUTPUT.name}: random blocks 0.3 m, delay {DELAY} "
             f"({DELAY * CTRL_MS} ms); one WandB eval video per tile.",
             "# reward = reward_480M from data.csv (430-480 M mean).",
             "tile,wandb_id,seed_pair,attributes,video_step,reward_480M"]
    for i, w in enumerate(RUNS):
        r = data.loc[w]
        attrs = attributes(r)
        vstep, path = last_video(w)
        videos.append(path)
        labels.append(f"{' + '.join(attrs) or 'baseline'}\n"
                      f"eval reward {r.reward_480M:.0f}")
        audit.append(f"{i},{w},{r.seed_pair},{'+'.join(attrs) or 'baseline'},"
                     f"{vstep},{r.reward_480M:.1f}")

    frames = sorted({_n_frames(v) for v in videos})
    audit.append(f"# frames per input video: {frames}; played at {FPS} fps "
                 f"(1 / ctrl_dt, real time)")
    AUDIT.write_text("\n".join(audit) + "\n")
    print("\n".join(audit))

    contact_sheet(videos, labels, CONTACT_SHEET)
    make_collage(videos, labels, OUTPUT)


if __name__ == "__main__":
    main()
