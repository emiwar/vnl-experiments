"""2x2 video collage of four factor-grid runs at 0.3 m / delay 5 (``factors.csv``).

    ../.venv/bin/python analysis/dm_control_suite/walker-terrain-first-look/make_video_factors.py

The same machinery as ``make_video.py`` (last WandB eval video at or before 480 M per
run, PIL labels, ``HQVideoWriter``), for a hand-picked set of runs from
``factors.csv``. Each label lists only the factors that are *on* for that run, with
``blind`` turned around into "block sensor" so every item is an addition:

* efference copy (``efference``), forward model (``forward_model``), servo (``servo``),
  block sensor (``not blind``). The privileged critic is on for all four, so it is not
  listed; a run with none of these is labelled "baseline".

The second label line is the run's ``reward_480M`` from ``factors.csv`` (mean eval
reward over 430-480 M, the report's readout), not a single eval point: one of the runs
(``sog4k5xi``) was trained to 2 G, so its final eval would not be comparable. The video
is one eval episode, so a tile can look better or worse than its number.

Writes ``eval_videos/collage_2x2_walker_terrain_factors_h0.3_d5.mp4`` (gitignored),
``video_factors_audit.txt`` and ``figures/video_factors_contact_sheet.png``.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_video import (  # noqa: E402  (this folder's)
    CTRL_MS, EVAL_DIR, FPS, _n_frames, contact_sheet, last_video, make_collage)

HERE = Path(__file__).resolve().parent
#: Tile order: top-left, top-right, bottom-left, bottom-right.
RUNS = ("sog4k5xi", "u8xdfjid", "d7ytv3fx", "0xmqmajf")
DELAY = 5

OUTPUT = EVAL_DIR / "collage_2x2_walker_terrain_factors_h0.3_d5.mp4"
AUDIT = HERE / "video_factors_audit.txt"
CONTACT_SHEET = HERE / "figures" / "video_factors_contact_sheet.png"


def attributes(r: pd.Series) -> list[str]:
    on = [("efference copy", r.efference), ("forward model", r.forward_model),
          ("servo", r.servo), ("block sensor", not r.blind)]
    if not r.privileged_critic:
        raise SystemExit(f"{r.wandb_id}: privileged critic off; the labels assume it is on")
    return [name for name, flag in on if flag]


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter
                            ).parse_args()
    factors = pd.read_csv(HERE / "factors.csv").set_index("wandb_id", drop=False)
    missing = [w for w in RUNS if w not in factors.index]
    if missing:
        raise SystemExit(f"not in factors.csv: {missing}")

    videos, labels = [], []
    audit = [f"# {OUTPUT.name}: blind/observed random blocks 0.3 m, delay {DELAY} "
             f"({DELAY * CTRL_MS} ms); one WandB eval video per tile.",
             "# reward = reward_480M from factors.csv (430-480 M mean).",
             "tile,wandb_id,seed_pair,attributes,video_step,reward_480M"]
    for i, w in enumerate(RUNS):
        r = factors.loc[w]
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
