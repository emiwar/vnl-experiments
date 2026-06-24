#!/usr/bin/env python3
"""Build a fixed evaluation dataset of 32 x 30 s clips from re-STAC'd rodent data.

This turns a long continuous STAC recording plus a list of manually annotated
"interesting" time periods into a fixed-length clip dataset that is plug-and-play
compatible with the rodent imitation env (the legacy flat-array
``reference_clips.h5`` format read by
``vnl_playground.tasks.reference_clips.ReferenceClips._load_legacy_format``).

Clip-generation rules (applied to each annotated ``MM:SS - MM:SS`` period):

* period < 60 s  -> one clip ``[start, start + CLIP_SECONDS)``;
  padding = ``max(0, CLIP_SECONDS - duration)`` (frames beyond the annotation).
* period >= 60 s -> ``floor(duration / CLIP_SECONDS)`` consecutive,
  non-overlapping clips starting at ``start`` (trailing remainder discarded);
  padding 0.

If more than ``N_CLIPS`` clips are generated, the ones that used the *least*
padding are kept (ties broken by source frame). If fewer than ``N_CLIPS`` are
generated the script errors out (so more data can be annotated) rather than
writing a short dataset.

Every clip is exactly ``CLIP_SECONDS * mocap_hz`` frames so the env's
``_load_legacy_format`` reshape (flat first axis -> ``(n_clips, n_frames, ...)``)
works. Provenance metadata (source video timestamps and frame ranges) is written
as extra datasets so each clip can be traced back to the raw video later.

Consuming the output:
    The env reshapes the flat arrays by ``clip_length``, so a consumer MUST set
        env_config.reference_data_path = "<output>.h5"
        env_config.clip_length        = CLIP_SECONDS * mocap_hz   # 1500
    Wiring this into eval_rodent_delays.py (iterating all clips) is a separate
    follow-up.

Run:
    cd vnl-experiments && ../.venv/bin/python -m vnl_experiments.tools.make_eval_clips
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np

# --- Defaults ---------------------------------------------------------------
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "art" / "2020_12_22_1"
DEFAULT_SOURCE = ASSET_DIR / "art_2020_12_22_1_new_STAC_from_Charles.h5"
DEFAULT_ANNOTATIONS = (
    Path(__file__).resolve().parents[2] / "manual_clip_extraction_timestamps.txt"
)
DEFAULT_OUTPUT = ASSET_DIR / "eval_clips_32x30s.h5"

MOCAP_HZ = 50
CLIP_SECONDS = 30
N_CLIPS = 32
CLIP_FRAMES = CLIP_SECONDS * MOCAP_HZ  # 1500

# Datasets copied verbatim (per-frame data is sliced; the rest copied whole).
_PER_FRAME_KEYS = ["qpos", "qvel", "xpos", "xquat", "kp_data", "marker_sites"]
_WHOLE_KEYS = ["names_qpos", "names_xpos", "kp_names", "offsets", "config"]

_TIME_RE = re.compile(r"^\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)\s*$")


def _to_seconds(minutes: str, seconds: str) -> int:
    return int(minutes) * 60 + int(seconds)


def parse_annotations(path: Path) -> list[tuple[int, int]]:
    """Parse a file of ``MM:SS - MM:SS`` lines into (start_sec, end_sec) tuples.

    Comment lines (starting with ``#``) and blank lines are skipped.
    """
    periods: list[tuple[int, int]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _TIME_RE.match(line)
        if m is None:
            raise ValueError(f"{path}:{lineno}: cannot parse period line: {raw!r}")
        start = _to_seconds(m.group(1), m.group(2))
        end = _to_seconds(m.group(3), m.group(4))
        if end <= start:
            raise ValueError(f"{path}:{lineno}: end <= start in {raw!r}")
        periods.append((start, end))
    return periods


class Clip:
    """A single generated clip with provenance back to its source period."""

    def __init__(self, src_start: int, src_end: int, clip_start: int):
        self.src_start = src_start          # annotated period start (s)
        self.src_end = src_end              # annotated period end (s)
        self.clip_start = clip_start        # this clip's start (s)
        self.clip_end = clip_start + CLIP_SECONDS
        # Padding (s) = portion of the clip that falls past the annotated
        # period. Clips always start at src_start or later, so only the tail
        # [src_end, clip_end] can be out-of-period.
        self.padding = max(0, self.clip_end - max(src_end, clip_start))
        self.start_frame = round(clip_start * MOCAP_HZ)
        self.end_frame = self.start_frame + CLIP_FRAMES

    @property
    def src_label(self) -> str:
        def fmt(s: int) -> str:
            return f"{s // 60:02d}:{s % 60:02d}"
        return f"{fmt(self.src_start)}-{fmt(self.src_end)}"


def generate_clips(periods: list[tuple[int, int]]) -> list[Clip]:
    """Expand annotated periods into fixed-length clips per the rules."""
    clips: list[Clip] = []
    for src_start, src_end in periods:
        duration = src_end - src_start
        if duration < 60:
            clips.append(Clip(src_start, src_end, src_start))
        else:
            n = duration // CLIP_SECONDS
            for i in range(n):
                clips.append(Clip(src_start, src_end, src_start + i * CLIP_SECONDS))
    return clips


def select_clips(clips: list[Clip], n_total_frames: int) -> list[Clip]:
    """Bounds-check, cap to N_CLIPS by least padding, return chronological order."""
    for c in clips:
        if c.end_frame > n_total_frames:
            raise ValueError(
                f"Clip starting {c.src_label} needs frame {c.end_frame} but the "
                f"source has only {n_total_frames} frames."
            )

    if len(clips) < N_CLIPS:
        raise SystemExit(
            f"Only {len(clips)} clips generated from the annotations, need "
            f"{N_CLIPS}. Annotate more interesting periods and re-run."
        )

    # Keep the N_CLIPS least-padded clips (tie-break by source frame for
    # determinism), then restore chronological order for the output.
    by_padding = sorted(clips, key=lambda c: (c.padding, c.start_frame))
    kept = by_padding[:N_CLIPS]
    dropped = by_padding[N_CLIPS:]
    kept.sort(key=lambda c: c.start_frame)

    # Overlap check (within the kept, chronological set).
    for a, b in zip(kept, kept[1:]):
        if b.start_frame < a.end_frame:
            raise ValueError(
                f"Clips overlap: {a.src_label} (frames {a.start_frame}-"
                f"{a.end_frame}) and {b.src_label} (frames {b.start_frame}-"
                f"{b.end_frame})."
            )

    max_pad = max((c.padding for c in kept), default=0)
    print(
        f"Generated {len(clips)} clips; kept {len(kept)}, dropped {len(dropped)}; "
        f"max retained padding {max_pad} s."
    )
    if dropped:
        dropped_desc = ", ".join(
            f"{c.src_label}@{c.clip_start}s(pad {c.padding}s)" for c in dropped
        )
        print(f"Dropped (most padding first): {dropped_desc}")
    return kept


def build_dataset(source: Path, clips: list[Clip], output: Path) -> None:
    """Slice + stack the source data for the kept clips and write the output h5."""
    with h5py.File(source, "r") as src:
        present = [k for k in _PER_FRAME_KEYS if k in src]
        sliced: dict[str, np.ndarray] = {k: [] for k in present}
        for c in clips:
            for k in present:
                sliced[k].append(src[k][c.start_frame : c.end_frame])
        stacked = {k: np.concatenate(v, axis=0) for k, v in sliced.items()}

        with h5py.File(output, "w") as dst:
            for k in present:
                dst.create_dataset(k, data=stacked[k])
            for k in _WHOLE_KEYS:
                if k in src:
                    src.copy(k, dst)

            # Provenance: one entry per clip in chronological (output) order.
            dst.create_dataset(
                "clip_src_start_sec", data=np.array([c.src_start for c in clips], np.int32)
            )
            dst.create_dataset(
                "clip_src_end_sec", data=np.array([c.src_end for c in clips], np.int32)
            )
            dst.create_dataset(
                "clip_clip_start_sec",
                data=np.array([c.clip_start for c in clips], np.int32),
            )
            dst.create_dataset(
                "clip_padding_sec", data=np.array([c.padding for c in clips], np.int32)
            )
            dst.create_dataset(
                "clip_start_frame",
                data=np.array([c.start_frame for c in clips], np.int64),
            )
            dst.create_dataset(
                "clip_end_frame", data=np.array([c.end_frame for c in clips], np.int64)
            )
            dst.create_dataset(
                "clip_src_label",
                data=np.array([c.src_label.encode("utf-8") for c in clips]),
            )

            dst.attrs["source_file"] = source.name
            dst.attrs["mocap_hz"] = MOCAP_HZ
            dst.attrs["n_frames_per_clip"] = CLIP_FRAMES
            dst.attrs["n_clips"] = len(clips)
            dst.attrs["clip_seconds"] = CLIP_SECONDS
            dst.attrs["description"] = (
                "Eval clips built by make_eval_clips.py. Periods <60s -> one "
                "30s clip from start (padded past the annotation); periods >=60s "
                "-> floor(dur/30) consecutive 30s clips. Kept the 32 least-padded. "
                "Consume with clip_length = n_frames_per_clip."
            )

    total = len(clips) * CLIP_FRAMES
    print(f"Wrote {output} ({len(clips)} clips x {CLIP_FRAMES} frames = {total}).")


def print_manifest(clips: list[Clip]) -> None:
    print(f"\n{'idx':>3} {'source':>13} {'clip_start':>10} {'frames':>17} {'pad_s':>5}")
    for i, c in enumerate(clips):
        print(
            f"{i:3d} {c.src_label:>13} {c.clip_start:8d} s "
            f"{c.start_frame:7d}-{c.end_frame:<7d} {c.padding:5d}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    with h5py.File(args.source, "r") as f:
        n_total_frames = f["qpos"].shape[0]

    periods = parse_annotations(args.annotations)
    print(f"Parsed {len(periods)} annotated periods from {args.annotations.name}.")
    clips = generate_clips(periods)
    kept = select_clips(clips, n_total_frames)
    print_manifest(kept)
    build_dataset(args.source, kept, args.output)


if __name__ == "__main__":
    main()
