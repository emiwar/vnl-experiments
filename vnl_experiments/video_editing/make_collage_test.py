"""Tests for the collage's planning and overlay logic.

The tiling itself needs real videos and ffmpeg, so what is tested here is everything
that decides *what* gets drawn: the clip plan, the caption lookup, and the death mark.
The death mark is the one worth a test -- it is the only thing in a free-run collage that
distinguishes a policy that is tracking from one that fell over ten seconds ago, and it is
driven by a sidecar field whose meaning (``terminated`` includes truncation at the clip
end) is easy to get backwards.
"""

import json
import tempfile
from pathlib import Path

from absl.testing import absltest
import numpy as np

from vnl_experiments.video_editing.make_collage import (
    CLIP_FRAMES,
    DEAD_COLOUR,
    FPS,
    LIVE_COLOUR,
    MIN_LABEL_SCALE,
    Overlay,
    build_plan,
    draw_label,
    read_captions,
)

HORIZON_S = CLIP_FRAMES / FPS


def _stats(tmp: Path, name: str, lifetimes) -> tuple[Path, str]:
    """A cell whose ``.stats.json`` says it lived ``lifetimes[c]`` seconds on clip c."""
    path = tmp / f"{name}.mp4"
    path.with_suffix(".stats.json").write_text(json.dumps({
        "n_eval_clips": len(lifetimes),
        "clip_length": CLIP_FRAMES,
        "per_clip": [
            {"clip_idx": c, "time_alive_s": t, "total_reward": 100.0 * (c + 1),
             "terminated": 1}
            for c, t in enumerate(lifetimes)],
    }))
    return path, name


class BuildPlanTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # Clip 0: everyone dies early. Clip 1: one tile reaches the horizon, so the
        # "terminated" flag there is a truncation, not a fall.
        self.grid = [[_stats(tmp, "a", [2.0, HORIZON_S]),
                      _stats(tmp, "b", [5.0, 4.0])]]
        self.stats = [json.loads(c[0].with_suffix(".stats.json").read_text())
                      for row in self.grid for c in row]

    def test_none_keeps_the_rendered_order(self):
        plan = build_plan(self.grid, self.stats, 3.0, "none", FPS)
        self.assertEqual([c for c, _ in plan], [0, 1])

    def test_lifetime_sorts_easiest_first(self):
        plan = build_plan(self.grid, self.stats, 3.0, "lifetime", FPS)
        self.assertEqual([c for c, _ in plan], [1, 0])  # clip 1 mean 17 s vs clip 0's 3.5

    def test_trim_cuts_a_clip_everyone_fell_out_of(self):
        (_, frames), = [p for p in build_plan(self.grid, self.stats, 3.0, "none", FPS)
                        if p[0] == 0]
        self.assertEqual(frames, round((5.0 + 3.0) * FPS))  # last death + tail

    def test_a_clip_someone_survived_plays_in_full(self):
        plan = dict(build_plan(self.grid, self.stats, 3.0, "none", FPS))
        self.assertEqual(plan[1], CLIP_FRAMES)


class CaptionTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = tmp / "captions.csv"
        self.path.write_text(
            "# a comment the reader must skip\n"
            "clip,start_frame,end_frame,coarse,text\n"
            "0,0,100,rear,Rearing\n"
            "0,200,300,still,Still / prone\n"   # deliberately sparse: 100..200 has none
            "1,0,50,locomote,Locomotion\n")
        self.segments = read_captions(self.path)

    def test_parses_comments_and_groups_by_clip(self):
        self.assertEqual(sorted(self.segments), [0, 1])
        self.assertEqual(self.segments[0][0], (0, 100, "Rearing"))

    def test_only_the_designated_cell_is_captioned(self):
        overlay = Overlay([[None]], captions=self.segments, caption_cell=(0, 0))
        self.assertEqual(overlay.caption(0, 0, 0, 10), "Rearing")
        self.assertIsNone(overlay.caption(0, 1, 0, 10))

    def test_a_gap_and_a_boundary_give_no_caption(self):
        overlay = Overlay([[None]], captions=self.segments, caption_cell=(0, 0))
        self.assertIsNone(overlay.caption(0, 0, 0, 150))  # the sparse gap
        self.assertIsNone(overlay.caption(0, 0, 0, 100))  # ranges are half-open
        self.assertIsNone(overlay.caption(0, 0, 9, 0))    # a clip with no rows


class DeathMarkTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # Cell (0,0) is the reference: no sidecar, so it can never be marked. Cell (0,1)
        # falls on clip 0 and is truncated at the horizon on clip 1.
        self.grid = [[(tmp / "reference.mp4", "Reference"),
                      _stats(tmp, "policy", [4.0, HORIZON_S])]]
        self.overlay = Overlay(self.grid, mark_deaths=True)

    def test_live_before_the_death_frame(self):
        text, colour = self.overlay.label(0, 1, "Policy", 0, round(3.9 * FPS))
        self.assertEqual((text, colour), ("Policy", LIVE_COLOUR))

    def test_marked_from_the_death_frame_on(self):
        text, colour = self.overlay.label(0, 1, "Policy", 0, round(4.0 * FPS))
        self.assertEqual(colour, DEAD_COLOUR)
        self.assertIn("4.0 s", text)

    def test_a_truncation_at_the_clip_end_is_not_a_death(self):
        # `terminated` is 1 here too -- the env ORs truncation into `done` -- so reading
        # the flag alone would mark a tile that tracked the whole clip.
        text, colour = self.overlay.label(0, 1, "Policy", 1, CLIP_FRAMES - 1)
        self.assertEqual((text, colour), ("Policy", LIVE_COLOUR))

    def test_a_cell_without_a_sidecar_is_never_marked(self):
        self.assertNotIn((0, 0), self.overlay.deaths)
        text, colour = self.overlay.label(0, 0, "Reference", 0, CLIP_FRAMES - 1)
        self.assertEqual((text, colour), ("Reference", LIVE_COLOUR))

    def test_no_overlay_state_means_no_mark(self):
        overlay = Overlay(self.grid, mark_deaths=False)
        self.assertEqual(overlay.deaths, {})
        self.assertEqual(overlay.label(0, 1, "Policy", 0, CLIP_FRAMES - 1),
                         ("Policy", LIVE_COLOUR))


class DrawLabelTest(absltest.TestCase):

    def test_a_long_label_is_shrunk_rather_than_clipped(self):
        # A death mark is appended at run time, so the text length is not known when the
        # preset is written; the right-hand columns must still carry pixels.
        tile = np.zeros((600, 480, 3), np.uint8)
        draw_label(tile, "Torque, proprioception 100 ms late  -  fell at 10.4 s")
        written = np.nonzero(tile.any(axis=2).any(axis=0))[0]
        self.assertLess(written.max(), tile.shape[1])
        self.assertGreater(written.max(), tile.shape[1] * 0.5)

    def test_it_gives_up_shrinking_at_the_floor(self):
        tile = np.zeros((600, 60, 3), np.uint8)
        draw_label(tile, "far too long to ever fit in sixty pixels")  # must not hang
        self.assertGreaterEqual(MIN_LABEL_SCALE, 0.0)

    def test_top_and_bottom_corners_do_not_collide(self):
        tile = np.zeros((600, 960, 3), np.uint8)
        draw_label(tile, "Name", corner="top")
        top = np.nonzero(tile.any(axis=2).any(axis=1))[0]
        tile[:] = 0
        draw_label(tile, "Caption", corner="bottom")
        bottom = np.nonzero(tile.any(axis=2).any(axis=1))[0]
        self.assertLess(top.max(), bottom.min())


if __name__ == "__main__":
    absltest.main()
