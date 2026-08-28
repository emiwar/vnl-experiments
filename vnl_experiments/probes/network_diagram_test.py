"""Guards the architecture strip's layout, which is the kind of thing that rots silently.

A diagram has no numbers to check, so what is worth testing is the wiring and the
invariants a reader relies on: every arrow connects boxes that exist, every box sits
inside the strip, and -- the whole point of the module -- a box is at the x coordinate of
the layer it stands for, so the strip lines up with the plot above it.

    ../.venv/bin/python -m pytest vnl_experiments/probes/network_diagram_test.py -q
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vnl_experiments.probes import network_diagram as nd  # noqa: E402

DIAGRAMS = [nd.DEFAULT_DIAGRAM, nd.ACTOR_ONLY_DIAGRAM]


# ---------------------------------------------------------------------------
# Layout integrity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("diagram", DIAGRAMS)
def test_every_link_connects_boxes_that_exist(diagram):
    """A typo'd key would otherwise surface as a KeyError at render time."""
    keys = {box.key for box in diagram.boxes}
    for link in diagram.links:
        assert link.src in keys, f"{link.src} -> {link.dst}: no such source"
        assert link.dst in keys, f"{link.src} -> {link.dst}: no such destination"


@pytest.mark.parametrize("diagram", DIAGRAMS)
def test_box_keys_are_unique(diagram):
    keys = [box.key for box in diagram.boxes]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("diagram", DIAGRAMS)
def test_boxes_stay_inside_the_strip(diagram):
    """Heights are axes fractions, so a bad one silently clips instead of erroring."""
    for box in diagram.boxes:
        assert 0.0 <= box.bottom, f"{box.key} hangs below the strip ({box.bottom:.3f})"
        assert box.top <= 1.0, f"{box.key} sticks out the top ({box.top:.3f})"


def test_actor_only_drops_the_encoder_entirely():
    """Both the boxes and the links that touch them -- a dangling link would raise."""
    keys = {box.key for box in nd.ACTOR_ONLY_DIAGRAM.boxes}
    assert not keys & {"reference", "latent", "enc1", "enc2", "enc3", "enc4"}
    for link in nd.ACTOR_ONLY_DIAGRAM.links:
        assert link.src in keys and link.dst in keys


# ---------------------------------------------------------------------------
# Alignment with the plot above -- the reason the module exists
# ---------------------------------------------------------------------------

def test_tick_positions_are_the_layer_coordinates():
    positions = nd.DEFAULT_DIAGRAM.tick_positions()
    assert positions == list(nd.LAYER_X.values())
    assert len(positions) == 11
    assert positions == sorted(positions), "layers must run left to right"


def test_tick_positions_and_labels_stay_in_step():
    assert len(nd.DEFAULT_DIAGRAM.tick_positions()) == \
        len(nd.DEFAULT_DIAGRAM.tick_labels())


def test_the_group_gaps_are_where_the_layout_says():
    """Gaps at 0, 6 and 11 separate input / predictor / decoder / output."""
    positions = nd.DEFAULT_DIAGRAM.tick_positions()
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert gaps == [2, 1, 1, 1, 1, 2, 1, 1, 1, 2]


def test_a_shared_x_axis_is_not_overridden():
    """The plot above owns the scale; the strip must not reset it (that is the bug that
    would make the boxes point at the wrong layers)."""
    fig, (ax, strip) = nd.figure_with_diagram(figsize=(6, 4))
    ax.set_xlim(-3, 15)
    nd.draw(strip)
    assert strip.get_xlim() == pytest.approx((-3, 15))
    plt.close(fig)


def test_standalone_axes_gets_the_diagrams_own_limits():
    fig, ax = plt.subplots()
    nd.draw(ax)
    assert ax.get_xlim() == pytest.approx(nd.DEFAULT_DIAGRAM.xlim)
    plt.close(fig)


def test_diagram_axes_hides_the_plots_tick_labels_by_default():
    """The boxes are the tick labels; drawn together they overprint each other."""
    fig, ax = plt.subplots()
    nd.diagram_axes(fig, ax)
    assert not any(t.get_visible() for t in ax.get_xticklabels())
    plt.close(fig)

    fig, ax = plt.subplots()
    nd.diagram_axes(fig, ax, hide_xticklabels=False)
    assert all(t.get_visible() for t in ax.get_xticklabels())
    plt.close(fig)


def test_diagram_axes_leaves_room_and_does_not_move_the_plots_left_edge():
    fig, ax = plt.subplots()
    before = ax.get_position()
    strip = nd.diagram_axes(fig, ax, height=0.3, pad=0.01)
    after = ax.get_position()
    assert after.height < before.height
    assert after.x0 == pytest.approx(before.x0)
    assert after.width == pytest.approx(before.width)
    assert strip.get_position().y0 == pytest.approx(before.y0)
    assert strip.get_position().y1 <= after.y0 + 1e-9
    plt.close(fig)


# ---------------------------------------------------------------------------
# Connector geometry
# ---------------------------------------------------------------------------

def test_equal_height_boxes_are_joined_by_a_straight_line():
    """The S-curve must fall out of a height difference, not be applied everywhere."""
    path = nd._horizontal_bezier(0.0, 0.5, 1.0, 0.5, slack=0.5)
    assert all(y == pytest.approx(0.5) for _, y in path.vertices)


def test_a_height_difference_gives_horizontal_tangents_at_both_ends():
    path = nd._horizontal_bezier(0.0, 0.2, 1.0, 0.8, slack=0.5)
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = path.vertices
    assert y1 == pytest.approx(y0), "must leave the source horizontally"
    assert y2 == pytest.approx(y3), "must arrive at the destination horizontally"
    assert x0 < x1 <= x2 < x3


def test_the_two_input_streams_leave_at_different_heights():
    """Their whole point is showing that the predictor joins two separate streams."""
    into_pred1 = [link for link in nd.DEFAULT_DIAGRAM.links
                  if (link.src, link.dst) == ("input", "pred1")]
    assert len(into_pred1) == 2
    assert into_pred1[0].src_y != into_pred1[1].src_y


# ---------------------------------------------------------------------------
# It renders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("diagram", DIAGRAMS)
def test_draw_emits_one_patch_per_box_and_link(diagram):
    fig, ax = plt.subplots()
    nd.draw(ax, diagram)
    assert len(ax.patches) == len(diagram.boxes) + len(diagram.links)
    plt.close(fig)


@pytest.mark.parametrize("diagram", DIAGRAMS)
def test_draw_labels_every_box(diagram):
    fig, ax = plt.subplots()
    nd.draw(ax, diagram, notes=False)
    extras = sum(len(box.extras) for box in diagram.boxes)
    labelled = sum(1 for link in diagram.links if link.label)
    assert len(ax.texts) == len(diagram.boxes) + extras + labelled
    plt.close(fig)


def test_notes_can_be_switched_off():
    fig, ax = plt.subplots()
    nd.draw(ax, nd.DEFAULT_DIAGRAM, notes=False)
    without = len(ax.texts)
    plt.close(fig)

    fig, ax = plt.subplots()
    nd.draw(ax, nd.DEFAULT_DIAGRAM, notes=True)
    assert len(ax.texts) == without + len(nd.DEFAULT_DIAGRAM.notes)
    plt.close(fig)


def test_the_delay_label_follows_the_delay():
    boxes = nd._actor_boxes(delay_label="t-20")
    input_box = next(b for b in boxes if b.key == "input")
    assert "action[t-20]" in input_box.label
    assert any("Obs[t-20]" == text for text, _, _ in input_box.extras)
