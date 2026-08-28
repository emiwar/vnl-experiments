"""Draw the rodent forward-model architecture as a strip under a layer-wise plot.

``analysis/implicit-forward-model/figures/delay_10_layers_obs_decoding_manual.png``
carried a hand-drawn version of this, made by editing the exported SVG in a vector
editor. That is a dead end -- the drawing does not survive re-running the analysis, and
in that case it did not survive re-exporting the SVG either. This module rebuilds it in
matplotlib so it can be regenerated, restyled, and re-used.

The point of the strip is **alignment**: every box sits at the x coordinate of the layer
it stands for, so a reader can drop straight down from a point on the decodability curve
to the box that produced it. Everything therefore lives in *data* coordinates on the x
axis and in axes fractions (0-1) on the y axis, and the intended use is an architecture
axes that shares its x axis with the plot above::

    from vnl_experiments.probes import network_diagram as nd

    xs = nd.DEFAULT_DIAGRAM.tick_positions()      # the layer x coordinates
    fig, (ax, arch) = nd.figure_with_diagram(figsize=(8.4, 6.4), height=0.42)
    ax.plot(xs, r2_explicit, marker="o")          # your usual plot
    nd.draw(arch)

or, to attach a strip to an axes you already have::

    arch = nd.diagram_axes(fig, ax, height=0.42)
    nd.draw(arch)

The plot's x tick labels are switched off when a strip is attached -- the boxes are the
tick labels. :meth:`Diagram.tick_labels` gives the short forms (``pred 1``, ``p̂``, ...)
for a plot drawn *without* a strip.

The x coordinates are the ones the original figure used -- ``-1`` for the input, then
1-5 for the predictor, 7-10 for the decoder, 12 for the output -- with a one-unit gap at
0, 6 and 11 that visually separates the three groups. They come from the layer table in
``analysis/implicit-forward-model/figure_for_report.ipynb``; ``LAYER_X`` below is the
single place they are written down, so a plot and its strip cannot drift apart.

Nothing here reads data, so it has no place in the extract/plot split -- it is styling,
like ``wandb_utils.style``.

Run ``python -m vnl_experiments.probes.network_diagram --out /tmp/arch.png`` to render
the strip on its own, which is the quickest way to check a change to the layout.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

#: The seaborn-darkgrid panel colour, which is what the hand-drawn original used for the
#: boxes -- they read as "part of the figure furniture" rather than as data.
BOX_FILL = "#EAEAF2"
ACTOR_TEXT = "#262626"
ACTOR_LINE = "#000000"
#: The encoder pathway is drawn muted throughout: this figure is about the actor, and the
#: encoder is context. Dotted outlines carry the same message as the grey.
MUTED_TEXT = "#8A8A93"
MUTED_LINE = "#7A7A82"


@dataclasses.dataclass(frozen=True)
class Style:
    box_fill: str = BOX_FILL
    text_color: str = ACTOR_TEXT
    line_color: str = ACTOR_LINE
    edge_color: str = "none"
    edge_style: str | tuple = "solid"
    edge_width: float = 0.0
    line_width: float = 1.9
    font_size: float = 7.0
    alpha: float = 1.0


ACTOR_STYLE = Style()
MUTED_STYLE = Style(text_color=MUTED_TEXT, line_color=MUTED_LINE,
                    edge_color=MUTED_LINE, edge_style=(0, (1, 1.6)),
                    edge_width=0.8, line_width=1.3, alpha=0.75)


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Box:
    """One layer, drawn as a tall narrow rectangle with a rotated label.

    ``x`` is a data coordinate on the shared axis; ``top`` and ``height`` are axes
    fractions. Boxes in a row are **top-aligned** and differ in height -- that is what
    makes the connectors between them curve, and it is how the original was drawn.

    ``extras`` places additional rotated labels inside the box at ``(x_frac, y_frac)``
    of its own bounding box; the input box uses it to carry ``Obs[t-k]`` above the
    action queue.
    """

    key: str
    x: float
    label: str
    height: float
    top: float
    style: Style = ACTOR_STYLE
    label_y: float = 0.5
    label_x: float = 0.5
    extras: tuple[tuple[str, float, float], ...] = ()
    tick_label: str | None = None
    #: Overrides ``style.font_size``. The two input boxes carry the longest strings in
    #: the figure and are the first thing to overflow when the strip is made shorter.
    font_size: float | None = None

    @property
    def bottom(self) -> float:
        return self.top - self.height

    @property
    def middle(self) -> float:
        return self.top - self.height / 2


@dataclasses.dataclass(frozen=True)
class Link:
    """An arrow from the right edge of one box to the left edge of another.

    Both ends leave and arrive **horizontally**, so a link between two boxes whose
    centres differ in height is an S-curve and a link between two equal boxes is a
    straight line -- no per-arrow tuning, and the picture stays right when a box height
    changes. (``connectionstyle="angle3"`` cannot express this: with both angles at 0 the
    two tangent lines are parallel and matplotlib has no control point to place. The
    cubic in :func:`_horizontal_bezier` is built directly instead.)

    ``src_y`` / ``dst_y`` override the attachment height as a fraction of the source or
    destination box, which is what lets the two input streams leave one box at different
    heights and converge on a single arrowhead. ``slack`` is how far the control points
    sit along the horizontal run: smaller is a tighter, steeper curve.
    """

    src: str
    dst: str
    style: Style = ACTOR_STYLE
    src_y: float | None = None
    dst_y: float | None = None
    slack: float = 0.5
    label: str | None = None
    label_pos: float = 0.55
    label_rotation: float = 78.0
    label_offset: tuple[float, float] = (0.12, 0.0)
    arrow: bool = True


@dataclasses.dataclass(frozen=True)
class Note:
    text: str
    x: float
    y: float
    ha: str = "left"
    va: str = "top"
    font_size: float = 8.0
    color: str = ACTOR_TEXT


@dataclasses.dataclass(frozen=True)
class Diagram:
    boxes: tuple[Box, ...]
    links: tuple[Link, ...]
    notes: tuple[Note, ...] = ()
    #: Width of every box, in x data units. 0.64 of the 1-unit layer spacing, matching
    #: the original's 40 px boxes on a 62.5 px grid.
    box_width: float = 0.64
    xlim: tuple[float, float] = (-1.9, 12.9)
    #: Multiplies every label size. The default geometry is tuned for a strip about
    #: 8 x 2.6 in; a shorter strip wants a smaller scale, since box heights are
    #: fractions of the axes but text is in points.
    font_scale: float = 1.0

    def by_key(self, key: str) -> Box:
        for box in self.boxes:
            if box.key == key:
                return box
        raise KeyError(f"no box {key!r}; have {[b.key for b in self.boxes]}")

    def tick_positions(self) -> list[float]:
        """The x coordinates a plot above this strip should put its layers at."""
        return [b.x for b in self.boxes if b.tick_label is not None]

    def tick_labels(self) -> list[str]:
        return [b.tick_label for b in self.boxes if b.tick_label is not None]


# ---------------------------------------------------------------------------
# The RodentForwardModel picture
# ---------------------------------------------------------------------------

#: Layer -> x coordinate, the one place the axis is defined. The gaps at 0, 6 and 11
#: separate input / predictor / decoder without a divider line.
LAYER_X = {
    "input": -1, "pred1": 1, "pred2": 2, "pred3": 3, "pred4": 4, "phat": 5,
    "dec1": 7, "dec2": 8, "dec3": 9, "dec4": 10, "out": 12,
}

ACTOR_TOP = 0.97
ENCODER_TOP = 0.41
#: Heights, in axes fractions, measured off the hand-drawn original. They are roughly
#: proportional to the rotated label's length, but were set by eye, so they are data
#: rather than a formula.
H_HIDDEN = 0.36
H_INPUT = 0.50
H_PHAT = 0.29
H_OUT = 0.19
H_ENCODER = 0.40
H_LATENT = 0.18
#: The two queue boxes hold the longest strings in the figure, so they get their own,
#: smaller size rather than forcing every other box down to fit them.
QUEUE_FONT = 6.0

_DELAY = "t-10"


def _actor_boxes(delay_label: str = _DELAY) -> list[Box]:
    boxes = [
        # One box, two labels: the delayed observation sits above the action queue,
        # and the two arrows leaving it are what say they are separate streams.
        Box("input", LAYER_X["input"], f"action[t-1]\n⋮\naction[{delay_label}]",
            H_INPUT, ACTOR_TOP, label_y=0.26, font_size=QUEUE_FONT,
            extras=((f"Obs[{delay_label}]", 0.72, 0.80),),
            tick_label="input\n(delayed +\nefference)"),
    ]
    for i in range(1, 5):
        boxes.append(Box(f"pred{i}", LAYER_X[f"pred{i}"],
                         f"Forward model\nHidden {i}", H_HIDDEN, ACTOR_TOP,
                         tick_label=f"pred {i}"))
    boxes.append(Box("phat", LAYER_X["phat"], "Forward model\nOutput",
                     H_PHAT, ACTOR_TOP, tick_label="p̂"))
    for i in range(1, 5):
        boxes.append(Box(f"dec{i}", LAYER_X[f"dec{i}"], f"Decoder\nHidden {i}",
                         H_HIDDEN, ACTOR_TOP, tick_label=f"dec {i}"))
    boxes.append(Box("out", LAYER_X["out"], "Torques", H_OUT, ACTOR_TOP,
                     tick_label="out"))
    return boxes


def _encoder_boxes() -> list[Box]:
    boxes = [
        Box("reference", LAYER_X["input"], "Reference [t-1]\n⋮\nReference [t-5]",
            H_ENCODER, ENCODER_TOP, style=MUTED_STYLE, font_size=QUEUE_FONT),
    ]
    for i in range(1, 5):
        boxes.append(Box(f"enc{i}", LAYER_X[f"pred{i}"], f"Encoder\nHidden {i}",
                         H_ENCODER, ENCODER_TOP, style=MUTED_STYLE))
    boxes.append(Box("latent", LAYER_X["phat"], "Latent", H_LATENT, ENCODER_TOP,
                     style=MUTED_STYLE))
    return boxes


def _links() -> list[Link]:
    links = [
        # The two input streams converge on one arrowhead: the predictor is what joins
        # the delayed observation to the efference copy.
        Link("input", "pred1", src_y=0.78),
        Link("input", "pred1", src_y=0.14),
    ]
    links += [Link(f"pred{i}", f"pred{i + 1}") for i in range(1, 4)]
    links += [Link("pred4", "phat"), Link("phat", "dec1")]
    links += [Link(f"dec{i}", f"dec{i + 1}") for i in range(1, 4)]
    links += [Link("dec4", "out")]
    links += [Link("reference", "enc1", style=MUTED_STYLE)]
    links += [Link(f"enc{i}", f"enc{i + 1}", style=MUTED_STYLE) for i in range(1, 4)]
    links += [
        Link("enc4", "latent", style=MUTED_STYLE),
        Link("latent", "dec1", style=MUTED_STYLE, label="Sampling"),
    ]
    return links


DEFAULT_NOTE = ("Forward model with autoregressive loss\n"
                "means $\\it{Forward\\ model\\ Output}$ is trained\n"
                "to predict Obs[t] (MSE loss)")

DEFAULT_DIAGRAM = Diagram(
    boxes=tuple(_actor_boxes() + _encoder_boxes()),
    links=tuple(_links()),
    notes=(Note(DEFAULT_NOTE, x=6.7, y=0.34),),
)

#: The same picture without the encoder row -- for a figure that only shows the actor
#: pathway, where the encoder is a distraction rather than context.
ACTOR_ONLY_DIAGRAM = dataclasses.replace(
    DEFAULT_DIAGRAM,
    boxes=tuple(_actor_boxes()),
    links=tuple(link for link in _links()
                if not {link.src, link.dst} & {"reference", "latent",
                                               *(f"enc{i}" for i in range(1, 5))}),
    notes=(),
)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_box(ax: Axes, box: Box, width: float, scale: float) -> None:
    style = box.style
    size = (box.font_size if box.font_size is not None else style.font_size) * scale
    patch = FancyBboxPatch(
        (box.x - width / 2, box.bottom), width, box.height,
        boxstyle="square,pad=0", linewidth=style.edge_width,
        facecolor=style.box_fill, edgecolor=style.edge_color,
        linestyle=style.edge_style, alpha=style.alpha, zorder=2,
    )
    ax.add_patch(patch)
    # rotation=90 stacks the lines of a multi-line label left-to-right, which is why the
    # action queue reads t-1 on the left and t-10 on the right.
    ax.text(box.x + (box.label_x - 0.5) * width, box.bottom + box.label_y * box.height,
            box.label, rotation=90, ha="center", va="center",
            multialignment="center", fontsize=size,
            color=style.text_color, zorder=3)
    for text, x_frac, y_frac in box.extras:
        ax.text(box.x + (x_frac - 0.5) * width, box.bottom + y_frac * box.height,
                text, rotation=90, ha="center", va="center",
                fontsize=size, color=style.text_color, zorder=3)


def _horizontal_bezier(x0: float, y0: float, x1: float, y1: float,
                       slack: float) -> Path:
    """A cubic from (x0,y0) to (x1,y1) leaving and arriving horizontally.

    Degenerates to a straight segment when ``y0 == y1``, so the same call draws the
    plain layer-to-layer arrows and the S-curves onto the offset boxes.
    """
    dx = (x1 - x0) * slack
    return Path([(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])


def _draw_link(ax: Axes, diagram: Diagram, link: Link) -> None:
    src, dst = diagram.by_key(link.src), diagram.by_key(link.dst)
    half = diagram.box_width / 2
    x0 = src.x + half
    x1 = dst.x - half
    y0 = (src.bottom + link.src_y * src.height) if link.src_y is not None else src.middle
    y1 = (dst.bottom + link.dst_y * dst.height) if link.dst_y is not None else dst.middle
    style = link.style
    patch = FancyArrowPatch(
        path=_horizontal_bezier(x0, y0, x1, y1, link.slack),
        arrowstyle="-|>" if link.arrow else "-",
        mutation_scale=11, linewidth=style.line_width,
        color=style.line_color, alpha=style.alpha,
        shrinkA=0, shrinkB=1, zorder=4,
    )
    ax.add_patch(patch)
    if link.label:
        lx = x0 + (x1 - x0) * link.label_pos + link.label_offset[0]
        ly = y0 + (y1 - y0) * link.label_pos + link.label_offset[1]
        ax.text(lx, ly, link.label, rotation=link.label_rotation,
                ha="center", va="center",
                fontsize=style.font_size * diagram.font_scale,
                color=style.text_color, zorder=5)


def draw(ax: Axes, diagram: Diagram = DEFAULT_DIAGRAM, *,
         notes: bool = True, clean: bool = True) -> Axes:
    """Draw ``diagram`` into ``ax``.

    ``ax`` keeps its x limits if they are already set by a shared axis; otherwise the
    diagram's own ``xlim`` is applied. y is always 0-1 in axes fractions.
    """
    for box in diagram.boxes:
        _draw_box(ax, box, diagram.box_width, diagram.font_scale)
    for link in diagram.links:
        _draw_link(ax, diagram, link)
    if notes:
        for note in diagram.notes:
            ax.text(note.x, note.y, note.text, ha=note.ha, va=note.va,
                    fontsize=note.font_size, color=note.color, zorder=5)
    ax.set_ylim(0, 1)
    # Only claim the x limits when nothing else owns them: on a shared axis the plot
    # above has already set the scale, and overriding it here would break alignment,
    # which is the one thing this strip exists to provide.
    if len(ax.get_shared_x_axes().get_siblings(ax)) <= 1:
        ax.set_xlim(*diagram.xlim)
    if clean:
        ax.set_axis_off()
        ax.set_facecolor("none")
    return ax


def diagram_axes(fig: Figure, plot_ax: Axes, *, height: float = 0.30,
                 pad: float = 0.01, hide_xticklabels: bool = True) -> Axes:
    """Add an architecture strip below ``plot_ax``, sharing its x axis.

    ``plot_ax`` is shrunk from the bottom to make room, so this works on an axes that is
    already laid out (including one from a gridspec or a figurefirst template). Both
    fractions are of ``plot_ax``'s height. Do not call ``tight_layout`` afterwards -- it
    would undo the placement; size the figure instead.

    The plot's own x tick labels are removed by default: the boxes *are* the tick labels,
    and because the strip abuts the plot the two overprint each other otherwise. Pass
    ``hide_xticklabels=False`` to keep them.
    """
    box = plot_ax.get_position()
    strip_height = box.height * height
    plot_ax.set_position([box.x0, box.y0 + strip_height + box.height * pad,
                          box.width, box.height - strip_height - box.height * pad])
    if hide_xticklabels:
        plot_ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax = fig.add_axes([box.x0, box.y0, box.width, strip_height], sharex=plot_ax)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return ax


def figure_with_diagram(*, figsize: tuple[float, float] = (8.4, 6.4),
                        height: float = 0.42, pad: float = 0.01,
                        hide_xticklabels: bool = True,
                        **subplot_kw) -> tuple[Figure, tuple[Axes, Axes]]:
    """A figure holding one plot axes with an architecture strip under it.

    Returns ``(fig, (plot_ax, diagram_ax))``. The strip shares x with the plot, so
    plotting against :meth:`Diagram.tick_positions` is all the alignment needs.

    The defaults give the strip about 2.6 in of an 6.4 in figure, which is what the box
    heights and font sizes below are tuned for. Making the strip much shorter without
    lowering ``Diagram.font_scale`` will push the long queue labels out of their boxes.
    """
    fig, plot_ax = plt.subplots(figsize=figsize, **subplot_kw)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.06)
    return fig, (plot_ax, diagram_axes(fig, plot_ax, height=height, pad=pad,
                                       hide_xticklabels=hide_xticklabels))


# ---------------------------------------------------------------------------
# Self-render, for checking a layout change
# ---------------------------------------------------------------------------

def _main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="network_diagram.png")
    parser.add_argument("--actor-only", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="draw a placeholder curve above the strip, to check that "
                             "the two axes line up")
    parser.add_argument("--width", type=float, default=8.0)
    parser.add_argument("--height", type=float, default=2.6)
    args = parser.parse_args(argv)

    diagram = ACTOR_ONLY_DIAGRAM if args.actor_only else DEFAULT_DIAGRAM
    if args.demo:
        xs = diagram.tick_positions()
        # Not data -- a shape that makes a misalignment obvious at a glance.
        ys = [0.70, 0.78, 0.80, 0.86, 0.87, 0.87, 0.76, 0.65, 0.59, 0.55, 0.37]
        fig, (ax, strip) = figure_with_diagram(figsize=(args.width, args.width * 0.8))
        ax.plot(xs, ys, marker="x", color="C0", label="placeholder, not data")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Forward-model-ness")
        ax.legend(frameon=False, fontsize=8)
        for x in xs:
            ax.axvline(x, color="0.85", lw=0.5, zorder=0)
        draw(strip, diagram)
    else:
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        draw(ax, diagram)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _main()
