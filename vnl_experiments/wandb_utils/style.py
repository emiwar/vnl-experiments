"""Consistent plotting style across all analysis figures.

Every ``plot.py`` should call :func:`apply_style` once at the top, and use
:data:`CONDITION_STYLE` (via :func:`color_for` / :func:`marker_for`) so that a
given experimental condition keeps the same colour and marker in every figure.

A condition's colour belongs *here*, not in a dict local to one folder: three of the
position/torque analyses each define the same ``MODE_COLOR`` privately, which is how one
manipulation ends up two colours in two figures. Add the key to :data:`CONDITION_STYLE`
instead.

The other three conventions this module exists to make easy -- see
``analysis/README.md`` §7 -- are :func:`reward_label` (name *which* reward is on the
axis), :func:`plot_seeds` (mean solid, seeds thin) and the ``ctrl_dt_ms`` argument of
:func:`add_ms_axis` (the two tracks have different control timesteps).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

_STYLE_FILE = Path(__file__).with_name("vnl.mplstyle")

#: The rodent's control step: 1 step = 10 ms (``ctrl_dt = 0.01 s``). The default for the
#: secondary ms axis. **dm_control_suite runs at ``ctrl_dt = 0.025``**, so those plots must
#: pass ``add_ms_axis(..., ctrl_dt_ms=25)`` -- the default would mislabel by 2.5x silently.
CTRL_DT_MS = 10

#: How a reward number is described on an axis, per source. Reward is close to meaningless
#: without this: the two worst measurement bugs in the project -- the in-training ``eval/*``
#: series that was really the train split, and the dm_control eval that was scaled 10x and
#: truncated -- both hid because a figure said "reward" where it meant "which reward".
REWARD_SOURCES: dict[str, str] = {
    # Rodent: the three offline eval datasets.
    "train": "train split",
    "old_eval": "held-out, old_eval",
    "new_eval": "held-out 30 s clips, new_eval",
    # Rodent: the run's own in-training series. Before 2026-08-20 this measured the
    # *train* split whatever it is called -- see rodent/README.md.
    "inline": "in-training eval",
    "final": "inline end-of-training eval",
    # dm_control_suite. Its "eval" is fresh episodes of the same task, not a held-out
    # set, and the training reward carries reward_scale = 10.
    "dmc_eval": "eval episodes, unscaled",
    "dmc_train": r"training rollouts, reward_scale $\times$10",
}

#: Per-seed vs mean line weights. A condition with more than one seed is drawn as thin
#: semi-transparent per-seed lines under a solid mean, so the reader sees the spread the
#: mean was taken over. See :func:`plot_seeds`.
SEED_LINE = dict(lw=0.9, alpha=0.35)
MEAN_LINE = dict(lw=2.2, alpha=0.95)

# Canonical colour + marker per condition. Keep these stable so figures across
# different questions are directly comparable. Colours are the matplotlib cycle
# colours already used in the original delay-sweep plot.
CONDITION_STYLE: dict[str, dict[str, str]] = {
    "efference": {"color": "C1", "marker": "o", "label": "With efference copy"},
    "no_efference": {"color": "C0", "marker": "s", "label": "No efference copy"},
    "forward_model": {"color": "C2", "marker": "^", "label": "Explicit forward model"},
    "efference_larger": {"color": "C3", "marker": "D", "label": "Efference, larger decoder"},
    "efference_deeper": {"color": "C4", "marker": "v", "label": "Efference, deeper decoder"},
    "efference_trunc": {"color": "C5", "marker": "P", "label": "Efference, truncated buffer"},
    # Imitation-target representation question (git f315e336).
    "absolute_reference": {"color": "C1", "marker": "o", "label": "Absolute, reference-root frame"},
    "absolute_current": {"color": "C2", "marker": "^", "label": "Absolute, current-root frame"},
    "relative": {"color": "C0", "marker": "s", "label": "Relative (baseline Imitation)"},
    # nnx-ppo update / seed reproducibility check.
    "baseline": {"color": "C0", "marker": "o", "label": "Original (old nnx-ppo, seed 42)"},
    "new_seed": {"color": "C1", "marker": "s", "label": "New nnx-ppo, seed 43"},
    "new_code_old_seed": {"color": "C3", "marker": "X", "label": "New nnx-ppo, seed 42 (test)"},
    # Forward-model loss vs architecture question.
    "pg_forward_model": {"color": "C4", "marker": "D", "label": "Policy-gradient FM (loss = 0)"},
    "fm0_untrained": {"color": "C7", "marker": ".", "label": "Untrained predictor (loss = 0, detached)"},
    # Action-noise robustness question. Colours match `forward_model` / `efference`
    # above so the arms read the same across questions; the wider-exploration
    # (min_std = 0.25) variants get their own hues -- same-hue/different-marker proved
    # illegible once both appeared in one panel.
    "expfm": {"color": "C2", "marker": "^", "label": "Explicit forward model"},
    "encdec": {"color": "C1", "marker": "o", "label": "Enc-dec with efference copy"},
    "expfm_std25": {"color": "C3", "marker": "*", "label": r"Explicit FM, min_std 0.25"},
    # Refactor regression check (2026-08-24). The three code epochs of the enc-dec net.
    "pre_refactor": {"color": "C0", "marker": "o", "label": "Pre-refactor (2026-08-11)"},
    "unregularized": {"color": "C3", "marker": "x", "label": "Refactor, regularisation zeroed"},
    "fixed": {"color": "C2", "marker": "s", "label": "Refactor, after fix"},
    # Recurrent-decoder question.
    "feedforward": {"color": "C1", "marker": "o", "label": "Feedforward decoder"},
    "recurrent": {"color": "C9", "marker": "D", "label": "Recurrent (LSTM) decoder"},
    # Recurrent-architecture comparison. `feedforward` / `forward_model` above are reused
    # so the arms read the same as in the sibling questions.
    "lstm": {"color": "C9", "marker": "D", "label": "LSTM decoder"},
    "gru": {"color": "C4", "marker": "s", "label": "GRU decoder"},
    "rnn": {"color": "C6", "marker": "v", "label": "Vanilla RNN decoder"},
    "pgfm_std25": {"color": "C4", "marker": "*", "label": r"Policy-gradient FM, min_std 0.25"},
    # Decoder-input ablations (2026-08-25). `ablate_efference` deliberately reuses
    # `no_efference`'s blue: it is the same manipulation, at one delay instead of a sweep.
    "ablate_intention": {"color": "C3", "marker": "v", "label": "No intention"},
    "ablate_proprioception": {"color": "C6", "marker": "P",
                              "label": "No proprioception"},
    "ablate_efference": {"color": "C0", "marker": "s", "label": "No efference copy"},
    # dm_control_suite flat-observation architectures (2026-09-10). Each reuses the hue
    # of its rodent analogue -- `encdec` C1, `forward_model` C2, `recurrent` C9 -- so an
    # architecture reads the same colour whichever track a figure is from.
    # Joint control mode (2026-09-16). `torque` reuses `delayed_mlp`'s C1 deliberately:
    # every delay analysis in this project so far *is* torque control, so the baseline
    # keeps the colour the reader already associates with it.
    "torque": {"color": "C1", "marker": "o", "label": "Torque control (servo_kp = 0)"},
    "servo": {"color": "C3", "marker": "D", "label": "Joint servo (equilibrium point)"},
    "delayed_mlp": {"color": "C1", "marker": "o", "label": "MLP (DelayedMLP)"},
    "flat_forward_model": {"color": "C2", "marker": "^",
                           "label": "Explicit forward model"},
    "flat_recurrent": {"color": "C9", "marker": "D", "label": "Recurrent"},
}


#: Colour and label per wall-clock bucket, for time-budget figures
#: (:mod:`vnl_experiments.wandb_utils.timing`). Kept here for the same reason as
#: :data:`CONDITION_STYLE`: a bucket should be the same colour in every figure that
#: draws one. The buckets are *not* conditions -- in a budget figure the condition is on
#: the categorical axis and the colour carries the bucket -- so they get their own dict
#: rather than entries in ``CONDITION_STYLE``, which would let a condition name collide
#: with a bucket name.
#:
#: Sequential greens for the periodic work (eval / video / checkpoint), grey for
#: training, and red for time lost to a stall, so a run that went wrong is separable
#: from a run that is merely busy at a glance.
BUCKET_STYLE: dict[str, dict[str, str]] = {
    "train_s": {"color": "#4c72b0", "label": "PPO step (training)"},
    "eval_s": {"color": "#dd8452", "label": "Eval rollouts"},
    "video_s": {"color": "#937860", "label": "Video render + upload"},
    "checkpoint_s": {"color": "#55a868", "label": "Checkpoint writes"},
    "overhead_s": {"color": "#8c8c8c", "label": "Other in-loop time"},
    "startup_tail_s": {"color": "#c7c7c7", "label": "Startup, compile, tail"},
    "stall_s": {"color": "#c44e52", "label": "Stalled (cluster outage)"},
}


def bucket_color(bucket: str) -> str:
    return BUCKET_STYLE.get(bucket, {}).get("color", "C7")


def bucket_label(bucket: str) -> str:
    return BUCKET_STYLE.get(bucket, {}).get("label", bucket)


def apply_style() -> None:
    """Apply the shared seaborn theme + matplotlib style. Call once per script."""
    sns.set_theme(style="ticks")
    plt.style.use(str(_STYLE_FILE))


def color_for(condition: str) -> str:
    return CONDITION_STYLE.get(condition, {}).get("color", "C7")


def marker_for(condition: str) -> str:
    return CONDITION_STYLE.get(condition, {}).get("marker", "o")


def label_for(condition: str) -> str:
    return CONDITION_STYLE.get(condition, {}).get("label", condition)


def reward_label(source: str, *, metric: str = "Episode reward",
                 per_step: bool = False) -> str:
    """A y-axis label that names *which* reward is plotted.

    ``reward_label("old_eval")`` -> ``"Episode reward (held-out, old_eval)"``. Pass a
    source not in :data:`REWARD_SOURCES` and it is used verbatim, so an unusual source is
    still named rather than dropped.

    Use this rather than a bare ``"Mean episode reward"``: eight different phrasings of
    that string are in the committed figures, none of which says what was measured.
    """
    described = REWARD_SOURCES.get(source, source)
    name = f"{metric} per step" if per_step else metric
    return f"{name} ({described})"


def plot_seeds(ax, df, *, x: str, y: str, seed_col: str = "seed",
               condition: str | None = None, color: str | None = None,
               scale: float = 1.0, label: str | None = None, marker_size: float = 3.5):
    """Draw one thin line per seed plus a solid mean, and return the mean series.

    Replicates within a ``(seed, x)`` cell are averaged **first**, so each seed
    contributes exactly one curve and the mean weights seeds equally rather than
    weighting whichever seed happened to be run twice.

    A single-seed condition is drawn as a mean line only -- one thin line under one solid
    line of the same colour reads as a spread that was never measured. Add the legend
    proxies with :func:`seed_legend_handles` once per figure, not once per condition.
    """
    import pandas as pd

    color = color if color is not None else color_for(condition or "")
    sub = df.dropna(subset=[y])
    if sub.empty:
        return pd.Series(dtype=float)

    per_seed = sub.groupby([seed_col, x])[y].mean().mul(scale)
    seeds = per_seed.index.get_level_values(0).unique()
    if len(seeds) > 1:
        for s in seeds:
            curve = per_seed.loc[s].sort_index()
            ax.plot(curve.index, curve.values, color=color, **SEED_LINE)

    mean = per_seed.groupby(x).mean().sort_index()
    ax.plot(mean.index, mean.values, color=color,
            marker=marker_for(condition or ""), ms=marker_size,
            label=label if label is not None else label_for(condition or ""),
            **MEAN_LINE)
    return mean


def seed_legend_handles(color: str = "0.4") -> list:
    """Proxy legend entries explaining the thin lines. One pair per figure."""
    from matplotlib.lines import Line2D

    return [Line2D([], [], color=color, **SEED_LINE, label="individual seed"),
            Line2D([], [], color=color, **MEAN_LINE, label="mean across seeds")]


def _short_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _repo_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def provenance(fig, here: Path | str, *inputs: Path | str) -> str:
    """Stamp a figure with the analysis, its input CSVs' hashes, commit and date.

    So a figure that has escaped into a slide deck can still be traced back to the exact
    committed data it was built from. Set ``VNL_NO_FOOTER=1`` to suppress the stamp for
    presentation figures; the returned string is written to ``figures/manifest.json``
    either way.
    """
    here = Path(here)
    parts = [here.name]
    parts += [f"{Path(p).name} {_short_hash(Path(p))}" for p in inputs]
    parts += [f"vnl-experiments {_repo_commit()}", date.today().isoformat()]
    text = "  ·  ".join(parts)
    if not os.environ.get("VNL_NO_FOOTER"):
        fig.text(0.005, 0.004, text, fontsize=4.5, color="0.55", ha="left", va="bottom")
    return text


def write_figure_manifest(here: Path | str, entries: dict[str, str]) -> Path:
    """Record ``{figure filename: provenance string}`` next to the figures."""
    path = Path(here) / "figures" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    return path


def add_ms_axis(ax, max_x: float, ctrl_dt_ms: float = CTRL_DT_MS):
    """Add a top x-axis expressing the bottom 'delay (steps)' axis in milliseconds.

    Returns the twin axis. Mirrors the bottom axis limits and converts tick labels using
    ``ctrl_dt_ms``, which defaults to the rodent's :data:`CTRL_DT_MS`. **Pass 25 for
    dm_control_suite** (``ctrl_dt = 0.025``); the default would label a 10-step delay as
    100 ms where it is 250, and nothing would raise.
    """
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ticks = ticks[(ticks >= 0) & (ticks <= max_x * 1.1)]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{int(t * ctrl_dt_ms)}" for t in ticks])
    ax2.set_xlabel("Observation delay (ms)")
    sns.despine(ax=ax2, top=False, right=True, left=True, bottom=True)
    return ax2
