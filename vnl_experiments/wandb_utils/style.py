"""Consistent plotting style across all analysis figures.

Every ``plot.py`` should call :func:`apply_style` once at the top, and use
:data:`CONDITION_STYLE` (via :func:`color_for` / :func:`marker_for`) so that a
given experimental condition keeps the same colour and marker in every figure.
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

# 1 control step = 10 ms (ctrl_dt = 0.01 s). Used for the secondary ms axis.
CTRL_DT_MS = 10

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
    # (min_std = 0.25) variants keep the hue and change the marker.
    "expfm": {"color": "C2", "marker": "^", "label": "Explicit forward model"},
    "encdec": {"color": "C1", "marker": "o", "label": "Enc-dec with efference copy"},
    "expfm_std25": {"color": "C2", "marker": "*", "label": r"Explicit FM, min_std 0.25"},
    "pgfm_std25": {"color": "C4", "marker": "*", "label": r"Policy-gradient FM, min_std 0.25"},
}


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


def add_ms_axis(ax, max_x: float):
    """Add a top x-axis expressing the bottom 'delay (steps)' axis in milliseconds.

    Returns the twin axis. Mirrors the bottom axis limits and converts tick labels
    using :data:`CTRL_DT_MS`.
    """
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ticks = ticks[(ticks >= 0) & (ticks <= max_x * 1.1)]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{int(t * CTRL_DT_MS)}" for t in ticks])
    ax2.set_xlabel("Observation delay (ms)")
    sns.despine(ax=ax2, top=False, right=True, left=True, bottom=True)
    return ax2
