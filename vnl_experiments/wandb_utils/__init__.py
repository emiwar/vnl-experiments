"""Shared helpers for WandB-based analyses.

See ``analysis/README.md`` for the analysis pipeline policy these support.
"""

from vnl_experiments.wandb_utils.comparability import (
    DEFAULT_INVARIANTS,
    comparability_report,
    git_commit_summary,
)
from vnl_experiments.wandb_utils.fetch import (
    PROVENANCE_KEYS,
    fetch_runs,
    records_to_df,
    run_record,
)
from vnl_experiments.wandb_utils.style import (
    BUCKET_STYLE,
    CONDITION_STYLE,
    CTRL_DT_MS,
    MEAN_LINE,
    REWARD_SOURCES,
    SEED_LINE,
    add_ms_axis,
    apply_style,
    behaviour_color,
    behaviour_label,
    bucket_color,
    bucket_label,
    color_for,
    label_for,
    marker_for,
    plot_seeds,
    provenance,
    reward_label,
    seed_legend_handles,
    write_figure_manifest,
)

# `index` and `pipeline` are imported as modules rather than re-exported piecemeal:
# `index.load` / `pipeline.resolve_selection` read better at the call site than bare
# names, and it keeps the (lazy) wandb import out of plot-only scripts.
from vnl_experiments.wandb_utils import index, pipeline, timing  # noqa: E402,F401

__all__ = [
    "index",
    "pipeline",
    "timing",
    "fetch_runs",
    "run_record",
    "records_to_df",
    "PROVENANCE_KEYS",
    "comparability_report",
    "git_commit_summary",
    "DEFAULT_INVARIANTS",
    "apply_style",
    "add_ms_axis",
    "behaviour_color",
    "behaviour_label",
    "color_for",
    "marker_for",
    "label_for",
    "provenance",
    "write_figure_manifest",
    "CONDITION_STYLE",
    "BUCKET_STYLE",
    "bucket_color",
    "bucket_label",
    "CTRL_DT_MS",
]
