"""Superseded by :mod:`vnl_experiments.probes.linear_decoding`.

This analysis is written up and frozen. The decoder moved into the package on 2026-08-19 so
that later probe analyses share it instead of copying it; the arithmetic is unchanged, and
``vnl_experiments/probes/linear_decoding_test.py`` pins it by re-decoding a recording and
matching this folder's committed ``data.csv`` R² to 1e-9.

Kept as a shim so ``extract.py`` here still runs. One difference if you do regenerate:
``decode_file`` now also returns ``pathway`` / ``stage_index`` / ``stage_label`` /
``target_degenerate`` / ``decode_version`` / ``probe_set_version``, so ``data.csv`` would
gain columns -- with identical values in the existing ones.

New analyses should import from the package directly.
"""

from vnl_experiments.probes.linear_decoding import (  # noqa: F401
    DEFAULT_MAX_SAMPLES,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
    LAMBDA_GRID,
    Recording,
    _r2,
    _ridge_fit,
    _ridge_predict,
    _rows,
    _split_clips,
    decode,
    decode_file,
    efference_queue,
    make_targets,
    valid_mask,
)


def load_recording(path):
    """Back-compat shim: the eager loader the frozen ``fm_mse_crosscheck.py`` expects.

    The package reads layers on demand (peak RAM ~1 GB instead of ~4 GB); this rebuilds the
    old all-at-once dict for the two scripts in this folder that still index it directly.
    """
    with Recording(path) as rec:
        return {"attrs": rec.attrs, "target": rec.target, "dones": rec.dones,
                "layers": {name: rec.layer(name) for name in rec.layer_names}}
