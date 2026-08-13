"""Manifest-backed store for derived per-run data (evals, histories, activations, videos).

See :mod:`vnl_experiments.artifacts.store` for the layout and the provenance model, and
``analysis/README.md`` for how analyses declare what they need.
"""

from vnl_experiments.artifacts.producers import PRODUCERS, Producer, get_producer
from vnl_experiments.artifacts.store import (
    KINDS,
    MANIFEST_PATH,
    Entry,
    Store,
    manifest_df,
    read_manifest,
    spec_id,
    store_root,
)

__all__ = [
    "Store",
    "Entry",
    "KINDS",
    "MANIFEST_PATH",
    "manifest_df",
    "read_manifest",
    "spec_id",
    "store_root",
    "Producer",
    "PRODUCERS",
    "get_producer",
]
