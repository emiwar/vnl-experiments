"""Linear probes on recorded network activations.

Reads the ``activations`` artifacts produced by
:mod:`vnl_experiments.delays.record_activations` and answers "how much of quantity X is
linearly present in layer L?". Question-independent, like the artifacts themselves, so
several analyses share it rather than each carrying a copy.

* :mod:`~vnl_experiments.probes.linear_decoding` -- the ridge decoder and the recording reader.
* :mod:`~vnl_experiments.probes.pathways` -- which depth each layer sits at, per pathway.
"""

from vnl_experiments.probes.linear_decoding import (
    DECODE_VERSION,
    Recording,
    decode,
    decode_file,
    degenerate_targets,
    efference_queue,
    make_targets,
    open_recording,
    valid_mask,
    valid_stats,
)
from vnl_experiments.probes.pathways import (
    ACTOR_STAGES,
    CRITIC_STAGES,
    ENCODER_STAGES,
    PROBE_SET_VERSION,
    REFERENCE_PROBES,
    resolve_stage,
    stage_axis,
    unclassified,
)

__all__ = [
    "ACTOR_STAGES", "CRITIC_STAGES", "DECODE_VERSION", "ENCODER_STAGES",
    "PROBE_SET_VERSION", "REFERENCE_PROBES", "Recording", "decode", "decode_file",
    "degenerate_targets", "efference_queue", "make_targets", "open_recording",
    "resolve_stage", "stage_axis", "unclassified", "valid_mask", "valid_stats",
]
