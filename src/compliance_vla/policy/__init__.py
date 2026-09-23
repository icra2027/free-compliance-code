"""Compliance-output VLA policy: architecture, losses, and dataset.

Importing this subpackage requires torch. The lerobot-backed policy modules
(compliance_policy, hybrid_policy, bi_act_policy, dataset) additionally require
lerobot and, for dataset/labels, the release tree's sibling directories -- see
_paths.py. `losses` and `force_encoder` need only torch, which is why they carry
the parts of the method that the test suite exercises directly.

Nothing is imported eagerly here: a missing optional dependency should surface
at the point of use, naming the module that needed it, rather than making the
whole subpackage unimportable.
"""

__all__ = [
    "bi_act_policy",
    "compliance_policy",
    "dataset",
    "force_encoder",
    "hybrid_policy",
    "labels",
    "losses",
]
