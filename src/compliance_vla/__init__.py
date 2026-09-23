"""Compliance labels from bilateral demonstrations, and the policy they supervise.

This package is the hardware-independent core of the accompanying paper: the
parts a reviewer can run, inspect and test without a robot. Each module maps to
one part of the method.

    geometry     contact-frame fitting and frame rotations
    extraction   windowed regression and the identifiability mask -> K(t) labels
    synthetic    a demonstration with known ground-truth stiffness, for testing
    wrench       sensorless wrench bias model (RFF ridge) and the noise floor
    benchmark    the offline "are these labels learnable?" check
    ensembling   temporal ensembling across overlapping action chunks
    safety       log-space stiffness rate limiting and the energy tank
    frames       contact-frame K -> base-frame diagonal, for the controller
    policy       the compliance-output VLA head, its losses and its dataset
                 (requires torch; the rest of this package does not)

`policy` is deliberately NOT imported here, so that importing compliance_vla on
a machine without torch works and the numpy-only core stays usable on its own.
"""

__all__ = [
    "benchmark",
    "ensembling",
    "extraction",
    "frames",
    "geometry",
    "safety",
    "synthetic",
    "wrench",
]

__version__ = "1.0.0"
