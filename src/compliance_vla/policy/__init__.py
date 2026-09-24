"""Compliance-output VLA policy: architecture, losses, and dataset.

Importing this subpackage requires torch. The lerobot-backed policy modules
(compliance_policy, hybrid_policy, bi_act_policy, dataset) additionally require
lerobot and, for dataset/labels, the release tree's sibling directories -- see
_paths.py. `losses` and `force_encoder` need only torch, which is why they carry
the parts of the method that the test suite exercises directly.

Nothing is imported eagerly here: a missing optional dependency should surface
at the point of use, naming the module that needed it, rather than making the
whole subpackage unimportable.

The SmolVLA training, serving, evaluation and diagnostic entrypoints live here
too, run as modules from the release root, e.g.
``python -m compliance_vla.policy.train_policy --policy b5 --seed 0``:

    training     train_policy, train_b5, tune_lambda, fit_manner_force_calibration
    serving      serve_policy, client_example
    rollouts     evaluate_gate3, score_ink_removal, controller_frame_utils
    diagnostics  diagnose_language_grounding, check_input_sensitivity,
                 check_state_convention_sensitivity,
                 analyze_operatorA_rollout_trajectories

The robot-side rollout node that talks to serve_policy is the ROS 2 package in
``smolvla_policy/deploy_vla/``.
"""

__all__ = [
    # model
    "bi_act_policy",
    "compliance_policy",
    "dataset",
    "force_encoder",
    "hybrid_policy",
    "labels",
    "losses",
    # entrypoints
    "analyze_operatorA_rollout_trajectories",
    "check_input_sensitivity",
    "check_state_convention_sensitivity",
    "client_example",
    "controller_frame_utils",
    "diagnose_language_grounding",
    "evaluate_gate3",
    "fit_manner_force_calibration",
    "score_ink_removal",
    "serve_policy",
    "train_b5",
    "train_policy",
    "tune_lambda",
]
