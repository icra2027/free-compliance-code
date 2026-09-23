"""Temporal ensembling across overlapping action chunks.

A chunk emitted by one forward pass and a chunk emitted by the next are not
guaranteed to agree where they overlap. Executing each freshly-queried chunk in
isolation and discarding the previous one wholesale at the boundary introduces a
discontinuity, and a rigid, non-compliant policy has nothing to physically
absorb it: in practice such a policy can stall, re-chasing a slightly different
target at every replan instead of making net progress, and the effect is worst
when boundaries are frequent.

Every buffered chunk's prediction for a given low-level step is therefore
combined by exponential recency weighting,

    a_hat(t) = sum_i w_i a_i(t) / sum_i w_i,   w_i = exp(-m * age_i(t))

where age_i(t) is the number of low-level steps since chunk i was queried. The
freshest available prediction for a step dominates, while older overlapping
predictions still smooth the boundary rather than being discarded outright.

Position, orientation (as a rotation vector), log K and the gripper command are
all blended as plain vectors. Treating the rotation vector linearly rather than
via rotation averaging is an approximation, and a deliberate one: the candidates
are different chunks' short-horizon predictions of what should be nearly the
same target, not arbitrary rotations to average.

m is a tuned knob, not a value assumed correct a priori. Setting m -> infinity
recovers naive single-chunk execution exactly, so disabling the mechanism is its
exact fallback behavior rather than an approximation of it -- which is also what
`enabled=False` does here.

Extracted from the deployment node so the policy-independent part can be tested
without ROS, a robot or a checkpoint.
"""

from collections import deque

import numpy as np

__all__ = ["ENSEMBLE_M_DEFAULT", "ChunkEnsembler"]

#: Starting value for the recency decay rate m. Tuned empirically, not assumed.
ENSEMBLE_M_DEFAULT = 0.05


class ChunkEnsembler:
    """Buffers overlapping action chunks and blends their predictions per step.

    Steps are ABSOLUTE low-level step indices. A chunk buffered with
    ``query_step = s`` supplies predictions for steps ``s .. s + len(chunk) - 1``,
    i.e. ``chunk[0]`` is its prediction for the step at which it was queried.
    """

    def __init__(self, ensemble_m: float = ENSEMBLE_M_DEFAULT, enabled: bool = True):
        if ensemble_m < 0.0:
            raise ValueError(f"ensemble_m must be non-negative, got {ensemble_m}")
        self.ensemble_m = float(ensemble_m)
        self.enabled = bool(enabled)
        self._buffer: deque[tuple[int, np.ndarray]] = deque()

    def __len__(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        """Drops all buffered chunks, e.g. between rollouts."""
        self._buffer.clear()

    def buffer_chunk(self, query_step: int, action_chunk: np.ndarray) -> None:
        """Records one chunk and drops any buffered chunk whose coverage has
        fully passed.

        Called once per policy response, before that chunk's steps are executed.
        A single prune here is sufficient: no chunk's coverage can expire again
        before the next call, because the step counter only advances within what
        is already covered until then.
        """
        chunk = np.asarray(action_chunk)
        if chunk.ndim != 2:
            raise ValueError(f"action_chunk must be 2-D (horizon, action_dim), got {chunk.shape}")
        if len(chunk) == 0:
            raise ValueError("action_chunk must contain at least one step")
        self._buffer.append((int(query_step), chunk))
        while self._buffer and (
            self._buffer[0][0] + self._buffer[0][1].shape[0] <= query_step
        ):
            self._buffer.popleft()

    def ensembled_action(self, step: int) -> np.ndarray | None:
        """Blends every buffered chunk's prediction for absolute step `step`.

        Returns None if no buffered chunk covers `step`. Callers must check:
        this should not happen when buffer_chunk is called for the chunk about
        to be played back, but a missing prediction must not be silently
        substituted with a stale one.

        With ``enabled=False`` this returns the newest covering chunk's raw,
        unweighted prediction -- exactly the pre-ensembling behavior, since
        chunks are appended in query order.
        """
        candidates: list[np.ndarray] = []
        ages: list[int] = []
        for query_step, chunk in self._buffer:
            if query_step <= step < query_step + chunk.shape[0]:
                candidates.append(chunk[step - query_step])
                ages.append(step - query_step)
        if not candidates:
            return None
        if not self.enabled:
            return candidates[-1]
        weights = np.exp(-self.ensemble_m * np.array(ages, dtype=np.float64))
        weights /= weights.sum()
        stacked = np.stack(candidates, axis=0).astype(np.float64)
        return (weights[:, None] * stacked).sum(axis=0).astype(np.float32)
