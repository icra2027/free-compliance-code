"""Forward kinematics for the Franka Emika Panda/FR3 arm.

The dataset logs `observation.state` as 7-DoF joint position, not Cartesian
follower pose, so the impedance-extraction pipeline (which needs the follower's
Cartesian pose x_f(t) to compute the pose error against the leader x_l(t))
has to recover it via FK. Uses the standard published Panda modified-DH
parameters (flange frame, no hand/TCP offset), consistent with how
`observation.leader_pose` / `action` are recorded elsewhere in this dataset.
"""

import numpy as np

# Modified-DH (Craig convention) parameters: a_{i-1} [m], alpha_{i-1} [rad], d_i [m]
# for joints 1-7, then a fixed flange offset.
_DH = [
    (0.0, 0.0, 0.333),
    (0.0, -np.pi / 2, 0.0),
    (0.0, np.pi / 2, 0.316),
    (0.0825, np.pi / 2, 0.0),
    (-0.0825, -np.pi / 2, 0.384),
    (0.0, np.pi / 2, 0.0),
    (0.088, np.pi / 2, 0.0),
]
_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D = 0.0, 0.0, 0.107


def _dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk(q):
    """Forward kinematics to the flange frame.

    q: (7,) joint angles [rad]. Returns 4x4 homogeneous transform, base -> flange.
    """
    q = np.asarray(q, dtype=np.float64)
    assert q.shape == (7,), f"expected 7 joint angles, got {q.shape}"
    T = np.eye(4)
    for (a, alpha, d), theta in zip(_DH, q):
        T = T @ _dh_transform(a, alpha, d, theta)
    T = T @ _dh_transform(_FLANGE_A, _FLANGE_ALPHA, _FLANGE_D, 0.0)
    return T


def fk_batch(Q):
    """Vectorized-ish FK over a batch. Q: (N,7) -> positions (N,3), rotmats (N,3,3)."""
    Q = np.asarray(Q, dtype=np.float64)
    N = Q.shape[0]
    pos = np.empty((N, 3))
    rot = np.empty((N, 3, 3))
    for i in range(N):
        T = fk(Q[i])
        pos[i] = T[:3, 3]
        rot[i] = T[:3, :3]
    return pos, rot


def rotvec_from_matrix(R):
    """SO(3) log map: rotation matrix -> axis-angle (rotation vector), matching
    the convention already used for `action` / `observation.leader_pose`."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near-pi: axis from the symmetric part of R (numerically stable branch).
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        signs = np.sign([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis * np.where(signs == 0, 1.0, signs)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return w * theta


def matrix_from_rotvec(rv):
    """SO(3) exp map: rotation vector -> rotation matrix."""
    rv = np.asarray(rv, dtype=np.float64)
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def pose_error(x_l, R_l, x_f, R_f):
    """e = x_l ⊖ x_f as a 6D vector: position difference + rotation-vector of
    the relative rotation R_f^{-1} R_l (so e_rot -> 0 as R_f -> R_l)."""
    e_pos = x_l - x_f
    R_err = R_f.T @ R_l
    e_rot = rotvec_from_matrix(R_err)
    return np.concatenate([e_pos, e_rot])


if __name__ == "__main__":
    # Self-test: FK should reproduce the known nullspace/start posture's EE
    # position, cross-checked against the recorded leader pose at the first
    # frame of a session (leader ~= follower before any teleop motion).
    q_start = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
    T = fk(q_start)
    print("FK(start posture) position:", T[:3, 3])
    print("FK(start posture) rotvec:", rotvec_from_matrix(T[:3, :3]))
