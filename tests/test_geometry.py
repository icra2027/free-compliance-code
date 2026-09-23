"""Contact-frame geometry: quaternion algebra and plane fitting.

The frame fit is the first step of extraction, and everything downstream is
expressed in the frame it returns, so a silently-wrong frame would corrupt every
label without failing anything loudly. These tests pin down the two properties
that protect against that: the fit is rejected when the data is not planar, and
the returned normal has a definite, checked orientation.
"""

import numpy as np
import pytest

from compliance_vla.geometry import (
    build_inplane_basis,
    fit_contact_frame,
    numerically_differentiate,
    quat_conjugate,
    quat_multiply,
    quat_to_rotmat,
    quat_to_rotvec,
)

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def test_quat_multiply_by_identity_is_identity():
    q = np.array([0.1, -0.2, 0.3, 0.927])
    q /= np.linalg.norm(q)
    assert np.allclose(quat_multiply(q, IDENTITY_QUAT), q)
    assert np.allclose(quat_multiply(IDENTITY_QUAT, q), q)


def test_quat_times_its_conjugate_is_identity():
    q = np.array([0.3, 0.1, -0.4, 0.857])
    q /= np.linalg.norm(q)
    product = quat_multiply(q, quat_conjugate(q))
    assert np.allclose(product, IDENTITY_QUAT, atol=1e-12)


def test_quat_to_rotvec_recovers_known_axis_angle():
    axis = np.array([0.0, 0.0, 1.0])
    angle = 0.7
    q = np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])
    rotvec = quat_to_rotvec(q)
    assert np.allclose(rotvec, axis * angle, atol=1e-12)


def test_quat_to_rotvec_is_zero_for_identity():
    assert np.allclose(quat_to_rotvec(IDENTITY_QUAT), np.zeros(3))


def test_quat_to_rotmat_is_orthonormal_with_unit_determinant():
    q = np.array([0.2, 0.3, -0.1, 0.927])
    q /= np.linalg.norm(q)
    R = quat_to_rotmat(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotvec_and_rotmat_agree_on_the_same_rotation():
    """The log map and the matrix form must describe one rotation, not two."""
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    angle = 0.9
    q = np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])
    v = np.array([0.3, -0.2, 0.8])

    rotated_by_matrix = quat_to_rotmat(q) @ v
    # Rodrigues' formula applied to the rotation vector from the log map.
    rv = quat_to_rotvec(q)
    theta = np.linalg.norm(rv)
    k = rv / theta
    rotated_by_rodrigues = (
        v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
    )
    assert np.allclose(rotated_by_matrix, rotated_by_rodrigues, atol=1e-12)


@pytest.mark.parametrize(
    "normal",
    [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),          # the seed-swap branch
        np.array([0.3, -0.4, 0.866]),
    ],
)
def test_inplane_basis_is_right_handed_and_orthonormal(normal):
    normal = normal / np.linalg.norm(normal)
    x_axis, y_axis = build_inplane_basis(normal)
    for a in (x_axis, y_axis):
        assert np.isclose(np.linalg.norm(a), 1.0)
        assert abs(np.dot(a, normal)) < 1e-12
    assert abs(np.dot(x_axis, y_axis)) < 1e-12
    assert np.allclose(np.cross(x_axis, y_axis), normal, atol=1e-12)


def _planar_contact(normal, n=400, noise=0.0, seed=0):
    """Points on a plane with the given normal, plus force pushing along it."""
    rng = np.random.default_rng(seed)
    x_axis, y_axis = build_inplane_basis(normal)
    uv = rng.uniform(-0.1, 0.1, size=(n, 2))
    pts = uv[:, :1] * x_axis + uv[:, 1:] * y_axis + np.array([0.4, 0.0, 0.3])
    if noise:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    force = np.tile(normal * 10.0, (n, 1))
    return pts, force


def test_fit_recovers_a_known_plane_normal():
    normal = np.array([0.3, -0.2, 0.93])
    normal /= np.linalg.norm(normal)
    pts, force = _planar_contact(normal)

    fit = fit_contact_frame(pts, force, contact_force_threshold=8.0)

    assert fit["ok"], fit.get("reason")
    assert fit["planarity_ratio"] < 1e-6
    # R's third column is the normal, by the documented return convention.
    assert np.allclose(fit["R"][:, 2], normal, atol=1e-8)


def test_fit_orients_the_normal_along_the_mean_contact_force():
    """The normal's SIGN is not determined by SVD, so it is fixed by the force.

    Flipping the measured force must flip the returned normal; otherwise every
    downstream sign -- pose error, stiffness, the whole impedance law -- would
    silently invert on some demonstrations and not others.
    """
    normal = np.array([0.0, 0.0, 1.0])
    pts, force = _planar_contact(normal)

    forward = fit_contact_frame(pts, force, contact_force_threshold=8.0)
    reversed_ = fit_contact_frame(pts, -force, contact_force_threshold=8.0)

    assert forward["ok"] and reversed_["ok"]
    assert np.dot(forward["R"][:, 2], normal) > 0
    assert np.allclose(forward["R"][:, 2], -reversed_["R"][:, 2], atol=1e-8)


def test_fit_is_rejected_when_points_are_not_planar():
    """A non-planar cloud must be refused outright, not returned as degenerate."""
    rng = np.random.default_rng(0)
    pts = rng.normal(0.0, 0.05, size=(500, 3))        # isotropic: no plane at all
    force = np.tile(np.array([0.0, 0.0, 10.0]), (500, 1))

    fit = fit_contact_frame(pts, force, contact_force_threshold=8.0)

    assert not fit["ok"]
    assert fit["R"] is None
    assert "planarity_ratio" in fit["reason"]


def test_fit_is_rejected_when_there_is_too_little_contact():
    normal = np.array([0.0, 0.0, 1.0])
    pts, force = _planar_contact(normal, n=10)

    fit = fit_contact_frame(pts, force, contact_force_threshold=8.0, min_contact_samples=50)

    assert not fit["ok"]
    assert fit["R"] is None
    assert "in-contact samples" in fit["reason"]


def test_firm_threshold_excludes_light_contact_from_the_plane_fit():
    """The two contact thresholds exist because light contact is not planar.

    A firm threshold must discard the non-planar light-contact samples that a
    low threshold would admit, which is the whole reason the extraction config
    carries two separate thresholds rather than one.
    """
    normal = np.array([0.0, 0.0, 1.0])
    firm_pts, firm_force = _planar_contact(normal, n=300)
    rng = np.random.default_rng(1)
    # Light contact: scattered off the plane, and pressing only gently.
    light_pts = rng.normal(0.0, 0.05, size=(300, 3)) + np.array([0.4, 0.0, 0.3])
    light_force = np.tile(normal * 3.0, (300, 1))

    pts = np.vstack([firm_pts, light_pts])
    force = np.vstack([firm_force, light_force])

    lenient = fit_contact_frame(pts, force, contact_force_threshold=2.0)
    firm = fit_contact_frame(pts, force, contact_force_threshold=8.0)

    assert not lenient["ok"], "a 2 N threshold should admit the non-planar light contact"
    assert firm["ok"], "an 8 N threshold should see only the planar firm contact"
    assert np.allclose(firm["R"][:, 2], normal, atol=1e-6)


def test_numerical_differentiation_recovers_a_known_derivative():
    t = np.linspace(0.0, 2.0, 2001)
    x = np.column_stack([np.sin(2 * np.pi * t), 0.5 * t])
    dx = numerically_differentiate(t, x, smooth_window=1)

    interior = slice(5, -5)
    assert np.allclose(dx[interior, 0], 2 * np.pi * np.cos(2 * np.pi * t)[interior], atol=1e-3)
    assert np.allclose(dx[interior, 1], 0.5, atol=1e-6)
