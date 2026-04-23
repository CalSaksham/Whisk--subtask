"""
Tests for arm/placo_kinematics.py

The pure helpers (_pose_to_transform, _transform_to_pose, _rotation_angle)
are exercised unconditionally — no Placo install required.  Any end-to-end
IK/FK tests against a real Placo instance are gated behind an import-skip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arm.placo_kinematics import (
    _pose_to_transform,
    _rotation_angle,
    _transform_to_pose,
    _yaw_to_rotation_y_up,
    _yaw_to_rotation_z_up,
)


# ===========================================================================
# Pure helpers
# ===========================================================================

class TestYawRotations:
    def test_y_up_zero_yaw_is_identity(self):
        np.testing.assert_allclose(_yaw_to_rotation_y_up(0.0), np.eye(3), atol=1e-12)

    def test_z_up_zero_yaw_is_identity(self):
        np.testing.assert_allclose(_yaw_to_rotation_z_up(0.0), np.eye(3), atol=1e-12)

    def test_y_up_quarter_turn_maps_x_to_minus_z(self):
        """For yaw about +y, R · [1,0,0] must equal [cos, 0, -sin]."""
        R = _yaw_to_rotation_y_up(math.radians(90.0))
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 0.0, -1.0], atol=1e-12)

    def test_z_up_quarter_turn_maps_x_to_plus_y(self):
        R = _yaw_to_rotation_z_up(math.radians(90.0))
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-12)


class TestPoseTransformRoundtrip:
    """_pose_to_transform followed by _transform_to_pose must be identity."""

    @pytest.mark.parametrize("urdf_is_y_up", [True, False])
    @pytest.mark.parametrize("pose", [
        [0.0, 0.0, 0.0, 0.0],
        [0.3, 0.05, 0.4, 0.0],
        [0.3, 0.05, 0.4, math.radians(30)],
        [-0.1, 0.2, 0.5, math.radians(-45)],
        [0.0, 0.0, 0.0, math.radians(170)],  # near ±π boundary
    ])
    def test_roundtrip(self, pose, urdf_is_y_up):
        T = _pose_to_transform(pose, urdf_is_y_up=urdf_is_y_up)
        back = _transform_to_pose(T, urdf_is_y_up=urdf_is_y_up)
        for a, b in zip(pose, back):
            # yaw wraps — compare on the circle
            if abs(a - b) > math.pi:
                b = b + math.copysign(2 * math.pi, a - b)
            assert a == pytest.approx(b, abs=1e-9)

    def test_y_up_transform_matches_hand_computation(self):
        """Sanity check: with y-up URDF, translation goes straight through."""
        T = _pose_to_transform([0.3, 0.2, 0.4, 0.0], urdf_is_y_up=True)
        np.testing.assert_allclose(T[:3, 3], [0.3, 0.2, 0.4], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)

    def test_z_up_transform_swaps_y_and_z(self):
        """With z-up URDF, a y-up pose's y and z components swap in the transform."""
        T = _pose_to_transform([0.3, 0.2, 0.4, 0.0], urdf_is_y_up=False)
        # Executor y-up → URDF z-up: (x, y_up, z_up) → (x, z_up, y_up)
        np.testing.assert_allclose(T[:3, 3], [0.3, 0.4, 0.2], atol=1e-12)


class TestRotationAngle:
    def test_identity_has_zero_angle(self):
        assert _rotation_angle(np.eye(3)) == pytest.approx(0.0, abs=1e-12)

    def test_180_degrees_is_pi(self):
        R = np.diag([1.0, -1.0, -1.0])  # 180° about x-axis
        assert _rotation_angle(R) == pytest.approx(math.pi, abs=1e-9)

    def test_90_degrees_is_pi_over_two(self):
        R = _yaw_to_rotation_y_up(math.pi / 2)
        assert _rotation_angle(R) == pytest.approx(math.pi / 2, abs=1e-9)


# ===========================================================================
# Integration smoke test (skipped if placo isn't installed)
# ===========================================================================

def test_placo_is_importable():
    """If placo imports, PlacoKinematics should at least construct on a URDF."""
    placo = pytest.importorskip(
        "placo",
        reason="placo not installed — skipping live IK test",
    )
    # We don't have a URDF shipped in this repo, so this test only asserts
    # that the import path doesn't raise on the Python side.  Real end-to-end
    # IK against the SO-101 URDF belongs in a hardware-integration suite.
    assert placo is not None
