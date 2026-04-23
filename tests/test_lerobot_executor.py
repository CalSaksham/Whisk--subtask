"""
Tests for arm/lerobot_executor.py

No real LeRobot SDK is imported — tests inject a stub SDK handle that
satisfies the Protocol contract used by :class:`LeRobotExecutor`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from arm.lerobot_executor import (
    LeRobotExecutor,
    min_singular_value,
    trapezoidal_duration,
)


# ===========================================================================
# Pure helpers
# ===========================================================================

class TestTrapezoidalDuration:
    def test_zero_distance_zero_time(self):
        assert trapezoidal_duration(0.0, v_max=1.0, a_max=1.0) == (0.0, 0.0, 0.0)

    def test_full_trapezoid_has_cruise_phase(self):
        # Long distance → trapezoidal, cruise > 0
        t_a, t_c, t_d = trapezoidal_duration(distance=10.0, v_max=1.0, a_max=1.0)
        assert t_c > 0
        assert t_a == pytest.approx(t_d)
        # accel-phase time = v_max / a_max = 1.0
        assert t_a == pytest.approx(1.0)
        # Distance covered: 2 * 0.5 * 1 + 1 * t_c = 10 → t_c = 9
        assert t_c == pytest.approx(9.0)

    def test_short_move_is_triangular_no_cruise(self):
        # Short distance — never reaches v_max.
        t_a, t_c, t_d = trapezoidal_duration(distance=0.1, v_max=1.0, a_max=1.0)
        assert t_c == 0.0
        assert t_a == pytest.approx(t_d)
        # Triangle: d = a * t_a^2 → t_a = sqrt(d/a) = sqrt(0.1)
        assert t_a == pytest.approx(math.sqrt(0.1))

    def test_handles_negative_distance(self):
        # Direction doesn't change the time profile.
        pos = trapezoidal_duration(1.0, 1.0, 1.0)
        neg = trapezoidal_duration(-1.0, 1.0, 1.0)
        assert pos == neg


class TestMinSingularValue:
    def test_identity_returns_one(self):
        assert min_singular_value(np.eye(3)) == pytest.approx(1.0)

    def test_rank_deficient_returns_zero(self):
        # A rank-2 3x3 matrix has one zero singular value.
        J = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        assert min_singular_value(J) == pytest.approx(0.0)


# ===========================================================================
# Stub SDK used by executor tests
# ===========================================================================

class _StubSDK:
    """
    Minimal SDK implementing the _SDKHandle Protocol.

    * IK: return the pose-as-joints for reachable poses; None for poses
      whose x > 1.0 (represents "out of workspace").
    * FK: inverse of IK — returns the 4-element pose.
    * Jacobian: identity 6x6 unless a "singular joint config" is triggered.
    * execute_joint_trajectory: records what was sent.
    """

    def __init__(self):
        self._joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._gripper = 0.08
        self.executed_trajectories: list[tuple[list[list[float]], list[float]]] = []
        self.singular_zone_joint_0: Optional[tuple[float, float]] = None

    def solve_ik(self, pose_xyz_yaw):
        if pose_xyz_yaw[0] > 1.0:      # out-of-workspace sentinel
            return None
        return [
            float(pose_xyz_yaw[0]),
            float(pose_xyz_yaw[1]),
            float(pose_xyz_yaw[2]),
            float(pose_xyz_yaw[3]),
            0.0,
            0.0,
        ]

    def forward_kinematics(self, joints):
        return [float(joints[0]), float(joints[1]), float(joints[2]), float(joints[3])]

    def jacobian(self, joints):
        # Normally identity (well-conditioned).  If joint[0] falls in the
        # configured singular zone, return a rank-deficient Jacobian.
        if self.singular_zone_joint_0 is not None:
            lo, hi = self.singular_zone_joint_0
            if lo <= joints[0] <= hi:
                J = np.eye(6)
                J[0, 0] = 0.0   # zero a row → rank-5 → min σ = 0
                return J
        return np.eye(6)

    def get_joint_positions(self):
        return list(self._joints)

    def execute_joint_trajectory(self, joint_waypoints, durations):
        # Record and "move" to the final commanded position.
        self.executed_trajectories.append((joint_waypoints, durations))
        if joint_waypoints:
            self._joints = list(joint_waypoints[-1])

    def set_gripper(self, width):
        self._gripper = float(width)


# ===========================================================================
# LeRobotExecutor — behavioural tests
# ===========================================================================

class TestLeRobotExecutorMoveTo:
    def _make(self, **kwargs):
        sdk = _StubSDK()
        # One stub implements both Protocols — pass it as both backends.
        ex = LeRobotExecutor(kinematics=sdk, motion=sdk, **kwargs)
        return sdk, ex

    def test_reachable_pose_succeeds_and_moves(self):
        sdk, ex = self._make()
        result = ex.move_to([0.3, 0.2, 0.4, 0.0])
        assert result["status"] == "success"
        # Stub reports pose = joints[:4]; executor drives joints to [0.3,0.2,0.4,0.0,0,0]
        assert sdk._joints[:4] == pytest.approx([0.3, 0.2, 0.4, 0.0])

    def test_unreachable_pose_errors_without_motion(self):
        sdk, ex = self._make()
        result = ex.move_to([1.5, 0.2, 0.4, 0.0])   # x > 1.0 → IK None
        assert result["status"] == "error"
        assert "IK" in result["reason"] or "unreachable" in result["reason"].lower()
        # No trajectory was streamed.
        assert sdk.executed_trajectories == []

    def test_singular_path_rejected(self):
        sdk, ex = self._make()
        # Put the singular zone between current (joint_0=0) and target (joint_0=0.5).
        sdk.singular_zone_joint_0 = (0.2, 0.3)
        result = ex.move_to([0.5, 0.0, 0.0, 0.0])
        assert result["status"] == "error"
        assert "singularity" in result["reason"].lower()
        assert sdk.executed_trajectories == []

    def test_trajectory_waypoints_are_dense(self):
        """A long move must stream more than one joint waypoint."""
        sdk, ex = self._make(control_rate_hz=100.0)
        ex.move_to([0.4, 0.0, 0.0, 0.0])
        assert len(sdk.executed_trajectories) == 1
        joint_wps, durations = sdk.executed_trajectories[0]
        assert len(joint_wps) >= 2
        # All per-step durations equal.
        assert all(d == pytest.approx(durations[0]) for d in durations)

    def test_trajectory_endpoints_match_target_joints(self):
        sdk, ex = self._make()
        ex.move_to([0.3, 0.1, 0.2, 0.5])
        joint_wps, _ = sdk.executed_trajectories[0]
        # Last streamed waypoint should reach the IK target.
        assert joint_wps[-1][:4] == pytest.approx([0.3, 0.1, 0.2, 0.5])


class TestLeRobotExecutorGripperAndPose:
    def test_set_gripper_passes_through(self):
        sdk = _StubSDK()
        ex = LeRobotExecutor(kinematics=sdk, motion=sdk)
        result = ex.set_gripper(0.04)
        assert result == {"status": "success", "gripper_width": 0.04}
        assert sdk._gripper == pytest.approx(0.04)

    def test_get_end_effector_pose_uses_fk(self):
        sdk = _StubSDK()
        sdk._joints = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0]
        ex = LeRobotExecutor(kinematics=sdk, motion=sdk)
        pose = ex.get_end_effector_pose()
        assert pose == pytest.approx([0.1, 0.2, 0.3, 0.4])
