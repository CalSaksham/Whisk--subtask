"""Tests for arm/trajectory.py"""

import pytest
from arm.grasp_configs import GRASP_CONFIGS
from arm.trajectory import (
    compute_grasp_trajectory,
    compute_place_trajectory,
    compute_trajectory,
)

# Shared fixtures
CURRENT = [0.0, 0.05, 0.0]
TARGET  = [0.3, 0.02, 0.4]
SAFE    = 0.30


# ===========================================================================
# compute_trajectory
# ===========================================================================

class TestComputeTrajectory:
    def test_returns_exactly_3_waypoints(self):
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert len(wps) == 3

    def test_lift_waypoint_is_at_safe_height(self):
        """First waypoint must have y == safe_height."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[0][1] == pytest.approx(SAFE)

    def test_lift_waypoint_preserves_current_xz(self):
        """Lift is straight up — x and z must not change."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[0][0] == pytest.approx(CURRENT[0])
        assert wps[0][2] == pytest.approx(CURRENT[2])

    def test_xz_translation_happens_at_safe_height(self):
        """Second waypoint (XZ transit) must stay at safe_height."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[1][1] == pytest.approx(SAFE)

    def test_xz_of_transit_waypoint_matches_target(self):
        """After transit, x and z must equal target x and z."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[1][0] == pytest.approx(TARGET[0])
        assert wps[1][2] == pytest.approx(TARGET[2])

    def test_final_waypoint_matches_target_pose(self):
        """Last waypoint must equal the full target pose."""
        wps = compute_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[-1] == pytest.approx(TARGET)

    def test_safe_height_clamped_when_too_low(self):
        """safe_height below current y must be silently clamped upward."""
        high_current = [0.0, 0.50, 0.0]   # arm is already above requested safe_height
        wps = compute_trajectory(high_current, TARGET, safe_height=0.10)
        # All transit waypoints must be at or above the current y (0.50)
        assert wps[0][1] >= high_current[1] - 1e-9
        assert wps[1][1] >= high_current[1] - 1e-9

    def test_same_start_and_end_pose_is_valid(self):
        """Edge case: current == target must not crash and must reach target."""
        pose = [0.3, 0.02, 0.4]
        wps  = compute_trajectory(pose, pose, safe_height=SAFE)
        assert len(wps) >= 1
        assert wps[-1] == pytest.approx(pose)


# ===========================================================================
# compute_grasp_trajectory
# ===========================================================================

class TestComputeGraspTrajectory:
    def _config(self):
        return GRASP_CONFIGS["matcha_cup"]

    def test_returns_exactly_4_waypoints(self):
        wps = compute_grasp_trajectory(CURRENT, TARGET, self._config())
        assert len(wps) == 4

    def test_lift_waypoint_at_safe_height(self):
        wps = compute_grasp_trajectory(CURRENT, TARGET, self._config(), safe_height=SAFE)
        assert wps[0][1] == pytest.approx(SAFE)

    def test_transit_waypoint_at_safe_height(self):
        wps = compute_grasp_trajectory(CURRENT, TARGET, self._config(), safe_height=SAFE)
        assert wps[1][1] == pytest.approx(SAFE)

    def test_approach_waypoint_respects_offset(self):
        """Third waypoint (pre-grasp hover) must be exactly approach_height_offset above target."""
        config = self._config()
        wps = compute_grasp_trajectory(CURRENT, TARGET, config, safe_height=SAFE)
        expected_approach_y = TARGET[1] + config["approach_height_offset"]
        assert wps[2][1] == pytest.approx(expected_approach_y)

    def test_grasp_waypoint_at_grasp_height_offset(self):
        """Final waypoint must be exactly grasp_height_offset above target."""
        config = self._config()
        wps = compute_grasp_trajectory(CURRENT, TARGET, config, safe_height=SAFE)
        expected_grasp_y = TARGET[1] + config["grasp_height_offset"]
        assert wps[-1][1] == pytest.approx(expected_grasp_y)

    def test_all_post_lift_waypoints_share_target_xz(self):
        """Waypoints 1–3 must all have target x and z (arm is above object)."""
        wps = compute_grasp_trajectory(CURRENT, TARGET, self._config(), safe_height=SAFE)
        for i, wp in enumerate(wps[1:], start=1):
            assert wp[0] == pytest.approx(TARGET[0]), f"waypoint {i} x mismatch"
            assert wp[2] == pytest.approx(TARGET[2]), f"waypoint {i} z mismatch"


# ===========================================================================
# compute_place_trajectory
# ===========================================================================

class TestComputePlaceTrajectory:
    def test_returns_exactly_3_waypoints(self):
        wps = compute_place_trajectory(CURRENT, TARGET)
        assert len(wps) == 3

    def test_lifts_to_safe_height(self):
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[0][1] == pytest.approx(SAFE)

    def test_transit_waypoint_at_safe_height(self):
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[1][1] == pytest.approx(SAFE)

    def test_final_y_is_place_height_above_target(self):
        """Arm stops at place_height_offset above target so object is set down gently."""
        offset = 0.02
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE, place_height_offset=offset)
        assert wps[-1][1] == pytest.approx(TARGET[1] + offset)

    def test_final_xz_matches_target(self):
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE)
        assert wps[-1][0] == pytest.approx(TARGET[0])
        assert wps[-1][2] == pytest.approx(TARGET[2])

    def test_custom_place_height_offset(self):
        """Caller-specified offset must be honoured."""
        wps = compute_place_trajectory(CURRENT, TARGET, safe_height=SAFE, place_height_offset=0.05)
        assert wps[-1][1] == pytest.approx(TARGET[1] + 0.05)
