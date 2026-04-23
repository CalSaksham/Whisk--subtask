"""
Trajectory planning for the SO-101 arm.

Coordinate convention
---------------------
  x — horizontal, positive to the robot's right
  y — vertical, positive upward
  z — depth, positive away from robot base

All planners use a *Y-ceiling* strategy: the arm always lifts to a safe
clearance height before any XZ movement so it never sweeps through cluttered
workspace at low altitude.  Each function returns a list of [x, y, z]
waypoints that the executor feeds to the arm one by one.

Dependencies: numpy only (no external robotics libraries required).
"""

from __future__ import annotations

import numpy as np


def compute_trajectory(
    current_pose: list[float],
    target_pose: list[float],
    safe_height: float = 0.30,
) -> list[list[float]]:
    """
    Compute a minimal Y-ceiling path from *current_pose* to *target_pose*.

    The planner always clamps *safe_height* to be at or above both the
    current and target y-coordinates, so this is safe even when the caller
    passes a value that is too low.

    Waypoints
    ---------
    1. Lift straight up to *safe_height* — x, z unchanged.
    2. Translate in XZ to directly above the target, staying at *safe_height*.
    3. Descend along Y to the target pose.

    Args:
        current_pose: [x, y, z] of the end-effector right now (metres).
        target_pose:  [x, y, z] of the destination (metres).
        safe_height:  Minimum Y clearance during transit (metres).  Clamped
                      upward if either endpoint is already above this value.

    Returns:
        List of three [x, y, z] waypoints.
    """
    cx, cy, cz = current_pose
    tx, ty, tz = target_pose

    # Never dip below either endpoint during transit.
    safe_y = float(max(safe_height, cy, ty))

    return [
        [cx, safe_y, cz],   # 1. lift
        [tx, safe_y, tz],   # 2. translate XZ at ceiling
        [tx, ty,     tz],   # 3. descend to target
    ]


def compute_grasp_trajectory(
    current_pose: list[float],
    target_pose: list[float],
    grasp_config: dict,
    safe_height: float = 0.30,
) -> list[list[float]]:
    """
    Compute a four-waypoint grasp trajectory respecting the object's config.

    The extra *approach* waypoint (step 3) gives the arm time to slow down
    and ensures the gripper is aligned before making contact with the object.

    Waypoints
    ---------
    1. Lift to *safe_height*.
    2. Translate XZ to directly above the target at *safe_height*.
    3. Descend to ``target_y + approach_height_offset`` (pre-grasp hover).
    4. Descend to ``target_y + grasp_height_offset`` (grasp contact point).

    Args:
        current_pose: [x, y, z] of the end-effector (metres).
        target_pose:  [x, y, z] of the object centroid (metres).
        grasp_config: Dict containing at minimum ``approach_height_offset``
                      and ``grasp_height_offset`` (see arm/grasp_configs.py).
        safe_height:  Transit clearance height (metres).

    Returns:
        List of four [x, y, z] waypoints.
    """
    cx, cy, cz = current_pose
    tx, ty, tz = target_pose

    approach_y = ty + float(grasp_config["approach_height_offset"])
    grasp_y    = ty + float(grasp_config["grasp_height_offset"])

    # Safe ceiling must clear the approach hover point too.
    safe_y = float(max(safe_height, cy, approach_y))

    return [
        [cx, safe_y,    cz],   # 1. lift
        [tx, safe_y,    tz],   # 2. translate XZ
        [tx, approach_y, tz],  # 3. pre-grasp hover
        [tx, grasp_y,   tz],   # 4. grasp contact
    ]


def compute_place_trajectory(
    current_pose: list[float],
    target_pose: list[float],
    safe_height: float = 0.30,
    place_height_offset: float = 0.02,
) -> list[list[float]]:
    """
    Compute a three-waypoint place trajectory for setting an object down.

    The arm stops at *place_height_offset* above the target surface before
    the caller opens the gripper, so the object is placed gently rather than
    dropped from height.

    Waypoints
    ---------
    1. Lift to *safe_height* (clears workspace while holding the object).
    2. Translate XZ to directly above the drop-off point.
    3. Descend to ``target_y + place_height_offset``.

    After reaching waypoint 3 the caller should call ``open_gripper()``.

    Args:
        current_pose:        [x, y, z] of the end-effector (metres).
        target_pose:         [x, y, z] of the drop-off surface point (metres).
        safe_height:         Transit clearance height (metres).
        place_height_offset: How far above surface to stop before releasing
                             (metres).  Default 2 cm prevents hard impacts.

    Returns:
        List of three [x, y, z] waypoints.
    """
    cx, cy, cz = current_pose
    tx, ty, tz = target_pose

    place_y = ty + float(place_height_offset)
    safe_y  = float(max(safe_height, cy, place_y))

    return [
        [cx, safe_y,  cz],   # 1. lift
        [tx, safe_y,  tz],   # 2. translate XZ
        [tx, place_y, tz],   # 3. descend to place height (open gripper next)
    ]
