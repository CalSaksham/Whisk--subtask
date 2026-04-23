"""
Placo-based kinematics backend for :class:`arm.lerobot_executor.LeRobotExecutor`.

Placo (Rhoban's QP-based kinematics/dynamics library,
https://github.com/rhoban/placo) is used to solve inverse kinematics,
forward kinematics, and Jacobians on a URDF description of the SO-101.

This module implements the :class:`KinematicsBackend` Protocol declared
in ``lerobot_executor``.  It does not touch motion control — the
executor composes this backend with a separate motion SDK.

Frame conventions
-----------------
The executor uses ``[x, y, z, yaw]`` end-effector poses where yaw is
rotation about world-vertical (**world +y**).  Placo — like most
robotics tooling — uses Z-up conventions by default.  The URDF for the
SO-101 should be published in robot-base frame with Z up; we therefore
apply a 90° coordinate swap between the executor's "y-up" convention and
Placo's "z-up" URDF internally (see :func:`_pose_to_transform` and
:func:`_transform_to_pose`).

If your URDF already uses y-up, pass ``urdf_is_y_up=True`` and the swap
is skipped.

Placo version sensitivity
-------------------------
Placo's Python API has shifted across releases.  The specific calls
flagged with ``# placo-api:`` comments may need renaming for your
installed version — the logic is stable but the method names aren't.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import placo
    _HAS_PLACO = True
except ImportError:  # pragma: no cover — skip path exercised in tests
    placo = None   # type: ignore[assignment]
    _HAS_PLACO = False


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without Placo)
# ---------------------------------------------------------------------------

def _yaw_to_rotation_y_up(yaw: float) -> np.ndarray:
    """Rotation matrix for *yaw* (radians) about world +y (y-up frame)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[c,    0.0, s  ],
         [0.0,  1.0, 0.0],
         [-s,   0.0, c  ]],
        dtype=float,
    )


def _yaw_to_rotation_z_up(yaw: float) -> np.ndarray:
    """Rotation matrix for *yaw* (radians) about world +z (z-up frame)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[c,   -s,   0.0],
         [s,    c,   0.0],
         [0.0,  0.0, 1.0]],
        dtype=float,
    )


# Coordinate swap sending y-up poses to z-up (and its inverse — the
# matrix is its own inverse).  Swap: (x, y, z)_yup → (x, z, y)_zup.
_Y_UP_TO_Z_UP: np.ndarray = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=float,
)


def _pose_to_transform(
    pose_xyz_yaw: list[float],
    urdf_is_y_up: bool,
) -> np.ndarray:
    """
    Convert an executor-frame ``[x, y, z, yaw]`` pose to a 4x4 homogeneous
    transform expressed in the URDF's base frame.
    """
    x, y, z, yaw = pose_xyz_yaw

    if urdf_is_y_up:
        R = _yaw_to_rotation_y_up(yaw)
        t = np.array([x, y, z], dtype=float)
    else:
        # Executor uses y-up; URDF uses z-up.  Swap axes for both
        # position and rotation (R' = S R S, since S = S^{-1}).
        R_yup = _yaw_to_rotation_y_up(yaw)
        R = _Y_UP_TO_Z_UP @ R_yup @ _Y_UP_TO_Z_UP
        t = _Y_UP_TO_Z_UP @ np.array([x, y, z], dtype=float)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def _transform_to_pose(T: np.ndarray, urdf_is_y_up: bool) -> list[float]:
    """
    Convert a 4x4 homogeneous transform (URDF frame) back to the
    executor's ``[x, y, z, yaw]`` (y-up) representation.

    Yaw is recovered assuming the end-effector's residual pitch and roll
    are negligible (true for a wrist aligned top-down or horizontally).
    """
    R = T[:3, :3]
    t = T[:3,  3]

    if urdf_is_y_up:
        x, y, z = float(t[0]), float(t[1]), float(t[2])
        # yaw about +y: R * [1,0,0]^T = [cos, 0, -sin]^T
        yaw = float(math.atan2(-R[2, 0], R[0, 0]))
    else:
        # Swap back to y-up: p_yup = S p_zup, R_yup = S R S
        t_yup = _Y_UP_TO_Z_UP @ t
        R_yup = _Y_UP_TO_Z_UP @ R @ _Y_UP_TO_Z_UP
        x, y, z = float(t_yup[0]), float(t_yup[1]), float(t_yup[2])
        yaw = float(math.atan2(-R_yup[2, 0], R_yup[0, 0]))

    return [x, y, z, yaw]


# ---------------------------------------------------------------------------
# Placo kinematics backend
# ---------------------------------------------------------------------------

class PlacoKinematics:
    """
    QP-based IK / FK / Jacobian on a URDF description, using Placo.

    Construct once with the URDF path and the end-effector frame name,
    then pass to :class:`LeRobotExecutor`.  This object is stateful
    internally (Placo mutates the robot model's joint configuration)
    but is safe to reuse across calls within a single thread.
    """

    def __init__(
        self,
        urdf_path: str,
        end_effector_frame: str,
        urdf_is_y_up: bool = False,
        max_iters: int = 300,
        position_tolerance: float = 1e-3,   # metres
        orientation_tolerance: float = 1e-2,  # radians (~0.57°)
        seed_joints: Optional[list[float]] = None,
    ) -> None:
        if not _HAS_PLACO:
            raise ImportError(
                "placo is not installed.  `pip install placo` (macOS users "
                "may need to build from source: "
                "https://github.com/rhoban/placo)."
            )

        self._urdf_is_y_up = bool(urdf_is_y_up)
        self._frame        = str(end_effector_frame)
        self._max_iters    = int(max_iters)
        self._pos_tol      = float(position_tolerance)
        self._ori_tol      = float(orientation_tolerance)
        self._seed_joints  = list(seed_joints) if seed_joints is not None else None

        # placo-api: RobotWrapper constructor signature has varied.  The
        # most portable form is a single positional URDF path; flags are
        # version-specific.
        self._robot = placo.RobotWrapper(urdf_path)

    # ------------------------------------------------------------------ IK
    def solve_ik(self, pose_xyz_yaw: list[float]) -> list[float] | None:
        """
        Solve IK for the end-effector pose and return joint positions,
        or None if the QP doesn't converge to tolerance within the
        iteration budget.
        """
        T_target = _pose_to_transform(pose_xyz_yaw, self._urdf_is_y_up)

        if self._seed_joints is not None:
            self._set_joints(self._seed_joints)
        # placo-api: update_kinematics() is the usual forward-propagation call.
        self._robot.update_kinematics()

        # Build a fresh solver per call — Placo tasks persist on the
        # solver instance, so reusing it between calls with different
        # targets would require explicit task removal.
        solver = placo.KinematicsSolver(self._robot)
        # placo-api: add_frame_task(frame_name, T_world_frame_target)
        frame_task = solver.add_frame_task(self._frame, T_target)
        # placo-api: configure(name, priority, position_weight, orientation_weight)
        frame_task.configure(self._frame, "soft", 1.0, 1.0)

        for _ in range(self._max_iters):
            # placo-api: solve(apply=True) integrates the velocity into
            # the robot state.  Some versions expose step() or require
            # a manual `q += qd * dt` update.
            solver.solve(True)
            self._robot.update_kinematics()

            T_current = self._get_frame_transform(self._frame)
            pos_err = float(np.linalg.norm(T_current[:3, 3] - T_target[:3, 3]))
            ori_err = _rotation_angle(T_current[:3, :3].T @ T_target[:3, :3])
            if pos_err < self._pos_tol and ori_err < self._ori_tol:
                return self._get_joints()

        return None

    # ------------------------------------------------------------------ FK
    def forward_kinematics(self, joints: list[float]) -> list[float]:
        self._set_joints(joints)
        self._robot.update_kinematics()
        T = self._get_frame_transform(self._frame)
        return _transform_to_pose(T, self._urdf_is_y_up)

    # ------------------------------------------------------------ Jacobian
    def jacobian(self, joints: list[float]) -> np.ndarray:
        self._set_joints(joints)
        self._robot.update_kinematics()
        # placo-api: frame_jacobian(frame, reference); reference is typically
        # "world" or "local_world_aligned".  The latter expresses the
        # twist at the frame origin aligned with world axes — a common
        # choice when subsequently checking singularities via SVD.
        J = self._robot.frame_jacobian(self._frame, "local_world_aligned")

        if not self._urdf_is_y_up:
            # Rotate the twist into the executor's y-up frame so that the
            # singular-value test matches what the executor expects.
            J = np.asarray(J).copy()
            J[:3, :] = _Y_UP_TO_Z_UP @ J[:3, :]  # linear part
            J[3:, :] = _Y_UP_TO_Z_UP @ J[3:, :]  # angular part
        return np.asarray(J)

    # ======================================================================
    # Internals — isolate version-sensitive calls behind a single method
    # ======================================================================

    def _get_frame_transform(self, frame: str) -> np.ndarray:
        # placo-api: method has been `get_T_world_frame`, `get_frame`,
        # `frame_T` across versions — try the most common first.
        if hasattr(self._robot, "get_T_world_frame"):
            return np.asarray(self._robot.get_T_world_frame(frame))
        return np.asarray(self._robot.frame_T(frame))  # fallback name

    def _get_joints(self) -> list[float]:
        # placo-api: robot.state.q (ndarray) is the standard accessor.
        q = np.asarray(self._robot.state.q).ravel()
        return [float(v) for v in q]

    def _set_joints(self, joints: list[float]) -> None:
        q = np.asarray(joints, dtype=float).ravel()
        self._robot.state.q = q


# ---------------------------------------------------------------------------
# Small pure helper — orientation error from a rotation matrix
# ---------------------------------------------------------------------------

def _rotation_angle(R: np.ndarray) -> float:
    """Return the magnitude of the rotation encoded in *R* (radians)."""
    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)
