"""
Real SO-101 arm executor using the LeRobot SDK.

Drop-in replacement for MockExecutor — implements the same three-method
Executor Protocol so nothing else in the codebase changes.

Setup
-----
    pip install lerobot          # HuggingFace LeRobot SDK
    # follow SO-101 hardware setup: https://github.com/huggingface/lerobot

Usage (swap into main.py)
--------------------------
    from arm.lerobot_executor import LeRobotExecutor
    executor = LeRobotExecutor()
    executor.connect()

    def execute(tool_call, pose_map):
        return dispatch_tool(tool_call, pose_map, executor)

    history = run_agent_loop(..., execute_tool_fn=execute, ...)

    executor.disconnect()

Coordinate frame
----------------
    x — robot right   (positive)
    y — robot up      (positive)
    z — robot forward (positive, away from base)
    All values in metres.

Safety
------
    WORKSPACE_LIMITS defines the Cartesian bounding box the arm will refuse
    to enter.  Tune these to your actual table layout before running.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Workspace safety limits  (metres, robot base frame)
# Edit these to match your physical table setup BEFORE first run.
# ---------------------------------------------------------------------------
WORKSPACE_LIMITS = {
    "x": (-0.10, 0.50),   # left / right
    "y": (0.00,  0.45),   # floor / ceiling (0 = table surface)
    "z": (0.20,  0.70),   # near / far from base
}

# Cartesian movement speed — lower is safer for first runs.
# Units depend on your SDK (often m/s or a 0–1 scale).
DEFAULT_SPEED = 0.05   # conservative; increase once you trust the trajectories

# How long to wait (seconds) for a waypoint move to complete before timing out.
MOVE_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class LeRobotExecutor:
    """
    SO-101 arm executor.

    Implements the :class:`arm.executor.Executor` Protocol so it is a
    drop-in replacement for MockExecutor with no changes elsewhere.

    All methods return ``{"status": "success"}`` or ``{"status": "error",
    "reason": str}`` — same contract as the mock.
    """

    def __init__(self, port: Optional[str] = None, speed: float = DEFAULT_SPEED) -> None:
        """
        Args:
            port:  Serial port the SO-101 is on, e.g. ``"/dev/ttyUSB0"`` or
                   ``"COM3"``.  If None, the SDK auto-detects.
            speed: Cartesian movement speed (tune for your setup).
        """
        self.port  = port
        self.speed = speed
        self._robot = None        # set by connect()
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Connect to the SO-101 and home / enable torque.

        Call this once before running the agent loop.
        """
        # ----------------------------------------------------------------
        # TODO: replace with your actual LeRobot SDK import and init.
        #
        # Example (SDK API subject to change — check lerobot docs):
        #
        #   from lerobot.common.robots.so101 import SO101Robot
        #   self._robot = SO101Robot(port=self.port)
        #   self._robot.connect()
        #   self._robot.go_to_rest()   # move to safe home pose
        # ----------------------------------------------------------------
        raise NotImplementedError(
            "LeRobotExecutor.connect(): fill in your LeRobot SDK calls here.\n"
            "See https://github.com/huggingface/lerobot for SO-101 setup."
        )
        self._connected = True
        logger.info("SO-101 connected on port %s", self.port or "auto")

    def disconnect(self) -> None:
        """Disable torque and close the serial connection cleanly."""
        if self._robot is not None:
            # TODO: self._robot.disconnect()
            pass
        self._connected = False
        logger.info("SO-101 disconnected.")

    # ------------------------------------------------------------------
    # Executor Protocol — three required methods
    # ------------------------------------------------------------------

    def get_end_effector_pose(self) -> list[float]:
        """
        Return current end-effector position as [x, y, z] in metres.

        Reads joint angles from the robot and runs forward kinematics.
        """
        self._require_connected()
        try:
            # ----------------------------------------------------------------
            # TODO: read joint angles and compute FK.
            #
            # Example:
            #   joints = self._robot.get_joint_positions()   # radians
            #   xyz = self._fk(joints)                        # your FK impl
            #   return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            # ----------------------------------------------------------------
            raise NotImplementedError("get_end_effector_pose: implement FK here")
        except Exception as exc:
            logger.error("get_end_effector_pose failed: %s", exc)
            raise

    def move_to(self, waypoint: list[float]) -> dict:
        """
        Move the end-effector to *waypoint* = [x, y, z] in metres.

        Steps:
        1. Safety-check the target is inside WORKSPACE_LIMITS.
        2. Run IK to get target joint angles.
        3. Send joint command; wait for completion.

        Returns:
            ``{"status": "success", "pose": [x, y, z]}`` on success.
            ``{"status": "error", "reason": str}`` if unreachable.
        """
        self._require_connected()

        x, y, z = waypoint

        # 1. Workspace safety check
        limit_error = self._check_workspace(x, y, z)
        if limit_error:
            return {"status": "error", "reason": limit_error}

        try:
            # ----------------------------------------------------------------
            # TODO: implement IK + joint command.
            #
            # Option A — SDK has Cartesian control built in:
            #   self._robot.move_to_cartesian(x, y, z,
            #                                 speed=self.speed,
            #                                 blocking=True)
            #
            # Option B — manual IK then joint command:
            #   joint_angles = self._ik(x, y, z)
            #   if joint_angles is None:
            #       return {"status": "error",
            #               "reason": f"IK failed for waypoint {waypoint}"}
            #   self._robot.set_joint_positions(joint_angles,
            #                                   speed=self.speed,
            #                                   blocking=True)
            # ----------------------------------------------------------------
            raise NotImplementedError("move_to: implement IK + joint command here")

        except TimeoutError:
            return {"status": "error",
                    "reason": f"move_to timed out after {MOVE_TIMEOUT_S}s"}
        except Exception as exc:
            logger.error("move_to(%s) failed: %s", waypoint, exc)
            return {"status": "error", "reason": str(exc)}

        return {"status": "success", "pose": waypoint}

    def set_gripper(self, width: float) -> dict:
        """
        Set gripper jaw opening to *width* metres (0.0 = fully closed).

        The SO-101 gripper is typically controlled by a single servo.
        Map the width (metres) to the servo's position units.

        Returns:
            ``{"status": "success", "gripper_width": width}``
        """
        self._require_connected()

        # Clamp to physical limits
        width = max(0.0, min(width, 0.08))

        try:
            # ----------------------------------------------------------------
            # TODO: convert width → servo position and send command.
            #
            # Example (tune the scale factor for your specific gripper):
            #   servo_pos = int(width / 0.08 * 1000)  # 0–1000 range
            #   self._robot.set_gripper(servo_pos)
            # ----------------------------------------------------------------
            raise NotImplementedError("set_gripper: map width to servo command here")

        except Exception as exc:
            logger.error("set_gripper(%.4f) failed: %s", width, exc)
            return {"status": "error", "reason": str(exc)}

        return {"status": "success", "gripper_width": width}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "LeRobotExecutor is not connected. Call executor.connect() first."
            )

    def _check_workspace(self, x: float, y: float, z: float) -> str | None:
        """
        Return an error string if (x, y, z) is outside WORKSPACE_LIMITS,
        or None if the point is safe.
        """
        checks = [("x", x), ("y", y), ("z", z)]
        for axis, val in checks:
            lo, hi = WORKSPACE_LIMITS[axis]
            if not (lo <= val <= hi):
                return (
                    f"Waypoint {axis}={val:.4f} outside safe range "
                    f"[{lo}, {hi}] — refusing to move."
                )
        return None

    def _ik(self, x: float, y: float, z: float) -> list[float] | None:
        """
        Compute inverse kinematics for target Cartesian position.

        Returns joint angles in radians, or None if the position is
        unreachable.

        Options:
        A) Use a dedicated IK library (e.g. ikpy, pinocchio, or your
           robot SDK's built-in IK solver).
        B) Use a pre-computed lookup table tuned for the SO-101's workspace.

        The SO-101 is a 5- or 6-DOF arm; the exact joint configuration
        depends on your build.  Fill in here once you have SDK access.
        """
        # TODO: implement IK
        # import ikpy.chain
        # chain = ikpy.chain.Chain.from_urdf_file("so101.urdf")
        # angles = chain.inverse_kinematics([x, y, z])
        # return list(angles)
        raise NotImplementedError(
            "_ik: implement inverse kinematics here.\n"
            "Options: ikpy, pinocchio, or the LeRobot SDK's built-in IK."
        )
