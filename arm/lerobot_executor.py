"""
Real SO-101 arm executor using the LeRobot SDK + ikpy for IK.

All three Executor Protocol methods are fully implemented.
Before running on hardware, complete the one-time setup below.

─────────────────────────────────────────────────────────────
ONE-TIME SETUP (do this before first real run)
─────────────────────────────────────────────────────────────

1. Install dependencies:
       pip install lerobot ikpy

2. Get the SO-101 URDF (needed for IK):
       python scripts/download_urdf.py
   This saves  assets/so101.urdf  in the project root.

3. Find your USB port:
       # Mac / Linux:
       ls /dev/tty.usb*        # Mac
       ls /dev/ttyUSB*         # Linux
       # Windows: check Device Manager → COM ports

4. Calibrate once (only needed when you first plug in or reassemble):
       python scripts/calibrate_arm.py --port /dev/tty.usbmodem...

5. Tune WORKSPACE_LIMITS below to match your real table layout.
   Wrong limits = arm slams into table.  Measure before running.

─────────────────────────────────────────────────────────────
SWAP INTO main.py (one line change)
─────────────────────────────────────────────────────────────
    from arm.lerobot_executor import LeRobotExecutor
    executor = LeRobotExecutor(port="/dev/tty.usbmodem...")
    executor.connect()
    # ... run_agent_loop(... execute_tool_fn=lambda tc, pm: dispatch_tool(tc, pm, executor))
    executor.disconnect()
─────────────────────────────────────────────────────────────

Coordinate frame
----------------
    x — robot right (positive)
    y — up          (positive)
    z — forward, away from base (positive)
    All units: metres.

Joint names (SO-101 has 5 arm joints + 1 gripper)
--------------------------------------------------
    shoulder_pan   — base rotation left/right
    shoulder_lift  — shoulder up/down
    elbow_flex     — elbow bend
    wrist_flex     — wrist up/down
    wrist_roll     — wrist rotation
    gripper        — 0 (open) → 100 (closed)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arm joint names (in order, gripper handled separately)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# ---------------------------------------------------------------------------
# Workspace safety limits (metres, robot base frame)
# ⚠️  TUNE THESE TO YOUR TABLE before first real run.
# The arm will refuse to move outside this box.
# ---------------------------------------------------------------------------
WORKSPACE_LIMITS = {
    "x": (-0.10, 0.50),   # left / right
    "y": (0.00,  0.45),   # table surface / max height
    "z": (0.20,  0.70),   # near base / far from base
}

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
DEFAULT_URDF_PATH = Path(__file__).parent.parent / "assets" / "so101.urdf"

# Joint position tolerance to declare "waypoint reached" (radians, ~1 degree)
JOINT_TOLERANCE_RAD = 0.02

# How long to wait for a waypoint to complete before giving up
MOVE_TIMEOUT_S = 10.0

# Seconds between joint-position polls while waiting for move to finish
POLL_INTERVAL_S = 0.05

# Gripper open/close travel in metres (physical measurement of your gripper)
GRIPPER_MAX_WIDTH_M = 0.08   # 8 cm = fully open

# Time to wait after sending a gripper command (it moves slower than the arm)
GRIPPER_SETTLE_S = 0.6


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class LeRobotExecutor:
    """
    SO-101 arm executor — real hardware, drop-in for MockExecutor.

    The LeRobot SDK talks to the arm over USB serial; ikpy solves
    inverse kinematics so the agent can command Cartesian waypoints.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        urdf_path: Path = DEFAULT_URDF_PATH,
        calibrate: bool = False,
    ) -> None:
        """
        Args:
            port:       USB serial port, e.g. ``"/dev/tty.usbmodem58760431551"``.
                        If None the SDK auto-detects (unreliable — prefer explicit).
            urdf_path:  Path to so101.urdf (run scripts/download_urdf.py first).
            calibrate:  Run full calibration on connect.  Only needed on first
                        plug-in or after mechanically reassembling the arm.
        """
        self.port       = port
        self.urdf_path  = Path(urdf_path)
        self.calibrate  = calibrate
        self._robot     = None
        self._chain     = None   # ikpy kinematic chain
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Connect to the SO-101 over USB and load the IK chain.

        Raises if the port is wrong or the URDF is missing.
        """
        self._connect_robot()
        self._load_ik_chain()
        self._connected = True
        logger.info("SO-101 connected (port=%s)", self.port or "auto")

    def disconnect(self) -> None:
        """Release torque and close the serial connection."""
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception as exc:
                logger.warning("Error during disconnect: %s", exc)
        self._connected = False
        logger.info("SO-101 disconnected.")

    # ------------------------------------------------------------------
    # Executor Protocol — the three methods the agent dispatcher calls
    # ------------------------------------------------------------------

    def get_end_effector_pose(self) -> list[float]:
        """
        Return current end-effector [x, y, z] in metres.

        Reads joint angles from the motors and runs forward kinematics
        through the ikpy chain.
        """
        self._require_connected()

        obs    = self._robot.get_observation()
        angles = self._obs_to_angle_array(obs)
        fk     = self._chain.forward_kinematics(angles)   # 4×4 homogeneous matrix
        x, y, z = float(fk[0, 3]), float(fk[1, 3]), float(fk[2, 3])
        return [x, y, z]

    def move_to(self, waypoint: list[float]) -> dict:
        """
        Move the end-effector to *waypoint* = [x, y, z] in metres.

        Steps:
        1. Workspace safety check.
        2. IK  →  joint angles.
        3. Send joint command to robot.
        4. Poll until joints reach target (or timeout).

        Returns:
            ``{"status": "success", "pose": [x,y,z]}``  or
            ``{"status": "error",   "reason": str}``
        """
        self._require_connected()

        x, y, z = waypoint

        # 1. Safety check
        err = self._check_workspace(x, y, z)
        if err:
            return {"status": "error", "reason": err}

        # 2. IK
        target_matrix          = np.eye(4)
        target_matrix[0:3, 3]  = [x, y, z]

        try:
            # ikpy needs an initial guess — use current joint angles
            obs     = self._robot.get_observation()
            current = self._obs_to_angle_array(obs)
            angles  = self._chain.inverse_kinematics_frame(
                target_matrix,
                initial_position=current,
            )
        except Exception as exc:
            return {"status": "error", "reason": f"IK failed: {exc}"}

        # angles[0] and angles[-1] are the fixed base/tip links in ikpy;
        # indices 1..5 are the five real arm joints.
        arm_angles = angles[1 : len(JOINT_NAMES) + 1]

        # 3. Send command
        action = {f"{name}.pos": float(a)
                  for name, a in zip(JOINT_NAMES, arm_angles)}
        self._robot.send_action(action)

        # 4. Wait for completion (SDK send_action is non-blocking)
        deadline = time.time() + MOVE_TIMEOUT_S
        while time.time() < deadline:
            obs    = self._robot.get_observation()
            errors = [abs(obs[f"{j}.pos"] - action[f"{j}.pos"])
                      for j in JOINT_NAMES]
            if max(errors) < JOINT_TOLERANCE_RAD:
                break
            time.sleep(POLL_INTERVAL_S)
        else:
            logger.warning("move_to timed out after %.1fs", MOVE_TIMEOUT_S)

        return {"status": "success", "pose": waypoint}

    def set_gripper(self, width: float) -> dict:
        """
        Set gripper opening to *width* metres (0.0 = closed, 0.08 = fully open).

        The SO-101 SDK uses a 0–100 scale for the gripper servo where
        0 = fully open and 100 = fully closed, so we invert the mapping.

        Returns:
            ``{"status": "success", "gripper_width": width}``
        """
        self._require_connected()

        # Clamp incoming value to physical range
        width = max(0.0, min(width, GRIPPER_MAX_WIDTH_M))

        # Map:  0.0 m (closed) → 100,   0.08 m (open) → 0
        gripper_pos = (1.0 - width / GRIPPER_MAX_WIDTH_M) * 100.0
        gripper_pos = max(0.0, min(100.0, gripper_pos))

        self._robot.send_action({"gripper.pos": gripper_pos})
        time.sleep(GRIPPER_SETTLE_S)   # gripper is slower than the arm joints

        return {"status": "success", "gripper_width": width}

    # ------------------------------------------------------------------
    # Internal: robot connection
    # ------------------------------------------------------------------

    def _connect_robot(self) -> None:
        """Instantiate SOFollower and open the serial connection."""
        from lerobot.robots.so_follower import SOFollower, SO101FollowerRobotConfig

        config = SO101FollowerRobotConfig(
            robot_type="so101_follower",
            id="whisk_arm",
            port=self.port,
        )
        self._robot = SOFollower(config)
        self._robot.connect(calibrate=self.calibrate)

    # ------------------------------------------------------------------
    # Internal: IK chain
    # ------------------------------------------------------------------

    def _load_ik_chain(self) -> None:
        """Load the SO-101 URDF into an ikpy kinematic chain."""
        import ikpy.chain

        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF not found at {self.urdf_path}.\n"
                "Run:  python scripts/download_urdf.py"
            )

        # active_links_mask: the 5 arm joints are active;
        # the implicit base link and the end-effector link are fixed.
        # Adjust the mask if your URDF has a different number of links.
        n_links       = 7   # base + 5 joints + end-effector
        active_mask   = [False] + [True] * len(JOINT_NAMES) + [False]

        self._chain = ikpy.chain.Chain.from_urdf_file(
            str(self.urdf_path),
            active_links_mask=active_mask,
        )
        logger.info("IK chain loaded from %s (%d links)", self.urdf_path, n_links)

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _obs_to_angle_array(self, obs: dict) -> list[float]:
        """
        Convert a robot observation dict to an ikpy-compatible angle list.

        ikpy expects one float per chain link (including fixed base and tip),
        so we pad the 5 real joint angles with zeros at each end.
        """
        real_angles = [obs[f"{j}.pos"] for j in JOINT_NAMES]
        return [0.0] + real_angles + [0.0]

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "Call executor.connect() before using the executor."
            )

    def _check_workspace(self, x: float, y: float, z: float) -> str | None:
        """
        Return an error message if (x, y, z) is outside WORKSPACE_LIMITS,
        or None if the point is safe.
        """
        for axis, val in [("x", x), ("y", y), ("z", z)]:
            lo, hi = WORKSPACE_LIMITS[axis]
            if not (lo <= val <= hi):
                return (
                    f"Waypoint {axis}={val:.4f} m is outside safe range "
                    f"[{lo}, {hi}] m — move refused."
                )
        return None
