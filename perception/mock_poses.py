"""
Mock AprilTag pose data for development and testing.

Simulates what a real camera + tag detection pipeline would return.
Poses are [x, y, z] in metres; y is the vertical axis (positive = up).
"""

import numpy as np

MOCK_POSES: dict[str, list[float]] = {
    "matcha_cup":   [0.30, 0.02, 0.40],
    "matcha_bowl":  [0.10, 0.02, 0.40],
    "matcha_whisk": [0.20, 0.02, 0.50],
    "matcha_scoop": [0.35, 0.02, 0.45],
    "cup_of_ice":   [0.00, 0.10, 0.35],   # taller than table objects — elevated cup
    "cup_of_water": [0.05, 0.10, 0.35],   # same height as ice cup, slightly offset
}


def get_mock_poses(noise_std: float = 0.002) -> dict[str, list[float]]:
    """
    Return a copy of MOCK_POSES with small Gaussian noise on each coordinate.

    The noise simulates realistic AprilTag detection jitter from camera
    resolution limits, tag warping, and minor calibration error.

    Args:
        noise_std: Standard deviation of per-axis Gaussian noise in metres.
                   Default 0.002 m (2 mm) matches typical apriltag accuracy.

    Returns:
        Dict mapping object name → noisy [x, y, z] pose in metres.
        Each call returns a new, independently noised dict.
    """
    noisy: dict[str, list[float]] = {}
    for obj, pose in MOCK_POSES.items():
        noise = np.random.normal(0.0, noise_std, size=3)
        noisy[obj] = [float(pose[i] + noise[i]) for i in range(3)]
    return noisy
