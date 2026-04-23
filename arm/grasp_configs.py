"""
Per-object grasp configurations for the SO-101 arm.

All height offsets are *positive* values (in metres) added to the object's
detected y-coordinate.  y is the vertical axis (positive = up).

Keys per config
---------------
approach_height_offset : float
    Y distance above the object pose to hover before descending.
    Larger values give the arm more time to decelerate and align.
grasp_height_offset : float
    Y distance above the object pose at which the gripper closes.
    0.0 = close at the detected centroid; positive = close higher on object.
wrist_angle : float
    Wrist rotation in degrees (0 = straight top-down grip).
grip_width : float
    Gripper jaw opening in metres when approaching the object.
"""

GRASP_CONFIGS: dict[str, dict] = {

    # ------------------------------------------------------------------ matcha_cup
    # A ceramic cup, ~8 cm tall, ~7 cm diameter.
    # Grip just below the rim so fingers clear the opening and the cup
    # can be lifted without catching the edge.
    "matcha_cup": {
        "approach_height_offset": 0.12,   # 12 cm hover — clears the rim comfortably
        "grasp_height_offset":    0.05,   # grip at 5 cm above centroid (below rim)
        "wrist_angle":            0.0,    # straight top-down: symmetric grip on cylinder
        "grip_width":             0.07,   # 7 cm — slightly wider than cup diameter
    },

    # ----------------------------------------------------------------- matcha_bowl
    # Wide, shallow bowl, ~5 cm tall, ~12 cm diameter.
    # Top-down pinch on the outer wall; wide grip to clear the bowl's radius.
    "matcha_bowl": {
        "approach_height_offset": 0.10,   # 10 cm — bowl is shorter, needs less clearance
        "grasp_height_offset":    0.02,   # grip near the rim edge
        "wrist_angle":            0.0,    # top-down
        "grip_width":             0.09,   # wide: fingers must reach around the bowl wall
    },

    # ---------------------------------------------------------------- matcha_whisk
    # Bamboo whisk, ~11 cm handle + 6 cm tine bundle.
    # Grip mid-handle well above the tines; slight wrist tilt to seat the
    # round handle securely between the gripper pads.
    "matcha_whisk": {
        "approach_height_offset": 0.15,   # extra clearance to avoid the tall tine bundle
        "grasp_height_offset":    0.04,   # grip 4 cm above centroid (mid-handle region)
        "wrist_angle":           15.0,    # tilt into the handle's natural lean angle
        "grip_width":             0.03,   # narrow handle (~2.5 cm); close firmly
    },

    # ---------------------------------------------------------------- matcha_scoop
    # Small bamboo scoop, ~14 cm handle, narrow.
    # Grip near the base of the handle so the scoop head stays level
    # and powder doesn't spill during the lift.
    "matcha_scoop": {
        "approach_height_offset": 0.10,
        "grasp_height_offset":    0.01,   # grasp near table level — handle lies flat
        "wrist_angle":           10.0,    # slight tilt to match the scoop's resting angle
        "grip_width":             0.025,  # very narrow handle
    },

    # ------------------------------------------------------------------ cup_of_ice
    # A cup filled with ice — heavier than it looks.
    # Grip lower on the body (not near the rim) for a stable centre of mass.
    "cup_of_ice": {
        "approach_height_offset": 0.12,
        "grasp_height_offset":    0.03,   # grip lower on body for stability under load
        "wrist_angle":            0.0,    # top-down: symmetric grip on cylinder
        "grip_width":             0.07,
    },

    # ----------------------------------------------------------------- cup_of_water
    # Same geometry as the ice cup.  Approach slowly (executor-level speed
    # control) to avoid sloshing; grip geometry is identical.
    "cup_of_water": {
        "approach_height_offset": 0.12,
        "grasp_height_offset":    0.03,   # same as ice cup — matching vessel
        "wrist_angle":            0.0,
        "grip_width":             0.07,
    },
}
