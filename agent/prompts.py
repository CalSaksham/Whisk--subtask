"""
Prompt templates and world-state formatting for the matcha-making agent.

Keeping token counts low is important: the state prompt is sent every step,
so format_state() deliberately truncates and summarises rather than dumps
the full history.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are the planning brain of a robotic arm making matcha tea.
Your job: issue exactly ONE tool call per turn as a raw JSON object.

Rules
-----
1. Output ONLY valid JSON — no prose, no markdown, no explanation.
2. Format: {"tool": "<name>", "args": {<key>: <value>}}
3. If a tool takes no arguments use "args": {}.
4. Choose the single best next action given the current world state.
5. Always call done() once the full task is complete.\
"""

# ---------------------------------------------------------------------------
# Tool catalogue (injected into the system message)
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS: str = """\
AVAILABLE TOOLS
===============
move_arm(target_object: str, action: str)
    Move the arm to an object and perform an action.
    target_object: one of the object names shown in OBJECTS.
    action: "grasp"  — descend with gripper open, then close on object
            "place"  — lower held object to surface, open gripper
            "hover"  — move above object without gripping

open_gripper()
    Fully open the gripper. Call before approaching an object to grasp.

close_gripper()
    Fully close the gripper. Call after a place action to release cleanly.

wait(seconds: float)
    Pause for the given number of seconds (e.g. let powder dissolve).

done()
    Signal that the task is fully complete. Ends the planning loop.\
"""


# ---------------------------------------------------------------------------
# State formatter
# ---------------------------------------------------------------------------

def format_state(
    poses: dict[str, list[float]],
    completed_steps: list[dict],
    last_result: dict | None,
    task_description: str,
) -> str:
    """
    Format current world state into a concise LLM prompt (< 500 tokens).

    Conciseness strategies:
    - Poses rounded to 3 decimal places.
    - Only the last 5 completed steps are shown.
    - Result dicts are summarised to status + optional reason.

    Args:
        poses:            Object-name → [x, y, z] from the detector.
        completed_steps:  All steps completed so far; only the last 5 shown.
        last_result:      Result dict from the previous tool call, or None.
        task_description: Natural-language task the robot must complete.

    Returns:
        Formatted string ready to be sent as the user message.
    """
    lines: list[str] = []

    # --- Object poses -------------------------------------------------------
    # Yaw is omitted from the prompt: the LLM doesn't pick approach geometry
    # (GRASP_CONFIGS does), so rendering yaw would just burn tokens.
    lines.append("OBJECTS:")
    for name, pose in poses.items():
        x, y, z = pose[0], pose[1], pose[2]
        lines.append(f"  {name}: x={x:.3f} y={y:.3f} z={z:.3f}")

    # --- Recent history (last 5 steps) -------------------------------------
    recent = completed_steps[-5:]
    if recent:
        lines.append("\nCOMPLETED (last 5):")
        for entry in recent:
            action = entry.get("action") or {}
            tool   = action.get("tool", "?")
            args   = action.get("args", {})
            status = entry.get("result", {}).get("status", "?")
            # Compact args rendering
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            lines.append(f"  [{entry['step']}] {tool}({args_str}) → {status}")

    # --- Last result --------------------------------------------------------
    if last_result is not None:
        status = last_result.get("status", "?")
        reason = last_result.get("reason", "")
        summary = status + (f" — {reason}" if reason else "")
        lines.append(f"\nLAST RESULT: {summary}")

    # --- Remaining task -----------------------------------------------------
    lines.append(f"\nTASK: {task_description}")

    return "\n".join(lines)
