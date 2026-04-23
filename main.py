"""
Entry point for the matcha-making robot agent (mock / development run).

Usage
-----
    cd whisk_agent
    ANTHROPIC_API_KEY=sk-ant-... python main.py

Everything runs against the mock executor and mock pose data — no physical
robot or camera required.  Replace get_mock_poses and mock_execute with real
implementations to run on hardware.
"""

from __future__ import annotations

from agent.loop import run_agent_loop
from agent.tools import dispatch_tool
from arm.executor import MockExecutor
from perception.mock_poses import get_mock_poses


def main() -> None:
    """Wire perception, execution, and the LLM agent loop together."""

    task = (
        "Make matcha: scoop matcha powder from the scoop into the bowl, "
        "add hot water from the cup of water, add ice from the cup of ice, "
        "whisk the mixture until frothy, then pour into the matcha cup."
    )

    # ---- Executor setup ----------------------------------------------------
    # MockExecutor logs every call.  To run on a real SO-101, swap this for a
    # LeRobotExecutor that sends commands over the hardware SDK.
    executor = MockExecutor()

    def mock_execute(tool_call: dict, pose_map: dict) -> dict:
        """
        Thin dispatch wrapper — the only line that needs changing for real HW.

        Signature matches what run_agent_loop expects:
            (tool_call: dict, pose_map: dict) → result: dict
        """
        return dispatch_tool(tool_call, pose_map, executor)

    # ---- Run ---------------------------------------------------------------
    print("=" * 60)
    print("MATCHA ROBOT — MOCK RUN")
    print("=" * 60)
    print(f"Task: {task}\n")

    history = run_agent_loop(
        task=task,
        get_poses_fn=get_mock_poses,
        execute_tool_fn=mock_execute,
        verbose=True,
    )

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Completed in {len(history)} steps")
    print("=" * 60)
    for entry in history:
        action = entry.get("action")
        tool   = action.get("tool", "(none)") if action else "(none)"
        status = entry["result"]["status"]
        print(f"  Step {entry['step']:>2}: {tool:<20} → {status}")


if __name__ == "__main__":
    main()
