"""VELMA pick-and-place benchmark and scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from inspect_robots.registry import scorer
from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import Score
from inspect_robots.task import Epochs, Task

if TYPE_CHECKING:
    from inspect_robots.rollout import Record


@dataclass(frozen=True)
class VelmaCubeInContainer:
    """Score whether the simulator ended with the cube inside its container."""

    name: str = "velma_cube_in_container"

    def __call__(self, record: Record, target: Target | None) -> Score:
        """Return one only after the sphere was removed, staged, and returned."""
        del target
        expected = (
            ("grasp", True, None),
            ("place", False, False),
            ("grasp", True, False),
            ("place", False, True),
        )
        phase = 0
        for step in record.steps:
            if phase == len(expected):
                break
            verb, grasping, contained = expected[phase]
            if step.action.meta.get("velma_tool_call") != verb:
                continue
            info = step.result.info
            if info.get("is_grasping") is not grasping:
                return Score(
                    value=False,
                    explanation=f"{verb} phase {phase + 1} did not reach the required grip state",
                )
            if contained is not None and info.get("cube_in_container") is not contained:
                return Score(
                    value=False,
                    explanation=f"{verb} phase {phase + 1} had an unexpected container state",
                )
            phase += 1
        success = phase == len(expected)
        return Score(
            value=success,
            explanation=(
                "sphere was removed, staged outside, re-grasped, and returned"
                if success
                else f"round trip incomplete: completed {phase} of {len(expected)} phases"
            ),
        )


@scorer("velma-cube-in-container")
def velma_cube_in_container() -> VelmaCubeInContainer:
    """Create the VELMA simulator's ground-truth placement scorer."""
    return VelmaCubeInContainer()


_DEFAULT_INSTRUCTION = (
    "Take the red sphere from inside the blue container, set it on the table at the "
    "staging point, then grasp it again and return it to the container. Use exactly "
    "one tool call per turn and only these four calls, in order: (1) velma_grasp on "
    "the sphere in the container with the supplied grasp_height_m, (2) velma_place "
    "at the staging point, (3) velma_grasp on the sphere at the staging point with "
    "its grasp_height_m, (4) velma_place back in the container. Do not call "
    "move_joints, sleep, velma_get_robot_state, velma_find_object, or "
    "velma_triangulate; the sphere is at the supplied coordinates and each tool "
    "result already reports is_grasping and cube_in_container. Report completion "
    "only after the final placement confirms cube_in_container=true."
)


def velma_pick_place_task(
    *,
    cube_x: float = 0.1,
    cube_y: float = -0.27,
    cube_z: float = 0.035,
    container_x: float = 0.10,
    container_y: float = -0.27,
    container_z: float = 0.0,
    staging_x: float = 0.21,
    staging_y: float = -0.15,
    staging_z: float = 0.015,
    max_steps: int = 20,
    epochs: int = 1,
    instruction: str | None = None,
) -> Task:
    """Build a reproducible sphere retrieval-and-return task for the VELMA simulator.

    Pass ``instruction`` (CLI: ``-T instruction="..."``) to replace the default
    procedure. The scorer judges VELMA's ``cube_in_container`` state field, which
    remains the simulator's internal name for the target object.
    """
    if instruction is None:
        instruction = (
            f"{_DEFAULT_INSTRUCTION} The sphere starts at x={cube_x:.3f}, "
            f"y={cube_y:.3f}, z={cube_z:.3f} metres inside the container. "
            f"First grasp it there with grasp_height_m={cube_z:.3f} and "
            "object_width_mm=30. Place it outside the container on the table at "
            f"x={staging_x:.3f}, y={staging_y:.3f}, z={staging_z:.3f}. "
            f"Grasp it again there with grasp_height_m={staging_z:.3f} and "
            "object_width_mm=30, then return it to the container at "
            f"x={container_x:.3f}, y={container_y:.3f}, z=0.035."
        )
    if not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    coordinates = {
        "cube_x": cube_x,
        "cube_y": cube_y,
        "cube_z": cube_z,
        "container_x": container_x,
        "container_y": container_y,
        "container_z": container_z,
        "staging_x": staging_x,
        "staging_y": staging_y,
        "staging_z": staging_z,
    }
    return Task(
        name="velma-pick-place",
        scenes=[
            Scene(
                id="red-sphere-out-and-back",
                instruction=instruction,
                target=Target(kind="cube_in_container", spec={"object": "sphere"}),
                init_seed=0,
                metadata=coordinates,
            )
        ],
        scorer=velma_cube_in_container(),
        max_steps=max_steps,
        epochs=Epochs(count=epochs, reducer="mean"),
    )
