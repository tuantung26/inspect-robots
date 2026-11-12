"""VELMA pick-and-place benchmark and scorer."""

from __future__ import annotations

import random
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


@dataclass(frozen=True)
class VelmaThreeCubesStacked:
    """Score the three-cube task from VELMA's server-side stack flag."""

    name: str = "velma_three_cubes_stacked"

    def __call__(self, record: Record, target: Target | None) -> Score:
        """Require a stable stack and three distinct verified object pickups."""
        del target
        final = record.steps[-1].result.info if record.steps else {}
        stacked = bool(final.get("stacked"))
        grasps = sum(
            1
            for step in record.steps
            if step.action.meta.get("velma_tool_call") == "grasp"
            and (
                step.result.info.get("holding_object") is True
                or step.result.info.get("is_grasping") is True
            )
        )
        identities = {
            step.result.info.get("active_object")
            for step in record.steps
            if step.action.meta.get("velma_tool_call") == "grasp"
            and step.result.info.get("holding_object") is True
            and step.result.info.get("active_object")
        }
        success = stacked and grasps >= 3 and (not identities or len(identities) >= 3)
        return Score(
            value=success,
            explanation=(
                "three cubes were removed and stacked"
                if success
                else f"stack not complete (stacked={stacked}, successful grasps={grasps})"
            ),
        )


@scorer("velma-three-cubes-stacked")
def velma_three_cubes_stacked() -> VelmaThreeCubesStacked:
    """Create the three-cube stacking scorer."""
    return VelmaThreeCubesStacked()


_DEFAULT_STACK_INSTRUCTION = (
    "There are three 30 mm cubes on the table. Their actual positions are unknown. "
    "Nominal SEARCH SEEDS only: red (0.085, -0.255), green (0.115, -0.255), "
    "yellow (0.100, -0.285), metres. These are not grasp coordinates; the cubes may "
    "have moved or exchanged positions. Pick red, then green, then yellow and "
    "stack them at (0.200, -0.150): first at z=0.015, second at "
    "z=0.045, third at z=0.075. Use mode='stack' for every velma_place. "
    "Before EVERY grasp call velma_measure_cube(colour, seed_x, seed_y, width_mm=30). "
    "Use its fresh measured xyz and set grasp_height_m to its measured z_m. "
    "Check holding_object after grasp; is_grasping is instantaneous pad contact. "
    "Initial coordinates are setup inputs, not live measurements. If a cube moves "
    "or grasp fails, call velma_measure_cube again for that colour "
    "before retrying. Pass the measured centre, grasp height and width. Never retry "
    "stale coordinates after a bump. At most two attempts per cube. Exactly one "
    "tool call per turn. For this known-size cube task do not call find_object "
    "or triangulate: velma_measure_cube already measures the centre in calibrated "
    "views. Do not call move_joints or open_gripper after grasp. "
    "Call done ONLY when stacked=true; otherwise give_up."
)


def velma_stack_three_task(
    *,
    instruction: str | None = None,
    max_steps: int = 40,
    epochs: int = 1,
    layout: str = "original",
    layout_seed: int = 0,
) -> Task:
    """Build the three-cube stacking task for the VELMA simulator."""
    cube_x, cube_y, cube_z = 0.085, -0.255, 0.035
    cube2_x, cube2_y, cube2_z = 0.115, -0.255, 0.035
    cube3_x, cube3_y, cube3_z = 0.100, -0.285, 0.035
    if layout not in ("original", "shifted", "permuted", "random"):
        raise ValueError("layout must be original, shifted, permuted or random")
    points = [(cube_x, cube_y), (cube2_x, cube2_y), (cube3_x, cube3_y)]
    if layout == "shifted":
        points = [(x + 0.010, y + 0.004) for x, y in points]
    elif layout == "permuted":
        # Six millimetres of source clearance avoids zero-gap contact impulses
        # when the solver's geom order changes under a colour permutation.
        points = [(0.1 + (x - 0.1) * 1.2, -0.265 + (y + 0.265) * 1.2) for x, y in points]
        points = [points[2], points[0], points[1]]
    elif layout == "random":
        points = [(0.1 + (x - 0.1) * 1.2, -0.265 + (y + 0.265) * 1.2) for x, y in points]
        generator = random.Random(layout_seed)
        generator.shuffle(points)
        dx, dy = generator.uniform(-0.008, 0.008), generator.uniform(-0.006, 0.006)
        points = [(x + dx, y + dy) for x, y in points]
    (cube_x, cube_y), (cube2_x, cube2_y), (cube3_x, cube3_y) = points
    container_x, container_y, container_z = 0.500, 0.500, 0.0
    stack_x, stack_y = 0.200, -0.150
    if instruction is None:
        instruction = _DEFAULT_STACK_INSTRUCTION
    if not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    metadata = {
        "layout": layout,
        "layout_seed": layout_seed,
        "cube_x": cube_x,
        "cube_y": cube_y,
        "cube_z": cube_z,
        "cube2_x": cube2_x,
        "cube2_y": cube2_y,
        "cube2_z": cube2_z,
        "cube3_x": cube3_x,
        "cube3_y": cube3_y,
        "cube3_z": cube3_z,
        "container_x": container_x,
        "container_y": container_y,
        "container_z": container_z,
        "stack_x": stack_x,
        "stack_y": stack_y,
        "stack_z1": 0.015,
        "stack_z2": 0.045,
        "stack_z3": 0.075,
    }
    return Task(
        name="velma-stack-three",
        scenes=[
            Scene(
                id="three-cubes-stack",
                instruction=instruction,
                target=Target(kind="stacked", spec={"count": 3}),
                init_seed=0,
                metadata=metadata,
            )
        ],
        scorer=velma_three_cubes_stacked(),
        max_steps=max_steps,
        epochs=Epochs(count=epochs, reducer="mean"),
    )
