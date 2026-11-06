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
        """Return one when the last recorded simulator state confirms placement."""
        del target
        info = record.steps[-1].result.info if record.steps else {}
        success = info.get("cube_in_container") is True
        return Score(
            value=success,
            explanation="cube is inside the container" if success else "cube was not placed inside",
        )


@scorer("velma-cube-in-container")
def velma_cube_in_container() -> VelmaCubeInContainer:
    """Create the VELMA simulator's ground-truth placement scorer."""
    return VelmaCubeInContainer()


def velma_pick_place_task(
    *,
    cube_x: float = 0.20,
    cube_y: float = 0.05,
    cube_z: float = 0.015,
    container_x: float = 0.10,
    container_y: float = -0.27,
    container_z: float = 0.0,
    max_steps: int = 20,
    epochs: int = 1,
) -> Task:
    """Build a reproducible pick-and-place task at a validated default VELMA layout."""
    coordinates = {
        "cube_x": cube_x,
        "cube_y": cube_y,
        "cube_z": cube_z,
        "container_x": container_x,
        "container_y": container_y,
        "container_z": container_z,
    }
    return Task(
        name="velma-pick-place",
        scenes=[
            Scene(
                id="red-cube-to-blue-container",
                instruction=(
                    "Pick up the red cube and place it inside the blue container. "
                    "Use velma_find_object to measure the cube, velma_triangulate to "
                    "refine its grasp point, velma_grasp with the measured width, then "
                    "measure the container and place the cube. Report completion only "
                    "when the placement tool confirms cube_in_container=true."
                ),
                target=Target(kind="cube_in_container", spec={"object": "cube"}),
                init_seed=0,
                metadata=coordinates,
            )
        ],
        scorer=velma_cube_in_container(),
        max_steps=max_steps,
        epochs=Epochs(count=epochs, reducer="mean"),
    )
