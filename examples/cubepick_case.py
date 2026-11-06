"""Run a small, reproducible end-to-end evaluation without robot hardware.

Run with:  uv run python examples/cubepick_case.py
"""

from __future__ import annotations

from inspect_robots import eval
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import episode_length, min_distance_to_goal, success_at_end
from inspect_robots.task import Epochs, Task


def main() -> None:
    """Evaluate the scripted policy on seeded cube layouts and report its metrics."""
    task = Task(
        name="cubepick-case",
        scenes=[
            Scene(
                id=f"layout-{index}",
                instruction="reach the cube",
                target=Target(kind="reach_object", spec={"object": "cube"}),
                init_seed=index,
            )
            for index in range(5)
        ],
        scorer=[success_at_end(), episode_length(), min_distance_to_goal()],
        max_steps=80,
        epochs=Epochs(count=2, reducer="mean"),
    )
    (log,) = eval(
        task,
        ScriptedPolicy(chunk_size=4),
        CubePickEmbodiment(),
        seed=0,
        log_dir="logs/cubepick-case",
    )

    print(f"status:  {log.status}")
    print(f"scenes:  {log.results.total_scenes}   trials: {log.results.total_trials}")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {value:.4g}" if value is not None else f"  {name}: n/a")
    if log.status != "success" or log.results.metrics.get("success_at_end") != 1.0:
        raise RuntimeError(f"CubePick case did not pass: {log.error or log.results.metrics}")


if __name__ == "__main__":
    main()
