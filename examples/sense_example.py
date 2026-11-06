from inspect_robots import eval
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

task = Task(
    name="cubepick-reach",
    scenes=[Scene(id=f"layout-{i}", instruction="reach the cube", init_seed=i) for i in range(5)],
    scorer=success_at_end(),
    max_steps=80,
)

(log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment())
print(log.status)                    # "success"
print(log.results.metrics)           # {"success_at_end": 1.0}