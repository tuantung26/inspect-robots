"""Tests for the VELMA MCP embodiment without a running simulator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_velma.embodiment import VelmaEmbodiment
from inspect_robots_velma.task import velma_cube_in_container, velma_pick_place_task


def _state(*, placed: bool = False) -> dict[str, object]:
    return {
        "robot_state": {
            "joint_angles_deg": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 10.0,
                "elbow_flex": 20.0,
                "wrist_flex": 30.0,
                "wrist_roll": 40.0,
            },
            "end_effector_m": {"x": 0.2, "y": 0.05, "z": 0.1},
            "gripper_openness_pct": 75.0,
            "is_grasping": placed,
            "cube_in_container": placed,
            "obstacle_hit": False,
        }
    }


def test_reset_maps_scene_metadata_and_canonical_state() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_call(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append((tool, dict(arguments)))
        if tool == "get_backend_info":
            return {"backend": "simulation"}
        return _state()

    embodiment = VelmaEmbodiment(_call=fake_call)
    task = velma_pick_place_task()
    observation = embodiment.reset(task.scenes[0])

    assert calls[0] == ("get_backend_info", {})
    assert calls[1][0] == "reset_scene"
    assert calls[1][1]["cube_x"] == 0.2
    assert observation.state["joint_pos"].shape == (5,)
    assert observation.state["eef_pos"].tolist() == [0.2, 0.05, 0.1]
    assert observation.state["gripper"].tolist() == [0.75]


def test_refuses_hardware_backend() -> None:
    embodiment = VelmaEmbodiment(_call=lambda tool, arguments: {"backend": "hardware"})

    report = embodiment.health()

    assert not report.ok
    assert "hardware" in report.summary()
    with pytest.raises(RuntimeError, match="hardware"):
        embodiment.reset(velma_pick_place_task().scenes[0])


def test_joint_action_is_sent_as_degrees() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_call(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append((tool, dict(arguments)))
        return {"backend": "simulation"} if tool == "get_backend_info" else _state()

    embodiment = VelmaEmbodiment(_call=fake_call)
    embodiment.reset(velma_pick_place_task().scenes[0])
    embodiment.step(Action(data=np.deg2rad(np.array([1, 2, 3, 4, 5], dtype=np.float64))))

    tool, arguments = calls[-1]
    assert tool == "set_joint_angles"
    assert arguments["shoulder_pan_deg"] == pytest.approx(1.0)
    assert arguments["wrist_roll_deg"] == pytest.approx(5.0)


def test_contributed_place_tool_produces_success_score() -> None:
    placed = False

    def fake_call(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal placed
        if tool == "get_backend_info":
            return {"backend": "simulation"}
        if tool == "place":
            placed = True
            return _state(placed=True)
        if tool == "get_robot_state":
            return _state(placed=placed)
        return _state()

    embodiment = VelmaEmbodiment(_call=fake_call)
    task = velma_pick_place_task()
    observation = embodiment.reset(task.scenes[0])
    place_tool = next(
        tool
        for tool in embodiment.contribute_tools(embodiment.info.action_space)
        if tool.name == "velma_place"
    )
    result = place_tool.run(
        {"x_m": 0.1, "y_m": -0.27, "z_m": 0.05},
        observation,
    )

    assert result.chunk is not None
    step = embodiment.step(result.chunk.actions[0])
    assert step.terminated
    assert step.termination_reason == "success"
    assert step.info["cube_in_container"] is True


def test_scorer_reads_final_recorded_step() -> None:
    from inspect_robots.rollout import Record, StepRecord

    obs = Observation()
    record = Record(scene_id="scene", epoch=0, seed=0)
    record.steps.append(
        StepRecord(
            t=0,
            observation=obs,
            action=Action(data=np.zeros(1)),
            result=StepResult(observation=obs, info={"cube_in_container": True}),
        )
    )

    score = velma_cube_in_container()(record, None)

    assert score.value is True
