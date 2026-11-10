"""Tests for the VELMA MCP embodiment without a running simulator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_velma.embodiment import VelmaEmbodiment, _mcp_failure
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
    assert calls[1][1]["cube_x"] == 0.1
    assert calls[1][1]["cube_z"] == 0.035
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


def test_grasp_forces_honest_coords_and_measured_width() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_call(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append((tool, dict(arguments)))
        return {"backend": "simulation"} if tool == "get_backend_info" else _state()

    embodiment = VelmaEmbodiment(_call=fake_call)
    observation = embodiment.reset(velma_pick_place_task().scenes[0])
    grasp_tool = next(
        tool
        for tool in embodiment.contribute_tools(embodiment.info.action_space)
        if tool.name == "velma_grasp"
    )
    grasp_tool.run(
        {
            "x_m": 0.21,
            "y_m": 0.04,
            "z_m": 0.02,
            "grasp_height_m": 0.015,
            "object_width_mm": 24.0,
            "trust_coords": False,
        },
        observation,
    )

    assert "grasp_height_m" in grasp_tool.schema["function"]["parameters"]["required"]
    tool, arguments = calls[-1]
    assert tool == "grasp"
    assert arguments["x_m"] == pytest.approx(0.21)
    assert arguments["trust_coords"] is True
    assert arguments["grip_width_mm"] == pytest.approx(34.0)
    assert arguments["grasp_height_m"] == pytest.approx(0.015)


def test_default_task_describes_sphere_out_and_back() -> None:
    task = velma_pick_place_task()
    scene = task.scenes[0]

    assert scene.id == "red-sphere-out-and-back"
    assert scene.target is not None
    assert scene.target.spec["object"] == "sphere"
    assert scene.metadata["cube_z"] == pytest.approx(0.035)
    assert scene.metadata["staging_x"] == pytest.approx(0.21)
    assert scene.metadata["staging_y"] == pytest.approx(-0.15)
    assert "grasp it again" in scene.instruction
    assert "grasp_height_m=0.015" in scene.instruction
    assert "z=0.035" in scene.instruction


def test_task_instruction_can_be_overridden() -> None:
    task = velma_pick_place_task(instruction="Stack the cube on the container.")
    assert task.scenes[0].instruction == "Stack the cube on the container."


def test_task_rejects_empty_instruction() -> None:
    with pytest.raises(ValueError, match="instruction"):
        velma_pick_place_task(instruction="  ")


def test_mcp_failure_unwraps_nested_connection_error() -> None:
    group = ExceptionGroup(
        "task group",
        [ExceptionGroup("inner", [ConnectionRefusedError(10061, "refused")])],
    )
    message = str(_mcp_failure("http://127.0.0.1:3001/mcp", group))
    assert "ConnectionRefusedError" in message
    assert "VELMA server is running" in message


def test_step_waits_for_object_to_leave_and_return() -> None:
    placed = True

    def fake_call(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if tool == "get_backend_info":
            return {"backend": "simulation"}
        return _state(placed=placed)

    embodiment = VelmaEmbodiment(_call=fake_call)
    observation = embodiment.reset(velma_pick_place_task().scenes[0])
    assert observation.extra["cube_in_container"] is True
    noop = Action(data=np.zeros(5), meta={"velma_tool_call": "grasp"})

    assert embodiment.step(noop).terminated is False  # still inside the container

    placed = False
    state = _state(placed=False)["robot_state"]
    assert isinstance(state, dict)
    embodiment._cached_state = state
    assert embodiment.step(noop).terminated is False  # left the container

    placed = True
    state = _state(placed=True)["robot_state"]
    assert isinstance(state, dict)
    embodiment._cached_state = state
    assert embodiment.step(noop).terminated is True  # came back


def test_scorer_reads_final_recorded_step() -> None:
    from inspect_robots.rollout import Record, StepRecord

    obs = Observation()
    record = Record(scene_id="scene", epoch=0, seed=0)
    phases = (
        ("grasp", True, False),
        ("place", False, False),
        ("grasp", True, False),
        ("place", False, True),
    )
    for t, (verb, is_grasping, cube_in_container) in enumerate(phases):
        record.steps.append(
            StepRecord(
                t=t,
                observation=obs,
                action=Action(data=np.zeros(1), meta={"velma_tool_call": verb}),
                result=StepResult(
                    observation=obs,
                    info={
                        "is_grasping": is_grasping,
                        "cube_in_container": cube_in_container,
                    },
                ),
            )
        )

    score = velma_cube_in_container()(record, None)

    assert score.value is True


def test_scorer_rejects_direct_place_without_round_trip() -> None:
    from inspect_robots.rollout import Record, StepRecord

    obs = Observation()
    record = Record(scene_id="scene", epoch=0, seed=0)
    record.steps.append(
        StepRecord(
            t=0,
            observation=obs,
            action=Action(data=np.zeros(1), meta={"velma_tool_call": "place"}),
            result=StepResult(
                observation=obs,
                info={"is_grasping": False, "cube_in_container": True},
            ),
        )
    )

    score = velma_cube_in_container()(record, None)

    assert score.value is False
