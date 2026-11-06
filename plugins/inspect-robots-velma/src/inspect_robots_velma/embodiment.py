"""Inspect Robots embodiment backed by the VELMA MCP server."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Any

import numpy as np

from inspect_robots.embodiment import (
    EmbodimentBase,
    EmbodimentInfo,
    HealthCheck,
    HealthReport,
)
from inspect_robots.guard import GuardContribution
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.tools import CONTROL, Tool, ToolLibrary, ToolResult
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
_LOW = np.deg2rad(np.array([-110.0, -100.0, -96.8, -95.0, -157.2]))
_HIGH = np.deg2rad(np.array([110.0, 100.0, 96.8, 95.0, 162.8]))
_NOOP = np.zeros(5, dtype=np.float64)

MCPCall = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
_ToolSpec = tuple[
    str,
    str,
    Mapping[str, Mapping[str, str]],
    tuple[str, ...],
    str,
]


async def _call_mcp_async(
    url: str, tool: str, arguments: Mapping[str, Any], timeout_s: float
) -> Mapping[str, Any]:
    """Call one VELMA tool over the MCP Streamable HTTP transport."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import TextContent

    timeout = timedelta(seconds=timeout_s)
    async with (
        streamablehttp_client(url, timeout=timeout, sse_read_timeout=timeout) as (
            read_stream,
            write_stream,
            _,
        ),
        ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timeout,
        ) as session,
    ):
        await session.initialize()
        result = await session.call_tool(tool, dict(arguments))
    if result.isError:
        details = [item.text for item in result.content if isinstance(item, TextContent)]
        detail = "; ".join(details)
        raise RuntimeError(f"VELMA MCP tool {tool!r} failed: {detail or 'unknown error'}")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for item in result.content:
        if isinstance(item, TextContent):
            try:
                payload = json.loads(item.text)
            except json.JSONDecodeError:
                return {"text": item.text}
            if isinstance(payload, dict):
                return payload
            return {"result": payload}
    return {}


def _default_call(url: str, timeout_s: float) -> MCPCall:
    """Create the synchronous bridge used by Inspect Robots' rollout thread."""

    def call(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return asyncio.run(_call_mcp_async(url, tool, arguments, timeout_s))

    return call


def _robot_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the robot-state object from a VELMA tool response."""
    state = payload.get("robot_state")
    if isinstance(state, Mapping):
        return state
    result = payload.get("result")
    if isinstance(result, Mapping):
        nested = result.get("robot_state")
        if isinstance(nested, Mapping):
            return nested
    raise ValueError("VELMA response did not contain robot_state")


def _state_arrays(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert VELMA's degree/metre state to canonical NumPy observation fields."""
    joints = state.get("joint_angles_deg")
    eef = state.get("end_effector_m")
    if not isinstance(joints, Mapping) or not isinstance(eef, Mapping):
        raise ValueError("VELMA state must include joint_angles_deg and end_effector_m")
    try:
        joint_pos = np.deg2rad(np.asarray([float(joints[name]) for name in _JOINTS]))
        eef_pos = np.asarray([float(eef[axis]) for axis in "xyz"], dtype=np.float64)
        gripper = np.asarray([float(state["gripper_openness_pct"]) / 100.0])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"VELMA returned an incomplete or invalid robot state: {exc}") from exc
    if not bool(np.all(np.isfinite(joint_pos))) or not bool(np.all(np.isfinite(eef_pos))):
        raise ValueError("VELMA returned non-finite joint or end-effector coordinates")
    if not bool(np.all(np.isfinite(gripper))):
        raise ValueError("VELMA returned non-finite gripper state")
    return {"joint_pos": joint_pos, "eef_pos": eef_pos, "gripper": gripper}


def _tool_schema(
    name: str,
    description: str,
    properties: Mapping[str, Mapping[str, str]],
    required: tuple[str, ...],
) -> dict[str, Any]:
    """Build the OpenAI-compatible JSON schema used by agent policies."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: dict(value) for key, value in properties.items()},
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


class VelmaEmbodiment(EmbodimentBase):
    """Connect to a simulation-only VELMA MCP server and expose it as an embodiment."""

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:3001/mcp",
        request_timeout_s: float = 300.0,
        _call: MCPCall | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must be an HTTP(S) MCP endpoint")
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive and finite")
        self._request = _call or _default_call(url, request_timeout_s)
        self._cached_state: Mapping[str, Any] | None = None
        self._cached_payload: Mapping[str, Any] = {}
        self.info = EmbodimentInfo(
            name="velma",
            action_space=Box(
                shape=(5,),
                low=_LOW.copy(),
                high=_HIGH.copy(),
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    frame="base",
                    dim_labels=_JOINTS,
                ),
            ),
            observation_space=ObservationSpace(
                state_keys=frozenset({"joint_pos", "eef_pos", "gripper"})
            ),
            control_hz=10.0,
            is_simulated=True,
            supported_target_kinds=frozenset({"cube_in_container"}),
            docs=(
                "VELMA SO-101 MuJoCo simulator. Joint actions are absolute radians in "
                f"this order: {', '.join(_JOINTS)}. Cartesian coordinates use metres "
                "in the arm-base frame (x forward, y left, z up). Use the contributed "
                "velma_find_object and velma_triangulate tools for measured object "
                "coordinates; do not infer coordinates from hidden ground truth. "
                "For pick-and-place, grasp the measured cube, then measure the "
                "container and place the cube inside it."
            ),
            environment_id="velma-so101-mujoco",
        )

    def health(self) -> HealthReport:
        """Verify that the connected VELMA endpoint is the simulation backend."""
        payload = self._request("get_backend_info", {})
        backend = payload.get("backend")
        if backend != "simulation":
            return HealthReport(
                (HealthCheck("simulation_only", False, f"VELMA reports backend={backend!r}"),)
            )
        return HealthReport((HealthCheck("simulation_only", True, "VELMA MuJoCo backend"),))

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset the VELMA scene from metadata and return its initial state."""
        del seed
        report = self.health()
        if not report.ok:
            raise RuntimeError(report.summary())
        metadata = scene.metadata
        coordinates = {
            key: float(metadata[key])
            for key in (
                "cube_x",
                "cube_y",
                "cube_z",
                "container_x",
                "container_y",
                "container_z",
            )
            if key in metadata
        }
        self._request("reset_scene", coordinates)
        return self._refresh_observation(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Apply one joint target or acknowledge a contributed VELMA tool call."""
        if action.meta.get("velma_tool_call") is None:
            target = np.asarray(action.data, dtype=np.float64).reshape(-1)
            if target.shape != (5,) or not bool(np.all(np.isfinite(target))):
                raise ValueError("VELMA joint action must contain five finite targets in radians")
            degrees = np.rad2deg(np.clip(target, _LOW, _HIGH))
            arguments = {
                f"{joint}_deg": float(value) for joint, value in zip(_JOINTS, degrees, strict=True)
            }
            self._cached_payload = self._request("set_joint_angles", arguments)
            state = _robot_state(self._cached_payload)
            self._cached_state = state
        elif self._cached_state is None:
            raise RuntimeError("VELMA tool action completed without a cached robot state")

        assert self._cached_state is not None
        observation = self._observation(self._cached_state, None)
        success = self._cached_state.get("cube_in_container") is True
        info = {
            key: self._cached_state[key]
            for key in ("cube_in_container", "is_grasping", "obstacle_hit")
            if key in self._cached_state
        }
        return StepResult(
            observation=observation,
            terminated=success,
            termination_reason="success" if success else None,
            info=info,
        )

    def contribute_tools(self, action_space: Box) -> ToolLibrary:
        """Expose VELMA's measured-perception and pick-and-place tools to LLM policies."""
        del action_space
        string = {"type": "string", "description": "Object description used by VELMA perception."}
        number = {"type": "number", "description": "Measured coordinate in metres."}
        tool_specs: tuple[_ToolSpec, ...] = (
            (
                "velma_find_object",
                "Locate a named object using VELMA's camera perception. "
                "Returns measured coordinates.",
                {"prompt": string},
                ("prompt",),
                "find_object",
            ),
            (
                "velma_triangulate",
                "Refine a coarse object location to a grasp-ready 3D point; check confidence.",
                {"prompt": string, "x_m": number, "y_m": number},
                ("prompt", "x_m", "y_m"),
                "triangulate",
            ),
            (
                "velma_grasp",
                "Gently grasp the object at a measured 3D point. Pass its measured width in mm.",
                {
                    "x_m": number,
                    "y_m": number,
                    "z_m": number,
                    "object_width_mm": {
                        "type": "number",
                        "description": "Object width measured by VELMA perception, in millimetres.",
                    },
                },
                ("x_m", "y_m", "z_m", "object_width_mm"),
                "grasp",
            ),
            (
                "velma_place",
                "Place the held object at a measured 3D container position.",
                {"x_m": number, "y_m": number, "z_m": number},
                ("x_m", "y_m", "z_m"),
                "place",
            ),
            (
                "velma_open_gripper",
                "Open the VELMA gripper.",
                {},
                (),
                "open_gripper",
            ),
            (
                "velma_get_robot_state",
                "Read current joints, end-effector pose, gripper, and task status.",
                {},
                (),
                "get_robot_state",
            ),
        )
        tools = []
        for order, (name, description, properties, required, endpoint) in enumerate(tool_specs):
            schema = _tool_schema(name, description, properties, required)
            tools.append(
                Tool(
                    name=name,
                    schema=schema,
                    run=self._contributed_tool(endpoint),
                    kind=CONTROL,
                    order=200 + order,
                )
            )
        return ToolLibrary(tools)

    def contribute_guards(self, action_space: Box) -> GuardContribution:
        """Use core-generated bounds and delta guards for joint-space commands."""
        del action_space
        return GuardContribution()

    def close(self) -> None:
        """Release no persistent resources; each MCP request owns its connection."""

    def _contributed_tool(self, endpoint: str) -> Callable[..., ToolResult]:
        """Create an LLM tool that executes one VELMA verb and records its result."""

        def run(arguments: dict[str, Any], observation: Observation) -> ToolResult:
            del observation
            result = self._request(endpoint, arguments)
            if endpoint == "get_robot_state":
                state = _robot_state(result)
                response = result
            else:
                response = result
                try:
                    state = _robot_state(result)
                except ValueError:
                    state = _robot_state(self._request("get_robot_state", {}))
            self._cached_payload = response
            self._cached_state = state
            note = json.dumps(result, sort_keys=True, default=str)
            action = Action(
                data=_NOOP.copy(),
                meta={"velma_tool_call": endpoint},
            )
            return ToolResult(
                chunk=ActionChunk(actions=[action], control_hz=self.info.control_hz),
                note=note,
            )

        return run

    def _refresh_observation(self, instruction: str | None) -> Observation:
        """Read a fresh state from VELMA and cache it for the next step."""
        self._cached_payload = self._request("get_robot_state", {})
        self._cached_state = _robot_state(self._cached_payload)
        return self._observation(self._cached_state, instruction)

    @staticmethod
    def _observation(state: Mapping[str, Any], instruction: str | None) -> Observation:
        """Build the canonical observation from a VELMA robot-state payload."""
        return Observation(
            state=_state_arrays(state),
            instruction=instruction,
            extra={
                key: state[key]
                for key in ("cube_in_container", "is_grasping", "obstacle_hit")
                if key in state
            },
        )


def velma_embodiment(**kwargs: Any) -> VelmaEmbodiment:
    """Construct the registered VELMA embodiment."""
    return VelmaEmbodiment(**kwargs)
