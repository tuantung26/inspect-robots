"""Inspect Robots embodiment backed by the VELMA MCP server."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import time as _time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
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
from inspect_robots.spaces import (
    CANONICAL_STATE_UNITS,
    ActionSemantics,
    Box,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.tools import CONTROL, Tool, ToolLibrary, ToolResult
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

from .dashboard_trace import archive_snapshot

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
_LOW = np.deg2rad(np.array([-110.0, -100.0, -96.8, -95.0, -157.2]))
_HIGH = np.deg2rad(np.array([110.0, 100.0, 96.8, 95.0, 162.8]))
_NOOP = np.zeros(5, dtype=np.float64)
# VELMA asks callers to pre-open the jaws to the measured width plus this margin.
_GRASP_WIDTH_MARGIN_MM = 10.0
# Absolute joint targets need one (5,) proprioceptive field; eef_pos and gripper
# complete the declared state so state_keys and the rich StateSpec agree.
_STATE_FIELDS = (
    StateField("joint_pos", (5,), CANONICAL_STATE_UNITS["joint_pos"]),
    StateField("eef_pos", (3,), CANONICAL_STATE_UNITS["eef_pos"]),
    StateField("gripper", (1,), CANONICAL_STATE_UNITS["gripper"]),
)

MCPCall = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
_ToolSpec = tuple[
    str,
    str,
    Mapping[str, Mapping[str, str]],
    tuple[str, ...],
    str,
]


def _mcp_failure(url: str, exc: ExceptionGroup) -> RuntimeError:
    """Flatten anyio's task-group error into the connection or tool failure behind it."""
    leaves: list[BaseException] = []

    def walk(group: ExceptionGroup) -> None:
        for item in group.exceptions:
            if isinstance(item, ExceptionGroup):
                walk(item)
            else:
                leaves.append(item)

    walk(exc)
    detail = "; ".join(f"{type(e).__name__}: {e}" for e in leaves) or str(exc)
    return RuntimeError(
        f"VELMA MCP request to {url} failed: {detail}. "
        "Check that the VELMA server is running (python main.py) and the URL is correct."
    )


async def _call_mcp_async(
    url: str, tool: str, arguments: Mapping[str, Any], timeout_s: float
) -> Mapping[str, Any]:
    """Call one VELMA tool over the MCP Streamable HTTP transport."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import ImageContent, TextContent

    timeout = timedelta(seconds=timeout_s)
    try:
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
    except ExceptionGroup as exc:
        raise _mcp_failure(url, exc) from None
    if result.isError:
        details = [item.text for item in result.content if isinstance(item, TextContent)]
        detail = "; ".join(details)
        raise RuntimeError(f"VELMA MCP tool {tool!r} failed: {detail or 'unknown error'}")
    structured = getattr(result, "structuredContent", None)
    images = [
        {"data": item.data, "mime_type": item.mimeType or "image/jpeg"}
        for item in result.content
        if isinstance(item, ImageContent)
    ]
    payload: dict[str, Any]
    if isinstance(structured, dict):
        payload = dict(structured)
    else:
        payload = {}
        for item in result.content:
            if isinstance(item, TextContent):
                try:
                    decoded = json.loads(item.text)
                except json.JSONDecodeError:
                    payload = {"text": item.text}
                    break
                payload = decoded if isinstance(decoded, dict) else {"result": decoded}
                break
    if images:
        order = payload.get("camera_order", [])
        if isinstance(order, list):
            for i, name in enumerate(order[: len(images)]):
                images[i]["camera"] = str(name)
        payload["_images"] = images
    return payload


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


def _decode_images(raw: Any) -> dict[str, np.ndarray]:
    """Decode VELMA's MCP JPEG frames into the Observation image mapping."""
    if not raw:
        return {}
    try:
        from PIL import Image as PILImage
    except ImportError:
        return {}
    images: dict[str, np.ndarray] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        data = item.get("data")
        if not isinstance(data, str):
            continue
        try:
            frame = PILImage.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        except Exception:
            continue
        images[str(item.get("camera", f"velma_{index}"))] = np.asarray(frame)
    return images


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


def _grasp_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Force VELMA's honest-perception grasp and size the jaws from the measured width."""
    args = dict(arguments)
    # VELMA's grasp defaults to trust_coords=False, which ignores the measured point
    # and drives the grasp from the cube's live simulator pose. An evaluation of
    # perception must use the coordinates the policy measured instead.
    args["trust_coords"] = True
    width = args.get("object_width_mm")
    if isinstance(width, int | float) and not isinstance(width, bool):
        args.setdefault("grip_width_mm", float(width) + _GRASP_WIDTH_MARGIN_MM)
    return args


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
        self._image_dir: Path | None = None
        self._image_step = 0
        self._mcp_events: list[dict[str, Any]] = []
        # A run only succeeds when the object has actually left the container and
        # come back. Starting inside (a retrieval task) must not terminate at step 1.
        self._seen_outside = False
        self._target_kind = "cube_in_container"
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
                state_keys=frozenset({"joint_pos", "eef_pos", "gripper"}),
                state=StateSpec(fields=_STATE_FIELDS),
            ),
            control_hz=10.0,
            is_simulated=True,
            supported_target_kinds=frozenset({"cube_in_container", "stacked"}),
            docs=(
                "VELMA SO-101 MuJoCo simulator. Joint actions are absolute radians in "
                f"this order: {', '.join(_JOINTS)}. Cartesian coordinates use metres "
                "in the arm-base frame (x forward, y left, z up). For known-size "
                "coloured cubes use velma_measure_cube directly. For other objects "
                "use velma_find_object and velma_triangulate. Do not infer metric "
                "coordinates from hidden ground truth or uncalibrated images. "
                "velma_grasp drives the grasp from the measured point you pass, never "
                "from the target's live simulator pose; pass grasp_height_m for spheres. "
                "For a pick-and-place task, grasp the measured object, then measure the "
                "container and place the object inside it."
                " For stacking use velma_measure_cube before each grasp, holding_object "
                "to verify a lift, and velma_place(mode='stack'). Only report done when "
                "stacked=true. Camera views are side, overhead and wrist; a wrist "
                "view can point away from the table. Metric xyz needs calibrated "
                "measurement, not visual guessing. The simulator internally uses "
                "object feedback for execution and collision checking; this is not "
                "a vision-only or hardware policy."
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
        self._target_kind = scene.target.kind if scene.target is not None else "cube_in_container"
        coordinates = {
            key: float(metadata[key])
            for key in (
                "cube_x",
                "cube_y",
                "cube_z",
                "container_x",
                "container_y",
                "container_z",
                "cube2_x",
                "cube2_y",
                "cube2_z",
                "cube3_x",
                "cube3_y",
                "cube3_z",
                "stack_x",
                "stack_y",
            )
            if key in metadata
        }
        self._request("reset_scene", coordinates)
        if os.environ.get("VELMA_CAPTURE_IMAGES", "0").strip() in ("1", "true", "True", "yes"):
            self._start_image_run()
        observation = self._refresh_observation(scene.instruction)
        self._seen_outside = observation.extra.get("cube_in_container") is not True
        return observation

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
        images = self._cached_payload.get("_images")
        if not images:
            images = self._capture_images()
        self._save_images(images, "step")
        observation = self._observation(self._cached_state, None, images)
        inside = self._cached_state.get("cube_in_container") is True
        if not inside:
            self._seen_outside = True
        success = (
            self._cached_state.get("stacked") is True
            if self._target_kind == "stacked"
            else inside and self._seen_outside
        )
        info = {
            key: self._cached_state[key]
            for key in (
                "cube_in_container",
                "is_grasping",
                "holding_object",
                "active_object",
                "obstacle_hit",
                "stacked",
            )
            if key in self._cached_state
        }
        snapshot_id = self._log_mcp("observation", "step", {}, info, images)
        observation = replace(observation, extra={**observation.extra, "mcp_event_id": snapshot_id})
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
                "velma_measure_cube",
                "Measure a known-size upright red/green/yellow cube in calibrated RGB "
                "views. Use before grasp and after any bump. xyz are fresh "
                "measurements; width is a supplied prior. On warning do not grasp. This "
                "avoids guessing metric coordinates from an uncalibrated image.",
                {
                    "colour": {"type": "string", "description": "red, green or yellow"},
                    "seed_x": number,
                    "seed_y": number,
                    "width_mm": number,
                },
                ("colour", "seed_x", "seed_y"),
                "measure_cube",
            ),
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
                "Refine a coarse object location to a grasp-ready 3D point; check "
                "confidence, then pass its x_m, y_m, z_m, and width_mm to velma_grasp.",
                {"prompt": string, "x_m": number, "y_m": number},
                ("prompt", "x_m", "y_m"),
                "triangulate",
            ),
            (
                "velma_grasp",
                "Gently grasp the object at a measured 3D point. The measured point "
                "drives the grasp. For known-size cubes pass the velma_measure_cube "
                "result directly; no extra wrist orbit is needed. For other objects "
                "pass triangulate coordinates and measured width in mm. For a sphere, "
                "explicitly pass grasp_height_m at its equator.",
                {
                    "x_m": number,
                    "y_m": number,
                    "z_m": number,
                    "grasp_height_m": {
                        "type": "number",
                        "description": (
                            "World z in metres for the gripper's grasp point; required for spheres."
                        ),
                    },
                    "object_width_mm": {
                        "type": "number",
                        "description": "Object width measured by VELMA perception, in millimetres.",
                    },
                    "object_yaw_deg": {
                        "type": "number",
                        "description": "Measured box yaw in degrees; omit if unknown.",
                    },
                },
                ("x_m", "y_m", "z_m", "grasp_height_m", "object_width_mm"),
                "grasp",
            ),
            (
                "velma_place",
                "Place a held object. For stacking pass mode='stack'; xyz means desired "
                "OBJECT centre, not TCP. Only place after holding_object=true; "
                "is_grasping is instantaneous contact. Check status and stacked.",
                {
                    "x_m": number,
                    "y_m": number,
                    "z_m": number,
                    "mode": {
                        "type": "string",
                        "description": "stack for precision stacking; container for bin placement",
                    },
                },
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
            payload = _grasp_arguments(arguments) if endpoint == "grasp" else dict(arguments)
            result = self._request(endpoint, payload)
            try:
                state = _robot_state(result)
            except ValueError:
                # Perception verbs return coordinates, not robot state.
                state = _robot_state(self._request("get_robot_state", {}))
            self._cached_payload = result
            self._cached_state = state
            note = json.dumps(result, sort_keys=True, default=str)
            self._log_mcp(
                "tool",
                endpoint,
                payload,
                result.get("message") if isinstance(result, Mapping) else result,
                result.get("_images") if isinstance(result, Mapping) else None,
            )
            action = Action(
                data=_NOOP.copy(),
                meta={"velma_tool_call": endpoint},
            )
            return ToolResult(
                chunk=ActionChunk(actions=[action], control_hz=self.info.control_hz),
                note=note,
            )

        return run

    def _start_image_run(self) -> None:
        """Create a folder for the images the policy actually receives."""
        self._image_dir = Path.cwd() / "logs" / "velma-image-runs" / _time.strftime("%Y%m%d-%H%M%S")
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._image_step = 0

    def _save_images(self, raw_images: Any, tag: str) -> None:
        """Persist the MCP JPEGs so the post-run report can show them."""
        if not raw_images or self._image_dir is None:
            return
        self._image_step += 1
        for index, item in enumerate(raw_images):
            if not isinstance(item, Mapping):
                continue
            data = item.get("data")
            if not isinstance(data, str):
                continue
            try:
                payload = base64.b64decode(data)
            except Exception:
                continue
            path = self._image_dir / f"{self._image_step:03d}-{tag}-{index}.jpg"
            path.write_bytes(payload)

    def _log_mcp(
        self,
        kind: str,
        name: str,
        arguments: Any,
        result: Any,
        images: Any,
    ) -> str:
        """Write the live MCP/tool trace consumed by the VELMA dashboard."""
        image_urls: list[str] = []
        camera_names: list[str] = []
        if isinstance(images, list):
            for item in images:
                if not isinstance(item, Mapping):
                    continue
                data = item.get("data")
                if not isinstance(data, str):
                    continue
                mime = item.get("mime_type") or "image/jpeg"
                image_urls.append(f"data:{mime};base64,{data}")
                camera_names.append(str(item.get("camera", f"image_{len(image_urls)}")))
        event = {
            "t": _time.time(),
            "kind": kind,
            "name": name,
            "args": arguments,
            "result": str(result)[:1200],
            "images": image_urls,
            "camera_names": camera_names,
        }
        snapshot_id = archive_snapshot(event)
        event["id"] = snapshot_id
        self._mcp_events.append(event)
        del self._mcp_events[:-12]
        try:
            path = Path(
                os.environ.get(
                    "VELMA_MCP_LOG",
                    str(Path.cwd() / "logs" / "velma-mcp-live.json"),
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {"updated": _time.time(), "events": self._mcp_events},
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError:
            pass
        return snapshot_id

    def _capture_images(self) -> list[Mapping[str, Any]]:
        """Capture side+wrist frames for the policy observation.

        Headless runs can render both cameras. Viewer mode has no offscreen
        context, so this returns [] and the observation stays text-only.
        Set VELMA_CAPTURE_IMAGES=0 to skip the extra capture.
        """
        if os.environ.get("VELMA_CAPTURE_IMAGES", "1").strip() in ("0", "false", "False", "no"):
            return []
        try:
            payload = self._request("capture_cameras", {})
        except Exception:
            return []
        images = payload.get("_images")
        return list(images) if isinstance(images, list) else []

    def _refresh_observation(self, instruction: str | None) -> Observation:
        """Read a fresh state from VELMA and cache it for the next step."""
        self._cached_payload = self._request("get_robot_state", {})
        self._cached_state = _robot_state(self._cached_payload)
        images = self._cached_payload.get("_images")
        if not images:
            images = self._capture_images()
        self._save_images(images, "obs")
        snapshot_id = None
        if isinstance(self._cached_state, Mapping):
            snapshot_id = self._log_mcp(
                "reset",
                "observation",
                {},
                {"stacked": self._cached_state.get("stacked")},
                images,
            )
        observation = self._observation(self._cached_state, instruction, images)
        return replace(observation, extra={**observation.extra, "mcp_event_id": snapshot_id})

    @staticmethod
    def _observation(
        state: Mapping[str, Any], instruction: str | None, raw_images: Any = None
    ) -> Observation:
        """Build the canonical observation from a VELMA robot-state payload."""
        return Observation(
            images=_decode_images(raw_images),
            state=_state_arrays(state),
            instruction=instruction,
            extra={
                key: state[key]
                for key in (
                    "cube_in_container",
                    "is_grasping",
                    "holding_object",
                    "active_object",
                    "obstacle_hit",
                    "stacked",
                )
                if key in state
            },
        )


def velma_embodiment(**kwargs: Any) -> VelmaEmbodiment:
    """Construct the registered VELMA embodiment."""
    return VelmaEmbodiment(**kwargs)
