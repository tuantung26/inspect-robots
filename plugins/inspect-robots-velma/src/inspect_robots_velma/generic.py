"""Manifest tasks and simulator adapter independent of object IDs and recipes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from inspect_robots.rollout import Record
from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import Score
from inspect_robots.spaces import Box
from inspect_robots.task import Task
from inspect_robots.tools import CONTROL, Tool, ToolLibrary
from inspect_robots.types import Observation, StepResult

from .embodiment import VelmaEmbodiment, _tool_schema


class GenericVelmaEmbodiment(VelmaEmbodiment):
    """Measured object-ID tools; setup coordinates never enter an observation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.info = replace(
            self.info,
            supported_target_kinds=frozenset({"manifest_goal"}),
            docs=(
                "SO101 simulator. Call velma_observe_scene for the catalog and capabilities. "
                "Object dimensions and colours are supplied priors. For containers, use "
                "inspect_container before moving and place_in for insertion; never use "
                "place_on or a tabletop command to bypass containment checks. If the "
                "requested set cannot fit, explain the geometry and give_up without moving. "
                "Poses are unknown: call "
                "velma_measure_object before each velma_grasp_object. On warnings re-observe; "
                "never invent coordinates. velma_place_object takes desired OBJECT CENTRE in "
                "robot-base metres, not TCP. Only boxes can be stacked; round objects can be "
                "placed using velma_place_at for a named destination. Prefer velma_place_on "
                "for stacking: it measures the support and computes centre height, avoiding "
                "confusion between object centre and top surface. Round objects can be "
                "placed on the floor. Use place_at for named destinations and place_on "
                "for stacking. Use numeric place_object only for explicit user-specified "
                "table destinations. When stacking at a named destination, pass its "
                "zone_id to place_on so layers align to that destination rather than "
                "accumulating small placement errors. Use numeric place_object for "
                "table destinations; table centre z is half the object's height. They can be "
                "placed on a table. Respect declared capabilities. Check holding_object after "
                "grasp and tool status after placement. Use velma_evaluate_goal to distinguish "
                "verified completion from your own conclusion. The simulator uses internal "
                "evaluation for the configured YAML goal, which can span multiple requests. "
                "Follow the latest user request; do not execute extra tasks merely to make "
                "the configured evaluator true. The simulator uses internal "
                "contact/state feedback for execution and evaluation; this is not vision-only."
            ),
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Explicit reset for evaluation; interactive requests do not invoke this."""
        del seed
        report = self.health()
        if not report.ok:
            raise RuntimeError(report.summary())
        expected = scene.metadata.get("manifest_digest")
        if expected and self._request("observe_scene", {}).get("scene_digest") != expected:
            raise ValueError("Manifest does not match the running VELMA scene")
        self._request("reset_manifest_scene", {})
        self._mcp_events.clear()
        self._target_kind = "manifest_goal"
        self._start_image_run()
        return self.observe(scene.instruction)

    def observe(self, instruction: str | None = None) -> Observation:
        """Fresh state and RGB, with only the server's public object catalog."""
        catalog = self._request("observe_scene", {})
        observation = self._refresh_observation(instruction)
        text = (instruction or "") + "\nScene catalog: " + json.dumps(catalog, ensure_ascii=False)
        return replace(observation, instruction=text)

    def step(self, action: Any) -> StepResult:
        """Acknowledge MCP action and return independent evaluation information."""
        # Raw framework move_joints is not enabled for manifest manipulation.
        if action.meta.get("velma_tool_call") is None:
            if action.terminal is not None:
                evaluation = self._request("evaluate_goal", {})
                return StepResult(
                    observation=self.observe(),
                    terminated=True,
                    termination_reason=action.terminal.kind,
                    info={"evaluation": evaluation},
                )
            raise ValueError("Use measured object tools rather than raw joint actions")
        result = super().step(action)
        evaluation = self._request("evaluate_goal", {})
        verified = evaluation.get("verified") is True
        return replace(
            result,
            terminated=verified,
            termination_reason="success" if verified else None,
            info={**result.info, "evaluation": evaluation},
        )

    def contribute_tools(self, action_space: Box) -> ToolLibrary:
        """Expose a compact object-independent manipulation surface."""
        del action_space
        identity = {"type": "string", "description": "Object ID from observe_scene"}
        number = {"type": "number", "description": "Desired object centre in robot-base metres"}
        specs: tuple[Any, ...] = (
            (
                "inspect_container",
                {"container_id": identity},
                ("container_id",),
                "Check capacity before moving. Feasible=false means the requested set cannot fit.",
            ),
            (
                "place_in",
                {"object_id": identity, "container_id": identity},
                ("object_id", "container_id"),
                "Insert held box; preflight capacity and require full containment.",
            ),
            (
                "place_on",
                {"object_id": identity, "support_id": identity, "zone_id": {"type": "string"}},
                ("object_id", "support_id"),
                "Stack on measured support. Pass zone_id to align layers at a named destination.",
            ),
            (
                "place_at",
                {"object_id": identity, "zone_id": {"type": "string"}},
                ("object_id", "zone_id"),
                "Place held object at a named destination centre from observe_scene.",
            ),
            (
                "observe_scene",
                {},
                (),
                "Read identities, dimensions, capabilities and destinations; no object poses.",
            ),
            (
                "measure_object",
                {"object_id": identity},
                ("object_id",),
                "Measure a fresh object pose in RGB. On warning do not grasp.",
            ),
            (
                "grasp_object",
                {"object_id": identity},
                ("object_id",),
                "Grasp using the latest fresh measurement; check holding_object.",
            ),
            (
                "place_object",
                {
                    "object_id": identity,
                    "x_m": number,
                    "y_m": number,
                    "z_m": number,
                    "mode": {"type": "string", "enum": ["table", "stack"]},
                },
                ("object_id", "x_m", "y_m", "z_m"),
                "Place held object at its desired CENTRE, not TCP. Stack boxes only.",
            ),
            ("get_robot_state", {}, (), "Read robot and grip state."),
            (
                "evaluate_goal",
                {},
                (),
                "Check independent simulator verification; unavailable does not mean success.",
            ),
        )
        return ToolLibrary(
            [
                Tool(
                    name="velma_" + endpoint,
                    schema=_tool_schema("velma_" + endpoint, description, properties, required),
                    run=self._contributed_tool(endpoint),
                    kind=CONTROL,
                    order=200 + i,
                )
                for i, (endpoint, properties, required, description) in enumerate(specs)
            ]
        )


@dataclass(frozen=True)
class ManifestGoalScore:
    """Score independent evaluation only; an agent's done is never proof."""

    name: str = "velma_manifest_goal"

    def __call__(self, record: Record, target: Target | None) -> Score:
        """Require evaluator evidence from the final step."""
        del target
        evaluation = record.steps[-1].result.info.get("evaluation", {}) if record.steps else {}
        return Score(
            value=evaluation.get("verified") is True,
            explanation="Simulator-verified goal"
            if evaluation.get("verified")
            else "Unverified or incomplete",
        )


def manifest_task(*, manifest: str, instruction: str | None = None, max_steps: int = 40) -> Task:
    """Create an eval task from launcher-normalized JSON, without scene-specific code."""
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    text = instruction or payload.get("instruction")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide a nonempty instruction")
    return Task(
        name="velma-task",
        scenes=[
            Scene(
                id="manifest",
                instruction=text,
                target=Target(kind="manifest_goal", spec={}),
                metadata={
                    "manifest_digest": hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest()
                },
            )
        ],
        scorer=ManifestGoalScore(),
        max_steps=max_steps,
    )


def generic_embodiment(**kwargs: Any) -> GenericVelmaEmbodiment:
    """Registry factory for manifest-backed VELMA worlds."""
    return GenericVelmaEmbodiment(**kwargs)
