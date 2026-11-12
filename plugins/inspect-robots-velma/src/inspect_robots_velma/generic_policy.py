"""LLM policy exposing measured-object verbs rather than raw joint commands."""

from typing import Any

from inspect_robots_agent.policy import LLMAgentPolicy

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy_adapter import ToolAdapter
from inspect_robots.tools import TERMINATION, ToolLibrary
from inspect_robots.types import ActionChunk, Observation

from .dashboard_trace import archive_agent_line


class ObjectToolAdapter(ToolAdapter):
    """Keep agent termination tools and the robot's declared manipulation verbs."""

    @property
    def tools(self) -> ToolLibrary:
        """Exclude default joint moves; every robot action goes through measured tools."""
        return ToolLibrary(
            t for t in super().tools if t.kind == TERMINATION or t.name.startswith("velma_")
        )


class ObjectAgentPolicy(LLMAgentPolicy):
    """Reuse the existing model/transcript loop with a restricted tool adapter."""

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Bind spaces normally, then replace only the tool-surface adapter."""
        super().bind(embodiment_info)
        adapter = ObjectToolAdapter(images="always")
        adapter.bind(embodiment_info)
        self._adapter = adapter

    def act(self, observation: Observation) -> ActionChunk:
        """Identify the exact input snapshot before inference or tool execution."""
        self._dashboard_snapshot = (
            observation.extra.get("mcp_event_id") if self._images == "always" else None
        )
        return super().act(observation)

    def _echo(self, text: str) -> None:
        super()._echo(text)
        archive_agent_line(text, getattr(self, "_dashboard_snapshot", None))

    def close(self) -> None:
        """Release provider connections when a session request finishes."""
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()


def object_agent_policy(**kwargs: Any) -> ObjectAgentPolicy:
    """Registry factory for manifest evaluation and interactive sessions."""
    return ObjectAgentPolicy(**kwargs)
