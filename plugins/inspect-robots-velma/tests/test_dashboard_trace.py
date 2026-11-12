"""Persistent snapshots and exact observation links without a paid model."""

import base64
import json
import struct
import zlib
from pathlib import Path

import pytest
from inspect_robots_agent.policy import LLMAgentPolicy

from inspect_robots.types import ActionChunk, Observation, Termination
from inspect_robots_velma.dashboard_trace import archive_snapshot
from inspect_robots_velma.embodiment import VelmaEmbodiment
from inspect_robots_velma.generic_policy import ObjectAgentPolicy


def _png(value: int) -> bytes:
    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes([0, value, 0, 0])))
        + chunk(b"IEND", b"")
    )


def test_snapshot_history_survives_live_log_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mcp-live.json"
    monkeypatch.setenv("VELMA_MCP_LOG", str(path))
    robot = VelmaEmbodiment(_call=lambda *args: {})
    for index in range(20):
        robot._log_mcp(
            "observation",
            "step",
            {},
            {},
            [
                {
                    "data": base64.b64encode(_png(index)).decode(),
                    "mime_type": "image/png",
                    "camera": "overhead",
                }
            ],
        )
    assert len(json.loads(path.read_text())["events"]) == 12
    history = [
        json.loads(line) for line in path.with_name("mcp-history.jsonl").read_text().splitlines()
    ]
    assert len(history) == 20
    assert len({event["id"] for event in history}) == 20
    assert all(event["camera_names"] == ["overhead"] for event in history)
    first = history[0]["images"][0].rsplit("/", 1)[1]
    assert (tmp_path / "mcp-images" / first).is_file()
    assert "base64" not in json.dumps(history)


def test_policy_links_output_to_observation_id_not_latest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VELMA_MCP_LOG", str(tmp_path / "mcp-live.json"))
    first = archive_snapshot({"kind": "observation", "images": [], "t": 1})
    archive_snapshot({"kind": "observation", "images": [], "t": 2})
    policy = object.__new__(ObjectAgentPolicy)
    policy._images = "always"
    monkeypatch.setattr(LLMAgentPolicy, "_echo", lambda *args: None)

    def fake_act(self: ObjectAgentPolicy, observation: Observation) -> ActionChunk:
        self._echo('[agent] << tool_call velma_measure_object({"object_id":"red_box"})')
        return ActionChunk(actions=[], terminal=Termination("done"))

    monkeypatch.setattr(LLMAgentPolicy, "act", fake_act)
    policy.act(Observation(extra={"mcp_event_id": first}))
    event = json.loads((tmp_path / "agent-events.jsonl").read_text())
    assert event["snapshot_id"] == first
    assert event["name"] == "velma_measure_object"
