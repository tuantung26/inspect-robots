"""Append-only dashboard metadata with content-addressed, on-demand images."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def trace_dir() -> Path:
    """Keep all artifacts in the same run as the existing MCP log."""
    return Path(
        os.environ.get("VELMA_MCP_LOG", str(Path.cwd() / "logs/velma-mcp-live.json"))
    ).parent


def append_trace(filename: str, event: dict[str, Any]) -> None:
    """A tracing failure must never abort a robot command."""
    try:
        folder = trace_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def archive_snapshot(event: dict[str, Any]) -> str:
    """Persist every event; remove image bytes from the JSON history."""
    identity = uuid.uuid4().hex
    archived = {**event, "id": identity, "images": []}
    refs: list[str] = []
    try:
        folder = trace_dir() / "mcp-images"
        for source in event.get("images", []):
            header, _, encoded = source.partition(",")
            suffix = {
                "data:image/jpeg;base64": ".jpg",
                "data:image/png;base64": ".png",
                "data:image/webp;base64": ".webp",
            }.get(header)
            if suffix is None:
                continue
            try:
                payload = base64.b64decode(encoded, validate=True)
            except ValueError:
                continue
            filename = hashlib.sha256(payload).hexdigest() + suffix
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / filename
            if not target.exists():
                target.write_bytes(payload)
            refs.append("/api/log-images/" + filename)
    except OSError:
        pass
    archived["images"] = refs
    append_trace("mcp-history.jsonl", archived)
    return identity


def archive_agent_line(text: str, snapshot_id: str | None) -> None:
    """Link visible model output to the exact observation passed to act()."""
    event: dict[str, Any] = {"id": uuid.uuid4().hex, "t": time.time(), "snapshot_id": snapshot_id}
    if text.startswith("[agent] << tool_call "):
        value = text[len("[agent] << tool_call ") :]
        name, _, arguments = value.partition("(")
        event.update(kind="tool_call", name=name, text=arguments.removesuffix(")"))
    elif text.startswith("[agent] << "):
        event.update(kind="assistant", text=text[len("[agent] << ") :])
    elif text.startswith("[agent] -- ") and not text.startswith("[agent] -- usage:"):
        event.update(kind="tool_result", text=text[len("[agent] -- ") :])
    else:
        return
    append_trace("agent-events.jsonl", event)
