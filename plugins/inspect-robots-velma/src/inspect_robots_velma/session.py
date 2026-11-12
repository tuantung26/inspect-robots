"""Local persistent VELMA session API; no UI and no implicit per-message reset."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from inspect_robots.scene import Scene

from .generic import GenericVelmaEmbodiment


class RobotSession:
    """Serialize instructions against one world; LLM conclusions are not verification."""

    def __init__(
        self,
        embodiment: GenericVelmaEmbodiment,
        *,
        model: str,
        status_file: Path,
        max_steps: int = 40,
        policy_factory: Any = None,
    ) -> None:
        self.embodiment = embodiment
        self.model = model
        self.status_file = status_file
        self.max_steps = max_steps
        self.policy_factory = policy_factory
        self.lock = threading.RLock()
        self.history: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {
            "stage": "created",
            "instruction": "",
            "verified": False,
            "session_id": os.environ.get("VELMA_SESSION_ID"),
        }
        self.worker: threading.Thread | None = None
        self._publish()

    def _publish(self, **updates: Any) -> None:
        self.state.update(updates, updated=time.time())
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self.status_file.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        for attempt in range(20):
            try:
                temp.replace(self.status_file)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.02)

    def status(self) -> dict[str, Any]:
        """Read status without taking the robot control lock or rendering images."""
        with self.lock:
            return {**self.state, "history": list(self.history)}

    def start(self) -> dict[str, Any]:
        """Explicit one-time scene reset; repeated start calls leave the world intact."""
        with self.lock:
            if self.state["stage"] != "created":
                return self.status()
            self.embodiment.reset(Scene(id="session", instruction="Awaiting instruction"))
            self._publish(stage="idle")
            return self.status()

    def observe(self) -> dict[str, Any]:
        """Read public scene and current robot state only when no command is executing."""
        with self.lock:
            if self.state["stage"] == "running":
                return {"status": "busy"}
            if self.state["stage"] in ("closed", "created"):
                return {"status": self.state["stage"]}
            return {
                "scene": self.embodiment._request("observe_scene", {}),
                "robot": self.embodiment._request("get_robot_state", {}),
                "cameras": self.embodiment._request("capture_cameras", {}),
            }

    def submit_instruction(self, instruction: str) -> dict[str, Any]:
        """Accept one instruction asynchronously; never queue competing movement."""
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 8000:
            raise ValueError("instruction must contain 1-8000 characters")
        with self.lock:
            if self.state["stage"] == "running":
                return {"status": "busy"}
            if self.state["stage"] in ("created", "closed"):
                raise ValueError("start an open session first")
            self._publish(
                stage="running",
                instruction=instruction,
                verified=False,
                agent_conclusion=None,
                error=None,
            )
            self.worker = threading.Thread(target=self._execute, args=(instruction,), daemon=True)
            self.worker.start()
            return {"status": "accepted"}

    def _execute(self, instruction: str) -> None:
        policy = None
        try:
            if self.policy_factory is None:
                from .generic_policy import ObjectAgentPolicy

                policy = ObjectAgentPolicy(
                    model=self.model, images="always", depth="off", transcript_echo=True
                )
            else:
                policy = self.policy_factory()
            policy.bind(self.embodiment.info)
            policy.bind_tools(self.embodiment.contribute_tools(self.embodiment.info.action_space))
            # Preserve sanitized conversation across requests without depending on
            # private provider-message or pending-motion state in the agent plugin.
            context = (
                "\nPrevious session conversation:\n" + json.dumps(self.history, ensure_ascii=False)
                if self.history
                else ""
            )
            observation = self.embodiment.observe(instruction)
            policy.reset(
                Scene(id="session", instruction=(observation.instruction or instruction) + context)
            )
            conclusion = "step_limit"
            for step in range(self.max_steps):
                observation = replace(observation, extra={**observation.extra, "env_step": step})
                chunk = policy.act(observation)
                if chunk.terminal is not None:
                    conclusion = chunk.terminal.kind
                    break
                for action in chunk.actions:
                    result = self.embodiment.step(action)
                    observation = result.observation
                # Do not end simply because the scene's optional evaluator goal
                # happens to match: subsequent natural-language requests may differ.
            evaluation = self.embodiment._request("evaluate_goal", {})
            transcript = policy.transcript()
            with self.lock:
                self.history.append(
                    {
                        "instruction": instruction,
                        "agent_conclusion": conclusion,
                        "transcript": transcript,
                        "evaluation": evaluation,
                    }
                )
                self._publish(
                    stage="idle",
                    agent_conclusion=conclusion,
                    verified=evaluation.get("verified") is True,
                    evaluation=evaluation,
                )
                self.status_file.with_name("session-history.json").write_text(
                    json.dumps(self.history, ensure_ascii=False), encoding="utf-8"
                )
        except Exception as exc:
            with self.lock:
                self._publish(stage="error", error=f"{type(exc).__name__}: {exc}", verified=False)
        finally:
            if policy is not None and callable(getattr(policy, "close", None)):
                closer = getattr(policy, "close", None)
                if callable(closer):
                    closer()

    def reset(self) -> dict[str, Any]:
        """Explicitly reset physical scene and clear the conversation."""
        with self.lock:
            if self.state["stage"] == "running":
                return {"status": "busy"}
            if self.state["stage"] == "closed":
                return {"status": "closed"}
            self.embodiment.reset(Scene(id="session", instruction="Awaiting instruction"))
            self.history.clear()
            self.status_file.with_name("session-history.json").write_text("[]", encoding="utf-8")
            self._publish(
                stage="idle", instruction="", verified=False, agent_conclusion=None, error=None
            )
            return self.status()

    def close(self) -> dict[str, Any]:
        """Close an idle session. This endpoint is not an emergency-stop mechanism."""
        with self.lock:
            if self.state["stage"] == "running":
                return {"status": "busy"}
            self.embodiment.close()
            self._publish(stage="closed")
            return self.status()


def serve(session: RobotSession, port: int) -> ThreadingHTTPServer:
    """Expose a loopback-only JSON API; reject browser cross-origin mutations."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/session/status":
                self.reply(session.status())
            elif self.path == "/session/observe":
                self.reply(session.observe())
            else:
                self.reply({"error": "not_found"}, 404)

        def do_POST(self) -> None:
            if self.headers.get("Origin") is not None:
                self.reply({"error": "cross_origin_not_allowed"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16000 or not self.headers.get("Content-Type", "").startswith(
                    "application/json"
                ):
                    raise ValueError("Expected application/json body up to 16000 bytes")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected JSON object")
                actions = {
                    "/session/start": session.start,
                    "/session/reset": session.reset,
                    "/session/close": session.close,
                }
                if self.path == "/session/submit_instruction":
                    result = session.submit_instruction(body.get("instruction", ""))
                elif self.path in actions:
                    result = actions[self.path]()
                else:
                    self.reply({"error": "not_found"}, 404)
                    return
                self.reply(result, 409 if result.get("status") == "busy" else 200)
            except (ValueError, TypeError) as exc:
                self.reply({"error": str(exc)}, 400)

        def reply(self, value: Any, code: int = 200) -> None:
            encoded = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args: Any) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    """Run the session service independently of task evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 500:
        parser.error("max-steps must be 1-500")
    session = RobotSession(
        GenericVelmaEmbodiment(url=args.url),
        model=args.model,
        status_file=args.status_file,
        max_steps=args.max_steps,
    )
    session.start()
    server = serve(session, args.port)
    if args.instruction:
        session.submit_instruction(args.instruction)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
