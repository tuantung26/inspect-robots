"""Persistent sessions and generic tool surface without paid models or hardware."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from inspect_robots.scene import Scene
from inspect_robots.types import ActionChunk, Observation, Termination
from inspect_robots_velma.generic import GenericVelmaEmbodiment, manifest_task
from inspect_robots_velma.session import RobotSession


def test_generic_catalog_tools_and_no_setup_coordinates(tmp_path: Path) -> None:
    state = {
        "joint_angles_deg": dict.fromkeys(
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"), 0
        ),
        "end_effector_m": dict.fromkeys("xyz", 0),
        "gripper_openness_pct": 50,
    }
    calls: list[str] = []

    def request(name: str, args: object) -> dict[str, object]:
        calls.append(name)
        if name == "get_backend_info":
            return {"backend": "simulation"}
        if name == "get_robot_state":
            return {"robot_state": state}
        if name == "observe_scene":
            return {"objects": [{"id": "parcel", "shape": "box", "size_m": [0.03] * 3}]}
        return {}

    robot = GenericVelmaEmbodiment(_call=request)
    observation = robot.reset(Scene(id="test", instruction="Move parcel"))
    assert observation.instruction is not None
    assert "parcel" in observation.instruction
    assert "position_m" not in observation.instruction
    assert "reset_manifest_scene" in calls
    names = robot.contribute_tools(robot.info.action_space).names
    assert "velma_measure_object" in names and "velma_grasp_object" in names
    assert "velma_grasp" not in names
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"instruction": "Move parcel", "objects": []}))
    assert manifest_task(manifest=str(path)).name == "velma-task"


def test_session_busy_history_reset_and_agent_done_not_verified(tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()
    robot = Mock()
    robot.observe.return_value = Observation(instruction="Public scene catalog")
    robot.reset.return_value = Observation()
    robot._request.return_value = {"available": False, "verified": False}
    robot.info = SimpleNamespace(action_space=None)
    policy = Mock()

    def act(obs: Observation) -> ActionChunk:
        entered.set()
        assert release.wait(5)
        return ActionChunk(actions=[], terminal=Termination("done"))

    policy.act.side_effect = act
    policy.transcript.return_value = [{"role": "assistant", "content": "done"}]
    session = RobotSession(
        robot, model="unused", status_file=tmp_path / "session.json", policy_factory=lambda: policy
    )
    session.start()
    session.start()
    assert robot.reset.call_count == 1
    assert session.submit_instruction("Move parcel")["status"] == "accepted"
    assert entered.wait(5)
    assert session.submit_instruction("Other task")["status"] == "busy"
    assert session.reset()["status"] == "busy"
    assert session.close()["status"] == "busy"
    release.set()
    assert session.worker is not None
    session.worker.join(5)
    assert session.status()["agent_conclusion"] == "done"
    assert session.status()["verified"] is False
    assert len(session.history) == 1
    session.submit_instruction("Bring it back")
    assert session.worker is not None
    session.worker.join(5)
    assert robot.reset.call_count == 1
    assert "Move parcel" in policy.reset.call_args.args[0].instruction
    session.reset()
    assert session.history == [] and robot.reset.call_count == 2
    session.close()
    assert session.status()["stage"] == "closed"


def test_object_adapter_excludes_unmeasured_joint_commands() -> None:
    from inspect_robots_velma.generic_policy import ObjectToolAdapter

    robot = GenericVelmaEmbodiment(_call=lambda *args: {})
    adapter = ObjectToolAdapter()
    adapter.bind(robot.info)
    adapter.add_tools(robot.contribute_tools(robot.info.action_space))
    assert "move_joints" not in adapter.tools.names
    assert "sleep" not in adapter.tools.names
    assert "velma_measure_object" in adapter.tools.names
    assert "done" in adapter.tools.names and "give_up" in adapter.tools.names


def test_session_http_api_and_cross_origin_rejection(tmp_path: Path) -> None:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    import pytest

    from inspect_robots_velma.session import serve

    robot = Mock()
    robot._request.return_value = {}
    session = RobotSession(robot, model="unused", status_file=tmp_path / "session.json")
    server = serve(session, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:

        def post(path: str, headers: dict[str, str] | None = None) -> dict[str, object]:
            req = Request(
                base + path, data=b"{}", headers=headers or {"Content-Type": "application/json"}
            )
            with urlopen(req, timeout=5) as response:
                result = json.load(response)
                assert isinstance(result, dict)
                return result

        assert post("/session/start")["stage"] == "idle"
        assert post("/session/start")["stage"] == "idle"
        assert robot.reset.call_count == 1
        with urlopen(base + "/session/observe", timeout=5) as response:
            assert "scene" in json.load(response)
        with pytest.raises(HTTPError) as caught:
            post(
                "/session/reset",
                {"Content-Type": "application/json", "Origin": "https://example.org"},
            )
        assert caught.value.code == 403
        assert post("/session/reset")["history"] == []
        assert post("/session/close")["stage"] == "closed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
