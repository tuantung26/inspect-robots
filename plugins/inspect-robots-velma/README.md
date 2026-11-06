# inspect-robots-velma

An Inspect Robots adapter for the VELMA SO-101 MuJoCo simulator. The adapter
connects to the VELMA Streamable HTTP MCP server, resets the simulator per scene,
exposes its measured-perception and pick-and-place tools to the `agent` policy,
and scores whether the cube ended inside the container.

The adapter deliberately refuses a VELMA server reporting the hardware backend.
This first integration is simulator-only.

## Install

From the Inspect Robots workspace:

```powershell
uv sync --all-packages --extra dev
```

The `velma` embodiment and `velma-pick-place` task then appear in
`inspect-robots list`.

## Start the VELMA simulator

In the VELMA checkout, confirm `config.yaml` uses `backend.type: simulation`,
then start the server using its robot environment. Perception tools need a
headless renderer:

```bash
SO101_VIEWER=0 MUJOCO_GL=egl ./venv/bin/python main.py
```

The MCP endpoint defaults to `http://127.0.0.1:3001/mcp`. The VELMA server must
provide `get_backend_info` so the adapter can verify that the connected backend
is simulation, not hardware.

## Run a pick-and-place evaluation

```powershell
$env:OPENAI_API_KEY = "..."
$env:PYTHONUTF8 = "1"
inspect-robots run --task velma-pick-place --policy agent --embodiment velma `
  -E url=http://127.0.0.1:3001/mcp `
  -P model=openai/gpt-4o-mini
```

Set the API key in the process environment or `.env`; do not commit it. The
task uses VELMA's perception tools rather than supplying hidden simulator
coordinates to the model. The scorer uses the simulator's `cube_in_container`
state. The initial benchmark is one known layout; change coordinates with
`-T cube_x=... -T cube_y=... -T container_x=... -T container_y=...` only after
checking that the layout is reachable.

The MCP server exposes VELMA's high-level `grasp` and `place` operations as
contributed policy tools. Calls are recorded in the agent transcript, but
Inspect Robots does not expand these operations into joint trajectories; its
action log contains a no-op placeholder for each high-level tool call. Use
VELMA's own logs when you need the simulator's internal actuation trace. The
current VELMA server does not stream camera frames in `get_robot_state`, so
this integration uses VELMA's `find_object` and `triangulate` tools for visual
localization. Raw camera video remains available through VELMA's own
`start_recording` / `stop_recording` tools.
