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

## Three-cube stacking task

`velma-stack-three` uses `models/so101/stack3.xml`: three 30 mm cubes are
clustered near the container centre, so the arm does not need a long reach.
The task asks the model to take them out and stack them at `(0.20, -0.15)` on
object-centre z levels `0.015 / 0.045 / 0.075`. VELMA computes a server-side `stacked`
boolean from the three cube poses; the scorer requires that flag and at least
three successful grasps.

From the Inspect Robots workspace, start VELMA with the stack scene and run:

```powershell
& "C:\Users\Hi\OneDrive\Documents\GitHub\run-velma-eval.ps1" `
    -Task velma-stack-three `
    -Scene models/so101/stack3.xml
```

## Start the VELMA simulator

In the VELMA checkout, confirm `config.yaml` uses `backend.type: simulation`,
then start the server from the repository root. `config.yaml`, `models/`, and
the perception venv are resolved relative to that directory. Use the sphere
scene for this task. It has low bin walls so the gripper can reach the sphere,
and rolling resistance so the sphere stays at the table staging point. If you
also want camera-based perception, run headless; this specific task uses known
setup coordinates and can run with the viewer.
On Linux:

```bash
SO101_SCENE=models/so101/sphere30.xml SO101_VIEWER=0 MUJOCO_GL=egl ./venv/bin/python main.py
```

On Windows, EGL is unavailable, so use MuJoCo's GLFW or default WGL context:

```powershell
$env:SO101_VIEWER = "0"
$env:SO101_SCENE = "models/so101/sphere30.xml"
$env:MUJOCO_GL = "glfw"
$env:PERCEPT_PYTHON = "C:\path\to\python.exe"   # optional: an interpreter with torch + ultralytics
.\venv\Scripts\python.exe main.py
```

The perception tools need a sidecar interpreter with torch and ultralytics.
Build `percept_venv` from `perception/requirements.txt` (CPU torch; both
`find_object` and `triangulate` use it), or point VELMA at an existing
interpreter with `PERCEPT_PYTHON`. `percept_gpu_venv`
(`perception/requirements-gpu.txt`, via `PERCEPT_GPU_PYTHON`) enables Falcon
open-vocabulary grounding; without it `find_object` falls back to FastSAM colour
detection and the detector's table-plane projection, which is enough for this
tabletop task but less accurate for raised objects. VELMA's default
metric-depth backend is a stub, so a real depth backend must be registered
before the coarse coordinates count as measurements. The MCP endpoint defaults
to `http://127.0.0.1:3001/mcp`. The VELMA server must provide
`get_backend_info` so the adapter can verify that the connected backend is
simulation, not hardware.

## Run a pick-and-place evaluation

```powershell
inspect-robots doctor --embodiment velma -E url=http://127.0.0.1:3001/mcp
$env:OPENAI_API_KEY = "..."
$env:PYTHONUTF8 = "1"
inspect-robots run --task velma-pick-place --policy agent --embodiment velma `
  -E url=http://127.0.0.1:3001/mcp `
  -P model=openai/gpt-4o-mini
```

Set the API key in the process environment or `.env`; do not commit it. The
task starts a 30 mm sphere inside the container, places it at a table staging
point, grasps it again, and returns it to the container. The scorer requires
all four state-confirmed phases, not merely a final `cube_in_container=true`.
Change the initial, container, or staging coordinates with `-T cube_x=...`,
`-T cube_y=...`, `-T container_x=...`, `-T container_y=...`,
`-T staging_x=...`, and `-T staging_y=...` only after checking reachability.

## Change the task

The goal text is a task argument, so a one-off run can reword it without editing
code (`-T instruction="..."` keeps the VELMA task and scorer):

```powershell
inspect-robots run --task velma-pick-place --policy agent --embodiment velma `
  -E url=http://127.0.0.1:3001/mcp `
  -T instruction="Pick up the red cube and put it in the blue container. Measure first, then grasp and place."
```

Coordinates and horizons are task arguments too (`-T max_steps=30`,
`-T container_y=-0.25`). A custom instruction replaces the default procedure,
so keep the sphere grasp heights and four-phase sequence explicit. The VELMA
server must use `SO101_SCENE=models/so101/sphere30.xml`; the adapter passes
`grasp_height_m` through because VELMA declines sphere grasps without it.

The MCP server exposes VELMA's high-level `grasp` and `place` operations as
contributed policy tools. `velma_grasp` forces VELMA's honest-perception path
(`trust_coords=true`) and pre-opens the jaws from the measured width, so the
coordinates the model measured drive the grasp instead of the cube's hidden
simulator pose. Calls are recorded in the agent transcript, but Inspect Robots
does not expand these operations into joint trajectories; its action log
contains a no-op placeholder for each high-level tool call. Use VELMA's own
logs when you need the simulator's internal actuation trace.

Legacy tasks may include nominal setup/search coordinates in their instructions.
The adapter also decodes camera frames into policy observations when capture is enabled. VELMA's
`start_recording` / `stop_recording` tools are not contributed to the policy;
use VELMA's own recorder if you need a video.


## Manifest tasks and persistent sessions

A compatible VELMA checkout provides `pipeline.cmd --scene <scene.yaml>
--instruction <request>`. The launcher generates MuJoCo XML and normalized scene
JSON, starts the host, and keeps a loopback session service running. The
`velma-generic` embodiment exposes object-ID measurement/grasp and destination
placement tools; `velma-agent` limits the policy to those tools and termination.
Install the `agent` extra when using these features outside the full workspace.

The `velma-task` task accepts `manifest=<generated scene.json>` and optional
`instruction`/`max_steps` for ordinary evaluations. Use embodiment
`velma-generic` and policy `velma-agent`; the running host must match the manifest
digest. Object setup poses are not part of the policy catalog. Dimensions and
colours are declared priors; metric poses come from calibrated RGB fitting.

`python -m inspect_robots_velma.session --url <MCP URL> --port 8780
--status-file <path>` runs the persistent service independently. Its endpoints
are GET `/session/status`, GET `/session/observe`, and POST `/session/start`,
`/session/submit_instruction`, `/session/reset`, `/session/close`. POST bodies
are JSON; submit requires `instruction`. Requests are serialized, returning
HTTP 409 while busy. Reset is explicit and clears conversation history. Close
is for idle sessions and does not implement an emergency stop.

Agent `done` and `evaluation.verified` are independent: verification applies to
the YAML's configured goal, which can span several requests. Simulator contact
and object state are still used internally for execution/evaluation. This is
not a hardware or vision-only adapter. VELMA's `SCENES.md` documents the scene
schema, supported geometry/capabilities, session examples and test runner.
