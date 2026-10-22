# Regression tests

Run from the repository root in an Open-RAIL development environment:

```bash
python -m pip install -e ".[dev,xpolicylab]"
git submodule update --init test/XPolicyLab
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q test --ignore=test/XPolicyLab
node --test test/camera_websocket.test.cjs
```

Node 22 or newer is required for the frontend transport test. Disabling pytest
plugin autoload keeps unrelated ROS or model-environment plugins out of these
tests. The XPolicyLab checkout contains independent model repositories and test
suites, so it is excluded from Open-RAIL test discovery. Tests requiring its
WebSocket implementation skip explicitly when that checkout or its protocol
dependencies are unavailable.

The dashboard has a separate real-browser regression check. It needs Node 22+
and a local Chromium binary, with no npm packages or running robot/model server:

```bash
CHROMIUM_BIN=/usr/bin/chromium-browser node test/dashboard_browser.cjs
# Optional screenshots from isolated mock data:
UI_SCREENSHOT_DIR=/tmp/openrail-ui-preview node test/dashboard_browser.cjs
```

This check serves local frontend assets and mocks HTTP/WebSocket responses. It
checks three workspace presets at eight desktop/tablet/mobile widths, browser
preference persistence, stale telemetry, HTTP fallback for a silent open socket,
camera expansion/Escape and focus restoration, chart controls, slider save counts,
keyboard-inert hidden panels, and local asset loading. Display-only controls are
also checked for unintended API mutations. Screenshots contain simulated data.

The camera transport tests use the same-origin `/visual-ws` endpoint. The Python
integration probe runs an isolated FastAPI relay with lifespan disabled, verifying
JPEG integrity from a real visual server through the actual frontend handler.

The suite runs without checkpoints or robot connections:

| Area | Coverage |
| --- | --- |
| Model registration | Every model type's CLI, factory configuration and checkpoint override |
| Lifecycle | Reset ordering, stale/invalid generations, legacy metadata, concurrency, failure retry and shutdown |
| ZMQ integration | Real TCP sockets and JPEG transport, history frames, request IDs, timeouts, late responses and policy errors |
| XPolicyLab | Observation/action mapping, real WebSocket RPCs, case preparation, concurrent generations, reset errors, restart detection and client cleanup |
| StarVLA | Camera ordering/layout, state transforms, checkpoint options, malformed inputs and output validation |
| Web configuration | HTTP patch/load requests, binding validation, runtime replacement and preservation on errors |
| Camera streams | Real visual server to frontend handler, binary JPEG integrity, disabled/depth cameras and relay text/binary frames |
| Client/robot barriers | Reset observation barriers, stale control rejection, pause ordering, language switching and nonblocking robot proxy |

Network probes use OS-assigned test ports. Robot SDK calls are replaced with
test doubles, and visual-server probes disable the runtime's port-killing
helper. These tests do not establish checkpoint quality, robot SDK units or
physical task success.

For an additional comparison against StarVLA's actual training transforms,
run the following in an environment containing StarVLA's dependencies:

```bash
export STARVLA_ROOT=/path/to/starvla
NO_ALBUMENTATIONS_UPDATE=1 python test/probe_starvla_training.py
```

This probe compares 400 synthetic states across `q99`, `min_max`, `mean_std`
and `binary` normalization, including constant dimensions, multiple state keys,
and read-only, noncontiguous arrays. It uses upstream transforms with synthetic
statistics and does not load model weights. Configure CUDA library paths as
required by that environment's PyTorch installation.
