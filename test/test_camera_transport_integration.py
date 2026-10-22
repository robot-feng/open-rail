"""Real visual WebSocket server -> actual frontend handler -> JPEG frame bytes."""

import hashlib
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import uvicorn

from client.core.visualize_server import VisualizeServer
from conf.client_conf import get_client_config
from web_client import server

NODE_CLIENT = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');
const frames = [];
let context;
const deadline = setTimeout(() => {
  console.error('camera frames timed out');
  vm.runInContext('disconnectCamWS()', context);
  process.exitCode = 1;
}, 4000);
context = vm.createContext({
  location:{protocol:'http:',hostname:'127.0.0.1',port:process.argv[1]},
  WebSocket, URL, ArrayBuffer, Blob, setTimeout, clearTimeout,
  handleCamWSMessage: () => {},
  decodeCameraBinaryFrame: (buffer) => {
    const bytes = Buffer.from(buffer);
    const headerSize = bytes.readUInt32BE(0);
    const header = JSON.parse(bytes.subarray(4, 4 + headerSize).toString());
    const payload = bytes.subarray(4 + headerSize);
    if (header.type !== 'camera_data_binary') throw Error('unexpected binary frame type');
    if (header.data_size !== payload.length) throw Error('incorrect JPEG size');
    frames.push({id:header.camera_id, hash:crypto.createHash('sha256').update(payload).digest('hex')});
    if (frames.length > 2) throw Error('disabled or depth camera was forwarded');
    if (frames.length === 2) {
      clearTimeout(deadline);
      setTimeout(() => {
        vm.runInContext('disconnectCamWS()', context);
        console.log(JSON.stringify(frames.sort((a,b) => a.id-b.id)));
      }, 100);
    }
  },
});
for (const file of ['constants.js','modules/websocket.js']) {
  vm.runInContext(fs.readFileSync('web_client/static/'+file,'utf8'), context);
}
vm.runInContext('connectCamWS()', context);
'''


def test_camera_frontend_receives_jpeg_frames_through_same_origin_relay(monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 22+ is required for the frontend transport probe")
    # Only an OS-assigned test port is used; never run the runtime's port-killing helper.
    monkeypatch.setattr(VisualizeServer, "kill_port", lambda self, port: None)
    config = get_client_config()
    config.visualize.host, config.visualize.port = "127.0.0.1", 0
    config.visualize.camera.open_wrist_right = False
    visual = VisualizeServer(config.visualize)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    relay_port = listener.getsockname()[1]
    # Disable lifespan: no SDK, runtime initialization, or shutdown of user services.
    relay = uvicorn.Server(uvicorn.Config(server.app, lifespan="off", log_level="error"))
    relay_thread = threading.Thread(target=relay.run, kwargs={"sockets": [listener]}, daemon=True)
    try:
        deadline = time.monotonic() + 5
        while visual.server is None and visual.main_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert visual.server is not None, "Visual WebSocket failed to start"
        config.visualize.port = visual.server.sockets[0].getsockname()[1]
        monkeypatch.setattr(server, "client_state", SimpleNamespace(config=config, lock=threading.Lock()))
        relay_thread.start()
        deadline = time.monotonic() + 5
        while not relay.started and relay_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert relay.started, "WebSocket relay failed to start"
        images = {}
        for name, value in (("head", 30), ("hand_left", 90), ("hand_right", 150)):
            ok, encoded = cv2.imencode(".jpg", np.full((8, 10, 3), value, dtype=np.uint8))
            assert ok
            images[f"cam.{name}"] = encoded.tobytes()
        images["cam.depth.head"] = b"depth must not be sent to the RGB panel"
        visual.update_image_data(images)
        result = subprocess.run(
            [node, "-e", NODE_CLIENT, str(relay_port)], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=8, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == [
            {"id": index, "hash": hashlib.sha256(images[key]).hexdigest()}
            for index, key in enumerate(("cam.head", "cam.hand_left"))
        ]
    finally:
        relay.should_exit = True
        if relay_thread.ident is not None:
            relay_thread.join(timeout=5)
        listener.close()
        visual.stop_server()
        visual.main_thread.join(timeout=3)
        assert not visual.main_thread.is_alive(), "Visual WebSocket thread did not stop"
        assert not relay_thread.is_alive(), "Relay server thread did not stop"
