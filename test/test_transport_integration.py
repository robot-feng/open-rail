"""Real TCP/ZMQ and JPEG transport in a subprocess, without robot or model weights."""

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from client.core.vla_client import VLAClient
from client.core.zmq_client import ZMQClient
from conf.zmq_conf import get_vla_zmq_config

SERVER_SCRIPT = r'''
import json
import sys
import threading
import time
from pathlib import Path
import numpy as np
import zmq
from conf.server_conf import get_vla_server_config
from server.core.vla_server import VLAServer
from server.core.zmq_server import ZMQServer

class Echo:
    cfg = {"model_path": "test://no-weights"}
    resets = 0

    def infer(self, sequence):
        data = sequence[0]
        obs = data["obs"]
        time.sleep(float(obs.get("delay", 0)))
        if obs.get("fail"):
            raise ValueError("injected policy failure")
        return {
            "type": "vla_action",
            "pred_action": np.tile([float(obs["state"][0]), self.resets], (2, 1)),
            "ref_timestamp": data["ref_timestamp"], "loc_timestamp": data["loc_timestamp"],
            "ext": {"rgb": obs["cam.head"][0, 0].tolist(), "language": obs["language"]},
        }

class StatefulEcho(Echo):
    def reset(self):
        self.resets += 1

config = get_vla_server_config()
config.zmq.port = "*"
transport = ZMQServer(config.zmq)
port = int(transport.router.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1])
model = StatefulEcho() if sys.argv[2] == "stateful" else Echo()
server = VLAServer(config, transport, model)
runner = threading.Thread(target=server.run, daemon=True)
runner.start()
ready = Path(sys.argv[1])
pending = ready.with_suffix(".tmp")
pending.write_text(json.dumps({"port": port}))
pending.replace(ready)
try:
    sys.stdin.readline()
finally:
    server.close()
    runner.join(timeout=2)
    if runner.is_alive():
        raise RuntimeError("receive thread did not stop")
'''


@pytest.mark.parametrize("model_kind", ["legacy", "stateful"])
def test_zmq_jpeg_requests_reset_timeout_recovery_and_shutdown(tmp_path, caplog, model_kind):
    ready = tmp_path / "ready.json"
    log_path = tmp_path / "server.log"
    client = None
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", SERVER_SCRIPT, str(ready), model_kind],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), log_path.read_text()
            config = get_vla_zmq_config()
            config.port = str(json.loads(ready.read_text())["port"])
            config.heartbeat_interval = 50
            client = VLAClient.__new__(VLAClient)
            client.config = SimpleNamespace(language=SimpleNamespace(auto_mode=False))
            client.task_language_manager = SimpleNamespace(get_current_language=lambda: "current task")
            client.logger = logging.getLogger("test.transport")
            client._request_id = 0
            client._model_reset_generation = 0
            client._inference_execution_lock = threading.Lock()
            client.realtime_data_manager = SimpleNamespace(clear=lambda: None)
            client._clear_raw_observe_queue = lambda: None
            client.vla_zmq = ZMQClient(config)
            bgr = np.full((8, 10, 3), [15, 70, 210], dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", bgr)
            assert ok
            expected_rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[0, 0, ::-1].tolist()

            def frame(value, **extra):
                return {
                    "obs": {"state": np.array([value]), "cam.head": memoryview(encoded),
                            "language": ["queued old task"], **extra},
                    "ref_timestamp": float(value), "loc_timestamp": time.perf_counter(),
                }

            for value in range(25):
                item = frame(value)
                data = [item, frame(value + 1)] if value % 2 else item
                response = client._request_inference(data, timeout_ms=3000)
                assert response is not None, log_path.read_text()
                action = response["data"]
                np.testing.assert_array_equal(
                    action["pred_action"], np.tile([value, int(model_kind == "stateful")], (2, 1))
                )
                assert action["ref_timestamp"] == value
                assert action["loc_timestamp"] == item["loc_timestamp"]
                assert action["ext"] == {"rgb": expected_rgb, "language": ["current task"]}
                assert response["meta"]["avg_infer_time"] > 0

            # A timed-out old response must not be consumed as the next episode's action.
            assert client._request_inference(frame(500, delay=0.2), timeout_ms=30) is None
            client.reset()
            with caplog.at_level(logging.WARNING, logger="test.transport"):
                response = client._request_inference(frame(999), timeout_ms=3000)
            assert response is not None, log_path.read_text()
            np.testing.assert_array_equal(
                response["data"]["pred_action"],
                np.tile([999, 2 * int(model_kind == "stateful")], (2, 1)),
            )
            # One policy failure must not prevent subsequent inference.
            assert client._request_inference(frame(700, fail=True), timeout_ms=100) is None
            assert client._request_inference(frame(701), timeout_ms=3000)["data"]["ref_timestamp"] == 701
            # Stateless workers may finish the new request before the old one;
            # this slower round also drains any late response from that old request.
            with caplog.at_level(logging.WARNING, logger="test.transport"):
                response = client._request_inference(frame(702, delay=0.3), timeout_ms=3000)
            assert response["data"]["ref_timestamp"] == 702
            assert "Drop stale response" in caplog.text
        finally:
            if client is not None:
                client.vla_zmq.release_data_socket()
                client.vla_zmq.close()
            if process.poll() is None:
                try:
                    process.communicate("stop\n", timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
                    pytest.fail(f"Server shutdown timed out:\n{log_path.read_text()}")
            assert process.returncode == 0, log_path.read_text()
