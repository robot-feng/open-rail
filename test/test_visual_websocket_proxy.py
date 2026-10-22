"""Exercise the Web UI's relay against a real local visual WebSocket server."""

import json
import socket
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from client.core.visualize_server import VisualizeServer
from conf.client_conf import get_client_config
from web_client import server


@pytest.fixture
def visual_relay(monkeypatch):
    config = get_client_config()
    config.visualize.host, config.visualize.port = "127.0.0.1", 0
    monkeypatch.setattr(VisualizeServer, "kill_port", lambda self, port: None)
    visual = VisualizeServer(config.visualize)
    client = TestClient(server.app)
    try:
        deadline = time.monotonic() + 5
        while visual.server is None and visual.main_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert visual.server is not None
        # The listener already owns its OS-assigned port; expose it to the relay.
        config.visualize.port = visual.server.sockets[0].getsockname()[1]
        monkeypatch.setattr(server, "client_state", SimpleNamespace(config=config, lock=threading.Lock()))
        yield visual, client
    finally:
        client.close()
        visual.stop_server()
        visual.main_thread.join(timeout=3)
        assert not visual.main_thread.is_alive()


def test_visual_relay_forwards_text_in_both_directions_and_cleans_up(visual_relay):
    visual, client = visual_relay
    with client.websocket_connect("/visual-ws") as websocket:
        assert websocket.receive_json()["type"] == "config"
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
        websocket.send_json({"type": "get_status"})
        assert websocket.receive_json()["data"]["connected_clients"] == 1
    deadline = time.monotonic() + 2
    while visual.clients and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not visual.clients


def test_visual_relay_preserves_binary_jpeg_packets(visual_relay):
    visual, client = visual_relay
    ok, encoded = cv2.imencode(".jpg", np.full((8, 10, 3), 90, dtype=np.uint8))
    assert ok
    with client.websocket_connect("/visual-ws") as websocket:
        assert websocket.receive_json()["type"] == "config"
        visual.update_image_data({"cam.head": encoded.tobytes()})
        packet = websocket.receive_bytes()
        header_size = int.from_bytes(packet[:4], "big")
        header = json.loads(packet[4:4 + header_size])
        assert header["type"] == "camera_data_binary"
        assert header["camera_id"] == 0
        assert header["data_size"] == len(encoded)
        assert packet[4 + header_size:] == encoded.tobytes()


def test_unavailable_visualizer_returns_a_retryable_close_code(monkeypatch):
    config = get_client_config()
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        config.visualize.port = reserved.getsockname()[1]
        monkeypatch.setattr(server, "client_state", SimpleNamespace(config=config, lock=threading.Lock()))
        client = TestClient(server.app)
        try:
            with (
                client.websocket_connect("/visual-ws") as websocket,
                pytest.raises(WebSocketDisconnect) as error,
            ):
                websocket.receive_text()
            assert error.value.code == 1013
        finally:
            client.close()
