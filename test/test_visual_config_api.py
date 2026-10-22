"""HTTP-level config tests; runtime shutdown is observed without loading an SDK."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest

from conf.client_conf import get_client_config
from web_client import server


@pytest.fixture
def config_runtime(monkeypatch):
    state = SimpleNamespace(
        config=get_client_config(), lock=threading.Lock(), vla_client=object(), robot=object(),
        running=True, paused=False, starting=False, stopping=False, config_switching=False,
        conf_file="test.yaml",
    )
    events = []

    def stop():
        events.append("stop")
        state.running = False

    def cleanup():
        events.append("cleanup")
        state.vla_client = state.robot = None

    async def broadcast(message):
        events.append(message)

    monkeypatch.setattr(server, "client_state", state)
    monkeypatch.setattr(server, "reset_resume_lock", asyncio.Lock())
    monkeypatch.setattr(server, "_shutdown_stop_if_running", stop)
    monkeypatch.setattr(server, "_cleanup", cleanup)
    monkeypatch.setattr(server, "_broadcast_to_web", broadcast)
    return state, events


def _request(endpoint, payload):
    async def request():
        transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(endpoint, json=payload)
    return asyncio.run(request())


def _patch(patch):
    return _request("/api/client/config/patch", {"patch": patch})


@pytest.mark.parametrize("patch", [
    {"visualize.port": 8877},
    {"visualize": {"port": 8877}},
    {"visualize.host": "127.0.0.1"},
    {"visualize.port": 8877, "controller.speed": 0.5},
])
def test_binding_change_replaces_runtime_and_broadcasts_stopped_state(config_runtime, patch):
    state, events = config_runtime
    old_config = state.config
    response = _patch(patch)
    assert response.status_code == 200, response.text
    assert response.json()["requires_restart"] is True
    assert events[:2] == ["stop", "cleanup"]
    assert events[2]["data"]["running"] is False
    assert state.vla_client is None
    assert state.config is not old_config
    assert old_config.visualize.port == 8765
    assert old_config.visualize.host == "0.0.0.0"
    if "visualize.host" in patch:
        assert state.config.visualize.host == "127.0.0.1"
    else:
        assert state.config.visualize.port == 8877
    if "controller.speed" in patch:
        assert state.config.controller.speed == 0.5


@pytest.mark.parametrize("patch", [
    {"visualize.port": 8765},
    {"visualize.port": "8765"},
    {"visualize.host": "0.0.0.0"},
    {"visualize.camera.open_head": False},
])
def test_unchanged_binding_and_display_settings_keep_runtime(config_runtime, patch):
    state, events = config_runtime
    runtime, config = state.vla_client, state.config
    response = _patch(patch)
    assert response.status_code == 200, response.text
    assert events == []
    assert state.vla_client is runtime
    assert state.config is config


@pytest.mark.parametrize("port", [-1, 0, 65536, 1.5, True, None, "invalid", "1.5"])
def test_invalid_port_does_not_stop_runtime_or_mutate_configuration(config_runtime, port):
    state, events = config_runtime
    runtime, config = state.vla_client, state.config
    response = _patch({"visualize.port": port})
    assert response.status_code in (400, 422), response.text[:200]
    assert events == []
    assert state.vla_client is runtime
    assert state.config is config
    assert state.config.visualize.port == 8765


@pytest.mark.parametrize("host", [None, "", "   ", 123])
def test_invalid_host_preserves_runtime(config_runtime, host):
    state, events = config_runtime
    response = _patch({"visualize.host": host})
    assert response.status_code in (400, 422), response.text[:200]
    assert events == []
    assert state.running


@pytest.mark.parametrize("busy_flag", ["starting", "stopping", "config_switching"])
def test_binding_changes_reject_busy_runtime(config_runtime, busy_flag):
    state, events = config_runtime
    setattr(state, busy_flag, True)
    response = _patch({"visualize.port": 8877})
    assert response.status_code == 409, response.text
    assert state.config.visualize.port == 8765
    assert events == []


def test_binding_can_change_before_runtime_exists(config_runtime):
    state, _events = config_runtime
    state.vla_client = state.robot = None
    state.running = False
    response = _patch({"visualize.port": 8877})
    assert response.status_code == 200, response.text
    assert response.json()["requires_restart"] is False
    assert state.config.visualize.port == 8877


@pytest.mark.parametrize("port", [8877, -1])
def test_config_file_binding_is_validated_before_runtime_replacement(
    config_runtime, monkeypatch, tmp_path, port
):
    state, events = config_runtime
    original = state.config
    monkeypatch.setattr(server, "ROOT", tmp_path)
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "test.yaml").write_text(f"visualize:\n  port: {port}\n")
    response = _request("/api/client/config/load", {"path": "test.yaml"})
    if port > 0:
        assert response.status_code == 200, response.text[:200]
        assert state.config.visualize.port == port
        assert events[:2] == ["stop", "cleanup"]
    else:
        assert response.status_code == 400, response.text[:200]
        assert state.config is original
        assert state.running
        assert events == []


def test_stale_config_replacement_preserves_current_runtime(config_runtime):
    state, events = config_runtime
    with pytest.raises(server.HTTPException) as error:
        asyncio.run(server._replace_client_config(
            get_client_config(), expected_config=get_client_config()
        ))
    assert error.value.status_code == 409
    assert state.running
    assert events == []
