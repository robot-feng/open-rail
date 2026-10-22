"""Use XPolicyLab's real WebSocket server/client with a deterministic policy."""

import asyncio
import socket
import threading
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest

from conf.models_conf import get_xpolicylab_config
from server.core.vla_server import VLAServer
from server.models.xpolicylab.xpolicylab import ModelVLA


@pytest.fixture
def xpl_runtime(monkeypatch):
    root = Path(__file__).resolve().parent / "XPolicyLab"
    if not (root / "client_server/ws/model_server.py").is_file():
        pytest.skip("Initialize the XPolicyLab submodule")
    pytest.importorskip("msgpack_numpy")
    pytest.importorskip("pydantic", minversion="2.5")
    monkeypatch.syspath_prepend(str(root.parent))
    monkeypatch.syspath_prepend(str(root))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    return root, PolicyServer, PolicyServerConfig


def _config(root, url):
    config = get_xpolicylab_config()
    config.repo_path = str(root)
    config.server_url = url
    config.expected_action_dim = 16
    config.ws_ping_interval_s = config.ws_ping_timeout_s = None
    config.connect_timeout_s = config.handshake_timeout_s = 2.0
    config.request_timeout_s = 2.0
    config.max_connect_attempts = 1
    config.connect_retry_delay_s = 0.01
    config.max_connect_seconds = 2.0
    config.close_timeout_s = 1.0
    return config


def _frame(value):
    return [{
        "obs": {
            "cam.head": np.full((8, 10, 3), [11, 31, 71], dtype=np.uint8)[:, ::-1],
            "cam.hand_left": np.full((8, 10, 3), [13, 37, 83], dtype=np.uint8),
            "cam.hand_right": np.full((8, 10, 3), [17, 41, 97], dtype=np.uint8),
            "state": np.full(16, value, dtype=np.float32)[None],
            "language": np.array([["pick up the cup"]]),
        },
        "ref_timestamp": 12.5, "loc_timestamp": 13.0,
    }]


class Policy:
    def __init__(self):
        self.observation = None
        self.reset_count = 0
        self.prepared = None
        self.fail_reset = False

    def prepare_case(self, metadata):
        self.prepared = metadata

    def reset(self):
        self.reset_count += 1
        if self.fail_reset:
            self.fail_reset = False
            raise RuntimeError("injected reset failure")
        self.observation = None

    def update_obs(self, observation):
        self.observation = observation

    def get_action(self):
        obs = self.observation
        assert obs["instruction"] == "pick up the cup"
        np.testing.assert_array_equal(obs["vision"]["cam_head"]["color"][0, 0], [11, 31, 71])
        value = float(obs["state"]["left_arm_joint_state"][0])
        if value == -1:
            return [{"wrong-key": [0]}]
        return [{
            "left_arm_joint_state": np.full(7, value),
            "right_arm_joint_state": np.full(7, value + 1),
            "left_ee_joint_state": [0.2], "right_ee_joint_state": [0.8],
        }] * 2


def _rail_server(model):
    server = VLAServer.__new__(VLAServer)
    server.model = model
    server._model_reset_lock = threading.Lock()
    server._client_reset_generations = {}
    return server


def test_real_websocket_prepare_concurrent_generations_and_error_recovery(xpl_runtime):
    root, PolicyServer, PolicyServerConfig = xpl_runtime

    async def probe():
        policy = Policy()
        remote = PolicyServer(policy, PolicyServerConfig(host="127.0.0.1", port=0))
        await remote.start()
        model = None
        try:
            config = _config(root, remote.url)
            config.prepare_case = True
            config.action_case_id = "case-1"
            config.case_meta = {"task": "cup", "options": {"seed": 7}}
            model = await asyncio.to_thread(ModelVLA, config)
            rail = _rail_server(model)
            for generation in range(3):
                outputs = await asyncio.gather(*[
                    asyncio.to_thread(
                        rail._infer_model, _frame(value), b"robot", {"reset_generation": generation}
                    )
                    for value in range(12)
                ])
                for value, output in enumerate(outputs):
                    expected = np.r_[np.full(7, value), np.full(7, value + 1), 0.2, 0.8]
                    np.testing.assert_allclose(output["pred_action"], np.tile(expected, (2, 1)))
                    assert output["ref_timestamp"] == 12.5
                assert policy.reset_count == generation + 1
            assert policy.prepared == {
                "task": "cup", "options": {"seed": 7}, "action_case_id": "case-1"
            }
            with pytest.raises(ValueError, match="missing configured key"):
                await asyncio.to_thread(rail._infer_model, _frame(-1), b"robot", {"reset_generation": 2})
            result = await asyncio.to_thread(rail._infer_model, _frame(99), b"robot", {"reset_generation": 3})
            assert result["pred_action"][0, 0] == 99
            assert policy.reset_count == 4
        finally:
            if model is not None:
                await asyncio.to_thread(model.close)
                assert not model.client._loop_thread.is_alive()
            await remote.stop()

    asyncio.run(probe())


def test_remote_reset_failure_is_retried_before_inference(xpl_runtime):
    root, PolicyServer, PolicyServerConfig = xpl_runtime

    async def probe():
        policy = Policy()
        policy.fail_reset = True
        remote = PolicyServer(policy, PolicyServerConfig(host="127.0.0.1", port=0))
        await remote.start()
        model = None
        try:
            model = await asyncio.to_thread(ModelVLA, _config(root, remote.url))
            rail = _rail_server(model)
            with pytest.raises(Exception, match="injected reset failure"):
                await asyncio.to_thread(rail._infer_model, _frame(1), b"robot", {"reset_generation": 0})
            assert policy.observation is None
            result = await asyncio.to_thread(rail._infer_model, _frame(2), b"robot", {"reset_generation": 0})
            assert result["pred_action"][0, 0] == 2
            assert policy.reset_count == 2
        finally:
            if model is not None:
                await asyncio.to_thread(model.close)
            await remote.stop()

    asyncio.run(probe())


def test_unreachable_server_does_not_leak_a_client_loop(xpl_runtime):
    root, _server, _config_type = xpl_runtime
    before = {thread.ident for thread in threading.enumerate() if thread.name == "ws-model-client"}
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        config = _config(root, f"ws://127.0.0.1:{reserved.getsockname()[1]}")
        with pytest.raises(ConnectionError, match="failed to connect"):
            ModelVLA(config)
    after = {thread.ident for thread in threading.enumerate() if thread.name == "ws-model-client"}
    assert after == before


def test_policy_server_restart_is_reported_instead_of_reusing_lost_history(xpl_runtime):
    root, PolicyServer, PolicyServerConfig = xpl_runtime
    from client_server.ws.protocol.exceptions import ServerRestartedError

    async def probe():
        remote = PolicyServer(Policy(), PolicyServerConfig(host="127.0.0.1", port=0))
        await remote.start()
        model = None
        replacement = None
        try:
            url = remote.url
            model = await asyncio.to_thread(ModelVLA, _config(root, url))
            await asyncio.to_thread(model.infer, _frame(1))
            await remote.stop()
            replacement = PolicyServer(
                Policy(), PolicyServerConfig(host="127.0.0.1", port=urlsplit(url).port)
            )
            await replacement.start()
            with pytest.raises(ServerRestartedError):
                await asyncio.to_thread(model.infer, _frame(2))
        finally:
            if model is not None:
                await asyncio.to_thread(model.close)
            await remote.stop()
            if replacement is not None:
                await replacement.stop()

    asyncio.run(probe())
