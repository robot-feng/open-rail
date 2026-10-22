from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from server.models.xpolicylab.xpolicylab import ModelVLA


def _install_fake_client(monkeypatch, client_type):
    package = ModuleType("client_server")
    package.__path__ = []
    ws_package = ModuleType("client_server.ws")
    ws_package.__path__ = []
    client_module = ModuleType("client_server.ws.model_client")
    client_module.WsModelClient = client_type
    monkeypatch.setitem(sys.modules, "client_server", package)
    monkeypatch.setitem(sys.modules, "client_server.ws", ws_package)
    monkeypatch.setitem(sys.modules, "client_server.ws.model_client", client_module)


def _config(tmp_path):
    repo = tmp_path / "XPolicyLab"
    client_module = repo / "client_server" / "ws" / "model_client.py"
    client_module.parent.mkdir(parents=True)
    client_module.touch()
    return {
        "repo_path": str(repo),
        "server_url": "ws://127.0.0.1:19000",
        "evaluation_id": "adapter-test",
        "trial_id": "trial-1",
        "camera_map": [{"source": "cam.head", "target": "cam_head"}],
        "state_fields": [
            {"name": "left_arm_joint_state", "start": 0, "end": 2},
            {"name": "left_ee_joint_state", "start": 2, "end": 3},
        ],
        "action_keys": ["left_arm_joint_state", "left_ee_joint_state"],
        "instruction_key": "language",
        "max_connect_attempts": 1,
    }


class _FakePolicyClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observation = None
        self.reset_count = 0
        self.prepared_case = None

    def call(self, func_name, obs=None):
        if func_name == "update_obs":
            self.observation = obs
            return None
        if func_name == "get_action":
            base = float(self.observation["state"]["left_arm_joint_state"][0])
            return [
                {
                    "left_arm_joint_state": np.array([base, base + 1], dtype=np.float32),
                    "left_ee_joint_state": np.array([0.5], dtype=np.float32),
                },
                {
                    "left_arm_joint_state": np.array([base + 2, base + 3], dtype=np.float32),
                    "left_ee_joint_state": np.array([0.75], dtype=np.float32),
                },
            ]
        if func_name == "reset":
            self.reset_count += 1
            self.observation = None
            return "reset-ok"
        if func_name == "prepare_case":
            self.prepared_case = obs
            return "prepared"
        raise AssertionError(f"unexpected call: {func_name}")

    def close(self):
        pass


def _make_model(monkeypatch, tmp_path):
    _install_fake_client(monkeypatch, _FakePolicyClient)
    return ModelVLA(_config(tmp_path))


def _frame(state, image, language="pick up the cup"):
    return {
        "obs": {
            "cam.head": image,
            "state": np.asarray(state, dtype=np.float32),
            "language": [language],
        },
        "ref_timestamp": 12.5,
        "loc_timestamp": 13.0,
    }


def test_maps_rgb_state_language_and_action_order(monkeypatch, tmp_path):
    model = _make_model(monkeypatch, tmp_path)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 0] = 17
    image[..., 2] = 231
    result = model.infer([_frame(np.arange(3), image)])

    assert result["type"] == "vla_action"
    assert result["ref_timestamp"] == 12.5
    assert result["loc_timestamp"] == 13.0
    np.testing.assert_array_equal(
        result["pred_action"],
        np.array([[0, 1, 0.5], [2, 3, 0.75]], dtype=np.float32),
    )
    observation = model.client.observation
    assert observation["instruction"] == "pick up the cup"
    np.testing.assert_array_equal(observation["vision"]["cam_head"]["color"], image)
    np.testing.assert_array_equal(observation["state"]["left_arm_joint_state"], [0, 1])
    np.testing.assert_array_equal(observation["state"]["left_ee_joint_state"], [2])


def test_reset_is_forwarded_to_the_xpolicylab_policy(monkeypatch, tmp_path):
    model = _make_model(monkeypatch, tmp_path)
    assert model.reset() == "reset-ok"
    assert model.client.reset_count == 1


def test_reset_generation_is_applied_once_before_its_inference(monkeypatch, tmp_path):
    model = _make_model(monkeypatch, tmp_path)
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    first = model.infer_with_reset([_frame([1, 0, 0], image)], 0, b"robot-client")
    repeated = model.infer_with_reset([_frame([2, 0, 0], image)], 0, b"robot-client")
    next_episode = model.infer_with_reset([_frame([3, 0, 0], image)], 1, b"robot-client")

    assert model.client.reset_count == 2
    assert first["pred_action"][0, 0] == 1
    assert repeated["pred_action"][0, 0] == 2
    assert next_episode["pred_action"][0, 0] == 3
    with pytest.raises(ValueError, match="Stale reset_generation"):
        model.infer_with_reset([_frame([4, 0, 0], image)], 0, b"robot-client")


def test_optional_case_preparation_is_forwarded(monkeypatch, tmp_path):
    _install_fake_client(monkeypatch, _FakePolicyClient)
    config = _config(tmp_path)
    config.update(
        action_case_id="case-7",
        prepare_case=True,
        case_meta={"task": "pick-and-place"},
    )
    model = ModelVLA(config)
    assert model.client.kwargs["action_case_id"] == "case-7"
    assert model.client.prepared_case == {"task": "pick-and-place"}


def test_inference_transactions_are_serialized(monkeypatch, tmp_path):
    class SlowClient(_FakePolicyClient):
        def call(self, func_name, obs=None):
            result = super().call(func_name, obs)
            time.sleep(0.002)
            return result

    _install_fake_client(monkeypatch, SlowClient)
    model = ModelVLA(_config(tmp_path))
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    def infer(value):
        output = model.infer([_frame([value, 0, 0], image)])
        return output["pred_action"][0, 0]

    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(pool.map(infer, range(6)))
    assert outputs == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "obs, message",
    [
        ({"state": np.zeros(3), "language": ["task"]}, "Missing Open-Rail camera"),
        (
            {"cam.head": np.zeros((4, 4)), "state": np.zeros(3), "language": ["task"]},
            "decoded HWC image",
        ),
        (
            {
                "cam.head": np.zeros((4, 4, 3), dtype=np.uint8),
                "state": np.array([0, np.nan, 2]),
                "language": ["task"],
            },
            "finite values",
        ),
    ],
)
def test_invalid_observations_fail_clearly(monkeypatch, tmp_path, obs, message):
    model = _make_model(monkeypatch, tmp_path)
    with pytest.raises((TypeError, ValueError), match=message):
        model._build_observation(obs)


def test_invalid_action_chunks_fail_clearly(monkeypatch, tmp_path):
    model = _make_model(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="missing configured key"):
        model._actions_to_array([{"left_arm_joint_state": [0, 1]}])
    with pytest.raises(ValueError, match="non-finite"):
        model._actions_to_array(
            [{"left_arm_joint_state": [np.nan, 1], "left_ee_joint_state": [0]}]
        )
    model._expected_action_dim = 4
    with pytest.raises(ValueError, match="action dimension"):
        model._actions_to_array(
            [{"left_arm_joint_state": [0, 1], "left_ee_joint_state": [0]}]
        )


def test_server_resets_the_model_once_per_client_generation():
    from server.core.vla_server import VLAServer

    class ResetSpy:
        calls = 0

        def reset(self):
            self.calls += 1

    server = VLAServer.__new__(VLAServer)
    server.model = ResetSpy()
    server._model_reset_lock = threading.Lock()
    server._client_reset_generations = {}
    client_id = b"test-client"

    server._reset_model_for_request(client_id, {"reset_generation": 0})
    server._reset_model_for_request(client_id, {"reset_generation": 0})
    assert server.model.calls == 1

    server._reset_model_for_request(client_id, {"reset_generation": 1})
    assert server.model.calls == 2
    with pytest.raises(ValueError, match="Invalid reset_generation"):
        server._reset_model_for_request(client_id, {"reset_generation": -1})


def test_client_carries_reset_generation_in_inference_metadata(monkeypatch):
    record_module = ModuleType("client.core.data_record_manager")
    record_module.DataRecordManager = type("DataRecordManager", (), {})
    monkeypatch.setitem(sys.modules, "client.core.data_record_manager", record_module)
    from client.core.vla_client import VLAClient

    class FakeZMQClient:
        metadata = None

        def sendMessage(self, data, meta):
            self.metadata = meta
            return True

        def recvMessage(self, timeout_ms):
            return {"meta": {"request_id": self.metadata["request_id"]}, "data": "ok"}

    client = VLAClient.__new__(VLAClient)
    client.config = SimpleNamespace(language=SimpleNamespace(auto_mode=True))
    client._request_id = 0
    client._model_reset_generation = 3
    client.vla_zmq = FakeZMQClient()
    assert client._request_inference({"obs": {}}, timeout_ms=100)["data"] == "ok"
    assert client.vla_zmq.metadata["reset_generation"] == 3

    client._inference_execution_lock = threading.Lock()
    client.realtime_data_manager = SimpleNamespace(clear=lambda: None)
    client._clear_raw_observe_queue = lambda: None
    client.image_process_time = 1.0
    client.reset()
    assert client._model_reset_generation == 4
    client._request_inference({"obs": {}}, timeout_ms=100)
    assert client.vla_zmq.metadata["reset_generation"] == 4
