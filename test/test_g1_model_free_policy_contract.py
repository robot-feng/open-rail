"""No-checkpoint G1 contract test across StarVLA, XPolicyLab, and the robot client.

The fake policies make the input/output boundaries deterministic. This checks
the adapters and command mapping; it does not claim model or task validation.
It starts with a synthetic decoded observation and does not verify the SDK's
feedback units or the checkpoint's state/action ordering.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from conf.robots_conf import get_agibot_g1_config
from server.models.starvla.starvla import ModelVLA as StarVLAModel

CAMERA_SOURCES = ("cam.head", "cam.hand_left", "cam.hand_right")
G1_STATE = np.asarray(
    [
        -1.0741696, 0.6110292, 0.2795141, -1.2838837, 0.7304570, 1.4952905,
        -0.1875875,
        1.0742918, -0.6111165, -0.2795490, 1.2839711, -0.7304046,
        -1.4954126, 0.1876050,
        0.0, 0.0018111,
        0.0, 0.4363829, 0.2180394, 0.3490646,
    ],
    dtype=np.float32,
)
# State values come from the audited pick_up_the_apple2 first frame. Camera
# arrays below are synthetic RGB sentinels so this contract test needs no data
# mount or camera decoder.
G1_ACTIONS = np.arange(32, dtype=np.float32).reshape(2, 16) / 100 - 0.1
G1_ACTIONS[:, 14:16] = [[0.2, 0.8], [0.0, 1.0]]
ACTION_KEYS = (
    "left_arm_joint_state",
    "right_arm_joint_state",
    "left_ee_joint_state",
    "right_ee_joint_state",
)


def _golden_observation():
    images = {
        "cam.head": np.full((5, 7, 3), [11, 31, 71], dtype=np.uint8),
        "cam.hand_left": np.full((5, 7, 3), [13, 37, 83], dtype=np.uint8),
        "cam.hand_right": np.full((5, 7, 3), [17, 41, 97], dtype=np.uint8),
    }
    return {
        **images,
        "state": G1_STATE.copy(),
        "language": ["pick up the apple"],
    }


def _install_starvla_fake(monkeypatch, tmp_path):
    repo = tmp_path / "starvla"
    (repo / "starVLA").mkdir(parents=True)
    wrapper = repo / "deployment" / "model_server" / "policy_wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.touch()

    class FixedPolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.metadata = {"action_chunk_size": len(G1_ACTIONS)}
            self.examples = None

        def predict_action(self, examples, **kwargs):
            self.examples = examples
            return {"actions": G1_ACTIONS[None, ...].copy()}

    deployment = ModuleType("deployment")
    deployment.__path__ = []
    model_server = ModuleType("deployment.model_server")
    model_server.__path__ = []
    wrapper_module = ModuleType("deployment.model_server.policy_wrapper")
    wrapper_module.PolicyServerWrapper = FixedPolicy
    monkeypatch.setitem(sys.modules, "deployment", deployment)
    monkeypatch.setitem(sys.modules, "deployment.model_server", model_server)
    monkeypatch.setitem(sys.modules, "deployment.model_server.policy_wrapper", wrapper_module)
    monkeypatch.syspath_prepend(str(repo))

    model = StarVLAModel(
        {
            "repo_path": str(repo),
            "model_path": "/fake/no-checkpoint",
            "device": "cpu",
            "use_bf16": False,
            "unnorm_key": None,
            "config_overrides": [],
            "camera_keys": list(CAMERA_SOURCES),
            "send_state": True,
            "normalize_state": False,
            "state_key": "state",
            "language_key": "language",
            "expected_state_dim": 20,
            "expected_action_dim": 16,
        }
    )
    return model


def _g1_xpolicylab_config(repo, url):
    return {
        "repo_path": str(repo),
        "server_url": url,
        "evaluation_id": "open-rail-g1-contract",
        "trial_id": "golden-frame-0",
        "ws_ping_interval_s": None,
        "ws_ping_timeout_s": None,
        "connect_timeout_s": 3.0,
        "handshake_timeout_s": 3.0,
        "request_timeout_s": 5.0,
        "max_connect_attempts": 1,
        "connect_retry_delay_s": 0.05,
        "max_connect_seconds": 3.0,
        "close_timeout_s": 1.0,
        "camera_map": [
            {"source": "cam.head", "target": "cam_head"},
            {"source": "cam.hand_left", "target": "cam_left_wrist"},
            {"source": "cam.hand_right", "target": "cam_right_wrist"},
        ],
        # RAIL G1 observation.state is [left arm, right arm, left gripper,
        # right gripper, head, waist] (20D). XPL policies consume named fields,
        # so spell out the G1 slices rather than using the generic 16D default.
        "state_fields": [
            {"name": "left_arm_joint_state", "start": 0, "end": 7},
            {"name": "right_arm_joint_state", "start": 7, "end": 14},
            {"name": "left_ee_joint_state", "start": 14, "end": 15},
            {"name": "right_ee_joint_state", "start": 15, "end": 16},
            {"name": "head_joint_state", "start": 16, "end": 18},
            {"name": "waist_joint_state", "start": 18, "end": 20},
        ],
        "action_keys": list(ACTION_KEYS),
        "expected_action_dim": 16,
        "instruction_key": "language",
    }


class _FixedXplPolicy:
    def __init__(self):
        self.observation = None
        self.last_observation = None
        self.reset_count = 0

    def update_obs(self, observation):
        self.observation = observation
        self.last_observation = observation

    def get_action(self):
        chunks = []
        for row in G1_ACTIONS:
            chunks.append(
                {
                    ACTION_KEYS[0]: row[0:7].copy(),
                    ACTION_KEYS[1]: row[7:14].copy(),
                    ACTION_KEYS[2]: row[14:15].copy(),
                    ACTION_KEYS[3]: row[15:16].copy(),
                }
            )
        return chunks

    def reset(self):
        self.reset_count += 1
        self.observation = None


def _install_fake_robot_sdk(monkeypatch):
    a2d_sdk = ModuleType("a2d_sdk")
    a2d_sdk.__path__ = []
    robot_sdk = ModuleType("a2d_sdk.robot")
    robot_sdk.RobotDds = object
    robot_sdk.CosineCamera = object
    monkeypatch.setitem(sys.modules, "a2d_sdk", a2d_sdk)
    monkeypatch.setitem(sys.modules, "a2d_sdk.robot", robot_sdk)
    # pandas is only used by RobotBody.load_action_data, outside this command path.
    monkeypatch.setitem(sys.modules, "pandas", ModuleType("pandas"))


def _dispatch_without_hardware(monkeypatch, actions):
    _install_fake_robot_sdk(monkeypatch)
    module_name = "client.robots.agibot_g1.body_robot"
    old_module = sys.modules.pop(module_name, None)
    try:
        body_module = importlib.import_module(module_name)

        class RobotSpy:
            def __init__(self):
                self.arm_targets = []
                self.gripper_targets = []
                self.other_calls = []

            def move_arm(self, target):
                self.arm_targets.append(np.asarray(target, dtype=np.float32))

            def move_gripper(self, target):
                self.gripper_targets.append(np.asarray(target, dtype=np.float32))

            def move_head(self, target):
                self.other_calls.append(("head", target))

            def move_waist(self, target):
                self.other_calls.append(("waist", target))

        robot = RobotSpy()
        body = body_module.RobotBody.__new__(body_module.RobotBody)
        body.robot = robot
        body.config = get_agibot_g1_config()
        body.action_layout = dict(body.config.action_layout)
        from client.utils.util import parse_action_layout

        assert parse_action_layout(body.action_layout)[0] == G1_ACTIONS.shape[1]
        body.gripper_count = 0
        body.head_count = 0
        for action in actions:
            body.control_robot(action)
        return robot
    finally:
        sys.modules.pop(module_name, None)
        if old_module is not None:
            sys.modules[module_name] = old_module


def test_g1_golden_observation_actions_match_starvla_and_xpolicylab(monkeypatch, tmp_path):
    """The no-weight probe covers the real XPL websocket and both adapter boundaries."""
    observation = _golden_observation()
    sequence = [
        {
            "obs": observation,
            "ref_timestamp": 12.5,
            "loc_timestamp": 13.0,
        }
    ]

    starvla = _install_starvla_fake(monkeypatch, tmp_path)
    starvla_result = starvla.infer(sequence)
    starvla_example = starvla.policy.examples[0]
    assert starvla_example["lang"] == "pick up the apple"
    np.testing.assert_array_equal(starvla_example["state"], G1_STATE[None, :])
    for key, image in zip(CAMERA_SOURCES, starvla_example["image"], strict=True):
        np.testing.assert_array_equal(image, observation[key])

    xpl_root = Path(__file__).resolve().parent / "XPolicyLab"
    if not (xpl_root / "client_server/ws/model_server.py").is_file():
        pytest.skip("Initialize XPolicyLab with git submodule update --init test/XPolicyLab")
    pytest.importorskip("msgpack_numpy", reason="Install Open-Rail's xpolicylab extra")
    pytest.importorskip("pydantic", minversion="2.5")
    xpl_parent = xpl_root.parent
    monkeypatch.syspath_prepend(str(xpl_parent))
    monkeypatch.syspath_prepend(str(xpl_root))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    from server.models.xpolicylab.xpolicylab import ModelVLA as XplModel

    fixed_policy = _FixedXplPolicy()

    async def run_xpl_round_trip():
        server = PolicyServer(
            fixed_policy,
            PolicyServerConfig(
                host="127.0.0.1",
                port=0,
                ws_ping_interval_s=None,
                ws_ping_timeout_s=None,
            ),
        )
        await server.start()
        xpl = None
        try:
            xpl = await asyncio.to_thread(
                XplModel,
                _g1_xpolicylab_config(xpl_root, server.url),
            )
            result = await asyncio.to_thread(xpl.infer, sequence)
            await asyncio.to_thread(xpl.reset)
            return result
        finally:
            if xpl is not None:
                await asyncio.to_thread(xpl.close)
            await server.stop()

    xpl_result = asyncio.run(run_xpl_round_trip())
    received = fixed_policy.last_observation
    assert fixed_policy.observation is None  # reset after inference cleared recurrent state
    assert fixed_policy.reset_count == 1
    assert received["instruction"] == "pick up the apple"
    for source, target in zip(
        CAMERA_SOURCES,
        ("cam_head", "cam_left_wrist", "cam_right_wrist"),
        strict=True,
    ):
        np.testing.assert_array_equal(received["vision"][target]["color"], observation[source])
    expected_state = {
        "left_arm_joint_state": G1_STATE[0:7],
        "right_arm_joint_state": G1_STATE[7:14],
        "left_ee_joint_state": G1_STATE[14:15],
        "right_ee_joint_state": G1_STATE[15:16],
        "head_joint_state": G1_STATE[16:18],
        "waist_joint_state": G1_STATE[18:20],
    }
    assert set(received["state"]) == set(expected_state)
    for key, expected in expected_state.items():
        np.testing.assert_array_equal(received["state"][key], expected)
    np.testing.assert_array_equal(starvla_result["pred_action"], G1_ACTIONS)
    np.testing.assert_array_equal(xpl_result["pred_action"], G1_ACTIONS)
    np.testing.assert_array_equal(starvla_result["pred_action"], xpl_result["pred_action"])
    assert starvla_result["ref_timestamp"] == xpl_result["ref_timestamp"] == 12.5
    assert starvla_result["loc_timestamp"] == xpl_result["loc_timestamp"] == 13.0

    starvla_robot = _dispatch_without_hardware(monkeypatch, starvla_result["pred_action"])
    xpl_robot = _dispatch_without_hardware(monkeypatch, xpl_result["pred_action"])
    for target in (starvla_robot, xpl_robot):
        assert len(target.arm_targets) == len(G1_ACTIONS)
        # G1's real config sends gripper commands every 40 control ticks;
        # a two-step probe therefore dispatches only the first gripper target.
        assert len(target.gripper_targets) == 1
        # G1's active Open-Rail policy layout is 16D; head/waist are manual.
        assert target.other_calls == []
        for actual, expected in zip(target.arm_targets, G1_ACTIONS[:, :14], strict=True):
            np.testing.assert_array_equal(actual, expected)
        # RAIL and SDK share the same 0=open, 1=closed command convention.
        expected_grippers = np.clip(G1_ACTIONS[:, 14:16], 0.0, 1.0)
        for actual, expected in zip(target.gripper_targets, expected_grippers[:1], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

    for actual, expected in zip(starvla_robot.arm_targets, xpl_robot.arm_targets, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        starvla_robot.gripper_targets,
        xpl_robot.gripper_targets,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
