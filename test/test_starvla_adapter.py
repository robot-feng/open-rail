from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from ml_collections import ConfigDict

from server.models.starvla.starvla import ModelVLA


class _FakePolicyServerWrapper:
    latest = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.metadata = {"action_chunk_size": 2, "state_dim": 4, "action_dim": 4}
        self.response = {"actions": np.arange(8, dtype=np.float32).reshape(1, 2, 4)}
        self.examples = None
        self.predict_kwargs = None
        type(self).latest = self

    def predict_action(self, examples, **kwargs):
        self.examples = examples
        self.predict_kwargs = kwargs
        return self.response


def _install_fake_wrapper(monkeypatch, tmp_path):
    repo = tmp_path / "starvla_repo"
    (repo / "starVLA").mkdir(parents=True)
    wrapper_file = repo / "deployment" / "model_server" / "policy_wrapper.py"
    wrapper_file.parent.mkdir(parents=True)
    wrapper_file.touch()

    deployment = ModuleType("deployment")
    deployment.__path__ = []
    model_server = ModuleType("deployment.model_server")
    model_server.__path__ = []
    policy_wrapper = ModuleType("deployment.model_server.policy_wrapper")
    policy_wrapper.PolicyServerWrapper = _FakePolicyServerWrapper
    monkeypatch.setitem(sys.modules, "deployment", deployment)
    monkeypatch.setitem(sys.modules, "deployment.model_server", model_server)
    monkeypatch.setitem(sys.modules, "deployment.model_server.policy_wrapper", policy_wrapper)
    # The adapter adds the repository to sys.path during lazy import.
    monkeypatch.setattr(sys, "path", list(sys.path))
    return repo


def _config(repo, **overrides):
    config = {
        "repo_path": str(repo),
        "model_path": "/checkpoints/test-policy",
        "device": "cpu",
        "use_bf16": False,
        "unnorm_key": "g1-test",
        "config_overrides": ["framework.action_model.action_horizon=2"],
        "camera_keys": ["cam.head", "cam.hand_left", "cam.hand_right"],
        "send_state": True,
        "normalize_state": False,
        "state_key": "state",
        "language_key": "language",
    }
    config.update(overrides)
    return ConfigDict(config)


def _observation(images=None, state=None, language=None):
    if images is None:
        images = {
            "cam.head": np.full((5, 7, 3), [11, 31, 71], dtype=np.uint8),
            "cam.hand_left": np.full((5, 7, 3), [13, 37, 83], dtype=np.uint8),
            "cam.hand_right": np.full((5, 7, 3), [17, 41, 97], dtype=np.uint8),
        }
    return {
        **images,
        "state": np.arange(4, dtype=np.float32) if state is None else state,
        "language": ["move the cup"] if language is None else language,
    }


def _frame(obs):
    return [{"obs": obs, "ref_timestamp": 12.5, "loc_timestamp": 13.0}]


def _make_model(monkeypatch, tmp_path, **config_overrides):
    repo = _install_fake_wrapper(monkeypatch, tmp_path)
    return ModelVLA(_config(repo, **config_overrides))


def test_builds_ordered_starvla_example_and_passes_checkpoint_options(monkeypatch, tmp_path):
    model = _make_model(monkeypatch, tmp_path)
    obs = _observation()
    output = model.infer(_frame(obs))

    policy = model.policy
    assert policy.kwargs == {
        "ckpt_path": "/checkpoints/test-policy",
        "device": "cpu",
        "use_bf16": False,
        "unnorm_key": "g1-test",
        "config_overrides": ["framework.action_model.action_horizon=2"],
    }
    example = policy.examples[0]
    np.testing.assert_array_equal(example["image"], [obs[key] for key in (
        "cam.head", "cam.hand_left", "cam.hand_right"
    )])
    assert example["lang"] == "move the cup"
    np.testing.assert_array_equal(example["state"], np.arange(4, dtype=np.float32)[None, :])
    assert policy.predict_kwargs == {"unnorm_key": "g1-test"}
    assert output["type"] == "vla_action"
    assert output["ref_timestamp"] == 12.5
    assert output["loc_timestamp"] == 13.0
    assert output["pred_action"].shape == (2, 4)
    np.testing.assert_array_equal(output["pred_action"], np.arange(8).reshape(2, 4))


def test_supports_plain_dict_model_config(monkeypatch, tmp_path):
    repo = _install_fake_wrapper(monkeypatch, tmp_path)
    model = ModelVLA(dict(_config(repo)))

    model.infer(_frame(_observation()))

    assert model.policy.predict_kwargs == {"unnorm_key": "g1-test"}


@pytest.mark.parametrize('interleaved_actions', [False, True])
def test_checkpoint_named_layout_uses_robot_definition(monkeypatch, tmp_path, interleaved_actions):
    from conf.robots_conf import get_agibot_g1_config

    repo = _install_fake_wrapper(monkeypatch, tmp_path)
    run_dir = tmp_path / 'run'
    (run_dir / 'checkpoints').mkdir(parents=True)
    canonical = [field['name'] for field in get_agibot_g1_config().io_fields]
    interleaved = [canonical[i] for i in [0, 2, 1, 3, 4, 5]]
    (run_dir / 'config.yaml').write_text(yaml.safe_dump({'rail_io': {
        'robot_type': 'agibot_g1',
        'state_keys': interleaved,
        'action_keys': interleaved if interleaved_actions else canonical,
    }}))
    model = ModelVLA(_config(repo, model_path=str(run_dir / 'checkpoints/model.pt')))
    model.policy_metadata.update(state_dim=20, action_dim=20)
    model.policy.response = {'actions': np.arange(40, dtype=np.float32).reshape(1, 2, 20)}
    original = np.arange(20, dtype=np.float32)
    result = model.infer(_frame(_observation(state=original)))

    expected_state = list(range(7)) + [14] + list(range(7, 14)) + list(range(15, 20))
    np.testing.assert_array_equal(model.policy.examples[0]['state'], [expected_state])
    expected_action = np.arange(40).reshape(2, 20)
    if interleaved_actions:
        expected_action = expected_action[:, list(range(7)) + list(range(8, 15)) + [7] + list(range(15, 20))]
    np.testing.assert_array_equal(result['pred_action'], expected_action)
    np.testing.assert_array_equal(original, np.arange(20))


@pytest.mark.parametrize("io_config, message", [
    ("agibot_g1", "must contain a mapping"),
    ({"robot_type": "missing"}, "Unknown checkpoint"),
    ({"robot_type": "agibot_g1", "state_keys": ["missing"]}, "Unknown robot field"),
    ({"robot_type": "agibot_g1", "state_keys": ["left_arm_joint_state"] * 2}, "distinct"),
    ({"robot_type": "agibot_g1", "action_keys": ["right_arm_joint_state"]}, "cover all robot fields"),
])
def test_rejects_invalid_checkpoint_layouts(monkeypatch, tmp_path, io_config, message):
    repo = _install_fake_wrapper(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"rail_io": io_config}))
    with pytest.raises(ValueError, match=message):
        ModelVLA(_config(repo, model_path=str(run_dir / "checkpoints/model.pt")))


def _make_native_model(monkeypatch, tmp_path, *, stats=None, **options):
    """Exercise the native loader without importing a GPU framework or weights."""
    from copy import deepcopy

    torch = pytest.importorskip("torch")
    repo = _install_fake_wrapper(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    checkpoint_config = {
        "framework": {
            "name": "TestPolicy", "state_dim": 4, "action_dim": 4,
            "n_action_steps": 2, "num_inference_steps": 20,
            "image_size": [8, 8], "image_keys": ["head", "left", "right"],
        },
        "datasets": {"vla_data": {"normalization_mode": "q99"}},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(checkpoint_config))
    if stats is None:
        stats = {"g1-test": {
            "state": {"offset": [1, 2, 3, 4], "scale": [1, 2, 4, 8]},
            "action": {"offset": [100, 200, 300, 400], "scale": [1, 10, 100, 1000]},
        }}

    def merge_config_overrides(config, overrides):
        result = deepcopy(config)
        for override in overrides:
            key, value = override.split("=", 1)
            parent = result
            parts = key.split(".")
            for part in parts[:-1]:
                parent = parent.setdefault(part, {})
            parent[parts[-1]] = yaml.safe_load(value)
        return result

    class NativePolicy:
        @classmethod
        def from_pretrained(cls, checkpoint, config_overrides):
            instance = cls()
            instance.loaded_config = merge_config_overrides(checkpoint_config, config_overrides)
            return instance

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

        def predict_action(self, examples, **kwargs):
            self.examples = examples
            horizon = self.loaded_config["framework"]["n_action_steps"]
            return {"normalized_actions": np.arange(
                1, horizon * 4 + 1, dtype=np.float32
            ).reshape(1, horizon, 4)}

    class Normalizer:
        def __init__(self, mode, values):
            self.mode = mode
            self.offset = torch.tensor(values["offset"], dtype=torch.float32)
            self.scale = torch.tensor(values["scale"], dtype=torch.float32)

        def forward(self, values):
            return (values - self.offset) / self.scale

        def inverse(self, values):
            return values * self.scale + self.offset

    modules = {
        "starVLA.model.framework.base_framework": {
            "baseframework": NativePolicy, "merge_config_overrides": merge_config_overrides,
        },
        "starVLA.model.framework.share_tools": {
            "read_mode_config": lambda path: (deepcopy(checkpoint_config), stats),
        },
        "starVLA.dataloader.gr00t_lerobot.transform.state_action": {"Normalizer": Normalizer},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    config = _config(
        repo, model_path=str(run_dir / "checkpoints/model.pt"), normalize_state=True,
        config_overrides=[],
    )
    config.update(options)
    return ModelVLA(config)


def test_native_loader_uses_effective_config_and_training_normalizers(monkeypatch, tmp_path):
    model = _make_native_model(
        monkeypatch, tmp_path,
        state_indices=[2, 0, 3, 1], action_indices=[1, 3, 0, 2],
        config_overrides=[
            "framework.n_action_steps=3", "framework.num_inference_steps=10",
            "datasets.vla_data.normalization_mode=min_max",
        ],
    )
    result = model.infer(_frame(_observation(state=np.array([10, 20, 30, 40]))))
    assert model._native_framework
    assert model.policy.device == "cpu"
    assert model.policy.evaluated
    assert model.policy_metadata["action_chunk_size"] == 3
    assert model.policy_metadata["num_inference_steps"] == 10
    assert model._state_normalizer.mode == model._action_normalizer.mode == "min_max"
    np.testing.assert_array_equal(model.policy.examples[0]["state"], [[29, 4, 9.25, 2]])
    np.testing.assert_array_equal(result["pred_action"], [
        [220, 4400, 101, 600], [260, 8400, 105, 1000], [300, 12400, 109, 1400],
    ])
    for image, key in zip(model.policy.examples[0]["image"], model._camera_keys):
        assert image.shape == (8, 8, 3)
        np.testing.assert_array_equal(image[0], np.zeros((8, 3), dtype=np.uint8))
        np.testing.assert_array_equal(image[3, 3], _observation()[key][0, 0])


@pytest.mark.parametrize("stats, unnorm_key, error, message", [
    ({"a": {}, "b": {}}, None, ValueError, "unnorm_key"),
    ({"a": {}}, "missing", KeyError, "Normalization key"),
])
def test_native_loader_rejects_ambiguous_statistics(
    monkeypatch, tmp_path, stats, unnorm_key, error, message
):
    with pytest.raises(error, match=message):
        _make_native_model(monkeypatch, tmp_path, stats=stats, unnorm_key=unnorm_key)


def test_reuses_checkpoint_state_transforms_in_training_key_order(monkeypatch, tmp_path):
    calls = []

    def state_transform(fields):
        calls.append("state")
        return {key: value + offset for (key, value), offset in zip(fields.items(), [10, 20])}

    class Transform:
        apply_to = ("state.arm", "state.gripper")

        def __call__(self, fields):
            return state_transform(fields)

    class ImageTransform:
        apply_to = ("video.head",)

        def __call__(self, fields):
            raise AssertionError("Image transforms must not run on packed state")

    processor = SimpleNamespace(
        state_keys=["state.arm", "state.gripper"],
        state_key_dims={"state.gripper": 1, "state.arm": 3},
        transform=SimpleNamespace(transforms=[ImageTransform(), Transform()]),
    )
    monkeypatch.setattr(
        _FakePolicyServerWrapper, "get_norm_processor", lambda self, key: processor, raising=False
    )
    model = _make_model(monkeypatch, tmp_path, normalize_state=True)
    observation = _observation()
    model.infer(_frame(observation))
    np.testing.assert_array_equal(model.policy.examples[0]["state"], [[10, 11, 12, 23]])
    np.testing.assert_array_equal(observation["state"], [0, 1, 2, 3])
    assert calls == ["state"]
    with pytest.raises(ValueError, match="checkpoint state dimension"):
        model._prepare_state(np.zeros((1, 5), dtype=np.float32))


def test_maps_state_before_normalization_and_action_after_inverse(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")

    class AffineNormalizer:
        def __init__(self, offset, scale):
            self.offset = torch.tensor(offset, dtype=torch.float32)
            self.scale = torch.tensor(scale, dtype=torch.float32)

        def forward(self, values):
            return (values - self.offset) / self.scale

        def inverse(self, values):
            return values * self.scale + self.offset

    # Use a permutation that differs from its inverse to catch direction errors.
    model = _make_model(
        monkeypatch, tmp_path,
        state_indices=[2, 0, 3, 1],
        action_indices=[1, 3, 0, 2],
    )
    model._native_framework = True
    model.cfg.normalize_state = True
    model._state_normalizer = AffineNormalizer([1, 2, 3, 4], [1, 2, 4, 8])
    model._action_normalizer = AffineNormalizer([100, 200, 300, 400], [1, 10, 100, 1000])

    def predict_action(examples, **kwargs):
        model.policy.examples = examples
        return {"normalized_actions": np.arange(1, 9, dtype=np.float32).reshape(1, 2, 4)}

    model.policy.predict_action = predict_action
    observation = _observation(state=np.array([10, 20, 30, 40], dtype=np.float32))
    output = model.infer(_frame(observation))

    np.testing.assert_array_equal(model.policy.examples[0]["state"], [[29, 4, 9.25, 2]])
    np.testing.assert_array_equal(output["pred_action"], [
        [220, 4400, 101, 600],
        [260, 8400, 105, 1000],
    ])
    np.testing.assert_array_equal(observation["state"], [10, 20, 30, 40])
    assert output["pred_action"].dtype == np.float32
    assert output["pred_action"].flags.c_contiguous


def test_selects_action_channels_from_the_validated_model_output(monkeypatch, tmp_path):
    model = _make_model(monkeypatch, tmp_path, action_indices=[3, 1])
    output = model.infer(_frame(_observation()))

    np.testing.assert_array_equal(output["pred_action"], [[3, 1], [7, 5]])
    assert model.policy_metadata["action_dim"] == 4
    assert model.policy_metadata["action_indices"] == [3, 1]


@pytest.mark.parametrize("field", ["state_indices", "action_indices"])
@pytest.mark.parametrize("indices", [[], [0, 0], [-1, 0], [0, 1.5], [0, True], [0, "1"], "0,1"])
def test_rejects_ambiguous_or_invalid_channel_maps(monkeypatch, tmp_path, field, indices):
    with pytest.raises(ValueError, match=field):
        _make_model(monkeypatch, tmp_path, **{field: indices})


@pytest.mark.parametrize("field", ["state_indices", "action_indices"])
def test_rejects_channel_maps_outside_the_source_vector(monkeypatch, tmp_path, field):
    model = _make_model(monkeypatch, tmp_path, **{field: [4]})
    with pytest.raises(ValueError, match=f"{field} exceeds source dimension 4"):
        model.infer(_frame(_observation()))


def test_rejects_mapped_state_width_incompatible_with_checkpoint(monkeypatch, tmp_path):
    model = _make_model(monkeypatch, tmp_path, state_indices=[2, 0])
    with pytest.raises(ValueError, match="state dimension 2 does not match checkpoint dimension 4"):
        model.infer(_frame(_observation()))
    assert model.policy.examples is None


def test_converts_chw_images_to_hwc_and_supports_image_only_checkpoint(monkeypatch, tmp_path):
    model = _make_model(monkeypatch, tmp_path, send_state=False)
    base_obs = _observation()
    base_images = {key: base_obs[key] for key in model._camera_keys}
    chw_images = {key: np.moveaxis(image, -1, 0) for key, image in base_images.items()}
    example = model._build_example(_observation(images=chw_images))

    assert [image.shape for image in example["image"]] == [(5, 7, 3)] * 3
    for key, image in zip(model._camera_keys, example["image"], strict=True):
        np.testing.assert_array_equal(image, base_images[key])
    assert "state" not in example


@pytest.mark.parametrize(
    "obs, message",
    [
        (_observation(images={"cam.head": np.zeros((4, 4, 3), dtype=np.uint8)}), "Missing configured"),
        (_observation(state=np.array([0, np.nan, 2, 3], dtype=np.float32)), "finite values"),
        (_observation(language=[]), "non-empty language"),
        (_observation(language=""), "non-empty language"),
        (_observation(language="   "), "non-empty language"),
    ],
)
def test_rejects_invalid_observations_before_model_call(monkeypatch, tmp_path, obs, message):
    model = _make_model(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=message):
        model.infer(_frame(obs))
    assert model.policy.examples is None


@pytest.mark.parametrize(
    "response, config, metadata, message",
    [
        ({"actions": np.zeros((1, 2, 5), dtype=np.float32)}, {}, {}, "action dimension"),
        ({"actions": np.zeros((1, 2, 5), dtype=np.float32)}, {"action_indices": [0, 1]}, {}, "action dimension"),
        ({"actions": np.zeros((1, 1, 4), dtype=np.float32)}, {}, {}, "action horizon"),
        ({"actions": np.zeros((2, 2, 4), dtype=np.float32)}, {}, {}, "batch size"),
        ({"actions": np.full((1, 2, 4), np.nan, dtype=np.float32)}, {}, {}, "non-finite"),
        ({"actions": np.zeros((1, 2, 4), dtype=np.float32)}, {}, {"action_chunk_size": 3}, "action horizon"),
    ],
)
def test_rejects_incompatible_policy_outputs(monkeypatch, tmp_path, response, config, metadata, message):
    model = _make_model(monkeypatch, tmp_path, **config)
    model.policy.response = response
    model.policy.metadata = {"action_chunk_size": 2, **metadata}
    model._action_horizon = model.policy.metadata["action_chunk_size"]

    with pytest.raises(ValueError, match=message):
        model.infer(_frame(_observation()))


class _FakeZMQServer:
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.language = None

    def set_heartbeat_info(self, **kwargs):
        if "lang_cmd" in kwargs:
            self.language = kwargs["lang_cmd"]

    def sendMessage(self, client_id, data, meta):
        self.result = (client_id, data, dict(meta))
        self.event.set()


def test_openrail_jpeg_transport_to_starvla_policy_wrapper(monkeypatch, tmp_path):
    from server.core.vla_server import VLAServer

    model = _make_model(monkeypatch, tmp_path, camera_keys=["cam.head"])
    zmq = _FakeZMQServer()
    server = VLAServer.__new__(VLAServer)
    server.config = ConfigDict({"image_pad_and_resize": False})
    server.zmq_server = zmq
    server.model = model
    server.image_decode_executor = ThreadPoolExecutor(max_workers=2)
    server.executor = ThreadPoolExecutor(max_workers=2)
    server.thread_lock = threading.Lock()
    server._model_reset_lock = threading.Lock()
    server._client_reset_generations = {}
    server.request_count = 0
    server.obs_info = {}
    server.act_info = {}
    server.avg_inference_time = 0

    bgr = np.full((8, 10, 3), [15, 70, 210], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    item = {
        "obs": {
            "cam.head": encoded.tobytes(),
            "state": np.arange(4, dtype=np.float32),
            "language": ["move the cup"],
        },
        "ref_timestamp": 22.0,
        "loc_timestamp": 23.0,
    }
    try:
        server.inference(b"robot", item, {"request_id": "starvla-joint"})
        assert zmq.event.wait(5), "Open-Rail inference callback timed out"
        client_id, output, _meta = zmq.result
        assert client_id == b"robot"
        assert output["type"] == "vla_action"
        assert output["pred_action"].shape == (2, 4)
        assert output["ref_timestamp"] == 22.0
        example = model.policy.examples[0]
        expected_rgb = cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        np.testing.assert_array_equal(example["image"][0], expected_rgb)
        assert example["lang"] == "move the cup"
        assert zmq.language == ["move the cup"]
    finally:
        server.executor.shutdown(wait=True)
        server.image_decode_executor.shutdown(wait=True)
