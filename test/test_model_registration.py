"""Exercise every model's CLI/factory routing without loading model dependencies."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

import run_server
from conf.models_conf import ModelType
from conf.server_conf import get_vla_server_config

MODEL_MODULES = {
    "mock": "server.models.mock.mock",
    "act": "server.models.act",
    "gr00t_n1": "server.models.gr00t.gr00t_n1",
    "gr00t_n1_5": "server.models.gr00t.gr00t_n1_5",
    "gr00t_n1_6": "server.models.gr00t.gr00t_n1_6",
    "rdt": "server.models.rdt",
    "smolvla": "server.models.smolvla",
    "go1": "server.models.go1",
    "pi0": "server.models.openpi.pi0",
    "pi05": "server.models.openpi.pi05",
    "pi05_pytorch": "server.models.openpi.pi05_pytorch",
    "tao": "server.models.tao",
    "dm05": "server.models.dm05",
    "wallx": "server.models.wallx",
    "starvla": "server.models.starvla",
    "xpolicylab": "server.models.xpolicylab",
}


@pytest.mark.parametrize("model_type", list(ModelType))
def test_cli_and_factory_route_each_model_to_its_own_configuration(monkeypatch, model_type):
    monkeypatch.setattr(sys, "argv", ["run_server.py", "--model_type", model_type.value])
    args = run_server.parse_args()
    config = run_server.override_config_with_args(get_vla_server_config(), args)
    expected_key = "gr00t" if model_type.value.startswith("gr00t_") else model_type.value
    expected = config.models[expected_key]
    module = ModuleType(MODEL_MODULES[model_type.value])
    module.ModelVLA = lambda cfg: SimpleNamespace(cfg=cfg)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = run_server.get_model(config.models)
    assert model.cfg is expected


@pytest.mark.parametrize("model_type", [m for m in ModelType if m.value not in ("mock", "xpolicylab")])
def test_checkpoint_override_preserves_other_models(model_type):
    config = get_vla_server_config()
    before = config.models.to_dict()
    args = SimpleNamespace(model_type=model_type.value, model_path="/test/checkpoint")
    run_server.override_config_with_args(config, args)
    expected_key = "gr00t" if model_type.value.startswith("gr00t_") else model_type.value
    assert config.models[expected_key].model_path == "/test/checkpoint"
    for key, value in before.items():
        if key not in ("type", expected_key):
            assert config.models[key].to_dict() == value


def test_remote_policy_rejects_a_local_checkpoint():
    config = get_vla_server_config()
    with pytest.raises(ValueError, match="--model_path is not used"):
        run_server.override_config_with_args(
            config, SimpleNamespace(model_type="xpolicylab", model_path="/wrong/location")
        )


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Invalid model type"):
        run_server.get_model(SimpleNamespace(type="unknown"))


def test_model_defaults_do_not_select_a_robot_layout():
    models = get_vla_server_config().models
    assert models.starvla.get('state_indices') is None
    assert models.starvla.get('action_indices') is None
    assert models.xpolicylab.state_fields == []
    assert models.xpolicylab.action_keys == []
