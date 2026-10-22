"""Open-Rail adapter for checkpoints served by the StarVLA framework."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _to_hwc(image: object, key: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"StarVLA camera {key!r} must be a 3D image, got {array.shape}")
    if array.shape[-1] in (1, 3, 4):
        return np.ascontiguousarray(array)
    if array.shape[0] in (1, 3, 4):
        return np.ascontiguousarray(np.moveaxis(array, 0, -1))
    raise ValueError(
        f"StarVLA camera {key!r} must be HWC or CHW with 1, 3, or 4 channels; "
        f"got {array.shape}"
    )


def _language_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("StarVLA requires a non-empty language instruction")
    return value


class ModelVLA:
    """Convert Open-Rail observations to StarVLA examples and return action chunks.

    The StarVLA repository is loaded lazily from ``config.repo_path``. Model
    loading, training-statistics lookup, and action un-normalization use its
    ``PolicyServerWrapper`` implementation.
    """

    def __init__(self, config):
        self.cfg = config
        repo_path = Path(config.get("repo_path", "")).expanduser().resolve()
        if not (
            repo_path.is_dir()
            and (repo_path / "starVLA").is_dir()
            and (repo_path / "deployment/model_server/policy_wrapper.py").is_file()
        ):
            raise FileNotFoundError(
                f"StarVLA source tree or PolicyServerWrapper not found at {repo_path}; "
                "set STARVLA_ROOT or models.starvla.repo_path"
            )
        repo_path_text = str(repo_path)
        if repo_path_text not in sys.path:
            sys.path.insert(0, repo_path_text)

        model_path = config.get("model_path")
        if not model_path or model_path == "/path/to/starvla/checkpoint":
            raise ValueError("Set models.starvla.model_path to a StarVLA checkpoint")

        camera_keys = config.get("camera_keys")
        if not isinstance(camera_keys, (list, tuple)) or not camera_keys:
            raise ValueError("models.starvla.camera_keys must be a non-empty ordered list")
        if any(not isinstance(key, str) or not key for key in camera_keys):
            raise ValueError("models.starvla.camera_keys must contain non-empty strings")
        if len(set(camera_keys)) != len(camera_keys):
            raise ValueError("models.starvla.camera_keys must not contain duplicates")
        self._camera_keys = tuple(camera_keys)
        self._send_state = bool(config.get("send_state", True))
        self._state_key = str(config.get("state_key", "state"))
        self._language_key = str(config.get("language_key", "language"))
        self._expected_state_dim = config.get("expected_state_dim")
        self._expected_action_dim = config.get("expected_action_dim")
        if self._expected_state_dim is not None and (
            type(self._expected_state_dim) is not int or self._expected_state_dim <= 0
        ):
            raise ValueError("expected_state_dim must be a positive integer or None")
        if self._expected_action_dim is not None and (
            type(self._expected_action_dim) is not int or self._expected_action_dim <= 0
        ):
            raise ValueError("expected_action_dim must be a positive integer or None")

        # Keep StarVLA's heavy dependencies out of the Open-Rail server unless
        # this backend is selected.
        try:
            from deployment.model_server.policy_wrapper import PolicyServerWrapper
        except ImportError as error:
            raise ImportError(
                "Could not import StarVLA's PolicyServerWrapper. Install the StarVLA "
                "runtime dependencies in this Python environment and verify repo_path."
            ) from error

        overrides = config.get("config_overrides", [])
        if isinstance(overrides, str) or not isinstance(overrides, (list, tuple)):
            raise TypeError("config_overrides must be a list of KEY=VALUE strings")
        self.policy = PolicyServerWrapper(
            ckpt_path=str(model_path),
            device=str(config.get("device", "cuda")),
            use_bf16=bool(config.get("use_bf16", False)),
            unnorm_key=config.get("unnorm_key"),
            config_overrides=list(overrides),
        )
        self.policy_metadata = self.policy.metadata
        self._action_horizon = self.policy_metadata.get("action_chunk_size")
        logger.info(
            "Loaded StarVLA checkpoint %s from %s (metadata=%s)",
            model_path,
            repo_path,
            self.policy_metadata,
        )

    def _build_example(self, obs: dict) -> dict:
        missing = [key for key in self._camera_keys if key not in obs]
        if missing:
            raise ValueError(
                f"Missing configured StarVLA camera(s) {missing}; available: {list(obs)}"
            )
        example = {
            "image": [_to_hwc(obs[key], key) for key in self._camera_keys],
            "lang": _language_text(obs.get(self._language_key)),
        }
        if self._send_state:
            if self._state_key not in obs:
                raise ValueError(f"Missing StarVLA state field {self._state_key!r}")
            state = np.asarray(obs[self._state_key], dtype=np.float32)
            if state.ndim == 1:
                state = state[None, :]
            if state.ndim != 2 or state.shape[0] != 1 or state.shape[1] == 0:
                raise ValueError(f"StarVLA state must have shape [D] or [1,D], got {state.shape}")
            if not np.isfinite(state).all():
                raise ValueError("StarVLA state must contain only finite values")
            if self._expected_state_dim is not None and state.shape[1] != self._expected_state_dim:
                raise ValueError(
                    f"StarVLA state dimension {state.shape[1]} does not match configured "
                    f"dimension {self._expected_state_dim}"
                )
            example["state"] = state
        return example

    def infer(self, sequence):
        if not sequence:
            raise ValueError("StarVLA inference requires at least one observation")
        data = sequence[0]
        example = self._build_example(data["obs"])
        kwargs = {}
        if self.cfg.get("unnorm_key") is not None:
            kwargs["unnorm_key"] = self.cfg.unnorm_key
        output = self.policy.predict_action(examples=[example], **kwargs)

        actions = np.asarray(output["actions"], dtype=np.float32)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"StarVLA returned batch size {actions.shape[0]}, expected 1")
            actions = actions[0]
        if actions.ndim != 2 or 0 in actions.shape:
            raise ValueError(f"StarVLA actions must have shape [T,D], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("StarVLA returned non-finite actions")
        if self._expected_action_dim is not None and actions.shape[1] != self._expected_action_dim:
            raise ValueError(
                f"StarVLA action dimension {actions.shape[1]} does not match configured "
                f"dimension {self._expected_action_dim}"
            )
        if self._action_horizon is not None and actions.shape[0] != self._action_horizon:
            raise ValueError(
                f"StarVLA action horizon {actions.shape[0]} does not match server metadata "
                f"{self._action_horizon}"
            )

        return {
            "type": "vla_action",
            "pred_action": actions,
            "ref_timestamp": data["ref_timestamp"],
            "loc_timestamp": data["loc_timestamp"],
            "ext": {},
        }
