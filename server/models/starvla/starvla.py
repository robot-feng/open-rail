"""Open-Rail adapter for checkpoints served by the StarVLA framework."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _parse_indices(indices, name: str) -> tuple[int, ...] | None:
    """Read source indices listed in the desired destination order."""
    if indices is None:
        return None
    if (
        not isinstance(indices, (list, tuple))
        or not indices
        or any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or index < 0
            for index in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise ValueError(
            f"{name} must be None or a non-empty list of distinct non-negative integers"
        )
    return tuple(int(index) for index in indices)


def _select_dimensions(values: np.ndarray, indices, name: str) -> np.ndarray:
    if indices is None:
        return values
    if max(indices) >= values.shape[-1]:
        raise ValueError(f"{name} exceeds source dimension {values.shape[-1]}: {indices}")
    return values[..., indices]


def _field_indices(fields, keys):
    """Resolve checkpoint field order using the robot-owned slices."""
    if not isinstance(keys, (list, tuple)) or not keys:
        raise ValueError("rail_io field keys must be a non-empty list of names")
    by_name = {field["name"]: field for field in fields}
    if any(not isinstance(key, str) or key not in by_name for key in keys):
        raise ValueError(f"Unknown robot field in checkpoint rail_io: {keys}")
    return _parse_indices([
        index for key in keys
        for index in range(by_name[key]["start"], by_name[key]["end"])
    ], "rail_io")


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
    if isinstance(value, np.ndarray):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("StarVLA requires a non-empty language instruction")
    return value


def _letterbox_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an RGB frame to the checkpoint image size without distortion."""
    from PIL import Image

    image = _to_hwc(image, "input")
    if image.shape[:2] == (height, width) and image.shape[2] == 3:
        return np.ascontiguousarray(image)

    pil_image = Image.fromarray(image).convert("RGB")
    source_width, source_height = pil_image.size
    scale = min(width / source_width, height / source_height)
    resized_width = min(width, max(1, int(round(source_width * scale))))
    resized_height = min(height, max(1, int(round(source_height * scale))))
    pil_image = pil_image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )

    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    offset = ((width - resized_width) // 2, (height - resized_height) // 2)
    canvas.paste(pil_image, offset)
    return np.asarray(canvas, dtype=np.uint8)


def _read_run_config(checkpoint: str):
    """Read optional run metadata without importing the model framework."""
    try:
        run_dir = Path(checkpoint).expanduser().resolve().parents[1]
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
    except (FileNotFoundError, IndexError):
        return {}
    if not isinstance(config, dict):
        raise ValueError("StarVLA run config.yaml must contain a mapping")
    return config


class ModelVLA:
    """Convert Open-Rail observations to StarVLA examples and return action chunks.

    The StarVLA repository is loaded lazily from ``config.repo_path``. Model
    loading and action un-normalization use StarVLA's own policy code. Wrapped
    checkpoints use ``PolicyServerWrapper``; native checkpoints use the
    StarVLA framework registry.
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
        self._state_indices = _parse_indices(config.get("state_indices"), "state_indices")
        self._action_indices = _parse_indices(config.get("action_indices"), "action_indices")
        run_config = _read_run_config(str(model_path))
        io_config = run_config.get("rail_io", {})
        if not isinstance(io_config, dict):
            raise ValueError("Checkpoint rail_io must contain a mapping")
        if io_config:
            from conf.robots_conf import get_robots_config

            if self._state_indices is not None or self._action_indices is not None:
                raise ValueError("Use checkpoint rail_io or explicit indices, not both")
            robots = get_robots_config()
            robot_type = io_config["robot_type"]
            if robot_type not in robots:
                raise ValueError(f"Unknown checkpoint rail_io.robot_type: {robot_type!r}")
            fields = robots[robot_type].get("io_fields")
            if not fields:
                raise ValueError(f"Robot {robot_type!r} has no io_fields definition")
            default_keys = [field["name"] for field in fields]
            self._state_indices = _field_indices(fields, io_config.get("state_keys", default_keys))
            native_action_indices = _field_indices(
                fields, io_config.get("action_keys", default_keys)
            )
            if sorted(native_action_indices) != sorted(_field_indices(fields, default_keys)):
                raise ValueError("Checkpoint rail_io.action_keys must cover all robot fields")
            # Reverse the checkpoint packing to restore canonical RAIL field order.
            self._action_indices = tuple(sorted(
                range(len(native_action_indices)), key=native_action_indices.__getitem__
            ))

        # Keep StarVLA's heavy dependencies out of the Open-Rail server unless
        # this backend is selected.
        self._native_framework = False
        self._state_normalizer = None
        self._action_normalizer = None
        framework_config = run_config.get("framework", {})
        if framework_config.get("name") and "action_model" not in framework_config:
            from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
            from starVLA.model.framework.base_framework import baseframework, merge_config_overrides
            from starVLA.model.framework.share_tools import read_mode_config

            overrides = config.get("config_overrides", [])
            if isinstance(overrides, str) or not isinstance(overrides, (list, tuple)):
                raise TypeError("config_overrides must be a list of KEY=VALUE strings")
            checkpoint_config, stats = read_mode_config(str(model_path))
            checkpoint_config = merge_config_overrides(checkpoint_config, overrides)
            self.policy = baseframework.from_pretrained(
                str(model_path), config_overrides=list(overrides)
            ).to(str(config.get("device", "cuda"))).eval()
            framework = checkpoint_config["framework"]
            stats_key = config.get("unnorm_key")
            if stats_key is None:
                if len(stats) != 1:
                    raise ValueError(
                        "Set models.starvla.unnorm_key for a checkpoint with multiple "
                        "normalization-statistics entries"
                    )
                stats_key = next(iter(stats))
            if stats_key not in stats:
                raise KeyError(f"Normalization key {stats_key!r} not found in {list(stats)}")
            mode = checkpoint_config.get("datasets", {}).get("vla_data", {}).get(
                "normalization_mode", "q99"
            )
            self._state_normalizer = Normalizer(mode, dict(stats[stats_key]["state"]))
            self._action_normalizer = Normalizer(mode, dict(stats[stats_key]["action"]))
            action_model = framework.get("action_model", {})
            chunk_size = framework.get("n_action_steps") or action_model.get("action_horizon")
            if chunk_size is None and "future_action_window_size" in action_model:
                chunk_size = int(action_model["future_action_window_size"]) + 1
            if chunk_size is None:
                raise ValueError("StarVLA framework config does not declare its action horizon")
            image_size = framework.get("image_size")
            self.policy_metadata = {
                "action_chunk_size": int(chunk_size),
                "action_dim": int(framework["action_dim"]),
                "state_dim": int(framework.get("state_dim", 0)),
                "image_size": list(image_size) if image_size is not None else None,
                "image_keys": list(framework.get("image_keys", [])),
                "normalization_key": stats_key,
                "num_inference_steps": framework.get("num_inference_steps"),
            }
            self._native_framework = True
        else:
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
            self.policy_metadata = dict(self.policy.metadata)
        self.policy_metadata["state_indices"] = (
            list(self._state_indices) if self._state_indices is not None else None
        )
        self.policy_metadata["action_indices"] = (
            list(self._action_indices) if self._action_indices is not None else None
        )
        self.policy_metadata["rail_io"] = io_config
        self._action_horizon = self.policy_metadata.get("action_chunk_size")
        self._state_processor = None
        if (
            self._send_state
            and config.get("normalize_state", True)
            and not self._native_framework
        ):
            self._state_processor = self.policy.get_norm_processor(config.get("unnorm_key"))
            if not self._state_processor.state_keys:
                raise ValueError("StarVLA checkpoint has no state keys; set send_state=False")
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
        images = [_to_hwc(obs[key], key) for key in self._camera_keys]
        image_size = self.policy_metadata.get("image_size")
        if self._native_framework and image_size is not None:
            height, width = image_size
            images = [_letterbox_rgb(image, height, width) for image in images]
        model_image_keys = self.policy_metadata.get("image_keys")
        if model_image_keys and len(model_image_keys) != len(images):
            raise ValueError(
                f"Checkpoint expects {len(model_image_keys)} camera views, "
                f"but {len(images)} are configured"
            )
        example = {
            "image": images,
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
            # Match the model's channel order before applying its channel statistics.
            state = _select_dimensions(state, self._state_indices, "state_indices")
            model_state_dim = self.policy_metadata.get("state_dim")
            if model_state_dim and state.shape[1] != model_state_dim:
                raise ValueError(
                    f"StarVLA state dimension {state.shape[1]} does not match checkpoint "
                    f"dimension {model_state_dim}"
                )
            example["state"] = self._prepare_state(state)
        return example

    def _prepare_state(self, state: np.ndarray) -> np.ndarray:
        if not self.cfg.get("normalize_state", True):
            return state
        if self._native_framework:
            import torch

            return np.ascontiguousarray(
                self._state_normalizer.forward(torch.from_numpy(state.copy())).numpy(),
                dtype=np.float32,
            )
        processor = self._state_processor
        if processor is None:
            return state
        keys = processor.state_keys
        dims = [processor.state_key_dims[key] for key in keys]
        if state.shape[1] != sum(dims):
            raise ValueError(
                f"StarVLA state dimension {state.shape[1]} does not match checkpoint "
                f"state dimension {sum(dims)} ({keys})"
            )
        # Apply only state transforms: image augmentation and action transforms
        # belong to their own paths. The processor already bound checkpoint stats.
        fields = dict(zip(keys, np.split(state.copy(), np.cumsum(dims)[:-1], axis=-1)))
        for transform in processor.transform.transforms:
            if any(key in keys for key in transform.apply_to):
                fields = transform(fields)
        normalized = np.concatenate([np.asarray(fields[key]) for key in keys], axis=-1)
        if not np.isfinite(normalized).all():
            raise ValueError("StarVLA state transforms produced non-finite values")
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def infer(self, sequence):
        if not sequence:
            raise ValueError("StarVLA inference requires at least one observation")
        data = sequence[0]
        example = self._build_example(data["obs"])
        kwargs = {}
        unnorm_key = self.cfg.get("unnorm_key")
        if unnorm_key is not None:
            kwargs["unnorm_key"] = unnorm_key
        if self._native_framework:
            import torch

            normalized = self.policy.predict_action(examples=[example], **kwargs)[
                "normalized_actions"
            ]
            actions = self._action_normalizer.inverse(
                torch.as_tensor(np.asarray(normalized), dtype=torch.float32)
            ).cpu().numpy()
        else:
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
        model_action_dim = self.policy_metadata.get("action_dim")
        if model_action_dim and actions.shape[1] != model_action_dim:
            raise ValueError(
                f"StarVLA action dimension {actions.shape[1]} does not match checkpoint "
                f"dimension {model_action_dim}"
            )
        if self.policy_metadata.get("rail_io") and len(self._action_indices) != actions.shape[1]:
            raise ValueError("Checkpoint rail_io.action_keys do not cover its action dimension")
        if self._action_horizon is not None and actions.shape[0] != self._action_horizon:
            raise ValueError(
                f"StarVLA action horizon {actions.shape[0]} does not match server metadata "
                f"{self._action_horizon}"
            )
        # Actions are now in physical units; return the configured RAIL order.
        actions = np.ascontiguousarray(
            _select_dimensions(actions, self._action_indices, "action_indices"),
            dtype=np.float32,
        )

        return {
            "type": "vla_action",
            "pred_action": actions,
            "ref_timestamp": data["ref_timestamp"],
            "loc_timestamp": data["loc_timestamp"],
            "ext": {},
        }
