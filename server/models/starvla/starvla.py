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
    if isinstance(value, np.ndarray):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("StarVLA requires a non-empty language instruction")
    return value


def _letterbox_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an RGB frame without changing its aspect ratio, then pad with black.

    The G1 DP training videos are stored as 224x224 letterboxed images. Applying
    this before StarVLA's own resize keeps live camera geometry consistent with
    those training frames.
    """
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


class _StarVLADiffusionPolicy:
    """Load StarVLA's flat-config DiffusionPolicy checkpoints."""

    def __init__(
        self, checkpoint: str, device: str, state_indices, num_inference_steps=None
    ):
        import json

        import torch
        from omegaconf import OmegaConf
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
        from starVLA.model.framework.VM4A.DiffusionPolicy import DiffusionPolicy

        self.checkpoint = Path(checkpoint).resolve()
        run_dir = self.checkpoint.parents[1]
        self.config = OmegaConf.load(run_dir / "config.yaml")
        framework_config = self.config.framework
        if framework_config.name != "DiffusionPolicy":
            raise ValueError("Expected a StarVLA DiffusionPolicy checkpoint")
        if num_inference_steps is not None:
            num_inference_steps = int(num_inference_steps)
            if num_inference_steps <= 0:
                raise ValueError("dp_num_inference_steps must be a positive integer")
            framework_config.num_inference_steps = num_inference_steps
        self.action_dim = int(framework_config.action_dim)
        self.state_dim = int(framework_config.state_dim)
        image_size = tuple(int(value) for value in framework_config.image_size)
        if len(image_size) != 2 or any(value <= 0 for value in image_size):
            raise ValueError("DiffusionPolicy image_size must be [height, width]")
        self.image_size = image_size
        self.image_keys = list(framework_config.image_keys)
        self.state_indices = np.asarray(state_indices, dtype=np.int64)
        if self.state_indices.shape != (self.state_dim,):
            raise ValueError(
                f"dp_state_indices must contain {self.state_dim} RAIL indices"
            )

        stats = json.loads((run_dir / "dataset_statistics.json").read_text())
        unnorm_key = "new_embodiment" if "new_embodiment" in stats else next(iter(stats))
        self._state_normalizer = Normalizer("q99", dict(stats[unnorm_key]["state"]))
        self._action_normalizer = Normalizer("q99", dict(stats[unnorm_key]["action"]))

        self.model = DiffusionPolicy(self.config).to(device).eval()
        weights = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.metadata = {
            "action_chunk_size": int(framework_config.n_action_steps),
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "image_size": list(self.image_size),
            "image_keys": self.image_keys,
            "normalization_key": unnorm_key,
            "num_inference_steps": int(framework_config.num_inference_steps),
        }

    def prepare_state(self, state: np.ndarray) -> np.ndarray:
        import torch

        if state.shape[1] != self.state_dim or np.max(self.state_indices) >= state.shape[1]:
            raise ValueError(
                f"RAIL state width {state.shape[1]} cannot map to DP state width {self.state_dim}"
            )
        ordered = state[:, self.state_indices]
        normalized = self._state_normalizer.forward(
            torch.from_numpy(ordered.copy())
        ).numpy()
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def predict_action(self, examples, **_kwargs):
        import torch

        normalized = self.model.predict_action(examples=examples)["normalized_actions"][0]
        actions = self._action_normalizer.inverse(torch.from_numpy(normalized)).numpy()
        return {"actions": actions[None, ...]}


def _is_flat_diffusion_policy_checkpoint(checkpoint: str) -> bool:
    try:
        from omegaconf import OmegaConf

        run_dir = Path(checkpoint).expanduser().resolve().parents[1]
        config = OmegaConf.load(run_dir / "config.yaml")
        framework = config.get("framework", {})
        return framework.get("name") == "DiffusionPolicy" and "action_model" not in framework
    except (OSError, IndexError, TypeError, ValueError):
        return False


class ModelVLA:
    """Convert Open-Rail observations to StarVLA examples and return action chunks.

    The StarVLA repository is loaded lazily from ``config.repo_path``. Model
    loading and action un-normalization use StarVLA's own policy code. Current
    wrapper checkpoints use ``PolicyServerWrapper``; older flat-config DP
    checkpoints use StarVLA's native ``DiffusionPolicy`` implementation.
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
        self._dp_action_state_indices = None
        if _is_flat_diffusion_policy_checkpoint(str(model_path)):
            self.policy = _StarVLADiffusionPolicy(
                checkpoint=str(model_path),
                device=str(config.get("device", "cuda")),
                state_indices=config.get("dp_state_indices", []),
                num_inference_steps=config.get("dp_num_inference_steps", 10),
            )
            self._dp_action_state_indices = np.asarray(
                config.get("dp_action_state_indices", []), dtype=np.int64
            )
            self._dp_rail_action_dim = int(config.get("dp_rail_action_dim", 20))
            if (
                self._dp_action_state_indices.shape != (self.policy.action_dim,)
                or np.any(self._dp_action_state_indices < 0)
                or np.any(self._dp_action_state_indices >= self._dp_rail_action_dim)
            ):
                raise ValueError(
                    "dp_action_state_indices must map every DP action dimension "
                    "into the RAIL action vector"
                )
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
        self.policy_metadata = self.policy.metadata
        self._action_horizon = self.policy_metadata.get("action_chunk_size")
        self._state_processor = None
        if (
            self._send_state
            and config.get("normalize_state", True)
            and not isinstance(self.policy, _StarVLADiffusionPolicy)
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
        if isinstance(self.policy, _StarVLADiffusionPolicy):
            height, width = self.policy.image_size
            images = [_letterbox_rgb(image, height, width) for image in images]
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
            if self._expected_state_dim is not None and state.shape[1] != self._expected_state_dim:
                raise ValueError(
                    f"StarVLA state dimension {state.shape[1]} does not match configured "
                    f"dimension {self._expected_state_dim}"
                )
            example["state"] = self._prepare_state(state)
        return example

    def _prepare_state(self, state: np.ndarray) -> np.ndarray:
        if isinstance(self.policy, _StarVLADiffusionPolicy):
            return self.policy.prepare_state(state)
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
        if self._dp_action_state_indices is not None:
            raw_state = np.asarray(data["obs"][self._state_key], dtype=np.float32).reshape(-1)
            if raw_state.size != self._dp_rail_action_dim:
                raise ValueError(
                    f"DP action expansion needs a {self._dp_rail_action_dim}D "
                    f"RAIL state, "
                    f"got {raw_state.size} values"
                )
            if actions.shape[1] != self._dp_action_state_indices.size:
                raise ValueError(
                    f"DP returned {actions.shape[1]} action values, expected "
                    f"{self._dp_action_state_indices.size}"
                )
            expanded = np.broadcast_to(
                raw_state,
                (actions.shape[0], self._dp_rail_action_dim),
            ).copy()
            expanded[:, self._dp_action_state_indices] = actions
            actions = expanded
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
