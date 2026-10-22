"""Bridge Open-Rail observations to a separately hosted XPolicyLab policy."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _as_instruction(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("XPolicyLab requires a non-empty language instruction")
    return value


class ModelVLA:
    """Adapt XPolicyLab's standard observation/action contract to Open-Rail.

    Policy weights and algorithm dependencies stay in the XPolicyLab policy
    server process. This adapter only uses XPolicyLab's WebSocket client to
    send observations and receive action chunks.
    """

    def __init__(self, config):
        self.cfg = config
        repo_path = Path(config.get("repo_path", "")).expanduser().resolve()
        client_module = repo_path / "client_server" / "ws" / "model_client.py"
        if not client_module.is_file():
            raise FileNotFoundError(
                f"XPolicyLab WebSocket client not found at {client_module}; "
                "set XPOLICYLAB_ROOT or models.xpolicylab.repo_path"
            )

        # XPolicyLab is an independent checkout/package. Add its root only
        # when this backend is selected so other Open-Rail models stay light.
        repo_path_text = str(repo_path)
        if repo_path_text in sys.path:
            sys.path.remove(repo_path_text)
        sys.path.insert(0, repo_path_text)
        try:
            from client_server.ws.model_client import WsModelClient
        except ImportError as error:
            raise ImportError(
                "Could not import XPolicyLab's WebSocket client. Install the "
                "optional Open-Rail xpolicylab dependencies and check repo_path."
            ) from error

        self._camera_map = self._parse_camera_map(config.get("camera_map"))
        self._state_fields = self._parse_state_fields(config.get("state_fields"))
        self._action_keys = self._parse_action_keys(config.get("action_keys"))
        self._instruction_key = str(config.get("instruction_key", "language"))
        if not self._instruction_key:
            raise ValueError("models.xpolicylab.instruction_key must not be empty")
        self._expected_action_dim = config.get("expected_action_dim")
        if self._expected_action_dim is not None and (
            type(self._expected_action_dim) is not int or self._expected_action_dim <= 0
        ):
            raise ValueError("expected_action_dim must be a positive integer or None")

        client_kwargs = {
            "url": str(config.get("server_url", "ws://127.0.0.1:19000")),
            "evaluation_id": str(config.get("evaluation_id", "open-rail")),
            "trial_id": str(config.get("trial_id", "open-rail")),
            "action_case_id": config.get("action_case_id"),
        }
        for key in (
            "ws_ping_interval_s",
            "ws_ping_timeout_s",
            "connect_timeout_s",
            "handshake_timeout_s",
            "request_timeout_s",
            "max_connect_attempts",
            "connect_retry_delay_s",
            "max_connect_seconds",
            "close_timeout_s",
        ):
            value = config.get(key)
            if value is not None:
                client_kwargs[key] = value

        self.client = WsModelClient(**client_kwargs)
        self._client_lock = threading.Lock()
        self._client_reset_generations = {}
        if bool(config.get("prepare_case", False)):
            if not client_kwargs["action_case_id"]:
                self.client.close()
                raise ValueError(
                    "models.xpolicylab.prepare_case requires a non-empty action_case_id"
                )
            case_meta = config.get("case_meta", {})
            if not isinstance(case_meta, Mapping):
                self.client.close()
                raise TypeError("models.xpolicylab.case_meta must be a mapping")
            try:
                with self._client_lock:
                    self.client.call("prepare_case", dict(case_meta))
            except Exception:
                self.client.close()
                raise
        logger.info(
            "Connected to XPolicyLab policy server %s (evaluation_id=%s, trial_id=%s)",
            client_kwargs["url"],
            client_kwargs["evaluation_id"],
            client_kwargs["trial_id"],
        )

    @staticmethod
    def _parse_camera_map(camera_map: Any) -> tuple[tuple[str, str], ...]:
        if not isinstance(camera_map, (list, tuple)) or not camera_map:
            raise ValueError("models.xpolicylab.camera_map must be a non-empty list")
        parsed = []
        sources = set()
        targets = set()
        for entry in camera_map:
            if not isinstance(entry, Mapping):
                raise TypeError("Each camera_map entry must be a mapping with source/target")
            source, target = entry.get("source"), entry.get("target")
            if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
                raise ValueError("camera_map source and target must be non-empty strings")
            if source in sources or target in targets:
                raise ValueError("camera_map source and target names must not contain duplicates")
            parsed.append((source, target))
            sources.add(source)
            targets.add(target)
        return tuple(parsed)

    @staticmethod
    def _parse_state_fields(state_fields: Any) -> tuple[tuple[str, int, int], ...]:
        if not isinstance(state_fields, (list, tuple)) or not state_fields:
            raise ValueError("models.xpolicylab.state_fields must be a non-empty list")
        parsed = []
        names = set()
        for entry in state_fields:
            if not isinstance(entry, Mapping):
                raise TypeError("Each state_fields entry must contain name, start, and end")
            name, start, end = entry.get("name"), entry.get("start"), entry.get("end")
            if not isinstance(name, str) or not name:
                raise ValueError("state_fields names must be non-empty strings")
            if name in names:
                raise ValueError(f"Duplicate XPolicyLab state field {name!r}")
            if type(start) is not int or type(end) is not int or start < 0 or end <= start:
                raise ValueError(f"Invalid slice for XPolicyLab state field {name!r}: {start}:{end}")
            parsed.append((name, start, end))
            names.add(name)
        return tuple(parsed)

    @staticmethod
    def _parse_action_keys(action_keys: Any) -> tuple[str, ...]:
        if not isinstance(action_keys, (list, tuple)) or not action_keys:
            raise ValueError("models.xpolicylab.action_keys must be a non-empty ordered list")
        if any(not isinstance(key, str) or not key for key in action_keys):
            raise ValueError("action_keys must contain non-empty strings")
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("action_keys must not contain duplicates")
        return tuple(action_keys)

    @staticmethod
    def _normalize_state(state: Any) -> dict[str, np.ndarray]:
        if isinstance(state, Mapping):
            normalized = {}
            for key, value in state.items():
                array = np.asarray(value, dtype=np.float32).reshape(-1)
                if array.size == 0 or not np.isfinite(array).all():
                    raise ValueError(f"Open-Rail state field {key!r} must be finite and non-empty")
                normalized[str(key)] = array
            if not normalized:
                raise ValueError("Open-Rail state mapping must not be empty")
            return normalized

        array = np.asarray(state, dtype=np.float32)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"Open-Rail state must be a vector or [1,D] array, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Open-Rail state must contain only finite values")
        return {"__flat_state__": array}

    def _build_observation(self, obs: Any) -> dict[str, Any]:
        if not isinstance(obs, Mapping):
            raise TypeError(f"Open-Rail observation must be a mapping, got {type(obs).__name__}")

        vision = {}
        for source, target in self._camera_map:
            if source not in obs:
                raise ValueError(
                    f"Missing Open-Rail camera {source!r}; available observation keys: {list(obs)}"
                )
            image = np.asarray(obs[source])
            if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"Camera {source!r} must be a decoded HWC image, got shape {image.shape}"
                )
            # Open-Rail already supplies decoded RGB images. XPolicyLab's
            # decoder leaves arrays alone, so do not decode or swap channels.
            vision[target] = {"color": np.ascontiguousarray(image)}

        if "state" not in obs:
            raise ValueError("Open-Rail observation is missing 'state'")
        normalized_state = self._normalize_state(obs["state"])
        if "__flat_state__" in normalized_state:
            flat_state = normalized_state["__flat_state__"]
            state = {}
            for name, start, end in self._state_fields:
                if end > flat_state.shape[0]:
                    raise ValueError(
                        f"State slice {name!r} [{start}:{end}] exceeds state dimension "
                        f"{flat_state.shape[0]}"
                    )
                state[name] = np.ascontiguousarray(flat_state[start:end])
        else:
            state = normalized_state

        if self._instruction_key not in obs:
            raise ValueError(f"Open-Rail observation is missing {self._instruction_key!r}")
        return {
            "vision": vision,
            "state": state,
            "instruction": _as_instruction(obs[self._instruction_key]),
        }

    def _actions_to_array(self, result: Any) -> np.ndarray:
        if isinstance(result, Mapping) and "actions" in result:
            result = result["actions"]

        if isinstance(result, np.ndarray):
            actions = np.asarray(result, dtype=np.float32)
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            if not result:
                raise ValueError("XPolicyLab returned an empty action chunk")
            if all(isinstance(step, Mapping) for step in result):
                rows = []
                for index, step in enumerate(result):
                    missing = [key for key in self._action_keys if key not in step]
                    if missing:
                        raise ValueError(
                            f"XPolicyLab action step {index} is missing configured key(s) {missing}; "
                            f"available: {list(step)}"
                        )
                    fields = [np.asarray(step[key], dtype=np.float32).reshape(-1) for key in self._action_keys]
                    if any(field.size == 0 for field in fields):
                        raise ValueError(f"XPolicyLab action step {index} contains an empty field")
                    rows.append(np.concatenate(fields))
                actions = np.stack(rows)
            else:
                actions = np.asarray(result, dtype=np.float32)
        else:
            actions = np.asarray(result, dtype=np.float32)

        if actions.ndim != 2 or 0 in actions.shape:
            raise ValueError(f"XPolicyLab actions must have shape [T,D], got {actions.shape}")
        if self._expected_action_dim is not None and actions.shape[1] != self._expected_action_dim:
            raise ValueError(
                f"XPolicyLab action dimension {actions.shape[1]} does not match configured "
                f"dimension {self._expected_action_dim}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("XPolicyLab returned non-finite actions")
        return np.ascontiguousarray(actions, dtype=np.float32)

    def _infer(self, sequence, reset_generation=None, client_id=None):
        if not isinstance(sequence, (list, tuple)) or not sequence:
            raise ValueError("XPolicyLab inference requires a non-empty observation sequence")
        data = sequence[0]
        if not isinstance(data, Mapping) or "obs" not in data:
            raise ValueError("Open-Rail sequence item must contain an 'obs' mapping")

        observation = self._build_observation(data["obs"])
        # The Open-Rail inference pool can call this adapter from several
        # workers. XPolicyLab's synchronous client and update/get state pair
        # are sequential APIs, so serialize reset + update/get as one transaction.
        with self._client_lock:
            if reset_generation is not None:
                if (
                    isinstance(reset_generation, bool)
                    or not isinstance(reset_generation, int)
                    or reset_generation < 0
                ):
                    raise ValueError(
                        f"Invalid reset_generation in request metadata: {reset_generation!r}"
                    )
                previous_generation = self._client_reset_generations.get(client_id)
                if previous_generation is not None and reset_generation < previous_generation:
                    raise ValueError(
                        f"Stale reset_generation {reset_generation}; client {client_id!r} "
                        f"is already at generation {previous_generation}"
                    )
                if previous_generation != reset_generation:
                    self.client.call("reset")
                    self._client_reset_generations[client_id] = reset_generation
            self.client.call("update_obs", observation)
            actions = self._actions_to_array(self.client.call("get_action"))
        return {
            "type": "vla_action",
            "pred_action": actions,
            "ref_timestamp": data["ref_timestamp"],
            "loc_timestamp": data["loc_timestamp"],
            "ext": {},
        }

    def infer(self, sequence):
        return self._infer(sequence)

    def infer_with_reset(self, sequence, reset_generation, client_id=None):
        """Apply client reset generations atomically with the associated inference."""
        return self._infer(sequence, reset_generation, client_id)

    def reset(self):
        with self._client_lock:
            return self.client.call("reset")

    def close(self):
        self.client.close()
