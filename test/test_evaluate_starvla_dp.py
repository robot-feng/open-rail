"""Offline evaluator regressions; predictions are deterministic and no SDK is imported."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_starvla_dp as evaluator


def _dataset(tmp_path, rows):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {key: {"names": evaluator.G1_NAMES} for key in ("action", "observation.state")}
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "features": features}))
    (root / "meta/tasks.jsonl").write_text("\n".join(json.dumps({
        "task_index": index, "task": task,
    }) for index, task in enumerate(("pick apple", "place apple"))) + "\n")
    pd.DataFrame({
        "observation.state": [np.zeros(20, dtype=np.float32) for _ in range(rows)],
        "action": [np.zeros(20, dtype=np.float32) for _ in range(rows)],
        "frame_index": np.arange(rows),
        "timestamp": (np.arange(rows) / 30).astype(np.float32),
        "episode_index": np.zeros(rows, dtype=np.int64),
        "task_index": np.arange(rows) % 2,
    }).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    return root


def _arguments(monkeypatch, tmp_path, root, extra=()):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"no model weights")
    monkeypatch.setattr(sys, "argv", [
        "evaluate_starvla_dp.py", "--dataset", str(root),
        "--checkpoint", str(checkpoint), "--starvla-root", str(tmp_path),
        "--output", str(tmp_path / "results"), "--stride", "1", *extra,
    ])


@pytest.mark.parametrize("rows", [1, 3])
def test_evaluates_windows_masks_tail_and_handles_single_latency(monkeypatch, tmp_path, rows):
    root = _dataset(tmp_path, rows)
    _arguments(monkeypatch, tmp_path, root)
    instructions = []

    class FixedModel:
        def __init__(self, config):
            self.policy_metadata = {"action_chunk_size": 2, "state_dim": 20, "action_dim": 20}

        def infer(self, sequence):
            instructions.append(sequence[0]["obs"]["language"])
            return {"pred_action": np.zeros((2, 20), dtype=np.float32)}

    monkeypatch.setattr(evaluator, "ModelVLA", FixedModel)
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(evaluator, "read_views", lambda root, path, origins: {
        key: {origin: np.zeros((8, 8, 3), dtype=np.uint8) for origin in origins}
        for key, _ in evaluator.CAMERAS
    })
    evaluator.main()
    report = json.loads((tmp_path / "results/report.json").read_text())
    assert report["windows"] == rows
    assert report["prediction"]["scored_frames"] == 2 * rows - 1
    assert report["prediction"]["all_20d_mixed_units_mse"] == 0
    assert instructions == ["pick apple", "place apple", "pick apple"][:rows]
    assert (report["inference_seconds_p95_excluding_first"] is None) == (rows == 1)
    assert (report["inference_seconds_mean_excluding_first"] is None) == (rows == 1)
    saved = np.load(tmp_path / "results/predictions.npz")
    assert saved["prediction"].shape == (rows, 2, 20)
    assert saved["valid"].sum() == 2 * rows - 1


@pytest.mark.parametrize("option", ["--stride", "--max-episodes"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_rejects_nonpositive_sampling_options(monkeypatch, tmp_path, option, value):
    _arguments(monkeypatch, tmp_path, tmp_path, [option, value])
    with pytest.raises(SystemExit) as error:
        evaluator.main()
    assert error.value.code == 2


@pytest.mark.parametrize("fps", [0, -30, float("nan"), float("inf")])
def test_rejects_invalid_dataset_fps_before_loading_model(monkeypatch, tmp_path, fps):
    root = _dataset(tmp_path, 1)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["fps"] = fps
    info_path.write_text(json.dumps(info))
    _arguments(monkeypatch, tmp_path, root)
    with pytest.raises(ValueError, match="fps must be finite and positive"):
        evaluator.main()


def test_rejects_noncanonical_dataset_order(monkeypatch, tmp_path):
    root = _dataset(tmp_path, 1)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["names"].reverse()
    info_path.write_text(json.dumps(info))
    _arguments(monkeypatch, tmp_path, root)
    with pytest.raises(ValueError, match="canonical RAIL G1"):
        evaluator.main()
