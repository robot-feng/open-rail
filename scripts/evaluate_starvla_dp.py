"""Evaluate a 20D G1 StarVLA checkpoint against a canonical RAIL LeRobot split offline."""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conf.models_conf import get_starvla_config
from server.models.starvla.starvla import ModelVLA

CAMERAS = (
    ("cam.head", "observation.images.head_rgb"),
    ("cam.hand_left", "observation.images.left_wrist_rgb"),
    ("cam.hand_right", "observation.images.right_wrist_rgb"),
)
ACTION_GROUPS = {
    "arm_14d": slice(0, 14),
    "gripper_2d": slice(14, 16),
    "head_2d": slice(16, 18),
    "waist_pitch_rad": slice(18, 19),
    "lift_m": slice(19, 20),
}
G1_NAMES = (
    [f"{side}_arm_joint{joint}" for side in ("left", "right") for joint in range(1, 8)]
    + ["left_gripper", "right_gripper", "head_yaw", "head_pitch", "waist_pitch", "lift"]
)


def read_views(root, parquet, origins):
    """Decode requested frames while preserving row-to-video alignment."""
    frames = {}
    for camera, video_key in CAMERAS:
        path = root / "videos" / parquet.parent.name / video_key / (parquet.stem + ".mp4")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        selected = {}
        wanted = set(origins)
        try:
            for index in range(max(origins) + 1):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Missing frame {index} in {path}")
                if index in wanted:
                    selected[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        frames[camera] = selected
    return frames


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mse_report(prediction, target, valid, names):
    errors = (prediction - target).astype(np.float64)
    scored = errors[valid]
    report = {
        "all_20d_mixed_units_mse": float(np.mean(scored ** 2)),
        "mse_by_dimension": np.mean(scored ** 2, axis=0).tolist(),
        "mae_by_dimension": np.mean(np.abs(scored), axis=0).tolist(),
        "dimensions": list(names),
        "scored_frames": int(valid.sum()),
    }
    for group, section in ACTION_GROUPS.items():
        part = scored[:, section]
        report[group] = {
            "mse": float(np.mean(part ** 2)),
            "rmse": float(np.sqrt(np.mean(part ** 2))),
            "mae": float(np.mean(np.abs(part))),
        }
    for index in (14, 15):
        binary = valid & ((target[..., index] < 0.1) | (target[..., index] > 0.9))
        if binary.any():
            report[f"gripper_{index - 14}_binary_accuracy"] = float(np.mean(
                (prediction[..., index][binary] >= 0.5)
                == (target[..., index][binary] >= 0.5)
            ))
            report[f"gripper_{index - 14}_binary_frames"] = int(binary.sum())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--starvla-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")

    logging.basicConfig(level=logging.WARNING)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.dataset.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be finite and positive")
    names = info["features"]["action"]["names"]
    if names != G1_NAMES or info["features"]["observation.state"]["names"] != G1_NAMES:
        raise ValueError("This evaluator requires canonical RAIL G1 state/action field names")
    tasks = {
        task["task_index"]: task["task"]
        for task in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())
    }

    cfg = get_starvla_config()
    cfg.repo_path = str(args.starvla_root.resolve())
    cfg.model_path = str(args.checkpoint.resolve())
    print("Loading checkpoint", cfg.model_path, flush=True)
    adapter = ModelVLA(cfg)
    horizon = int(adapter.policy_metadata["action_chunk_size"])
    action_dim = int(adapter.policy_metadata["action_dim"])
    state_dim = int(adapter.policy_metadata["state_dim"])
    if state_dim != 20 or action_dim != 20 or len(names) != 20:
        raise ValueError(
            f"This evaluator requires a 20D G1 checkpoint and canonical RAIL dataset; "
            f"got state/action dimensions {state_dim}/{action_dim} and {len(names)} action names"
        )

    predictions, targets, holds, masks, identifiers, latencies = [], [], [], [], [], []
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if args.max_episodes:
        files = files[:args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No LeRobot episodes found under {root / 'data'}")

    started = time.perf_counter()
    for episode_number, path in enumerate(files):
        df = pd.read_parquet(path)
        states = np.stack(df["observation.state"]).astype(np.float32)
        actions = np.stack(df["action"]).astype(np.float32)
        if states.shape[1] != state_dim or actions.shape[1] != action_dim:
            raise ValueError(f"Unexpected state/action shape in {path}: {states.shape}/{actions.shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite state/action in {path}")
        np.testing.assert_array_equal(df.frame_index, np.arange(len(df)))
        expected_times = (np.arange(len(df), dtype=np.float64) / fps).astype(np.float32)
        np.testing.assert_allclose(df.timestamp, expected_times, rtol=0, atol=1e-6)

        origins = list(range(0, len(df), args.stride))
        views = read_views(root, path, origins)
        episode_id = int(df.episode_index.iloc[0])
        for origin in origins:
            torch.manual_seed(args.seed + episode_id * 100000 + origin)
            observation = {key: images[origin] for key, images in views.items()}
            observation.update(
                state=states[origin],
                language=tasks[int(df.task_index.iloc[origin])],
            )
            stamp = float(df.timestamp.iloc[origin])
            before = time.perf_counter()
            output = adapter.infer([{
                "obs": observation,
                "ref_timestamp": stamp,
                "loc_timestamp": stamp,
            }])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - before)

            prediction = output["pred_action"]
            if prediction.shape != (horizon, action_dim):
                raise ValueError(f"Unexpected prediction shape {prediction.shape}")
            rows = np.arange(origin, origin + horizon)
            valid = rows < len(df)
            target = actions[np.minimum(rows, len(df) - 1)]
            hold = np.broadcast_to(states[origin], (horizon, state_dim)).copy()
            predictions.append(prediction)
            targets.append(target)
            holds.append(hold)
            masks.append(valid)
            identifiers.append((episode_id, origin))

        if (episode_number + 1) % 5 == 0 or episode_number == len(files) - 1:
            print(
                f"episodes={episode_number + 1}/{len(files)} "
                f"windows={len(identifiers)} elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

    prediction, target, hold, valid = map(np.stack, (predictions, targets, holds, masks))
    ids = np.asarray(identifiers)
    np.savez_compressed(
        args.output / "predictions.npz",
        prediction=prediction,
        target=target,
        hold_current_state=hold,
        valid=valid,
        episode_frame=ids,
        inference_seconds=latencies,
    )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "dataset": str(root),
        "episodes": len(files),
        "windows": len(ids),
        "horizon": horizon,
        "action_dim": action_dim,
        "state_indices": adapter.policy_metadata.get("state_indices"),
        "action_indices": adapter.policy_metadata.get("action_indices"),
        "rail_io": adapter.policy_metadata.get("rail_io"),
        "stride": args.stride,
        "seed": args.seed,
        "fps": fps,
        "inference_seconds_mean_excluding_first": (
            float(np.mean(latencies[1:])) if len(latencies) > 1 else None
        ),
        "inference_seconds_p95_excluding_first": (
            float(np.quantile(latencies[1:], 0.95)) if len(latencies) > 1 else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": "Observation at frame t; compare all predicted actions with rows t..t+horizon-1; repeated episode-tail targets are masked.",
        "prediction": mse_report(prediction, target, valid, names),
        "hold_current_state_baseline": mse_report(hold, target, valid, names),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
