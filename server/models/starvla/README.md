# StarVLA adapter

This adapter runs a StarVLA checkpoint inside Open-Rail's model server. It
uses StarVLA's native model and normalization code for both wrapped VLA
checkpoints and flat-config Diffusion Policy checkpoints.

## Runtime setup

Run Open-Rail with a Python environment that has the dependencies required by
the selected StarVLA checkpoint. `STARVLA_ROOT` must point to the StarVLA source
checkout, or set `models.starvla.repo_path` in `conf/models_conf.py`.

```bash
export STARVLA_ROOT=/path/to/starvla
python run_server.py --model_type starvla --model_path /path/to/checkpoint
```

For the StarVLA environment used with this checkout, start the server with:

```bash
scripts/run_starvla_server.sh data/DP20/checkpoints/steps_25000_pytorch_model.pt
```

Pass a checkpoint file as the first argument; relative paths resolve from your
current directory. The launcher uses the StarVLA Python environment and sets
the source and native library paths. Override `STARVLA_ROOT` or
`STARVLA_PYTHON` when moving to another machine. Extra server arguments, such
as `--debug`, can follow the checkpoint path.

Install Open-Rail's declared `ml-collections` dependency directly in the
StarVLA environment once:

```bash
/data/conda-envs/starvla/bin/python -m pip install ml-collections==1.1.0
```

The adapter automatically selects the StarVLA algorithm using
`framework.name` in the checkpoint's run directory:

```text
data/DP20/
├── config.yaml               # framework.name, dimensions and inference settings
├── dataset_statistics.json   # Training normalization statistics
└── checkpoints/
    └── steps_25000_pytorch_model.pt
```

The adapter reads `framework.name` and loads the selected native framework
from StarVLA's registry. Keep the configuration and statistics with the
checkpoint when moving a trained model.

The checkpoint must be loadable by StarVLA's `baseframework.from_pretrained`
and include its training configuration and normalization statistics. For
multi-dataset checkpoints, set `unnorm_key` in `get_starvla_config()` in
`conf/models_conf.py`.

## Native framework checkpoints

Flat-config checkpoints are instantiated by
`baseframework.from_pretrained`, the same StarVLA framework registry used by
training. State normalization and action un-normalization use the co-located
statistics; a checkpoint can declare `datasets.vla_data.normalization_mode`.
When it is absent, the adapter uses StarVLA's standard `q99` transform.

By default, the returned action vector keeps the checkpoint's dimension and
order. Optional `rail_io` metadata in the checkpoint's `config.yaml` declares
its named input/output fields. Denoising settings such as
`num_inference_steps` are also read from that configuration.

## Observation contract

Edit `get_starvla_config()` in `conf/models_conf.py` to match the checkpoint:

- `camera_keys`: ordered Open-Rail image keys. The default is
  `cam.head`, `cam.hand_left`, `cam.hand_right`; the order must match training.
- `send_state`: include `state` when the checkpoint was trained with
  proprioception. Set it to `False` for image-only checkpoints.
- `normalize_state`: defaults to `True`; reuses the checkpoint's ordered
  state keys, dimensions, and training-time state transforms. Set it to
  `False` only when the input is already in the model's expected state space.
- `language_key` and `state_key`: keys in each Open-Rail observation.
- `device`, `config_overrides`: StarVLA loading options. `use_bf16` is passed
  to wrapped checkpoints; native frameworks keep their own dtype behavior.

Robot field names, slices and physical units are defined once in
`robots.<type>.io_fields` in `conf/robots_conf.py`. Model adapters read that
configuration directly. RAIL's observation/action transport, recording,
visualization and control use their existing canonical vectors.

The current DP20 checkpoint used interleaved training states and canonical
RAIL actions. Its run-directory `config.yaml` carries this compatibility
declaration; keep it with the weights when moving the model:

```yaml
rail_io:
  robot_type: agibot_g1
  state_keys:
    - left_arm_joint_state
    - left_ee_joint_state
    - right_arm_joint_state
    - right_ee_joint_state
    - head_joint_state
    - waist_joint_state
  # action_keys defaults to the robot's canonical field order.
```

Field widths come from the robot definition. The adapter packs states in
`state_keys` order **before** normalization, then restores canonical action
order **after** un-normalization. `action_keys` must cover every robot field;
partial vectors are rejected rather than shifted into other joints. Online
and offline inference use the same
declared robot schema. Models trained entirely in RAIL
order can omit `rail_io`; both vectors then pass through in their original order.
Explicit `state_indices`/`action_indices` remain supported for older manually
configured integrations and cannot be combined with checkpoint `rail_io`.

Open-Rail passes decoded HWC RGB images and the instruction to StarVLA. Flat G1
Diffusion Policy checkpoints resize each view to the checkpoint's `image_size`
with aspect ratio preserved and black padding, matching the 224x224 letterboxed
training videos. The ordered keys map `cam.head`, `cam.hand_left`, and
`cam.hand_right` to `cam_high_rgb`, `cam_left_wrist_rgb`, and
`cam_right_wrist_rgb`; the wrist views are not swapped. Wrapped checkpoints keep
the order defined by their checkpoint processor. Camera order, physical units,
and action mode must still match training.

## Offline action error

In the StarVLA environment, using the same import and library paths as the
server, evaluate a canonical RAIL 20D G1 LeRobot split with:

```bash
python scripts/evaluate_starvla_dp.py \
  --dataset /path/to/dataset/val \
  --checkpoint /path/to/DP20/checkpoints/steps_25000_pytorch_model.pt \
  --starvla-root "$STARVLA_ROOT" \
  --output /path/to/evaluation
```

The default stride is 64. Each observation is compared with its full predicted
action chunk, while repeated episode-tail targets are excluded from scores.
The evaluator checks canonical G1 state/action names and uses each frame's
instruction from `meta/tasks.jsonl`.
The report includes per-dimension MSE, arm, gripper, head and waist errors, a
hold-current-state baseline, latency, and saved predictions.
Latency statistics exclude the first inference and are `null` for a single
window. Mixed-unit MSE is a diagnostic, not a grasp-success measurement;
camera placement, live feedback and action timing still need separate validation.

## Adapter regression tests

With pytest and the adapter's runtime dependencies installed:

```bash
python -m pytest -q test/test_starvla_adapter.py \
  test/test_evaluate_starvla_dp.py test/test_starvla_launcher.py
```

These tests use deterministic fake policies and a fake launcher executable.
They cover configuration overrides, named field order, normalization order,
image transport, output validation and evaluator edge cases without model
weights or robot access.
