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

The checkpoint must be loadable by StarVLA's `baseframework.from_pretrained`
and include its training configuration and normalization statistics. For
multi-dataset checkpoints, set `unnorm_key` in `get_starvla_config()` in
`conf/models_conf.py`.

## G1 Diffusion Policy checkpoint

The flat-config StarVLA `DiffusionPolicy` checkpoint format is also supported.
For the included G1 model, `get_starvla_config()` defines the 20D state
permutation and maps the model's 8D right-arm plus right-gripper output into a
full 20D RAIL action. Dimensions `[7..13, 15]` come from the model; every other
dimension is copied from the current state. This mapping matches the checkpoint
statistics and the G1 dataset's action layout; review it when using a checkpoint
trained with a different joint order.

The G1 DP adapter defaults to `dp_num_inference_steps = 10`, matching the
setting used in the recent real-time run. Change this value in
`get_starvla_config()` to adjust denoising steps without editing checkpoint
files.

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
- `device`, `use_bf16`, `config_overrides`: StarVLA loading options.
- `expected_state_dim` and `expected_action_dim`: optional width checks.

Open-Rail passes decoded HWC RGB images and the instruction to StarVLA. Flat G1
Diffusion Policy checkpoints resize each view to the checkpoint's `image_size`
with aspect ratio preserved and black padding, matching the 224x224 letterboxed
training videos. The ordered keys map `cam.head`, `cam.hand_left`, and
`cam.hand_right` to `cam_high_rgb`, `cam_left_wrist_rgb`, and
`cam_right_wrist_rgb`; the wrist views are not swapped. Wrapped checkpoints keep
the order defined by their checkpoint processor. Camera order, physical units,
and action mode must still match training.
