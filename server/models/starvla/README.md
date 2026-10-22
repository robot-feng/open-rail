# StarVLA adapter

This adapter runs a StarVLA checkpoint inside Open-Rail's model server. It
loads StarVLA's `PolicyServerWrapper`, so checkpoint loading and action
un-normalization stay in the StarVLA implementation.

## Runtime setup

Run Open-Rail with a Python environment that has the dependencies required by
the selected StarVLA checkpoint. `STARVLA_ROOT` must point to the StarVLA source
checkout; it defaults to `/data/EvaClient-StarVLA/starVLA`.

```bash
export STARVLA_ROOT=/data/EvaClient-StarVLA/starVLA
python run_server.py --model_type starvla --model_path /path/to/checkpoint
```

The checkpoint must be loadable by StarVLA's `baseframework.from_pretrained`
and include its training configuration and normalization statistics. For
multi-dataset checkpoints, set `unnorm_key` in `get_starvla_config()` in
`conf/models_conf.py`.

## Observation contract

Edit `get_starvla_config()` in `conf/models_conf.py` to match the checkpoint:

- `camera_keys`: ordered Open-Rail image keys. The default is
  `cam.head`, `cam.hand_left`, `cam.hand_right`; the order must match training.
- `send_state`: include `state` when the checkpoint was trained with
  proprioception. Set it to `False` for image-only checkpoints.
- `language_key` and `state_key`: keys in each Open-Rail observation.
- `device`, `use_bf16`, `config_overrides`: StarVLA loading options.
- `expected_state_dim` and `expected_action_dim`: optional width checks.

Open-Rail passes decoded HWC RGB images and the instruction to StarVLA. The
adapter does not resize images, reorder action dimensions, or convert action
semantics. Keep camera preprocessing, state/action ordering, units, and the
checkpoint's action mode aligned with training and the robot client.
