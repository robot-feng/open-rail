# XPolicyLab adapter

This adapter connects Open-Rail to a separately running XPolicyLab policy
server. XPolicyLab loads the selected policy, checkpoint, CUDA runtime, and
model-specific dependencies on its own server. Open-Rail sends observations
over XPolicyLab's WebSocket protocol and receives an action chunk.

## Runtime setup

Keep an XPolicyLab checkout at `test/XPolicyLab` (the repository submodule),
or set `XPOLICYLAB_ROOT` / `models.xpolicylab.repo_path` to its location. The
Open-Rail process imports XPolicyLab's small WebSocket client from that
checkout; it does **not** import a policy or install the XPolicyLab model
environment. Install only the bridge's optional protocol dependencies in the
Open-Rail environment:

```bash
pip install -e ".[xpolicylab]"
```

On the machine/environment that has the selected policy installed, launch its
XPolicyLab WebSocket server. Set `policy_name` and the policy's checkpoint,
robot, and task values in that policy's `deploy.yml` first:

```bash
cd test/XPolicyLab
python -m client_server.ws --config-path policy/<POLICY>/deploy.yml --host 0.0.0.0 --port 19000
```

Then start Open-Rail with:

```bash
python run_server.py --model_type xpolicylab
```

Set `server_url` in `get_xpolicylab_config()` to a host/port reachable from
Open-Rail. `--model_path` is intentionally rejected for this model type: the
checkpoint is selected in XPolicyLab's `deploy.yml`.
Policies that need XPolicyLab's `prepare_case` hook can enable
`prepare_case`, set `action_case_id`, and provide optional `case_meta` there.

## Observation mapping

Open-Rail sends flattened camera keys, a packed state vector, and a language
instruction. The adapter maps them to XPolicyLab's standard observation:

```text
{
  "vision": {"cam_head": {"color": RGB_HWC}, ...},
  "state": {"left_arm_joint_state": ..., ...},
  "instruction": "..."
}
```

Set `camera_map`, `state_fields`, `action_keys`, and `instruction_key` in
`conf/models_conf.py` to match the selected policy and robot. `camera_map`
maps Open-Rail camera names to XPolicyLab camera names. `state_fields` maps
half-open slices of Open-Rail's flat state vector to XPolicyLab state keys.
`state_fields` and `action_keys` start empty and must be configured before
starting this backend. They do not select a robot, vector width, or joint/EE
pose representation. Use the robot's layout and the selected policy's contract
to define these mappings; the adapter rejects empty mappings before connecting.

Open-Rail images arrive already decoded as RGB HWC arrays. The adapter passes
them through without decoding or swapping channels; XPolicyLab's server also
leaves image arrays intact.

## Action mapping

XPolicyLab policies normally return a list of action dictionaries, one per
time step. `action_keys` specifies the exact concatenation order for each
dictionary. For example, a policy using two 7-joint arms and two grippers could
use this order (configure matching `state_fields` slices as well):

```text
left_arm_joint_state, right_arm_joint_state,
left_ee_joint_state, right_ee_joint_state
```

The resulting array has shape `[T, D]` and is returned as Open-Rail's
`pred_action`. Configure this order and the policy's action type to match the
target robot's `action_layout`; the adapter does not convert units,
normalization, absolute/delta semantics, or joint conventions.
Set `expected_action_dim` to the robot's action width when you want the bridge
to reject a mismatched policy output before it reaches the robot client.

The bridge uses XPolicyLab's synchronous client from Open-Rail's inference
workers. It serializes `update_obs` + `get_action` as one transaction because
both the client and stateful policy interface are sequential.
When the Open-Rail client resets, it sends a reset generation with the next
inference request; the Open-Rail server forwards `reset()` to XPolicyLab once
before processing that generation's first observation.
The Open-Rail server serializes reset and inference for models exposing a
`reset()` method. Run one active robot client per stateful policy server;
policy history is shared by that server instance.
