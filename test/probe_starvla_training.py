"""Compare the adapter with real StarVLA training transforms, without loading weights.

Run in a StarVLA environment with STARVLA_ROOT set to its source checkout.
"""

import os
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np


def main():
    repo = Path(os.environ["STARVLA_ROOT"]).expanduser().resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deployment.model_server import policy_norm_processor as upstream
    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
    from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
        StateActionToTensor,
        StateActionTransform,
    )

    from server.models.starvla.starvla import ModelVLA

    class ProbeConfig:
        state_keys = ("state.arm", "state.gripper")
        action_keys = ("action.arm", "action.gripper")
        state_key_dims: ClassVar = {"state.gripper": 1, "state.arm": 3}
        action_key_dims: ClassVar = {"action.gripper": 1, "action.arm": 3}
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

        def __init__(self, mode):
            self.mode = mode

        def transform(self):
            return ComposedModalityTransform(transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: self.mode for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: self.mode for key in self.action_keys},
                ),
            ])

    stats = {
        "mean": [0., 0., 0., 2.], "std": [1., 2., 3., 0.],
        "min": [-2., -2., -2., 2.], "max": [4., 4., 4., 2.],
        "q01": [-1., -1., -1., 2.], "q99": [3., 3., 3., 2.],
    }
    tag = "__openrail_state_probe__"
    upstream.DATASET_NAMED_MIXTURES[tag] = [("synthetic", 1.0, tag)]
    upstream.read_mode_config = lambda path: (
        {"datasets": {"vla_data": {"data_mix": tag}}},
        {"synthetic": {"state": stats, "action": stats}},
    )
    rng = np.random.default_rng(20261022)
    samples = rng.uniform(-10, 10, (100, 4)).astype(np.float32)
    comparisons = 0
    for mode in ("q99", "min_max", "mean_std", "binary"):
        upstream.ROBOT_TYPE_CONFIG_MAP[tag] = ProbeConfig(mode)
        processor = upstream.PolicyNormProcessor("/unused/synthetic-statistics")
        model = ModelVLA.__new__(ModelVLA)
        model._state_processor = processor
        for sample in samples:
            state = np.repeat(sample, 2)[None, ::2]
            state.flags.writeable = False
            original = state.copy()
            actual = model._prepare_state(state)
            fields = processor.transform({
                "state.arm": state[:, :3].copy(), "state.gripper": state[:, 3:].copy(),
            })
            expected = np.concatenate([np.asarray(fields[key]) for key in processor.state_keys], axis=-1)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(state, original)
            assert actual.dtype == np.float32 and actual.flags.c_contiguous
            assert np.isfinite(actual).all()
            comparisons += 1
    print(f"PASS: {comparisons} exact training-transform comparisons across four modes; no weights loaded.")


if __name__ == "__main__":
    main()
