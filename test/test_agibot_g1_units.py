"""Offline G1 observation/command contract checks; no hardware SDK is loaded."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

from conf.robots_conf import get_agibot_g1_config


class FakeCamera:
    def __init__(self, names):
        self.timestamp = 0

    def get_latest_image(self, name):
        self.timestamp += 1
        return np.zeros((2, 3, 3), dtype=np.uint8), self.timestamp

    def close(self):
        pass


class FakeRobot:
    def __init__(self):
        self.values = {
            'arm': np.linspace(-1, 1, 14, dtype=np.float32),
            'gripper': np.array([35, 120], dtype=np.float32),
            'head': np.array([30, -20], dtype=np.float32),
            'waist': np.array([0.35, 0.22], dtype=np.float32),
        }
        for values in self.values.values():
            values.flags.writeable = False
        self.move_head = Mock()

    def __getattr__(self, name):
        if name.endswith('_joint_states_nearest'):
            group = name.removesuffix('_joint_states_nearest')
            return lambda timestamp: (self.values[group], timestamp)
        raise AttributeError(name)

    def shutdown(self):
        pass


class G1UnitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sdk = ModuleType('a2d_sdk.robot')
        sdk.RobotDds = FakeRobot
        sdk.CosineCamera = FakeCamera
        spec = importlib.util.spec_from_file_location(
            'client.robots.agibot_g1._unit_test_body',
            Path(__file__).resolve().parents[1] / 'client/robots/agibot_g1/body_robot.py',
        )
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'a2d_sdk': ModuleType('a2d_sdk'), 'a2d_sdk.robot': sdk}):
            spec.loader.exec_module(cls.module)

    def make_body(self, unit='degrees'):
        config = get_agibot_g1_config()
        config.camera.names = {'head': 'head'}
        config.head_feedback_unit = unit
        with patch.object(self.module.time, 'sleep'):
            body = self.module.RobotBody(config)
        self.addCleanup(body.close)
        return body

    def test_complete_first_and_repeated_observation(self):
        body = self.make_body()
        sdk_head = body.robot.values['head'].copy()
        expected = np.concatenate([
            body.robot.values['arm'], [35.0 / 120.0, 1.0],
            [np.pi / 6, -np.pi / 9], body.robot.values['waist'],
        ]).astype(np.float32)
        # Make any loss of previously written fields during growth deterministic.
        original_empty = np.empty
        def empty_with_sentinel(*args, **kwargs):
            result = original_empty(*args, **kwargs)
            result.fill(-999)
            return result
        with patch.object(self.module.np, 'empty', side_effect=empty_with_sentinel):
            first = body.retrieve_observation()['obs.state']
        np.testing.assert_allclose(first, expected, atol=1e-6, rtol=0)
        np.testing.assert_allclose(body.current_state[16:18], expected[16:18], atol=1e-6)
        np.testing.assert_array_equal(body.robot.values['head'], sdk_head)
        body.robot.values['arm'] = np.zeros(14, dtype=np.float32)
        second = body.retrieve_observation()['obs.state']
        np.testing.assert_array_equal(second[:14], np.zeros(14))
        np.testing.assert_allclose(second[16:18], expected[16:18], atol=1e-6)
        np.testing.assert_allclose(first, expected, atol=1e-6, rtol=0)
        self.assertEqual(first.dtype, np.float32)

    def test_radian_feedback_is_not_converted_again(self):
        body = self.make_body('radians')
        body.robot.values['head'] = np.array([0.3, -0.2], dtype=np.float32)
        # Exercise the full first-frame path in this mode too.
        result = body.retrieve_observation()
        np.testing.assert_array_equal(result['obs.state'][16:18], body.robot.values['head'])

    def test_invalid_unit_does_not_open_hardware(self):
        config = get_agibot_g1_config()
        config.head_feedback_unit = 'invalid'
        with patch.object(self.module, 'Camera') as camera, patch.object(self.module, 'Robot') as robot:
            with self.assertRaisesRegex(ValueError, 'head_feedback_unit'):
                self.module.RobotBody(config)
            camera.assert_not_called()
            robot.assert_not_called()

    def test_invalid_feedback_is_rejected_in_both_unit_modes(self):
        for unit in ('degrees', 'radians'):
            body = self.make_body(unit)
            for head in ([0], [0, 1, 2], [0, float('nan')], [float('inf'), 0]):
                with self.subTest(unit=unit, head=head):
                    body.robot.values['head'] = head
                    with self.assertLogs(body.logger, level='WARNING'):
                        self.assertIsNone(body.retrieve_observation())

    def test_head_targets_remain_radians_on_both_command_paths(self):
        body = self.make_body()
        target = [np.pi / 6, -np.pi / 9]
        body.execute_action({'head': target})
        body.robot.move_head.assert_called_once_with(target)
        body.robot.move_head.reset_mock()
        body.robot.move_arm = Mock()
        body.robot.move_gripper = Mock()
        action = np.zeros(18)
        action[14:16] = [0.25, 0.75]
        action[16:18] = target
        body.control_robot(action)
        body.robot.move_head.assert_called_once_with(target)
        body.robot.move_gripper.assert_called_once_with([0.25, 0.75])

    def test_gripper_feedback_is_scaled_to_dataset_contract(self):
        body = self.make_body()
        body.robot.values['gripper'] = np.array([0.2173333333, 60.0], dtype=np.float32)
        result = body.retrieve_observation()
        np.testing.assert_allclose(
            result['obs.state'][14:16],
            [0.2173333333 / 120.0, 0.5],
            atol=1e-7,
            rtol=0,
        )


if __name__ == '__main__':
    unittest.main()
