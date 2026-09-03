import unittest
from pathlib import Path

import numpy as np
import yaml

from sim2sim.mujoco.adapter import LEG_JOINTS, R1MujocoAdapter
from sim2sim.mujoco.foothold import FixedFootholdGenerator, FootPose
from sim2sim.mujoco.math_utils import world_to_stance_foot_goal


SIM2SIM_ROOT = Path(__file__).resolve().parents[1]
CONFIG = SIM2SIM_ROOT / "mujoco/config/r1_flat_model_1000.yaml"


class FootholdTests(unittest.TestCase):
    def setUp(self):
        left = FootPose(np.array([0.0, 0.10, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        right = FootPose(np.array([0.0, -0.10, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.feet = (left, right)

    def test_stance_transform_shape_and_values(self):
        target = np.array([[0.12, 0.10, 0.0], [0.0, -0.10, 0.0]])
        goal = world_to_stance_foot_goal(
            target, np.zeros(2), self.feet[1].position, self.feet[1].quaternion, 0.0
        )
        self.assertEqual(goal.shape, (10,))
        np.testing.assert_allclose(goal[:3], [0.12, 0.20, 0.0], atol=1e-7)

    def test_goal_is_frozen_until_phase_switch(self):
        generator = FixedFootholdGenerator(half_cycle_steps=30)
        initial = generator.reset(self.feet)
        for _ in range(29):
            self.assertFalse(generator.advance(self.feet))
            np.testing.assert_array_equal(generator.goal, initial)
        self.assertTrue(generator.advance(self.feet))
        self.assertEqual(generator.swing_foot_index, 1)


class OfficialModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG.read_text())
        scene = Path(cls.config["robot"]["official_scene"])
        if not scene.exists():
            raise unittest.SkipTest(f"Official unitree_mujoco scene missing: {scene}")
        cls.adapter = R1MujocoAdapter(scene, cls.config)

    def test_joint_mapping_and_observation(self):
        self.assertEqual(len(set(self.adapter.leg_qpos)), len(LEG_JOINTS))
        self.assertEqual(len(set(self.adapter.leg_actuators)), len(LEG_JOINTS))
        self.assertEqual(self.adapter.observation().shape, (54,))
        self.assertTrue(np.isfinite(self.adapter.observation()).all())

    def test_zero_action_short_smoke(self):
        self.adapter.reset()
        for _ in range(10):
            self.adapter.apply_action(np.zeros(12))
        self.assertTrue(np.isfinite(self.adapter.data.qpos).all())
        self.assertAlmostEqual(self.adapter.data.time, 0.1, places=6)

    def test_exported_policy_walks_for_five_seconds(self):
        from sim2sim.mujoco.policy import NumpyMLPPolicy

        policy_path = (
            Path(__file__).resolve().parents[2]
            / "artifacts/sim2sim/r1_flat_model_1000/policy_weights.npz"
        )
        if not policy_path.exists():
            self.skipTest("Exported model_1000 artifact is not present")
        policy = NumpyMLPPolicy.load(policy_path)
        self.adapter.reset()
        for _ in range(500):
            action = policy(self.adapter.observation())
            self.assertEqual(action.shape, (12,))
            self.adapter.apply_action(action)
            self.assertFalse(self.adapter.has_fallen)
        self.assertGreater(self.adapter.data.qpos[0], 1.5)


if __name__ == "__main__":
    unittest.main()
