"""Checks the scientific data contract, rather than matching a training loss."""
import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from charger.data import SkillDataset, policy_observation
from charger.stages import boundaries


class ContractTests(unittest.TestCase):
    def test_relative_joint_targets_roundtrip_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            episodes = []
            with h5py.File(Path(root)/'trajectory.h5', 'w') as file:
                for i in range(5):
                    key = f'traj_{i}'
                    episodes.append(dict(seed=i, trajectory=key, boundaries=dict(A=[0,2], B=[2,4], C=[4,6])))
                    g = file.create_group(key)
                    g.create_dataset('actions', data=np.stack((np.arange(6), -np.ones(6)), -1).astype(np.float32))
                    g.create_dataset('obs/agent/qpos', data=(np.arange(7)*.1).astype(np.float32)[:, None])
                    g.create_dataset('obs/agent/qvel', data=np.zeros((7,1), np.float32))
                    g.create_dataset('obs/sensor_data/hand_camera/rgb', data=np.zeros((7,8,8,3), np.uint8))
            (Path(root)/'stages.json').write_text(json.dumps(dict(episodes=episodes)))
            data = SkillDataset(root, 'B', 'train', horizon=4, proprio=True, relative_actions=True)
            original = data[0]
            _, state, normalized = original
            physical = data.low+(normalized.numpy()+1)/2*(data.high-data.low)
            physical[:,0] += state[0].item()
            self.assertTrue(np.allclose(physical, [[2,-1],[3,-1],[3,-1],[3,-1]]))
            data.preload()
            for a,b in zip(original, data[0]):
                self.assertTrue(torch.equal(a,b))
            data.file.close()

    def test_boundary_action_belongs_to_previous_skill(self):
        labels = [dict(A=True, B=False, C=False), dict(A=True, B=True, C=False),
                  dict(A=False, B=False, C=True)]
        self.assertEqual(boundaries(labels), dict(A=[0, 1], B=[1, 2], C=[2, 3]))
        with self.assertRaises(ValueError):
            boundaries(labels[:2])

    def test_policy_does_not_read_privileged_fields(self):
        obs = dict(sensor_data={'hand_camera': {'rgb': np.zeros((1, 16, 16, 3), np.uint8)}},
                   extra={'charger_pose': object(), 'goal_pose': object()})
        rgb, state = policy_observation(obs, 'hand_camera')
        self.assertEqual(tuple(rgb.shape), (1, 3, 16, 16))
        self.assertEqual(tuple(state.shape), (1, 0))

    def test_multiview_channel_order_is_explicit(self):
        obs = {'sensor_data': {
            'hand_camera': {'rgb': np.zeros((1, 8, 8, 3), np.uint8)},
            'base_camera': {'rgb': np.full((1, 8, 8, 3), 255, np.uint8)}}}
        rgb, _ = policy_observation(obs, 'hand_camera,base_camera')
        self.assertEqual(tuple(rgb.shape), (1, 6, 8, 8))
        self.assertTrue((rgb[:, :3] == 0).all())
        self.assertTrue((rgb[:, 3:] == 255).all())

    def test_seed_split_and_action_chunks_do_not_cross_stage(self):
        with tempfile.TemporaryDirectory() as root:
            episodes = []
            with h5py.File(Path(root) / 'trajectory.h5', 'w') as f:
                for i in range(10):
                    key = f'traj_{i}'
                    episodes.append(dict(seed=i // 2, trajectory=key,
                                         boundaries=dict(A=[0, 2], B=[2, 4], C=[4, 6])))
                    g = f.create_group(key)
                    g.create_dataset('actions', data=np.arange(6, dtype=np.float32)[:, None])
                    g.create_dataset('obs/sensor_data/hand_camera/rgb', data=np.zeros((7, 16, 16, 3), np.uint8))
            (Path(root) / 'stages.json').write_text(json.dumps(dict(episodes=episodes)))
            train = SkillDataset(root, 'B', 'train', horizon=4)
            val = SkillDataset(root, 'B', 'val', horizon=4)
            train_keys = {key for key, _, _ in train.index}
            val_keys = {key for key, _, _ in val.index}
            mapping = {e['trajectory']: e['seed'] for e in episodes}
            self.assertFalse({mapping[k] for k in train_keys} & {mapping[k] for k in val_keys})
            _, _, actions = train[1]
            self.assertTrue(torch.equal(actions, torch.ones(4, 1)))
            old = train[0]
            train.preload()
            cached = train[0]
            for a, b in zip(old, cached):
                self.assertTrue(torch.equal(a, b))
            train.file.close()


if __name__ == '__main__':
    unittest.main()
