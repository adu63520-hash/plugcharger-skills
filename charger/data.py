import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .splits import split_seeds


def policy_observation(obs, camera, proprio=False):
    rgb = torch.cat([torch.as_tensor(obs['sensor_data'][name]['rgb'])
                     for name in camera.split(',')], dim=-1)
    if rgb.ndim == 3:
        rgb = rgb[None]
    rgb = rgb.permute(0, 3, 1, 2).contiguous()
    if proprio:
        q = torch.as_tensor(obs['agent']['qpos']).reshape(len(rgb), -1)
        v = torch.as_tensor(obs['agent']['qvel']).reshape(len(rgb), -1)
        state = torch.cat((q, v), -1)
    else:
        state = torch.empty(len(rgb), 0, device=rgb.device)
    return rgb, state


class SkillDataset(Dataset):
    def __init__(self, root, stage, split, camera='hand_camera', horizon=16, proprio=False,
                 relative_actions=False, cache=None):
        from pathlib import Path
        self.path = str(Path(root) / 'trajectory.h5')
        self.manifest = json.loads((Path(root) / 'stages.json').read_text())
        self.camera, self.horizon, self.proprio = camera, horizon, proprio
        self.relative_actions = relative_actions
        # Optional memory-mapped frame cache built by charger.cache_frames. HDF5 RGB is
        # gzip chunked and shuffling makes random access ~80x slower than sequential.
        self.memmap = None
        self.memmap_index = None
        self.index = []
        episodes = self.manifest['episodes']
        seeds = sorted({e['seed'] for e in episodes})
        if len(seeds) < 2:
            raise ValueError('Need >=2 distinct seeds for disjoint train/validation sets')
        # Seed grouping prevents repeated resets leaking across splits.
        groups = split_seeds(episodes)
        selected = [e for e in episodes if e['seed'] in groups[split]]
        if not selected:
            raise ValueError(f'Empty {split} split')
        with h5py.File(self.path) as f:
            key = selected[0]['trajectory']
            g = f[key]
            cameras = list(g['obs/sensor_data'])
            if any(name not in cameras for name in camera.split(',')):
                raise ValueError(f'Camera {camera!r} absent; available: {cameras}')
            self.action_dim = g['actions'].shape[-1]
            # Fit normalization exclusively on training seeds, even for val dataset.
            arrays = []
            for e in episodes:
                if e['seed'] not in groups['train']:
                    continue
                start, end = e['boundaries'][stage]
                g_train = f[e['trajectory']]
                a = g_train['actions'][start:end]
                if relative_actions:
                    indices = np.minimum(np.arange(len(a))[:, None]+np.arange(horizon), len(a)-1)
                    a = a[indices].copy()
                    qpos = g_train['obs/agent/qpos'][start:end, :self.action_dim-1]
                    a[:, :, :-1] -= qpos[:, None]
                    a = a.reshape(-1, self.action_dim)
                arrays.append(a)
            actions = np.concatenate(arrays)
            self.low = actions.min(0)
            self.high = actions.max(0)
            self.proprio_dim = (g['obs/agent/qpos'].shape[-1] + g['obs/agent/qvel'].shape[-1]) if proprio else 0
        for e in selected:
            start, end = e['boundaries'][stage]
            self.index.extend((e['trajectory'], t, end) for t in range(start, end))
        self.file = None
        self.cache = None
        self.episode = None
        if cache:
            self._load_memmap(Path(cache), stage, set(map(str, selected and [e['trajectory'] for e in selected])))

    def _load_memmap(self, cache, stage, keys):
        info = json.loads((cache / f'{stage}_index.json').read_text())
        if info['camera'] != self.camera or info['horizon'] != self.horizon:
            raise ValueError(f'Frame cache built for camera={info["camera"]} horizon={info["horizon"]}, '
                             f'requested camera={self.camera} horizon={self.horizon}')
        self.memmap = {
            'rgb': np.load(cache / f'{stage}_{self.camera.replace(",", "-")}_rgb.npy', mmap_mode='r'),
            'qpos': np.load(cache / f'{stage}_qpos.npy', mmap_mode='r'),
            'qvel': np.load(cache / f'{stage}_qvel.npy', mmap_mode='r'),
            'actions': np.load(cache / f'{stage}_actions.npy', mmap_mode='r'),
        }
        self.memmap_index = {}
        for e in info['episodes']:
            self.memmap_index[e['trajectory']] = (e['offset'], e['start'], e['end'])
        missing = keys - set(self.memmap_index)
        if missing:
            raise ValueError(f'Frame cache missing {len(missing)} episodes in split')
        return self

    def preload(self):
        """Decode each selected episode once, rather than gzip chunks per sample."""
        self.cache = {}
        with h5py.File(self.path, 'r') as f:
            for key in dict.fromkeys(key for key, _, _ in self.index):
                g = f[key]
                indices = [t for k, t, _ in self.index if k == key]
                start, end = min(indices), max(indices) + 1
                rgb = np.concatenate([g[f'obs/sensor_data/{name}/rgb'][start:end]
                                      for name in self.camera.split(',')], axis=-1)
                actions = g['actions'][start:end]
                state = (np.concatenate((g['obs/agent/qpos'][start:end],
                                         g['obs/agent/qvel'][start:end]), -1)
                         if self.proprio else np.empty((end-start, 0), np.float32))
                anchor = g['obs/agent/qpos'][start:end, :self.action_dim-1] if self.relative_actions else None
                self.cache[key] = (start, rgb, actions, state, anchor)
        return self

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        if self.cache is not None:
            key, t, end = self.index[item]
            start, images, actions, state, anchors = self.cache[key]
            j = t - start
            a = actions[j:min(j+self.horizon, end-start)]
            a = np.concatenate((a, np.repeat(a[-1:], self.horizon-len(a), axis=0)))
            if self.relative_actions:
                a[:, :-1] -= anchors[j]
            a = 2*(a-self.low)/np.maximum(self.high-self.low, 1e-6)-1
            return (torch.from_numpy(images[j]).permute(2, 0, 1),
                    torch.tensor(state[j], dtype=torch.float32),
                    torch.tensor(a, dtype=torch.float32))
        if self.memmap is not None:
            key, t, end = self.index[item]
            offset, start, stage_end = self.memmap_index[key]
            j = offset + (t - start)
            span = min(t + self.horizon, end) - t
            a = self.memmap['actions'][j:j + span]
            a = np.concatenate((a, np.repeat(a[-1:], self.horizon - len(a), axis=0)))
            if self.relative_actions:
                a = a.copy()
                a[:, :-1] -= self.memmap['qpos'][j, :self.action_dim-1]
            else:
                a = a.copy()
            a = 2 * (a - self.low) / np.maximum(self.high - self.low, 1e-6) - 1
            if self.proprio:
                state = np.concatenate((self.memmap['qpos'][j], self.memmap['qvel'][j]))
            else:
                state = np.empty(0, np.float32)
            return (torch.from_numpy(np.ascontiguousarray(self.memmap['rgb'][j])).permute(2, 0, 1).clone(),
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(a, dtype=torch.float32))
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        key, t, end = self.index[item]
        # Decoding one frame at a time from a gzip-chunked dataset costs ~10 ms
        # because the same chunk is re-inflated on every access, making the loader
        # ~10x slower than the GPU step. Fetch the resident episode's skill range
        # once and index into it; frames are contiguous within an episode because
        # the index is built in order.
        if self.episode is None or self.episode[0] != key:
            g = self.file[key]
            lo = min(v for k2, v, _ in self.index if k2 == key)
            hi = max(v for k2, v, _ in self.index if k2 == key) + self.horizon + 1
            hi = min(hi, len(g['actions']))
            if self.proprio or self.relative_actions:
                hi = min(hi, len(g['obs/agent/qpos']), len(g['obs/agent/qvel']))
            rgb = np.concatenate([g[f'obs/sensor_data/{name}/rgb'][lo:hi]
                                  for name in self.camera.split(',')], axis=-1)
            actions = g['actions'][lo:hi]
            # qpos/qvel are only touched when proprio or relative anchoring needs them.
            if self.proprio or self.relative_actions:
                qpos = g['obs/agent/qpos'][lo:hi]
                qvel = g['obs/agent/qvel'][lo:hi]
            else:
                qpos = qvel = None
            self.episode = (key, lo, rgb, qpos, qvel, actions)
        _, lo, rgb, qpos, qvel, actions = self.episode
        j = t - lo
        image = rgb[j]
        proprio = np.concatenate((qpos[j], qvel[j])) if self.proprio else np.empty(0)
        # Padding repeats the final action of this skill, never the next skill's action.
        a = actions[j:min(j + self.horizon, end - lo)]
        a = np.concatenate((a, np.repeat(a[-1:], self.horizon - len(a), axis=0)))
        if self.relative_actions:
            a[:, :-1] -= qpos[j, :self.action_dim-1]
        a = 2 * (a - self.low) / np.maximum(self.high - self.low, 1e-6) - 1
        return (torch.from_numpy(image.copy()).permute(2, 0, 1),
                torch.tensor(proprio, dtype=torch.float32), torch.tensor(a, dtype=torch.float32))
