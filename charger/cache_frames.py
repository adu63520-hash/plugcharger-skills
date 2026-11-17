"""Export one skill's frames to memory-mappable arrays so training is not I/O bound.

The HDF5 RGB datasets are gzip chunked. Sequential reads are cheap, but the
training loader shuffles, so each batch touches dozens of different episodes and
re-inflates their chunks: measured 4031 ms/batch vs 50 ms/batch sequential, i.e.
~4 s per optimisation step against 60 ms of GPU work.

This writes a compact per-skill cache that numpy can mmap, so random access costs a
page fault instead of a gunzip. Only the frames the skill needs (plus horizon padding)
are copied, and the original data is never modified.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--stage', choices=list('ABC'), required=True)
    p.add_argument('--camera', default='hand_camera')
    p.add_argument('--out', required=True)
    p.add_argument('--horizon', type=int, default=16)
    args = p.parse_args()
    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'stages.json').read_text())
    cameras = args.camera.split(',')
    index = []
    with h5py.File(root / 'trajectory.h5', 'r') as f:
        for entry in manifest['episodes']:
            start, end = entry['boundaries'][args.stage]
            g = f[entry['trajectory']]
            total = len(g['actions'])
            # Horizon padding may read past the stage end, but never past the episode.
            hi = min(end + args.horizon + 1, total)
            index.append(dict(trajectory=entry['trajectory'], seed=entry['seed'],
                              lo=start, hi=hi, start=start, end=end))
    total_frames = sum(e['hi'] - e['lo'] for e in index)
    shape = (total_frames, 128, 128, 3 * len(cameras))
    print(dict(stage=args.stage, episodes=len(index), frames=total_frames,
               gb=round(total_frames * shape[1] * shape[2] * shape[3] / 2**30, 2)), flush=True)
    rgb_path = out / f'{args.stage}_{args.camera.replace(",", "-")}_rgb.npy'
    qpos = np.zeros((total_frames, 9), np.float32)
    qvel = np.zeros((total_frames, 9), np.float32)
    actions = np.zeros((total_frames, 8), np.float32)
    offset = 0
    # Written with open_memmap so the full cache is never resident at once.
    array = np.lib.format.open_memmap(rgb_path, mode='w+', dtype=np.uint8, shape=shape)
    with h5py.File(root / 'trajectory.h5', 'r') as f:
        for entry in index:
            g = f[entry['trajectory']]
            lo, hi = entry['lo'], entry['hi']
            for c_index, camera in enumerate(cameras):
                block = g[f'obs/sensor_data/{camera}/rgb'][lo:hi]
                array[offset:offset + len(block), :, :, c_index * 3:(c_index + 1) * 3] = block
            qpos[offset:offset + hi - lo] = g['obs/agent/qpos'][lo:hi]
            qvel[offset:offset + hi - lo] = g['obs/agent/qvel'][lo:hi]
            actions[offset:offset + hi - lo] = g['actions'][lo:hi]
            offset += hi - lo
            print(dict(episode=entry['trajectory'], frames=hi - lo), flush=True)
    array.flush()
    del array
    np.save(out / f'{args.stage}_qpos.npy', qpos)
    np.save(out / f'{args.stage}_qvel.npy', qvel)
    np.save(out / f'{args.stage}_actions.npy', actions)
    # Offsets are stored so frames can be located without re-reading the HDF5.
    for n, entry in enumerate(index):
        entry['offset'] = sum(e['hi'] - e['lo'] for e in index[:n])
    (out / f'{args.stage}_index.json').write_text(json.dumps(
        dict(stage=args.stage, camera=args.camera, horizon=args.horizon,
             shape=list(shape), episodes=index), indent=2))
    print(dict(done=args.stage, rgb_mb=round(rgb_path.stat().st_size / 2**20, 1)), flush=True)


if __name__ == '__main__':
    main()
