import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from .splits import split_seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    args = p.parse_args()
    root = Path(args.data)
    manifest = json.loads((root/'stages.json').read_text())
    identities = [(a['seed'], a['repeat']) for a in manifest['attempts']]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate reset identities in final dataset')
    groups = split_seeds(manifest['episodes'])
    hashes = Counter()
    frames = Counter()
    cameras = set()
    with h5py.File(root/'trajectory.h5') as file:
        if len(file) != len(manifest['episodes']):
            raise ValueError('H5 and stage manifest episode count mismatch')
        for entry in manifest['episodes']:
            g = file[entry['trajectory']]
            actions = g['actions'][:]
            if not np.isfinite(actions).all():
                raise ValueError('Non-finite expert actions')
            hashes[hashlib.sha256(actions.tobytes()).hexdigest()] += 1
            previous = 0
            for stage, (start, end) in entry['boundaries'].items():
                if start != previous or not start < end <= len(actions):
                    raise ValueError(f'Invalid {stage} boundary in {entry["trajectory"]}')
                frames[stage] += end-start
                previous = end
            for camera in g['obs/sensor_data']:
                rgb = g[f'obs/sensor_data/{camera}/rgb']
                if len(rgb) != len(actions)+1 or rgb.dtype != np.uint8:
                    raise ValueError('Image/action alignment or image dtype incorrect')
                cameras.add(camera)
    report = dict(attempts=len(identities), successful=len(manifest['episodes']),
                  independent_seeds=len({s for s, _ in identities}),
                  duplicate_attempts=len(manifest.get('duplicate_attempts', [])),
                  identical_action_sequences=sum(n-1 for n in hashes.values()),
                  stage_frames=dict(frames), cameras=sorted(cameras),
                  splits={name: sorted(values) for name, values in groups.items()},
                  failures=dict(Counter(a.get('error') for a in manifest['attempts'] if not a['success'])))
    (root/'audit.json').write_text(json.dumps(report, indent=2))
    print({k:v for k,v in report.items() if k not in ['splits','failures']}, flush=True)


if __name__ == '__main__':
    main()
