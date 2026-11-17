"""Merge independent official trajectory shards without losing stage provenance."""
import argparse
import copy
import json
import h5py
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--deduplicate', action='store_true')
    p.add_argument('--state-only', action='store_true')
    p.add_argument('--attempts-from')
    p.add_argument('inputs', nargs='+')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'trajectory.h5').exists():
        raise FileExistsError('Choose a new merged output directory')
    merged = None
    seen = set()
    copies = []
    duplicates = []
    official_merged = None
    for root in map(Path, args.inputs):
        data = json.loads((root/'stages.json').read_text())
        official = json.loads((root/'trajectory.json').read_text())
        if args.state_only:
            official['env_info']['env_kwargs'].update(obs_mode='none', render_backend='none')
        by_id = {f'traj_{e["episode_id"]}': e for e in official['episodes']}
        if official_merged is None:
            official_merged = copy.deepcopy(official)
            official_merged['episodes'] = []
        elif official['env_info'] != official_merged['env_info']:
            raise ValueError(f'Incompatible environment metadata in {root}')
        if merged is None:
            merged = copy.deepcopy(data)
            merged.update(episodes=[], attempts=[], sources=args.inputs)
        for attempt in data['attempts']:
            identity = (attempt['seed'], attempt['repeat'])
            if identity in seen:
                if not args.deduplicate:
                    raise ValueError(f'Duplicate reset identity {identity}')
                duplicates.append(dict(source=str(root), attempt=attempt))
                continue
            seen.add(identity)
            item = copy.deepcopy(attempt)
            if item['success']:
                original_key = item['trajectory']
                item['trajectory'] = f'traj_{len(merged["episodes"])}'
                episode = copy.deepcopy(by_id[original_key])
                episode['episode_id'] = len(merged['episodes'])
                official_merged['episodes'].append(episode)
                copies.append((root/'trajectory.h5', original_key, item['trajectory']))
                merged['episodes'].append(item)
            merged['attempts'].append(item)
    # Keep the first reset identity regardless of outcome; never select by success.
    with h5py.File(out/'trajectory.inprogress.h5', 'w') as destination:
        for source in dict.fromkeys(path for path, _, _ in copies):
            with h5py.File(source, 'r') as origin:
                for path, old, new in copies:
                    if path == source:
                        if args.state_only:
                            group = destination.create_group(new)
                            for child in origin[old]:
                                if child != 'obs':
                                    origin.copy(origin[old][child], group, child)
                        else:
                            origin.copy(old, destination, new)
    (out/'trajectory.inprogress.h5').replace(out/'trajectory.h5')
    (out/'trajectory.json').write_text(json.dumps(official_merged, indent=2))
    merged['duplicate_attempts'] = duplicates
    if args.state_only:
        merged.update(obs_mode='none', render_backend='none')
    if args.attempts_from:
        source_manifest = json.loads((Path(args.attempts_from)/'stages.json').read_text())
        successful = {(e['seed'], e['repeat']): e for e in merged['episodes']}
        merged['attempts'] = [successful[(e['seed'], e['repeat'])] if e['success'] else e
                              for e in source_manifest['attempts']]
        merged['duplicate_attempts'] += source_manifest.get('duplicate_attempts', [])
    (out/'stages.json').write_text(json.dumps(merged, indent=2))
    print(dict(attempts=len(merged['attempts']), successful=len(merged['episodes'])), flush=True)


if __name__ == '__main__':
    main()
