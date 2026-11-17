"""Split an already-recorded episode into per-skill HDF5 files without re-simulating.

Why this exists
---------------
`collect.py` records the full expert episode in one pass (reset -> grasp -> align -> insert).
The default layout keeps one episode per trajectory and `charger.data` slices it by stage, so
this is NOT needed for normal training. Two measured properties of this task motivate it:

  * The source recording can continue past the end of the last skill. Frames after the end of
    the task are not part of any skill, so they are never written into a per-skill file.
    Measured on `data/formal_rgb`: the recorded `terminated` flag starts exactly at the recorded
    C boundary end in 569/569 episodes, so `--truncate-at-termination` currently removes nothing
    there. It is kept because the cut is defined by the env's own signal, not by a constant.
  * Per-skill files make one skill's data easy to inspect, subset or re-render without
    touching the multi-GB source file.

Nothing is re-simulated: actions, observations, states and labels are copied from the source,
and every output episode records the source episode and step range it came from.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# Datasets copied verbatim (slice [start, hi)) when present. `rewards`/`success` are
# observation-aligned (length T+1) for some recordings and action-aligned (T) for others, so
# they are clamped to the action range instead of assumed.
ALIGNED = ('success', 'terminated', 'truncated')


def copy_tree(source, target):
    for name in source:
        child = source[name]
        if isinstance(child, h5py.Dataset):
            target.create_dataset(name, data=child[:])
        else:
            copy_tree(child, target.create_group(name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stage', choices=list('ABC'), required=True)
    p.add_argument('--truncate-at-termination', action='store_true',
                   help='Cut the skill at the first step where the env reports terminated/truncated.')
    p.add_argument('--horizon', type=int, default=16,
                   help='Extra frames kept past the stage end for action-chunk padding.')
    args = p.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'trajectory.h5').exists():
        raise FileExistsError('Use a fresh output directory')
    manifest = json.loads((root / 'stages.json').read_text())
    official = json.loads((root / 'trajectory.json').read_text())
    resets = {f'traj_{e["episode_id"]}': e['reset_kwargs'] for e in official['episodes']}
    episodes = []
    with h5py.File(root / 'trajectory.h5', 'r') as source, \
            h5py.File(out / 'trajectory.inprogress.h5', 'w') as target:
        for entry in manifest['episodes']:
            g = source[entry['trajectory']]
            start, end = entry['boundaries'][args.stage]
            total = len(g['actions'])
            stop = total
            if args.truncate_at_termination and 'terminated' in g:
                hits = np.flatnonzero(np.asarray(g['terminated'][:]))
                if len(hits):
                    stop = min(stop, int(hits[0]) + 1)
            # [start, min(end, stop)) is the skill; up to `horizon` extra frames are kept for
            # action-chunk padding, never past the termination point.
            skill_end = min(end, stop)
            hi = min(max(skill_end, start) + args.horizon, stop, total)
            hi = max(hi, start)
            name = f'traj_{len(episodes)}'
            group = target.create_group(name)
            group.create_dataset('actions', data=g['actions'][start:hi])
            for field in ALIGNED:
                if field in g:
                    group.create_dataset(field, data=g[field][start:min(hi, len(g[field]))])
            copy_tree(g['obs'], group.create_group('obs'))
            if 'env_states' in g:
                copy_tree(g['env_states'], group.create_group('env_states'))
            target.flush()
            episodes.append(dict(
                seed=entry['seed'], trajectory=name,
                boundaries={args.stage: [start, skill_end]},
                source=entry['trajectory'], source_range=[start, hi],
                label_extended_past_termination=bool(end > stop)))
    (out / 'trajectory.inprogress.h5').replace(out / 'trajectory.h5')
    (out / 'stages.json').write_text(json.dumps(dict(
        stage=args.stage, env_id=manifest.get('env_id'), episodes=episodes,
        source=str(root.resolve()), truncated_at_termination=args.truncate_at_termination,
        horizon=args.horizon), indent=2))
    (out / 'trajectory.json').write_text(json.dumps(dict(
        env_info=official['env_info'],
        episodes=[dict(episode_id=i, episode_seed=e['seed'],
                       reset_kwargs=resets[e['source']]) for i, e in enumerate(episodes)]), indent=2))
    print(dict(stage=args.stage, episodes=len(episodes),
               label_extended_past_termination=sum(e['label_extended_past_termination'] for e in episodes)),
          flush=True)


if __name__ == '__main__':
    main()
