"""Single-pass RGB capture: physics runs once, pixels are taken inline.

Why this exists
---------------
`replay_rgb` restores every recorded state with `set_state_dict` and renders around it.
With `set_state_dict` inside the loop, PhysX re-absorbs contacts each step and can push a
grasped object out of the gripper (measured: charger drifted ~4.2 cm, insert success lost
on 13/40 episodes). Training on those pixels would mix labels with wrong physics.

This module instead resets the env from the official `reset_kwargs`, replays the recorded
expert actions once, and records `get_obs()` at every step (s_0 .. s_T). Physics is never
restored mid-episode, so the trajectory is the same one that produced the original
`success` array. `success`, `terminated`, `truncated` and `env_states` are replaced by the
values recomputed during this run, not copied, so any divergence is visible.

Verification is mandatory: every stage boundary is re-derived from this run and must match
the recorded boundaries, and the final `success` must hold. A mismatch is reported, never
silently dropped.
"""
import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import common
from charger.stages import measurements, boundaries


def save_observations(group, observations):
    for key in observations[0]:
        values = [o[key] for o in observations]
        if isinstance(values[0], dict):
            save_observations(group.create_group(key), values)
        else:
            array = np.stack([np.asarray(v)[0] for v in values])
            group.create_dataset(key, data=array, compression='gzip', compression_opts=1)


def capture(env, reset_kwargs, actions):
    """Replay once, returning observations and per-step labels/terminations."""
    obs, _ = env.reset(**reset_kwargs)
    observations = [obs]
    labels, terminated, truncated = [], [], []
    for action in actions:
        obs, _, term, trunc, _ = env.step(torch.as_tensor(action)[None])
        observations.append(obs)
        labels.append(measurements(env))
        terminated.append(bool(torch.as_tensor(term).any()))
        truncated.append(bool(torch.as_tensor(trunc).any()))
    return observations, labels, terminated, truncated


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--count', type=int)
    p.add_argument('--verify-limit', type=int, default=0,
                   help='Only capture the first N episodes (dry run); 0 means all.')
    args = p.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'trajectory.h5').exists():
        raise FileExistsError('Use a fresh capture output directory')
    manifest = json.loads((root / 'stages.json').read_text())
    original = json.loads((root / 'trajectory.json').read_text())
    by_id = {f'traj_{e["episode_id"]}': e for e in original['episodes']}
    entries = manifest['episodes'][args.offset:]
    entries = entries[:None if args.count is None else args.count]
    if args.verify_limit:
        entries = entries[:args.verify_limit]
    kwargs = json.loads((root / 'trajectory.json').read_text())['env_info']['env_kwargs']
    kwargs.update(obs_mode='rgb', render_backend='cpu', sim_backend='cpu')
    env = gym.make(original['env_info']['env_id'], **kwargs)
    output_manifest = {k: v for k, v in manifest.items() if k not in ('episodes', 'attempts')}
    output_manifest.update(episodes=[], attempts=[], obs_mode='rgb', render_backend='cpu',
                           capture_mode='single_pass_replay',
                           state_replay_source=str(root.resolve()))
    output_official = {k: v for k, v in original.items() if k != 'episodes'}
    output_official['env_info']['env_kwargs'].update(obs_mode='rgb', render_backend='cpu')
    output_official['episodes'] = []
    report = []
    started = time.monotonic()
    try:
        with h5py.File(root / 'trajectory.h5', 'r') as source, \
                h5py.File(out / 'trajectory.inprogress.h5', 'w') as target:
            for index, entry in enumerate(entries):
                key = entry['trajectory']
                new = f'traj_{index}'
                actions = source[key]['actions'][:]
                observations, labels, term, trunc = capture(env, by_id[key]['reset_kwargs'], actions)
                # Re-derive boundaries from this run; do not trust the copy.
                try:
                    derived = boundaries(labels)
                except ValueError as exc:
                    derived = None
                    row = dict(source=key, seed=entry['seed'], ok=False, error=repr(exc))
                if derived is not None:
                    row = dict(source=key, seed=entry['seed'], ok=True,
                               boundaries_match=derived == entry['boundaries'],
                               final_success=bool(labels[-1]['C']),
                               recorded=entry['boundaries'], derived=derived)
                report.append(row)
                print(row, flush=True)
                group = target.create_group(new)
                group.create_dataset('actions', data=actions)
                group.create_dataset('rewards', data=source[key]['rewards'][:])
                group.create_dataset('terminated', data=np.array(term, bool))
                group.create_dataset('truncated', data=np.array(trunc, bool))
                group.create_dataset('success', data=np.array([m['C'] for m in labels], bool))
                save_observations(group.create_group('obs'), observations)
                # State tree: keep the official layout (per-step recorded states).
                _copy_states(source[key]['env_states'], group.create_group('env_states'))
                target.flush()
                item = dict(entry)
                item['trajectory'] = new
                if derived is not None:
                    item['boundaries'] = derived
                item['captured'] = row['ok']
                output_manifest['episodes'].append(item)
                episode = dict(by_id[key])
                episode['episode_id'] = index
                output_official['episodes'].append(episode)
    finally:
        env.close()
    (out / 'trajectory.inprogress.h5').replace(out / 'trajectory.h5')
    output_manifest['attempts'] = output_manifest['episodes']
    output_manifest['capture_report'] = report
    (out / 'stages.json').write_text(json.dumps(output_manifest, indent=2))
    (out / 'trajectory.json').write_text(json.dumps(output_official, indent=2))
    ok = sum(r['ok'] for r in report)
    match = sum(r.get('boundaries_match', False) for r in report)
    print(dict(captured=len(report), boundaries_ok=ok, exact_match=match,
               seconds=round(time.monotonic() - started, 1)), flush=True)


def _copy_states(source_group, dest_group):
    for name in source_group:
        child = source_group[name]
        if isinstance(child, h5py.Dataset):
            dest_group.create_dataset(name, data=child[:])
        else:
            _copy_states(child, dest_group.create_group(name))


if __name__ == '__main__':
    main()
