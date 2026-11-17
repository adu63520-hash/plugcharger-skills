"""Render exact recorded states; retain official actions, states and stage indices."""
import argparse
import copy
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import common


def read_tree(group):
    return {k: read_tree(v) if isinstance(v, h5py.Group) else v[:] for k, v in group.items()}


def state_at(tree, i):
    return {k: state_at(v, i) if isinstance(v, dict) else torch.as_tensor(v[i])[None]
            for k, v in tree.items()}


def save_observations(group, observations):
    for key in observations[0]:
        values = [o[key] for o in observations]
        if isinstance(values[0], dict):
            save_observations(group.create_group(key), values)
        else:
            array = np.stack([np.asarray(v)[0] for v in values])
            group.create_dataset(key, data=array, compression='gzip', compression_opts=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--count', type=int)
    args = p.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'trajectory.h5').exists():
        raise FileExistsError('Use a fresh replay output directory')
    manifest = json.loads((root/'stages.json').read_text())
    original = json.loads((root/'trajectory.json').read_text())
    by_id = {f'traj_{e["episode_id"]}': e for e in original['episodes']}
    entries = manifest['episodes'][args.offset:None if args.count is None else args.offset+args.count]
    kwargs = copy.deepcopy(original['env_info']['env_kwargs'])
    kwargs.update(obs_mode='rgb', render_backend='cpu', sim_backend='cpu')
    kwargs['max_episode_steps'] = original['env_info']['max_episode_steps']
    env = gym.make(original['env_info']['env_id'], **kwargs)
    output_manifest = copy.deepcopy(manifest)
    output_manifest.update(episodes=[], attempts=[], obs_mode='rgb', render_backend='cpu',
                           state_replay_source=str(root.resolve()))
    output_official = copy.deepcopy(original)
    output_official['episodes'] = []
    output_official['env_info']['env_kwargs'].update(obs_mode='rgb', render_backend='cpu')
    try:
        with h5py.File(root/'trajectory.h5', 'r') as source, h5py.File(out/'trajectory.inprogress.h5', 'w') as target:
            for index, entry in enumerate(entries):
                key = entry['trajectory']
                new = f'traj_{index}'
                source.copy(key, target, new)
                if 'obs' in target[new]:
                    del target[new]['obs']
                states = read_tree(source[key]['env_states'])
                env.reset(**by_id[key]['reset_kwargs'])
                observations = []
                # Render only; physics is never stepped here. `success`, `terminated`
                # and `env_states` are copied verbatim from the source recording, so
                # RGB generation cannot alter labels or trajectories.
                for t in range(len(source[key]['actions'])+1):
                    env.unwrapped.set_state_dict(state_at(states, t))
                    observations.append(copy.deepcopy(common.to_numpy(env.unwrapped.get_obs())))
                save_observations(target[new].create_group('obs'), observations)
                target.flush()
                item = copy.deepcopy(entry)
                item['trajectory'] = new
                output_manifest['episodes'].append(item)
                output_manifest['attempts'].append(item)
                episode = copy.deepcopy(by_id[key])
                episode['episode_id'] = index
                output_official['episodes'].append(episode)
                print(dict(source=key, trajectory=new, seed=entry['seed'], frames=len(observations)), flush=True)
    finally:
        env.close()
    if args.offset == 0 and args.count is None:
        successful = {(e['seed'], e['repeat']): e for e in output_manifest['episodes']}
        output_manifest['attempts'] = [successful[(e['seed'], e['repeat'])] if e['success'] else e
                                       for e in manifest['attempts']]
    (out/'trajectory.inprogress.h5').replace(out/'trajectory.h5')
    (out/'stages.json').write_text(json.dumps(output_manifest, indent=2))
    (out/'trajectory.json').write_text(json.dumps(output_official, indent=2))


if __name__ == '__main__':
    main()
