"""Validate real saved boundaries through sequential expert replay."""
import argparse
import json
from pathlib import Path
import h5py
import torch
import gymnasium as gym
import mani_skill.envs
from charger.stages import measurements


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--render-backend', default='none')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    root = Path(args.data)
    manifest = json.loads((root / 'stages.json').read_text())
    official = json.loads((root / 'trajectory.json').read_text())
    resets = {f'traj_{e["episode_id"]}': e['reset_kwargs'] for e in official['episodes']}
    env = gym.make('PlugCharger-v1', obs_mode='none', control_mode='pd_joint_pos',
                   sim_backend='cpu', render_backend=args.render_backend, max_episode_steps=400)
    rows = []
    try:
        with h5py.File(root / 'trajectory.h5') as f:
            episodes = manifest['episodes'][:args.limit] if args.limit else manifest['episodes']
            for episode in episodes:
                key = episode['trajectory']
                g = f[key]
                row = dict(trajectory=key, boundaries={})
                # Full original action sequence, zero resets between skills.
                env.reset(**resets[key])
                # Stage boundaries are the first step at which the label holds.
                # Later steps may legitimately drop A/B (e.g. release during
                # insertion), so checking at `end` would flag valid expert data.
                first = {stage: end - 1 for stage, (_, end) in episode['boundaries'].items()}
                for t, action in enumerate(g['actions'][:], 1):
                    env.step(torch.as_tensor(action)[None])
                    for stage, s in first.items():
                        if s == t:
                            result = measurements(env)
                            row['boundaries'][stage] = result
                            if not result[stage]:
                                raise AssertionError(f'{key} replay boundary {stage} failed at {t}')
                row['expert_replay_success'] = measurements(env)['C']
                if not row['expert_replay_success']:
                    raise AssertionError(f'{key} expert replay did not succeed')
                rows.append(row)
                print(row, flush=True)
    finally:
        env.close()
    (root / 'verification.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
