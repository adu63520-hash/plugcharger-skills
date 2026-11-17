"""Locate the first point at which two evaluation runs disagree, for one seed.

Motivation: two runs of the same evaluation config (`charger.evaluate`, identical checkpoint,
seeds and budgets) agreed on only 82/100 A-stage outcomes and disagreed on the first step of
some episodes while the robot was still 20 cm from the object. Physics replays are bitwise
deterministic in this install, so the difference must enter through an input to the policy.

This replays chain steps one at a time and, at each replan, compares the model input
(quantised RGB + proprio) against a stored recording. It reports the FIRST step whose policy
input differs, and how many pixels differ.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch

from .data import policy_observation
from .evaluate import Policy
from .stages import measurements


def record(seed, budget, action_steps, checkpoint, out):
    """Run one episode, saving the exact policy inputs at every replan."""
    env = gym.make('PlugCharger-v1', obs_mode='rgb', sim_backend='cpu', control_mode='pd_joint_pos',
                   render_backend='cpu', reward_mode='sparse', max_episode_steps=400 + budget)
    path = Path(checkpoint)
    policy = Policy(path / 'best.pt' if path.is_dir() else path, 'cuda')
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    steps, rows = 0, []
    while steps < budget:
        rgb, state = policy_observation(obs, policy.camera, policy.proprio)
        actions = policy.actions(obs)
        rows.append(dict(step=steps, rgb=rgb[0].numpy().copy(), state=state[0].numpy().copy(),
                         actions=actions.cpu().numpy().copy(),
                         distance=measurements(env)['distance']))
        for action in actions[:min(action_steps, budget - steps)]:
            obs, _, term, trunc, _ = env.step(action[None])
            steps += 1
            if measurements(env)['A'] or bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
                env.close()
                np.savez(out, **{'rgb': np.stack([r['rgb'] for r in rows]),
                                 'state': np.stack([r['state'] for r in rows]),
                                 'actions': np.stack([r['actions'] for r in rows])},
                         steps=np.array([r['step'] for r in rows]),
                         distance=np.array([r['distance'] for r in rows]))
                return len(rows)
    env.close()
    np.savez(out, **{'rgb': np.stack([r['rgb'] for r in rows]),
                     'state': np.stack([r['state'] for r in rows]),
                     'actions': np.stack([r['actions'] for r in rows])},
             steps=np.array([r['step'] for r in rows]), distance=np.array([r['distance'] for r in rows]))
    return len(rows)


def compare(a, b):
    x, y = np.load(a), np.load(b)
    n = min(len(x['steps']), len(y['steps']))
    for i in range(n):
        skip_equal = 'steps' in x and 'steps' in y and x['steps'][i] == y['steps'][i]
        if skip_equal and np.array_equal(x['rgb'][i], y['rgb'][i]) and np.allclose(
                x['state'][i], y['state'][i], atol=0, rtol=0):
            continue
        return dict(first_differ_at_replan=int(x['steps'][i]), x_steps=int(x['steps'][i]),
                    y_steps=int(y['steps'][i]),
                    rgb_pixels_differ=int((x['rgb'][i] != y['rgb'][i]).sum()),
                    state_maxdiff=float(np.abs(x['state'][i] - y['state'][i]).max()),
                    action_maxdiff=float(np.abs(x['actions'][i] - y['actions'][i]).max()),
                    distance_x=float(x['distance'][i]), distance_y=float(y['distance'][i]))
    return dict(first_differ_at_replan=None, identical_replans=n,
                distance_x=float(x['distance'][-1]), distance_y=float(y['distance'][-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--budget', type=int, default=192)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--checkpoint', default='runs/formal/A')
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ('run1.npz', 'run2.npz'):
        path = out / name
        replans = record(args.seed, args.budget, args.action_steps, args.checkpoint, path)
        print(dict(file=str(path), replans=replans), flush=True)
        paths.append(str(path))
    print(json.dumps(compare(*paths)), flush=True)


if __name__ == '__main__':
    main()
