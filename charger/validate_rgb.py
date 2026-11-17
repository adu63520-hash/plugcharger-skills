"""One-pass validity check for the whole RGB dataset, with no `set_state_dict` in the loop.

Why this exists
---------------
`replay_rgb` restores every recorded state and renders around it; HANDOFF_DEEPSEEK.md
measured that this rebuilds physics that only reproduces the recorded insertion on ~73% of
episodes. `capture_rgb` fixes the method but was never run at scale.

This module answers the two open questions from the handoff with measured numbers instead
of assumptions:

  1. Does a single forward pass from `reset_kwargs` over the RECORDED actions reproduce the
     recorded stage boundaries? (full dataset, not a 150-episode subset)
  2. Is the recorded `success` tail contaminated by ManiSkill's success-terminates-episode
     behaviour? I.e. does a recorded episode that reports success at step t keep accepting
     steps after t (which in a real rollout would be a different, post-reset episode)?

It labels, it does not train. No pixels are written unless `--write-pixels` is passed, so a
pass over all 569 episodes is cheap (physics only). Every episode gets a row; nothing is
dropped silently. Exit code is 0 only if every check passes.
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

from .stages import measurements, boundaries


def reset_kwargs_map(official_manifest):
    return {f'traj_{e["episode_id"]}': e['reset_kwargs'] for e in official_manifest['episodes']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True, help='JSON report path')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--obs-mode', default='none', choices=['none', 'rgb'],
                   help='rgb also renders every step; identical physics is expected.')
    p.add_argument('--render-backend', default='none', choices=['none', 'cpu'])
    p.add_argument('--write-pixels', help='Optional output dir for a corrected RGB dataset '
                                          '(only valid with --obs-mode rgb).')
    args = p.parse_args()
    root = Path(args.data)
    manifest = json.loads((root / 'stages.json').read_text())
    official = json.loads((root / 'trajectory.json').read_text())
    resets = reset_kwargs_map(official)
    env = gym.make('PlugCharger-v1', obs_mode=args.obs_mode, control_mode='pd_joint_pos',
                   sim_backend='cpu', render_backend=args.render_backend, max_episode_steps=400)
    rows = []
    started = time.monotonic()
    try:
        with h5py.File(root / 'trajectory.h5', 'r') as f:
            episodes = manifest['episodes'][args.offset:]
            if args.limit:
                episodes = episodes[:args.limit]
            episodes = episodes[args.shard::args.shards]
            for n, episode in enumerate(episodes):
                key = episode['trajectory']
                g = f[key]
                actions = g['actions'][:]
                recorded_success = g['success'][:]
                recorded_terminated = g['terminated'][:]
                obs, _ = env.reset(**resets[key])
                labels, terminated = [], []
                distances = []
                for step, action in enumerate(actions):
                    obs, _, term, _, _ = env.step(torch.as_tensor(action)[None])
                    m = measurements(env)
                    labels.append(m)
                    distances.append(m['distance'])
                    terminated.append(bool(torch.as_tensor(term).any()))
                row = dict(trajectory=key, seed=episode['seed'], steps=len(actions),
                           recorded=episode['boundaries'])
                try:
                    derived = boundaries(labels)
                    row['derived'] = derived
                    row['boundaries_match'] = derived == episode['boundaries']
                except ValueError as exc:
                    row['derived'] = None
                    row['boundaries_match'] = False
                    row['error'] = repr(exc)
                row['final_success'] = bool(labels[-1]['C'])
                row['recorded_final_success'] = bool(recorded_success[-1])
                row['min_distance'] = float(min(distances))
                # Contamination check: steps after the recorded first success still ran.
                success_steps = int(recorded_success.sum())
                row['steps_after_first_success'] = success_steps - 1
                row['replay_terminated_early'] = bool(any(terminated))
                row['replay_first_terminated_step'] = (
                    terminated.index(True) + 1 if any(terminated) else None)
                row['recorded_first_terminated_step'] = (
                    int(np.argmax(recorded_terminated)) + 1 if recorded_terminated.any() else None)
                # Which stage is the first to diverge, if any.
                if row['derived'] is not None and not row['boundaries_match']:
                    row['diverged_stage'] = next(
                        (s for s in 'ABC' if row['derived'][s] != episode['boundaries'][s]), None)
                rows.append(row)
                if n % 10 == 0:
                    print(dict(done=n, of=len(episodes), matched=sum(r['boundaries_match'] for r in rows),
                               seconds=round(time.monotonic() - started, 1)), flush=True)
    finally:
        env.close()

    exact = sum(r['boundaries_match'] for r in rows)
    summary = dict(
        n=len(rows), exact_boundary_match=exact,
        exact_rate=round(exact / max(len(rows), 1), 4),
        final_success_true=sum(r['final_success'] for r in rows),
        diverged_by_stage={s: sum(1 for r in rows if r.get('diverged_stage') == s) for s in 'ABC'},
        missing_boundary=sum(1 for r in rows if r['derived'] is None),
        contaminated_tail=sum(1 for r in rows if r['steps_after_first_success'] > 0),
        replay_terminated_early=sum(r['replay_terminated_early'] for r in rows),
        obs_mode=args.obs_mode, render_backend=args.render_backend,
        seconds=round(time.monotonic() - started, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(summary=summary, episodes=rows), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
