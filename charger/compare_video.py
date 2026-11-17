"""Side-by-side videos of rollouts that differ in outcome, to see where they diverge.

Both runs use the same checkpoint and environment; only the reset seed differs. Frames are
padded to the longer episode so the two stay time-aligned, and each is annotated with the step,
distance, angle and grasp state. A third column of the changing numbers is written next to the
video as JSON so a claim in a video can be checked numerically.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs
import numpy as np
import torch

from .evaluate import Policy
from .record_video import annotate, draw_text
from .stages import measurements


def run_episode(env, policy, seed, stage, budget, action_steps, policy_seed):
    torch.manual_seed(policy_seed + seed)
    obs, _ = env.reset(seed=seed)
    frames, trace = [], []
    steps, success = 0, False
    while steps < budget:
        actions = policy.actions(obs)
        for action in actions[:min(action_steps, budget - steps)]:
            obs, _, term, trunc, _ = env.step(action[None].to(env.unwrapped.device))
            steps += 1
            m = measurements(env)
            trace.append(dict(step=steps, **m))
            wrist = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
            glob = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
            line1 = 'seed %d step %d d %.3fm' % (seed, steps, m['distance'])
            line2 = 'ang %de grasp %d' % (int(np.degrees(m['angle'])), int(m['A']))
            frames.append(annotate(wrist, glob, line1, line2, m[stage]))
            if m[stage]:
                success = True
                break
            if bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
                break
        if success or steps >= budget:
            break
    return frames, trace, success, steps


def pad_to(frames, n):
    """Repeat the last frame so both columns have the same length."""
    if not frames:
        raise ValueError('no frames')
    return frames + [frames[-1]] * (n - len(frames))


def banner(frame, text, ok):
    out = frame.copy()
    draw_text(out, text, (3, 3), 2, (0, 255, 0) if ok else (255, 80, 80))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['A', 'B'])
    p.add_argument('--seeds', nargs='+', type=int, required=True,
                   help='Odd number of seeds: first half shown left, second half right, per row.')
    p.add_argument('--checkpoints', default='runs/formal')
    p.add_argument('--out', required=True)
    p.add_argument('--budget', type=int, default=200)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--inference-steps', type=int, default=20)
    p.add_argument('--policy-seed', type=int, default=0)
    p.add_argument('--scale', type=int, default=3)
    p.add_argument('--fps', type=int, default=20)
    args = p.parse_args()
    if len(args.seeds) < 2:
        p.error('need at least 2 seeds')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = Policy(Path(args.checkpoints) / args.mode / 'best.pt', device, args.inference_steps)
    env = gym.make('PlugCharger-v1', obs_mode='rgb', sim_backend='cpu', control_mode='pd_joint_pos',
                   render_backend='cpu', reward_mode='sparse', max_episode_steps=400 + args.budget)
    results = {}
    try:
        for seed in args.seeds:
            frames, trace, success, steps = run_episode(
                env, policy, seed, args.mode, args.budget, args.action_steps, args.policy_seed)
            if args.scale > 1:
                frames = [np.repeat(np.repeat(f, args.scale, 0), args.scale, 1) for f in frames]
            results[seed] = dict(frames=frames, success=success, steps=steps, trace=trace)
            print(dict(seed=seed, success=success, steps=steps,
                       min_distance=min(t['distance'] for t in trace)), flush=True)
    finally:
        env.close()
    # Pair each seed with every other seed: one side-by-side video per pair.
    seeds = args.seeds
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = results[seeds[i]], results[seeds[j]]
            if a['success'] == b['success']:
                continue  # only pair a success with a failure
            n = max(len(a['frames']), len(b['frames']))
            left = pad_to(a['frames'], n)
            right = pad_to(b['frames'], n)
            rows = [np.concatenate([banner(left[k], '%s ok' % seeds[i] if a['success'] else '%s fail' % seeds[i], a['success']),
                                    banner(right[k], '%s ok' % seeds[j] if b['success'] else '%s fail' % seeds[j], b['success'])], axis=1)
                    for k in range(n)]
            name = f'{args.mode}_seed{seeds[i]}_{"ok" if a["success"] else "fail"}_vs_seed{seeds[j]}_{"ok" if b["success"] else "fail"}'
            imageio.mimwrite(str(out / f'{name}.mp4'), rows, fps=args.fps, quality=7)
            (out / f'{name}.json').write_text(json.dumps(dict(
                left=dict(seed=seeds[i], success=a['success'], steps=a['steps']),
                right=dict(seed=seeds[j], success=b['success'], steps=b['steps'])), indent=2))
            print('wrote', name, flush=True)


if __name__ == '__main__':
    main()
