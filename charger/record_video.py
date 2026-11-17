"""Record a rollout with both cameras and an annotated overlay.

Why this exists
---------------
`charger.evaluate` stores only the camera the policy consumes (wrist), and only writes a video
for failures. Diagnosing a failure needs more: both views at once, and the numbers that define
each skill, drawn on the frame.

This module runs one episode, keeps every step of both cameras, and writes:

  * `*.mp4`   - wrist on top, global view below, annotated (step, distance, angle, grasp, skill)
  * `*.json`  - the per-step measurements, so the video can be checked against numbers

Labels are drawn from privileged environment state (distance, angle, grasp). They are diagnostic
output only; the policy never sees them.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs
import numpy as np
import torch

from .evaluate import Policy, rollout
from .stages import measurements

# 5x7 bitmap digits for annotation without any font dependency.
_GLYPHS = {
    '0': ('01110', '10001', '10011', '10101', '11001', '10001', '01110'),
    '1': ('00100', '01100', '00100', '00100', '00100', '00100', '01110'),
    '2': ('01110', '10001', '00001', '00010', '00100', '01000', '11111'),
    '3': ('11111', '00010', '00100', '00010', '00001', '10001', '01110'),
    '4': ('00010', '00110', '01010', '10010', '11111', '00010', '00010'),
    '5': ('11111', '10000', '11110', '00001', '00001', '10001', '01110'),
    '6': ('00110', '01000', '10000', '11110', '10001', '10001', '01110'),
    '7': ('11111', '00001', '00010', '00100', '01000', '01000', '01000'),
    '8': ('01110', '10001', '10001', '01110', '10001', '10001', '01110'),
    '9': ('01110', '10001', '10001', '01111', '00001', '00010', '01100'),
    'm': ('00000', '00000', '11010', '10101', '10101', '10101', '10101'),
    '.': ('00000', '00000', '00000', '00000', '00000', '01100', '01100'),
    '-': ('00000', '00000', '00000', '11111', '00000', '00000', '00000'),
    ' ': ('00000', '00000', '00000', '00000', '00000', '00000', '00000'),
    'd': ('00001', '00001', '01111', '10001', '10001', '10001', '01111'),
    'e': ('00000', '00000', '01110', '10001', '11111', '10000', '01110'),
    'g': ('00000', '01111', '10001', '01111', '00001', '00001', '01110'),
    'A': ('01110', '10001', '10001', '11111', '10001', '10001', '10001'),
    'B': ('11110', '10001', '10001', '11110', '10001', '10001', '11110'),
    'C': ('01110', '10001', '10000', '10000', '10000', '10001', '01110'),
    'K': ('10001', '10010', '10100', '11000', '10100', '10010', '10001'),
    'S': ('01111', '10000', '10000', '01110', '00001', '00001', '11110'),
    'T': ('11111', '00100', '00100', '00100', '00100', '00100', '00100'),
    'P': ('11110', '10001', '10001', '11110', '10000', '10000', '10000'),
    '=': ('00000', '00000', '11111', '00000', '11111', '00000', '00000'),
    '/': ('00001', '00010', '00010', '00100', '01000', '01000', '10000'),
}


def draw_text(frame, text, origin=(3, 3), scale=1, color=(255, 255, 255)):
    """Draw text with the bitmap font. `text` must use supported characters."""
    x0, y0 = origin
    h, w = frame.shape[:2]
    x = x0
    for ch in text:
        glyph = _GLYPHS.get(ch, _GLYPHS[' '])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit != '1':
                    continue
                ys = y0 + row * scale
                xs = x + col * scale
                if 0 <= ys < h and 0 <= xs < w:
                    frame[ys:ys + scale, xs:xs + scale] = color
        x += 6 * scale
    return frame


def annotate(wrist, global_view, line1, line2, ok):
    """Stack the two views and put a caption on each."""
    color = (0, 255, 0) if ok else (255, 80, 80)
    top = wrist.copy()
    bottom = global_view.copy()
    draw_text(top, line1, (3, 3), 2, color)
    draw_text(top, 'wrist', (3, top.shape[0] - 10), 1, (200, 200, 200))
    draw_text(bottom, line2, (3, 3), 2, color)
    draw_text(bottom, 'global', (3, bottom.shape[0] - 10), 1, (200, 200, 200))
    return np.concatenate([top, bottom], axis=0)


def expert_prefix(env, data_root, trajectory, stage):
    """Replay the expert up to a skill boundary so contacts are real (as charger.evaluate does).

    B and C must start from a held-out expert prefix: restoring `env_states` would not restore
    PhysX contact forces, so a plain reset leaves the charger un-grasped and the skill cannot
    be judged.
    """
    import h5py
    root = Path(data_root)
    manifest = json.loads((root / 'stages.json').read_text())
    episode = next(e for e in manifest['episodes'] if e['trajectory'] == trajectory)
    t = episode['boundaries'][stage][0]
    with h5py.File(root / 'trajectory.h5', 'r') as f:
        actions = f[trajectory]['actions'][:t]
    for action in actions:
        env.step(torch.as_tensor(action)[None])
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['A', 'B'])
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--checkpoints', default='runs/formal')
    p.add_argument('--data', default='data/formal_rgb')
    p.add_argument('--out', required=True)
    p.add_argument('--budget', type=int, default=200)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--inference-steps', type=int, default=20)
    p.add_argument('--policy-seed', type=int, default=0)
    p.add_argument('--render-backend', default='cpu', choices=['cpu', 'gpu'])
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--scale', type=int, default=2,
                   help='Integer upscale of each camera view; 128px source needs 2-3 to read the caption.')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = Policy(Path(args.checkpoints) / args.mode / 'best.pt', device, args.inference_steps)
    env = gym.make('PlugCharger-v1', obs_mode='rgb', sim_backend='cpu', control_mode='pd_joint_pos',
                   render_backend=args.render_backend, reward_mode='sparse',
                   max_episode_steps=400 + args.budget)
    # For B, "seed" selects a held-out expert episode: the skill starts from its A boundary.
    starts = {}
    if args.mode != 'A':
        manifest = json.loads((Path(args.data) / 'stages.json').read_text())
        official = json.loads((Path(args.data) / 'trajectory.json').read_text())
        resets = {f'traj_{e["episode_id"]}': e['reset_kwargs'] for e in official['episodes']}
        for e in manifest['episodes']:
            starts.setdefault(e['seed'], e)
    rows = []
    try:
        for seed in args.seeds:
            torch.manual_seed(args.policy_seed + seed)
            frames = []
            if args.mode == 'A':
                obs, _ = env.reset(seed=seed)
            else:
                entry = starts[seed]
                obs, _ = env.reset(**resets[entry['trajectory']])
                expert_prefix(env, args.data, entry['trajectory'], args.mode)
                obs = env.unwrapped.get_obs()
            m = measurements(env)
            trace = []
            steps, success = 0, False
            while steps < args.budget:
                actions = policy.actions(obs)
                for action in actions[:min(args.action_steps, args.budget - steps)]:
                    obs, _, term, trunc, _ = env.step(action[None].to(env.unwrapped.device))
                    steps += 1
                    m = measurements(env)
                    trace.append(dict(step=steps, **m))
                    wrist = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
                    glob = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
                    if args.scale > 1:
                        wrist = np.repeat(np.repeat(wrist, args.scale, 0), args.scale, 1)
                        glob = np.repeat(np.repeat(glob, args.scale, 0), args.scale, 1)
                    line1 = '%s step %d d %.3fm' % (args.mode, steps, m['distance'])
                    line2 = 'ang %de grasp %d' % (int(np.degrees(m['angle'])), int(m['A']))
                    frames.append(annotate(wrist, glob, line1, line2, m[args.mode]))
                    if m[args.mode]:
                        success = True
                        break
                    if bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
                        break
                if success or steps >= args.budget:
                    break
            name = f'{args.mode}_seed{seed}_{"ok" if success else "fail"}'
            imageio.mimwrite(str(out / f'{name}.mp4'), frames, fps=args.fps, quality=7)
            row = dict(seed=seed, success=success, steps=steps,
                       final_distance=m['distance'], final_angle_deg=float(np.degrees(m['angle'])),
                       min_distance=float(min(t['distance'] for t in trace)) if trace else None,
                       ever_grasped=bool(any(t['A'] for t in trace)),
                       trace=trace)
            rows.append(row)
            print({k: v for k, v in row.items() if k != 'trace'}, flush=True)
    finally:
        env.close()
    (out / f'{args.mode}_report.json').write_text(json.dumps(rows, indent=2))
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
