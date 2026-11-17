"""Real rollouts. Skill chains never reset or restore at internal boundaries."""
import argparse
import json
import math
from pathlib import Path
import h5py
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from .data import policy_observation
from .model import VisionDiffusion
from .stages import measurements
from .splits import split_seeds


class Policy:
    def __init__(self, path, device, inference_steps=20):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.model = VisionDiffusion(**ckpt['model_config']).to(device).eval()
        self.model.load_state_dict(ckpt['model'])
        self.camera = ckpt['args']['camera']
        self.proprio = ckpt['args']['proprio']
        self.low = torch.as_tensor(ckpt['low'], device=device)
        self.high = torch.as_tensor(ckpt['high'], device=device)
        self.device = device
        self.inference_steps = inference_steps
        self.checkpoint_path = str(path)
        self.relative_actions = ckpt['args'].get('relative_actions', False)

    def actions(self, obs):
        rgb, state = policy_observation(obs, self.camera, self.proprio)
        actions = self.model.sample(rgb.to(self.device), state.to(self.device), self.inference_steps)[0]
        physical = self.low + (actions + 1) / 2 * (self.high - self.low)
        if self.relative_actions:
            qpos = torch.as_tensor(obs['agent']['qpos'], device=self.device).reshape(1, -1)
            physical[:, :-1] += qpos[0, :physical.shape[-1]-1]
        return physical


def rollout(env, obs, policy, stage, budget, action_steps, frames=None, trace=None):
    steps = 0
    while steps < budget:
        actions = policy.actions(obs)
        for action in actions[:min(action_steps, budget - steps)]:
            if hasattr(policy, 'correct'):
                action = policy.correct(obs, action)
            obs, _, terminated, truncated, _ = env.step(action[None].to(env.unwrapped.device))
            steps += 1
            m = measurements(env)
            if trace is not None:
                trace.append(dict(stage=stage, step=steps, **m))
            if frames is not None:
                camera = policy.camera if hasattr(policy, 'camera') else policy.base.camera
                rgb, _ = policy_observation(obs, camera)
                views = [rgb[0, j:j+3].permute(1, 2, 0).cpu().numpy()
                         for j in range(0, rgb.shape[1], 3)]
                frames.append(np.concatenate(views, axis=1))
            if m[stage]:
                return obs, True, steps
            if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                return obs, False, steps
    return obs, False, steps


def interval(successes, n):
    z = 1.96
    p = successes / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    delta = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [max(0., center-delta), min(1., center+delta)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['A', 'B', 'C', 'AB', 'BC', 'ABC', 'Bprime', 'ABprime', 'BprimeC', 'ABprimeC'], required=True)
    p.add_argument('--residual')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--checkpoints', default='runs')
    p.add_argument('--data', default='data/charger')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--inference-steps', type=int, default=20)
    p.add_argument('--start-seed', type=int, default=10000)
    p.add_argument('--policy-seed', type=int, default=0)
    p.add_argument('--render-backend', default='gpu', choices=['gpu', 'cpu'])
    p.add_argument('--budget', type=int, default=150)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--out', default='runs/evaluation.json')
    p.add_argument('--video-dir')
    args = p.parse_args()
    if args.episodes < 1 or args.action_steps < 1 or args.budget < 1:
        p.error('episodes, action-steps and budget must be positive')
    torch.manual_seed(args.policy_seed)
    stages = args.mode.replace('Bprime', 'B')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policies = {s: Policy(Path(args.checkpoints) / s / 'best.pt', device, args.inference_steps) for s in stages}
    if 'Bprime' in args.mode:
        if not args.residual:
            p.error('Bprime evaluation requires --residual')
        from .residual import ResidualPolicy
        policies['B'] = ResidualPolicy(policies['B'], args.residual)
    env = gym.make('PlugCharger-v1', obs_mode='rgb', sim_backend='cpu',
                   control_mode='pd_joint_pos', render_backend=args.render_backend, reward_mode='sparse',
                   max_episode_steps=400 + args.budget * len(stages))
    rows = []
    data = None
    try:
        if stages[0] != 'A':
            manifest = json.loads((Path(args.data) / 'stages.json').read_text())
            chosen = split_seeds(manifest['episodes'])[args.split]
            starts = [e for e in manifest['episodes'] if e['seed'] in chosen]
            # Do not silently recycle identical starts to inflate sample count.
            if args.offset+args.episodes > len(starts):
                raise ValueError(f'Only {len(starts)} held-out starts; reduce --episodes or collect more data')
            data = h5py.File(Path(args.data) / 'trajectory.h5')
            metadata = json.loads((Path(args.data) / 'trajectory.json').read_text())
            official = {f'traj_{e["episode_id"]}': e for e in metadata['episodes']}
        for i in range(args.offset, args.offset+args.episodes):
            torch.manual_seed(args.policy_seed + i)
            if stages[0] == 'A':
                seed = args.start_seed + i
                obs, _ = env.reset(seed=seed)
            else:
                entry = starts[i]
                seed = entry['seed']
                obs, _ = env.reset(**official[entry['trajectory']]['reset_kwargs'])
                t = entry['boundaries'][stages[0]][0]
                # PhysX contacts/controller history are absent from env_states.
                # Replay a held-out expert prefix to obtain real grasp contacts.
                for action in data[entry['trajectory']]['actions'][:t]:
                    obs, _, _, _, _ = env.step(torch.as_tensor(action)[None])
                previous = 'A' if stages[0] == 'B' else 'B'
                if not measurements(env)[previous]:
                    raise RuntimeError(f'Expert prefix failed for {entry["trajectory"]}')
            row = dict(seed=seed, stages={})
            frames = [] if args.video_dir else None
            trace = []
            for stage in stages:
                # Phase-specific noise keeps paired C rollouts comparable even
                # when B and Bprime consume different numbers of action chunks.
                torch.manual_seed(args.policy_seed+i*17+ord(stage))
                obs, success, steps = rollout(env, obs, policies[stage], stage, args.budget, args.action_steps, frames, trace)
                row['stages'][stage] = dict(success=success, steps=steps)
                if not success:
                    break
            row['success'] = len(row['stages']) == len(stages) and all(v['success'] for v in row['stages'].values())
            row['trace'] = trace
            if not row['success'] and args.video_dir and frames:
                import imageio.v2 as imageio
                folder = Path(args.video_dir)
                folder.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(str(folder/f'{args.mode}_seed{seed}.mp4'), frames, fps=20)
            rows.append(row)
            print({k: v for k, v in row.items() if k != 'trace'}, flush=True)
    finally:
        env.close()
        if data is not None:
            data.close()
    successes = sum(r['success'] for r in rows)
    result = dict(config=vars(args), successes=successes, n=len(rows),
                  success_rate=successes/len(rows), wilson_95=interval(successes, len(rows)), episodes=rows)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print({k: result[k] for k in ['successes', 'n', 'success_rate', 'wilson_95']})


if __name__ == '__main__':
    main()
