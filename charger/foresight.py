"""Offline downstream rollout labels and binomial visual foresight fitting."""
import argparse
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18
import gymnasium as gym
import mani_skill.envs
from .evaluate import Policy, rollout
from .data import policy_observation
from .stages import measurements
from .splits import split_seeds


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Linear(512, 1)

    def forward(self, rgb):
        return self.encoder(rgb.float()/255.).squeeze(-1)


def binomial_nll(logits, successes, trials):
    return (F.binary_cross_entropy_with_logits(logits, successes/trials, reduction='none')*trials).mean()


def load_starts(root, split='train'):
    root = Path(root)
    manifest = json.loads((root/'stages.json').read_text())
    chosen = split_seeds(manifest['episodes'])[split]
    selected = [e for e in manifest['episodes'] if e['seed'] in chosen]
    metadata = json.loads((root/'trajectory.json').read_text())
    official = {f'traj_{e["episode_id"]}': e for e in metadata['episodes']}
    return selected, official


def expert_start(env, file, entry, official, stage='B'):
    reset = official[entry['trajectory']]['reset_kwargs']
    obs, _ = env.reset(**reset)
    prefix = file[entry['trajectory']]['actions'][:entry['boundaries'][stage][0]]
    for action in prefix:
        obs, _, _, _, _ = env.step(torch.as_tensor(action)[None])
    if not measurements(env)['A' if stage == 'B' else 'B']:
        raise RuntimeError(f'Invalid expert prefix {entry["trajectory"]}')
    return obs, reset, list(prefix)


def collect(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = Policy(Path(args.checkpoints)/'B'/'best.pt', device)
    downstream = Policy(Path(args.checkpoints)/'C'/'best.pt', device)
    starts, official = load_starts(args.data)
    env = gym.make('PlugCharger-v1', obs_mode='rgb', sim_backend='cpu',
                   render_backend=args.render_backend, control_mode='pd_joint_pos',
                   reward_mode='sparse', max_episode_steps=1000)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, images, counts = [], [], []
    try:
        with h5py.File(Path(args.data)/'trajectory.h5') as file:
            for i, entry in enumerate(starts[:args.states]):
                torch.manual_seed(args.seed+i)
                obs, reset, prefix = expert_start(env, file, entry, official)
                terminal = False
                executed = []
                for _ in range(0, args.budget, args.action_steps):
                    actions = base.actions(obs)
                    for action in actions[:min(args.action_steps, args.budget-len(executed))]:
                        a = action.detach().cpu().numpy()
                        obs, _, _, _, _ = env.step(torch.as_tensor(a)[None])
                        executed.append(a)
                        if measurements(env)['B']:
                            terminal = True
                            break
                    if terminal:
                        break
                if not terminal:
                    print(dict(seed=entry['seed'], base_B_success=False), flush=True)
                    continue
                rgb, _ = policy_observation(obs, base.camera.split(',')[0])
                image = rgb[0].cpu().numpy()
                full_prefix = np.asarray(prefix+executed, dtype=np.float32)
                successes = 0
                for repeat in range(args.repeats):
                    obs, _ = env.reset(**reset)
                    for a in full_prefix:
                        obs, _, _, _, _ = env.step(torch.as_tensor(a)[None])
                    if not measurements(env)['B']:
                        raise RuntimeError('B terminal prefix replay did not reproduce success')
                    # Independently randomized C; B prefix is a fixed action sequence.
                    torch.manual_seed(args.seed+100000+i*args.repeats+repeat)
                    _, success, _ = rollout(env, obs, downstream, 'C', args.budget, args.action_steps)
                    successes += int(success)
                images.append(image)
                counts.append([successes, args.repeats])
                row = dict(seed=entry['seed'], trajectory=entry['trajectory'],
                           successes=successes, repeats=args.repeats,
                           prefix_sha256=hashlib.sha256(full_prefix.tobytes()).hexdigest())
                rows.append(row)
                np.savez_compressed(out/f'prefix_{i}.npz', actions=full_prefix,
                                    reset_json=json.dumps(reset), rgb=image)
                np.savez_compressed(out/'labels.npz', rgb=np.stack(images), counts=np.array(counts),
                                    seeds=np.array([r['seed'] for r in rows]))
                (out/'manifest.json').write_text(json.dumps(dict(config=vars(args), rows=rows), indent=2))
                print(row, flush=True)
    finally:
        env.close()
    if not rows:
        raise RuntimeError('Base B produced no successful terminal states; train base first')


def fit(args):
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arrays = np.load(Path(args.data)/'labels.npz')
    rgb = torch.as_tensor(arrays['rgb'], device=device)
    counts = torch.as_tensor(arrays['counts'], dtype=torch.float32, device=device)
    seeds = arrays['seeds']
    unique = np.unique(seeds)
    if len(unique) < 5:
        raise ValueError('Need at least 5 independent terminal-state seeds')
    np.random.default_rng(42).shuffle(unique)
    val = np.isin(seeds, unique[:max(1,len(unique)//5)])
    train_idx = torch.as_tensor(np.flatnonzero(~val), device=device)
    val_idx = torch.as_tensor(np.flatnonzero(val), device=device)
    net = Predictor().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for step in range(1, args.steps+1):
        net.train()
        idx = train_idx[torch.randint(len(train_idx), (args.batch_size,), device=device)]
        logits = net(rgb[idx])
        k, n = counts[idx].unbind(-1)
        loss = binomial_nll(logits, k, n)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                logits = net(rgb[val_idx])
                k, n = counts[val_idx].unbind(-1)
                value = binomial_nll(logits, k, n).item()
                brier = (logits.sigmoid()-k/n).square().mean().item()
            if value < best:
                best = value
                torch.save(dict(model=net.state_dict(), config=vars(args), validation_nll=value,
                                brier=brier, train_seeds=unique[max(1,len(unique)//5):].tolist(),
                                validation_seeds=unique[:max(1,len(unique)//5)].tolist()), out/'best.pt')
            print(dict(step=step, train_nll=loss.item(), validation_nll=value, brier=brier), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--data', required=True)
    c.add_argument('--checkpoints', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--states', type=int, default=256)
    c.add_argument('--repeats', type=int, default=5)
    c.add_argument('--budget', type=int, default=150)
    c.add_argument('--action-steps', type=int, default=4)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--render-backend', default='cpu')
    f = sub.add_parser('fit')
    f.add_argument('--data', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--steps', type=int, default=2000)
    f.add_argument('--batch-size', type=int, default=16)
    f.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    (collect if args.command == 'collect' else fit)(args)


if __name__ == '__main__':
    main()
