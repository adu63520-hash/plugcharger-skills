"""PPO on a residual over frozen B, with offline C-success terminal rewards."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from .evaluate import Policy
from .foresight import Predictor, load_starts, expert_start
from .residual import ActorCritic, features
from .data import policy_observation
from .stages import measurements


def advantages(rewards, dones, values, next_value, gamma=.99, lam=.95):
    result = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        following = next_value if t == len(rewards)-1 else values[t+1]
        keep = 1-dones[t]
        delta = rewards[t]+gamma*following*keep-values[t]
        gae = delta+gamma*lam*keep*gae
        result[t] = gae
    return result, result+values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--predictor')
    p.add_argument('--out', required=True)
    p.add_argument('--reward', choices=['foresight', 'local'], default='foresight')
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--rollout-steps', type=int, default=256)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--alpha', type=float, default=.1)
    p.add_argument('--budget', type=int, default=150)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--render-backend', default='cpu')
    args = p.parse_args()
    if args.reward == 'foresight' and not args.predictor:
        p.error('foresight reward requires a trained --predictor')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = Policy(args.base, device)
    base.model.requires_grad_(False).eval()
    predictor = None
    if args.predictor:
        predictor = Predictor().to(device).eval().requires_grad_(False)
        predictor.load_state_dict(torch.load(args.predictor, map_location=device, weights_only=False)['model'])
    starts, official = load_starts(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args), indent=2))
    base_hash = hashlib.sha256(Path(args.base).read_bytes()).hexdigest()
    env = gym.make('PlugCharger-v1', obs_mode='rgb', sim_backend='cpu',
                   render_backend=args.render_backend, control_mode='pd_joint_pos',
                   reward_mode='sparse', max_episode_steps=1000)
    low = torch.as_tensor(env.action_space.low.reshape(-1), device=device)
    high = torch.as_tensor(env.action_space.high.reshape(-1), device=device)
    if not torch.isfinite(torch.cat((low, high))).all():
        raise ValueError('Residual requires finite controller action bounds')
    network = ActorCritic(256+base.model.config['action_dim'], len(low)).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4, eps=1e-5)
    transitions = 0
    completed, successful = 0, 0
    ep_steps = 0
    buffer = []
    file = h5py.File(Path(args.data)/'trajectory.h5')

    def reset():
        entry = starts[int(rng.integers(len(starts)))]
        obs, _, _ = expert_start(env, file, entry, official)
        return obs

    def next_base(obs):
        nonlocal buffer
        if not buffer:
            buffer = list(base.actions(obs)[:args.action_steps])
        return buffer.pop(0)

    obs = reset()
    base_action = next_base(obs)
    current = features(base, obs, base_action)[0]
    try:
        for update in range(1, args.updates+1):
            warmup = update <= args.warmup
            batch = {k: [] for k in ['features', 'raw', 'logprob', 'value', 'reward', 'done']}
            for _ in range(args.rollout_steps):
                with torch.no_grad():
                    distribution, value = network(current[None])
                    raw = torch.zeros_like(distribution.mean) if warmup else distribution.sample()
                    logprob = distribution.log_prob(raw).sum(-1)
                    delta = args.alpha*raw[0].tanh()*(high-low)/2
                    action = (base_action+delta).clamp(low, high)
                next_obs, _, terminated, truncated, _ = env.step(action[None].to(env.unwrapped.device))
                ep_steps += 1
                transitions += 1
                success = measurements(env)['B']
                done = success or ep_steps >= args.budget or bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any())
                reward = 0.
                if success:
                    reward = 1.
                    if args.reward == 'foresight':
                        rgb, _ = policy_observation(next_obs, base.camera.split(',')[0])
                        with torch.no_grad():
                            reward = predictor(rgb.to(device)).sigmoid().item()
                for key, item in [('features', current), ('raw', raw[0]),
                                  ('logprob', logprob[0]), ('value', value[0]),
                                  ('reward', torch.tensor(reward, device=device)),
                                  ('done', torch.tensor(float(done), device=device))]:
                    batch[key].append(item.detach())
                if done:
                    completed += 1
                    successful += int(success)
                    obs = reset()
                    buffer = []
                    ep_steps = 0
                else:
                    obs = next_obs
                base_action = next_base(obs)
                current = features(base, obs, base_action)[0]
            batch = {k: torch.stack(v) for k, v in batch.items()}
            with torch.no_grad():
                _, next_value = network(current[None])
                adv, returns = advantages(batch['reward'], batch['done'], batch['value'], next_value[0])
                adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
            for _ in range(args.epochs):
                indices = torch.randperm(args.rollout_steps, device=device)
                for idx in indices.split(args.batch_size):
                    distribution, value = network(batch['features'][idx])
                    new_logprob = distribution.log_prob(batch['raw'][idx]).sum(-1)
                    ratio = (new_logprob-batch['logprob'][idx]).exp()
                    actor_loss = -torch.minimum(ratio*adv[idx], ratio.clamp(.8, 1.2)*adv[idx]).mean()
                    value_loss = .5*(value-returns[idx]).square().mean()
                    entropy = distribution.entropy().sum(-1).mean()
                    loss = value_loss if warmup else actor_loss+.5*value_loss-.001*entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), .5)
                    optimizer.step()
            row = dict(update=update, transitions=transitions, completed=completed,
                       local_success_rate=successful/max(1,completed), warmup=warmup,
                       mean_reward=batch['reward'].mean().item())
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            torch.save(dict(network=network.state_dict(), network_config=network.config,
                            alpha=args.alpha, action_low=low.cpu(), action_high=high.cpu(),
                            base_checkpoint=str(Path(args.base).resolve()), base_sha256=base_hash,
                            config=vars(args), update=update, optimizer=optimizer.state_dict()), out/'last.tmp')
            (out/'last.tmp').replace(out/'last.pt')
            print(row, flush=True)
    finally:
        file.close()
        env.close()
    if hashlib.sha256(Path(args.base).read_bytes()).hexdigest() != base_hash:
        raise RuntimeError('Frozen base checkpoint changed during residual training')


if __name__ == '__main__':
    main()
