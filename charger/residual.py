"""Frozen visual base policy plus per-control-step bounded residual actions."""
import torch
import hashlib
from pathlib import Path
from torch import nn
from torch.distributions import Normal
from .data import policy_observation


@torch.no_grad()
def features(base, obs, action):
    rgb, proprio = policy_observation(obs, base.camera, base.proprio)
    latent = base.model.encode(rgb.to(base.device), proprio.to(base.device))
    representation = action.clone()
    if getattr(base, 'relative_actions', False):
        qpos = torch.as_tensor(obs['agent']['qpos'], device=base.device).reshape(1, -1)
        representation[:-1] -= qpos[0, :len(action)-1]
    normalized_action = 2*(representation-base.low)/(base.high-base.low).clamp_min(1e-6)-1
    return torch.cat((latent, normalized_action.reshape(1, -1)), -1).detach()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim)
        self.shared = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.5))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, x):
        z = self.shared(x)
        return Normal(self.actor(z), self.log_std.clamp(-5, 0).exp()), self.critic(z).squeeze(-1)


class ResidualPolicy:
    def __init__(self, base, checkpoint):
        self.base = base
        self.device = base.device
        data = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if 'base_sha256' in data:
            if hashlib.sha256(Path(base.checkpoint_path).read_bytes()).hexdigest() != data['base_sha256']:
                raise ValueError('Residual checkpoint does not match the frozen base B')
        self.network = ActorCritic(**data['network_config']).to(self.device).eval()
        self.network.load_state_dict(data['network'])
        self.alpha = data['alpha']
        self.low = torch.as_tensor(data['action_low'], device=self.device)
        self.high = torch.as_tensor(data['action_high'], device=self.device)

    def actions(self, obs):
        return self.base.actions(obs)

    @torch.no_grad()
    def correct(self, obs, base_action):
        distribution, _ = self.network(features(self.base, obs, base_action))
        delta = self.alpha * distribution.mean[0].tanh() * (self.high-self.low)/2
        return (base_action+delta).clamp(self.low, self.high)
