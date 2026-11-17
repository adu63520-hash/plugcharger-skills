"""ResNet18 from scratch, conditioned action DDPM (compact reference model)."""
import math
import torch
from torch import nn
from torchvision.models import resnet18


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet18(weights=None)
        def replace_norm(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.GroupNorm(child.num_features//16, child.num_features))
                else:
                    replace_norm(child)
        replace_norm(network)
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        self.keypoints = nn.Conv2d(512, 32, 1)
        self.projection = nn.Linear(64, 256)

    def forward(self, x):
        logits = self.keypoints(self.backbone(x))
        b, c, h, w = logits.shape
        weights = logits.reshape(b, c, -1).softmax(-1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing='ij')
        coordinates = torch.stack((xx.flatten(), yy.flatten()), -1).to(weights.dtype)
        return self.projection((weights@coordinates).reshape(b, -1))


class VisionDiffusion(nn.Module):
    def __init__(self, action_dim, proprio_dim=0, horizon=16, diffusion_steps=100, encoder_kind='gap', num_views=1):
        super().__init__()
        self.config = dict(action_dim=action_dim, proprio_dim=proprio_dim,
                           horizon=horizon, diffusion_steps=diffusion_steps, encoder_kind=encoder_kind, num_views=num_views)
        if encoder_kind == 'spatial':
            self.encoder = SpatialEncoder()
        else:
            self.encoder = resnet18(weights=None)
            self.encoder.fc = nn.Linear(512, 256)
        self.condition = nn.Linear(256*num_views + proprio_dim, 256)
        self.action_in = nn.Linear(action_dim, 256)
        self.position = nn.Parameter(torch.randn(1, horizon, 256) * .02)
        self.time = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256))
        layer = nn.TransformerEncoderLayer(256, 4, 512, dropout=0., batch_first=True)
        self.denoiser = nn.TransformerEncoder(layer, 3)
        self.output = nn.Linear(256, action_dim)
        # Cosine schedule reaches near-pure noise even with only 100 steps.
        # A 100-step unscaled linear 1e-4..0.02 schedule would leave ~36% signal.
        grid = torch.linspace(0, 1, diffusion_steps + 1)
        cumulative = torch.cos((grid + .008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, .999)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha_bar', (1 - beta).cumprod(0))

    def encode(self, rgb, proprio):
        # RGB uint8 BCHW; fixed scaling, no pretrained normalization/weights.
        b, channels, h, w = rgb.shape
        views = rgb.reshape(b*self.config['num_views'], 3, h, w)
        visual = self.encoder(views.float()/255.).reshape(b, -1)
        return self.condition(torch.cat((visual, proprio), -1))

    def noise_prediction(self, noisy, t, condition):
        frequencies = torch.exp(torch.arange(32, device=t.device) * (-math.log(10000) / 31))
        phase = t[:, None] * frequencies[None]
        time = self.time(torch.cat((phase.sin(), phase.cos()), -1))
        x = self.action_in(noisy) + self.position + condition[:, None] + time[:, None]
        return self.output(self.denoiser(x))

    def loss(self, rgb, proprio, actions):
        t = torch.randint(len(self.beta), (len(actions),), device=actions.device)
        noise = torch.randn_like(actions)
        a = self.alpha_bar[t, None, None]
        noisy = a.sqrt() * actions + (1 - a).sqrt() * noise
        prediction = self.noise_prediction(noisy, t, self.encode(rgb, proprio))
        return (prediction - noise).square().mean()

    @torch.no_grad()
    def sample(self, rgb, proprio, inference_steps=None):
        condition = self.encode(rgb, proprio)
        x = torch.randn(len(rgb), self.config['horizon'], self.config['action_dim'], device=rgb.device)
        if inference_steps is not None and inference_steps < len(self.beta):
            indices = torch.linspace(len(self.beta)-1, 0, inference_steps, device=rgb.device).long()
            for j, i in enumerate(indices):
                t = i.expand(len(rgb))
                eps = self.noise_prediction(x, t, condition)
                a = self.alpha_bar[i]
                clean = ((x-(1-a).sqrt()*eps)/a.sqrt()).clamp(-1, 1)
                previous = self.alpha_bar[indices[j+1]] if j+1 < len(indices) else torch.ones((), device=rgb.device)
                x = previous.sqrt()*clean+(1-previous).sqrt()*eps
            return x.clamp(-1, 1)
        for i in reversed(range(len(self.beta))):
            t = torch.full((len(rgb),), i, device=rgb.device, dtype=torch.long)
            eps = self.noise_prediction(x, t, condition)
            clean = ((x-(1-self.alpha_bar[i]).sqrt()*eps)/self.alpha_bar[i].sqrt()).clamp(-1, 1)
            previous = self.alpha_bar[i-1] if i else torch.ones((), device=rgb.device)
            mean = (previous.sqrt()*self.beta[i]/(1-self.alpha_bar[i])*clean
                    +(1-self.beta[i]).sqrt()*(1-previous)/(1-self.alpha_bar[i])*x)
            if i:
                variance = self.beta[i] * (1 - self.alpha_bar[i - 1]) / (1 - self.alpha_bar[i])
                x = mean + variance.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)
