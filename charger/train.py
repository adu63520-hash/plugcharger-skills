import argparse
import json
import copy
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import SkillDataset
from .model import VisionDiffusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/charger')
    p.add_argument('--stage', choices=list('ABC'), required=True)
    p.add_argument('--camera', default='hand_camera')
    p.add_argument('--proprio', action='store_true')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--out', default='runs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--val-batches', type=int, default=64)
    p.add_argument('--preload', action='store_true')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--encoder-kind', choices=['gap', 'spatial'], default='spatial')
    p.add_argument('--relative-actions', action='store_true')
    p.add_argument('--frame-cache',
                   help='Directory from charger.cache_frames; makes shuffled RGB reads '
                        'memory-mapped instead of re-inflating HDF5 chunks.')
    args = p.parse_args()
    if args.relative_actions and not args.proprio:
        p.error('relative action anchoring requires explicit --proprio')
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = device == 'cuda'
    train = SkillDataset(args.data, args.stage, 'train', args.camera, proprio=args.proprio,
                         relative_actions=args.relative_actions, cache=args.frame_cache)
    val = SkillDataset(args.data, args.stage, 'val', args.camera, proprio=args.proprio,
                       relative_actions=args.relative_actions, cache=args.frame_cache)
    if args.preload:
        train.preload()
        val.preload()
    print(dict(stage=args.stage, train_frames=len(train), val_frames=len(val),
               device=device, proprio=train.proprio_dim), flush=True)
    model = VisionDiffusion(train.action_dim, train.proprio_dim, encoder_kind=args.encoder_kind,
                            num_views=len(args.camera.split(','))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device == 'cuda')
    loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=0, pin_memory=device == 'cuda')
    # Fixed random validation subset samples across episodes, not only first seeds.
    generator = torch.Generator().manual_seed(123)
    valid_indices = torch.randperm(len(val), generator=generator)[:args.val_batches*args.batch_size]
    validation = DataLoader(torch.utils.data.Subset(val, valid_indices), args.batch_size, shuffle=False, num_workers=0)
    out = Path(args.out) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    iterator = iter(loader)
    best = float('inf')
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['raw_model'])
        ema.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_step = checkpoint['step']
        best = checkpoint['best_validation_loss']
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        lr_factor = min(1., step/200) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = 1e-4 * lr_factor
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.amp and device == 'cuda'):
            loss = model.loss(*(x.to(device, non_blocking=True) for x in batch))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            decay = min(.995, (1+step)/(10+step))
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 1-decay)
            for target, source in zip(ema.buffers(), model.buffers()):
                target.copy_(source)
        if step % 100 == 0:
            print(dict(stage=args.stage, step=step, train_loss=loss.item(),
                       steps_per_second=(step-start_step)/(time.monotonic()-started)), flush=True)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            total, count = 0., 0
            # Fixed validation diffusion noise without perturbing training RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
                torch.manual_seed(1234)
                with torch.no_grad():
                    for batch in validation:
                        n = len(batch[0])
                        total += ema.loss(*(x.to(device) for x in batch)).item() * n
                        count += n
            value = total / count
            ckpt = dict(model=ema.state_dict(), model_config=model.config,
                        args=vars(args), low=train.low, high=train.high,
                        step=step, validation_loss=value)
            if value < best:
                best = value
                torch.save(ckpt, out / 'best.tmp')
                (out / 'best.tmp').replace(out / 'best.pt')
            ckpt.update(raw_model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), best_validation_loss=best)
            torch.save(ckpt, out / 'last.tmp')
            (out / 'last.tmp').replace(out / 'last.pt')
            with (out / 'metrics.jsonl').open('a') as log:
                row = dict(step=step, train_loss=loss.item(), validation_loss=value)
                log.write(json.dumps(row) + '\n')
            print(row, flush=True)


if __name__ == '__main__':
    main()
