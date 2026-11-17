"""Run `charger.train` as a child process and relaunch it after an unexpected exit.

The parent owns one stage. It keeps relaunching the child with `--resume <out>/<stage>/last.pt`
until the child reports completion, so a GPU TDR, driver reset or VRAM blip costs at most
`--eval-every` steps instead of the whole run.

Completion is decided from the child's own logs, not from the exit code alone:
  * `--steps` reached (metric rows only ever print after a checkpoint write), or
  * nothing left to do (`last.pt` already at `--steps`), or
  * the child exits with a non-zero status and no new training progress was made
    (a real error, not an interruption) - the parent stops instead of looping forever.
Every parent/child decision is written to `<log-dir>/<stage>.supervisor.log`.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def last_step(path):
    if not path.exists():
        return 0
    best = 0
    for line in path.read_text().splitlines():
        try:
            best = max(best, int(json.loads(line)['step']))
        except Exception:
            continue
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=list('ABC'), required=True)
    p.add_argument('--data', default='data/formal_rgb')
    p.add_argument('--camera', default='hand_camera')
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--val-batches', type=int, default=64)
    p.add_argument('--frame-cache', default='data/frame_cache')
    p.add_argument('--out', default='runs/formal')
    p.add_argument('--log-dir', default='runs/logs')
    p.add_argument('--retries', type=int, default=20)
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                   help='Remaining arguments are forwarded to charger.train verbatim.')
    args = p.parse_args()
    out = Path(args.out) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics = out / 'metrics.jsonl'
    supervisor = (log_dir / f'{args.stage}.supervisor.log').open('a', encoding='utf-8')

    def note(message):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
        print(line, flush=True)
        supervisor.write(line + '\n')
        supervisor.flush()

    if metrics.exists() and last_step(metrics) >= args.steps:
        note(f'already complete at step {last_step(metrics)}/{args.steps}; nothing to do')
        return 0

    base = [sys.executable, '-u', '-m', 'charger.train',
            '--data', args.data, '--stage', args.stage, '--camera', args.camera,
            '--steps', str(args.steps), '--batch-size', str(args.batch_size),
            '--eval-every', str(args.eval_every), '--val-batches', str(args.val_batches),
            '--amp', '--proprio', '--relative-actions', '--out', args.out]
    if args.frame_cache:
        base += ['--frame-cache', args.frame_cache]
    base += args.extra

    attempts = 0
    while True:
        before = last_step(metrics)
        last = out / 'last.pt'
        command = list(base)
        if last.exists():
            command += ['--resume', str(last)]
        attempts += 1
        note(f'attempt {attempts}: starting at step {before} '
             f'({"resume" if last.exists() else "from scratch"})')
        started = time.monotonic()
        child_log = log_dir / f'{args.stage}.train.log'
        with child_log.open('a', encoding='utf-8') as sink:
            sink.write(f'\n===== attempt {attempts} start {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n')
            sink.flush()
            code = subprocess.call(command, cwd=str(HERE), stdout=sink, stderr=subprocess.STDOUT)
        after = last_step(metrics)
        note(f'attempt {attempts}: child exit={code} after {time.monotonic()-started:.0f}s, '
             f'step {before} -> {after}')
        if after >= args.steps:
            note(f'complete: step {after}/{args.steps}')
            return 0
        if code == 0:
            note('child reported a clean exit without reaching --steps; stopping')
            return 1
        if after <= before and attempts > 1:
            note('child failed and made no progress; treating as a real error and stopping')
            return 1
        if attempts > args.retries:
            note(f'giving up after {args.retries} retries')
            return 1
        time.sleep(10)


if __name__ == '__main__':
    sys.exit(main())
