"""Shard the one-pass validity check over several processes and merge the reports.

Physics-only validation of data/formal_rgb takes ~1-2 s per episode, so the whole dataset is
cheap to check. Each shard is a separate process against its own CPU SAPIEN env; the merged
summary is written once every shard has finished.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_shard(root, index, count, args, out_dir):
    report = out_dir / f'shard_{index}.json'
    command = [sys.executable, '-u', '-m', 'charger.validate_rgb',
               '--data', args.data, '--out', str(report),
               '--shard', str(index), '--shards', str(count),
               '--obs-mode', args.obs_mode, '--render-backend', args.render_backend]
    if args.limit:
        command += ['--limit', str(args.limit)]
    log = out_dir / f'shard_{index}.log'
    print(f'[shard {index}/{count}] {" ".join(command)}', flush=True)
    with log.open('w', encoding='utf-8') as sink:
        code = subprocess.call(command, cwd=str(root), stdout=sink, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError(f'shard {index} exited with {code}; see {log}')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/formal_rgb')
    p.add_argument('--out', required=True)
    p.add_argument('--shards', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--obs-mode', default='none', choices=['none', 'rgb'])
    p.add_argument('--render-backend', default='none', choices=['none', 'cpu'])
    args = p.parse_args()
    out = Path(args.out)
    out_dir = out.parent / (out.stem + '_shards')
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    with ThreadPoolExecutor(args.shards) as pool:
        reports = list(pool.map(lambda i: run_shard(root, i, args.shards, args, out_dir),
                                range(args.shards)))
    episodes, summaries = [], []
    for report in reports:
        payload = json.loads(report.read_text())
        episodes.extend(payload['episodes'])
        summaries.append(payload['summary'])
    episodes.sort(key=lambda r: r['seed'])
    n = len(episodes)
    exact = sum(r['boundaries_match'] for r in episodes)
    summary = dict(
        n=n,
        exact_boundary_match=exact,
        exact_rate=round(exact / max(n, 1), 4),
        final_success_true=sum(r['final_success'] for r in episodes),
        missing_boundary=sum(1 for r in episodes if r['derived'] is None),
        diverged_by_stage={s: sum(1 for r in episodes if r.get('diverged_stage') == s) for s in 'ABC'},
        contaminated_tail=sum(1 for r in episodes if r['steps_after_first_success'] > 0),
        replay_terminated_early=sum(r['replay_terminated_early'] for r in episodes),
        min_distance_over_5mm=sum(1 for r in episodes if r['min_distance'] <= 5e-3),
        diverged_seeds=[r['seed'] for r in episodes if not r['boundaries_match']],
        obs_mode=args.obs_mode, render_backend=args.render_backend,
        shards=args.shards,
        seconds=sum(s['seconds'] for s in summaries))
    out.write_text(json.dumps(dict(summary=summary, episodes=episodes), indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'diverged_seeds'}), flush=True)


if __name__ == '__main__':
    main()
