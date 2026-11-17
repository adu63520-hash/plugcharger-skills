"""Run `charger.evaluate` for several skill chains and summarise the success rates.

`charger.evaluate` evaluates one chain (`--mode`) per process. Formal comparisons need several
chains under the same policy seeds, so this wrapper runs them back to back and prints one table.

Each chain is written to its own JSON file. A chain that crashes does not stop the rest; its
error is recorded in the summary and it is counted as a failure of the batch, never silently
dropped.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def wilson(successes, n):
    if n == 0:
        return [None, None]
    z = 1.96
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - delta), 4), round(min(1.0, centre + delta), 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modes', nargs='+', required=True,
                   help='Every value must be a --mode accepted by charger.evaluate. '
                        'Python slices them as a list: --modes BC BprimeC ABC ABprimeC')
    p.add_argument('--checkpoints', default='runs')
    p.add_argument('--residual', help='Required by Bprime modes; passed through to evaluate.')
    p.add_argument('--data', default='data/formal_rgb')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--budget', type=int, default=150)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--render-backend', default='cpu', choices=['cpu', 'gpu'])
    p.add_argument('--seed', type=int, default=0, help='Policy seed; use several for paired runs.')
    p.add_argument('--out', required=True, help='Output directory for one JSON per chain.')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in args.modes:
        path = out / f'{mode}_seed{args.seed}.json'
        command = [sys.executable, '-u', '-m', 'charger.evaluate',
                   '--mode', mode, '--checkpoints', args.checkpoints, '--data', args.data,
                   '--episodes', str(args.episodes), '--budget', str(args.budget),
                   '--action-steps', str(args.action_steps), '--render-backend', args.render_backend,
                   '--policy-seed', str(args.seed), '--out', str(path)]
        if args.residual:
            command += ['--residual', args.residual]
        print(f'[{mode}] ' + ' '.join(command), flush=True)
        code = subprocess.call(command)
        row = dict(mode=mode, policy_seed=args.seed, exit_code=code)
        if code == 0 and path.exists():
            result = json.loads(path.read_text())
            row.update(successes=result['successes'], n=result['n'],
                       success_rate=round(result['success_rate'], 4),
                       wilson_95=[round(x, 4) for x in result['wilson_95']],
                       stages={s: round(sum(e['stages'][s]['success'] for e in result['episodes']
                                            if s in e['stages'])
                                        / max(1, sum(1 for e in result['episodes'] if s in e['stages'])), 4)
                               for s in (result['config']['mode'].replace('Bprime', 'B'))})
        else:
            # A crashed chain is a failed measurement, reported as such rather than omitted.
            row.update(successes=None, n=None, success_rate=None, wilson_95=None)
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(config=vars(args), rows=rows)
    (out / 'batch_summary.json').write_text(json.dumps(summary, indent=2))
    print('\nchain            success   rate   wilson95')
    for row in rows:
        if row['success_rate'] is None:
            print(f"{row['mode']:<16} FAILED (exit {row['exit_code']})")
        else:
            print(f"{row['mode']:<16} {row['successes']:>3}/{row['n']:<4} {row['success_rate']:.3f}  {row['wilson_95']}")


if __name__ == '__main__':
    main()
