"""Paired seed-level comparisons; retain negative and null experiment results."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def compare(left, right):
    def by_seed(data):
        rows = data['episodes']
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError('Comparison requires one rollout per independent seed')
        return {r['seed']: int(r['success']) for r in rows}
    a, b = by_seed(left), by_seed(right)
    if set(a) != set(b):
        raise ValueError('Paired experiments must use exactly the same reset seeds')
    keys = sorted(a)
    delta = np.array([b[k]-a[k] for k in keys], dtype=float)
    rng = np.random.default_rng(42)
    samples = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
    losses, gains = int((delta == -1).sum()), int((delta == 1).sum())
    n = losses+gains
    p = min(1., 2*sum(math.comb(n, k) for k in range(min(losses, gains)+1))/2**n) if n else 1.
    return dict(n=len(keys), baseline_sr=sum(a.values())/len(a),
                changed_sr=sum(b.values())/len(b), delta=float(delta.mean()),
                paired_bootstrap_95=np.quantile(samples, [.025, .975]).tolist(),
                discordant_gains=gains, discordant_losses=losses, mcnemar_exact_p=p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bc', required=True)
    p.add_argument('--bprimec', required=True)
    p.add_argument('--abc', required=True)
    p.add_argument('--abprimec', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    bc = compare(load(args.bc), load(args.bprimec))
    abc = compare(load(args.abc), load(args.abprimec))
    result = dict(BC_to_BprimeC=bc, ABC_to_ABprimeC=abc,
                  descriptive_hypothesis_observed=bc['delta'] > 0 and abc['delta'] < 0,
                  both_paired_intervals_exclude_zero=bc['paired_bootstrap_95'][0] > 0 and abc['paired_bootstrap_95'][1] < 0)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    main()
