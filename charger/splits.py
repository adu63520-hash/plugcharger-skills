import numpy as np


def split_seeds(episodes):
    seeds = np.array(sorted({e['seed'] for e in episodes}))
    np.random.default_rng(42).shuffle(seeds)
    n = len(seeds)
    n_val = max(1, n//10) if n >= 2 else 0
    n_test = max(1, n//10) if n >= 5 else 0
    return dict(val=set(seeds[:n_val].tolist()),
                test=set(seeds[n_val:n_val+n_test].tolist()),
                train=set(seeds[n_val+n_test:].tolist()))
