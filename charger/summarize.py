"""Compact, committable summary of measured results.

Full evaluation traces are multi-MB and are not committed; this reduces them to the numbers a
reader needs (success rate, Wilson interval, per-stage rates, training curves) plus the
per-episode outcomes, and writes `runs/summary/results.json`.

Only results that were actually produced are summarised. A missing file is reported as missing,
never replaced by a zero or an estimate.
"""
import argparse
import json
import math
from pathlib import Path

EVALUATIONS = [
    ('A_100.json', 'A single-view, new seeds 10000-10099, budget 200'),
    ('A_100_rep.json', 'A single-view, identical config re-run (reproducibility check)'),
    ('A_dual_100.json', 'A dual-view (hand+base camera), new seeds 10000-10099'),
    ('AB_192.json', 'AB chain, new seeds, no reset between skills'),
    ('AB_100.json', 'AB chain with budget 150 (A stage cut short; not comparable to AB_192)'),
    ('B_50.json', 'B started from held-out expert A-prefixes, 50 episodes'),
]
TRAINING = [
    ('formal/A', 'A'),
    ('formal/B', 'B'),
    ('dual/A', 'A (dual-view)'),
    ('formal_labels_invalid/C', 'C (trained on invalid labels, isolated on purpose)'),
]


def wilson(successes, n):
    if not n:
        return [None, None]
    z = 1.96
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - delta), 4), round(min(1.0, centre + delta), 4)]


def summarise_evaluation(path, description):
    result = json.loads(path.read_text())
    config, episodes = result['config'], result['episodes']
    row = dict(file=path.name, description=description, mode=config['mode'],
               checkpoints=config['checkpoints'], episodes=config['episodes'],
               start_seed=config['start_seed'], budget=config['budget'],
               action_steps=config['action_steps'], inference_steps=config['inference_steps'],
               policy_seed=config['policy_seed'], success=result['successes'], n=result['n'],
               success_rate=round(result['success_rate'], 4),
               wilson_95=[round(x, 4) for x in result['wilson_95']])
    stages = config['mode'].replace('Bprime', 'B')
    for stage in stages:
        entered = [e for e in episodes if stage in e['stages']]
        won = sum(1 for e in entered if e['stages'][stage]['success'])
        row[f'stage_{stage}'] = dict(success=won, n=len(entered),
                                     rate=round(won / len(entered), 4) if entered else None,
                                     wilson_95=wilson(won, len(entered)))
    row['per_episode'] = [dict(seed=e['seed'], success=e['success'],
                               stages={s: dict(success=v['success'], steps=v['steps'])
                                       for s, v in e['stages'].items()})
                          for e in episodes]
    return row


def summarise_training(root, name, stage):
    metrics = root / name / 'metrics.jsonl'
    if not metrics.exists():
        return None
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    best = min(rows, key=lambda r: r['validation_loss'])
    config_path = root / name / 'config.json'
    return dict(run=name, stage=stage,
                config=json.loads(config_path.read_text()) if config_path.exists() else None,
                steps_completed=rows[-1]['step'],
                first_val_loss=round(rows[0]['validation_loss'], 6),
                final_val_loss=round(rows[-1]['validation_loss'], 6),
                best_val_loss=round(best['validation_loss'], 6), best_step=best['step'],
                curve=[dict(step=r['step'], train_loss=round(r['train_loss'], 6),
                            validation_loss=round(r['validation_loss'], 6)) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='runs/summary/results.json')
    args = p.parse_args()
    root = Path(args.runs)
    summary = dict(evaluations=[], training=[], missing=[])
    for name, description in EVALUATIONS:
        path = root / 'eval' / name
        if path.exists():
            summary['evaluations'].append(summarise_evaluation(path, description))
        else:
            summary['missing'].append(f'eval/{name}')
    for name, stage in TRAINING:
        row = summarise_training(root, name, stage)
        if row is not None:
            summary['training'].append(row)
        else:
            summary['missing'].append(f'{name}/metrics.jsonl')
    validation = root / 'validation' / 'physics_all.json'
    if validation.exists():
        data = json.loads(validation.read_text())['summary']
        data.pop('diverged_seeds', None)
        summary['data_validation'] = data
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(out=str(out), evaluations=len(summary['evaluations']),
                          training=len(summary['training']), missing=summary['missing'])))


if __name__ == '__main__':
    main()
