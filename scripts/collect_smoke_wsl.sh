#!/bin/bash
# Smoke test: collect a handful of attempts in WSL, verifying that failed trajectories persist.
set -euo pipefail
cd /mnt/f/work/robot
PY=/home/liudeli/.venvs/plugcharger/bin/python
OUT=${1:-data/wsl_smoke}
SEEDS=${2:-6}
rm -rf "$OUT"
export OMP_NUM_THREADS=1
export LP_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
"$PY" -c "import mplib, mani_skill, torch; print('mplib', getattr(mplib,'__version__','?')); print('mani_skill', mani_skill.__version__); print('cuda', torch.cuda.is_available())"
echo "--- collecting $SEEDS attempts ---"
"$PY" -u -m charger.collect --out "$OUT" --seeds "$SEEDS" --repeats 1 \
    --obs-mode rgb --render-backend cpu 2>&1 | tail -40
echo "--- result ---"
"$PY" - "$OUT" <<'PYEOF'
import json, sys, h5py
out = sys.argv[1]
m = json.load(open(f'{out}/stages.json'))
print('attempts', len(m['attempts']), 'episodes', len(m['episodes']))
print('failure_counts', m.get('failure_counts'))
with h5py.File(f'{out}/trajectory.h5', 'r') as f:
    keys = sorted(f.keys())
    print('saved trajectories:', keys)
    for k in keys[:3]:
        print(' ', k, 'actions', f[k]['actions'].shape,
              'cams', list(f[k]['obs/sensor_data'].keys()) if 'obs' in f[k] else 'no-obs')
for a in m['attempts']:
    print(' ', a.get('seed'), 'success' if a.get('success') else 'FAILED@' + str(a.get('failed_stage')),
          'traj=' + str(a.get('trajectory')))
PYEOF
