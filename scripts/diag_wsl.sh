#!/bin/bash
# Diagnose which interpreter actually runs and whether CPU rendering works in WSL.
set -uo pipefail
cd /mnt/f/work/robot
PY=/home/liudeli/.venvs/plugcharger/bin/python

echo "=== interpreter identity ==="
echo "which python -> $(command -v python)"
echo "venv python  -> $PY"
"$PY" -c "import sys; print('sys.executable:', sys.executable); print('sys.prefix:', sys.prefix); print('base_prefix:', sys.base_prefix)"

echo
echo "=== venv pyvenv.cfg ==="
cat /home/liudeli/.venvs/plugcharger/pyvenv.cfg 2>/dev/null || echo "no pyvenv.cfg"

echo
echo "=== env vars affecting python ==="
env | grep -i -E 'python|virtual|conda|pyenv' || echo "(none)"

echo
echo "=== try CPU render backend (physics + rgb, no GPU) ==="
export OMP_NUM_THREADS=1
"$PY" - <<'PYEOF' 2>&1 | tail -15
import gymnasium as gym, mani_skill.envs, numpy as np
env = gym.make('PlugCharger-v1', obs_mode='rgb', control_mode='pd_joint_pos',
               sim_backend='cpu', render_backend='cpu', reward_mode='sparse', max_episode_steps=100)
obs, _ = env.reset(seed=0)
rgb = obs['sensor_data']['hand_camera']['rgb']
print('CPU render OK, rgb shape', tuple(rgb.shape), 'dtype', rgb.dtype)
env.close()
PYEOF
