#!/bin/bash
# Probe the WSL environment for the collection pipeline.
echo "=== whoami / home ==="
whoami
HOME_DIR=$(eval echo ~)
echo "$HOME_DIR"

echo
echo "=== venvs under home ==="
ls -d "$HOME_DIR"/.venvs/* 2>/dev/null || echo "no ~/.venvs"

echo
echo "=== python binaries ==="
for p in "$HOME_DIR"/.venvs/plugcharger/bin/python /usr/bin/python3 python3; do
  if [ -x "$p" ] || command -v "$p" >/dev/null 2>&1; then
    echo "FOUND: $p -> $($p --version 2>&1)"
  else
    echo "MISSING: $p"
  fi
done

echo
echo "=== mplib ==="
PY="$HOME_DIR/.venvs/plugcharger/bin/python"
if [ -x "$PY" ]; then
  "$PY" -c "import mplib; print('mplib OK', getattr(mplib,'__version__','?'))" 2>&1 | tail -3
  "$PY" -c "import mani_skill, sapien, torch; print('mani_skill', mani_skill.__version__); print('sapien', sapien.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -5
  "$PY" -c "import h5py, gymnasium; print('h5py', h5py.__version__); print('gymnasium', gymnasium.__version__)" 2>&1 | tail -3
else
  echo "plugcharger venv not found"
fi

echo
echo "=== project path visible from WSL ==="
ls -d /mnt/f/work/robot 2>/dev/null && ls /mnt/f/work/robot | head -20 || echo "/mnt/f/work/robot NOT VISIBLE"

echo
echo "=== nvidia ==="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi in WSL"
