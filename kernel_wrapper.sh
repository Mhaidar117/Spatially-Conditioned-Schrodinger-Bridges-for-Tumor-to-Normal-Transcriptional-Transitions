#!/bin/bash
# Fixes OpenMP crash when launching Jupyter kernel
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/Users/michaelhaidar/.matplotlib
exec /opt/homebrew/Caskroom/miniforge/base/envs/rl_env/bin/python -m ipykernel_launcher "$@"
