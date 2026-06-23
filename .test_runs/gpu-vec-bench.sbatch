#!/bin/bash
# GPU bench of the VECTORIZED monster step (NETHAX_VEC_MONSTERS=1) in NON-vendor
# (training) mode. The serial-scan path ran a single B=1 step for >8h unfinished
# on the A100 (job 24053786). The vectorized path collapses the 27.3M serial
# op-executions (68k-eqn body x400) into wide vmap ops; CPU jaxpr + sanity already
# confirm the collapse (turn-loop scan body 68253->2744 eqns, warm 0.2s/step).
# This job confirms GPU exec drops from hours to sub-second and scales with B.
#
#SBATCH --job-name=nethax-gpu-vec
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=gpu-vec-bench-%j.out
#SBATCH --error=gpu-vec-bench-%j.err

set -euo pipefail
cd /n/netscratch/sompolinsky_lab/Lab/rsiegelmann/nethax_parity

export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache_gpu_vec
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export NETHAX_BENCH_NO_PARITY=1     # default Threefry training mode (vec path)
export NETHAX_VEC_MONSTERS=1        # vectorized simultaneous-move monster step
export NETHAX_SINGLE_LEVEL=1
export NETHAX_DUNGEON_LEVELS=1
export NETHAX_GROUND_STACK=4
export NETHAX_MAX_MONSTERS=128

echo "[sbatch] node:    $(hostname)"
echo "[sbatch] commit:  $(git rev-parse --short HEAD)"
echo "[sbatch] NO_PARITY=$NETHAX_BENCH_NO_PARITY VEC=$NETHAX_VEC_MONSTERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source .venv/bin/activate

PYTHONPATH=. python -u .test_runs/_bench_jit_explicit.py \
    --env MiniHack-Room-Monster-15x15-v0 \
    --batches 1,1024,2048,4096 \
    --warm 10 \
    --compile-watchdog 2400 \
    --json .test_runs/gpu_vec_$SLURM_JOB_ID.json

echo "[sbatch] DONE rc=$?"
