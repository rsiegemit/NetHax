#!/bin/bash
# Isolated GPU measurement of monsters_step_all (no FOV/timer/obs): the real
# per-op rate behind the ~1ms/serial-op model, and the NETHAX_VEC_MONSTERS payoff.
#   Run A: vec ON  -> should compile+exec fast and COMPLETE (collapsed chain).
#   Run B: vec OFF -> 27.3M serial chain, bounded by a 1200s exec watchdog so we
#          can read its rate (time/serial_ops) without an 8h hang.
#
#SBATCH --job-name=nethax-gpu-mstep
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=gpu-monster-step-%j.out
#SBATCH --error=gpu-monster-step-%j.err

set -uo pipefail
cd /n/netscratch/sompolinsky_lab/Lab/rsiegelmann/nethax_parity
export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache_gpu_ms
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export NETHAX_BENCH_NO_PARITY=1
echo "[sbatch] node $(hostname)  commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source .venv/bin/activate

echo "================ RUN A: NETHAX_VEC_MONSTERS=1 (vectorized) ================"
NETHAX_VEC_MONSTERS=1 PYTHONPATH=. python -u .test_runs/_bench_monster_step.py \
    --batches 1,64,1024 --warm 20 \
    --json .test_runs/gpu_mstep_vec1_$SLURM_JOB_ID.json

# Run B: serial path but with a SMALL monster cap so it COMPLETES on GPU and
# yields the per-serial-op rate directly (serial ops ~= MAX_MONSTERS * 68k body).
# 8 monsters -> ~544k serial ops -> completes in minutes; rate = exec1 / 544k.
echo "================ RUN B: VEC=0 serial, NETHAX_MAX_MONSTERS=8 (rate probe) ==="
NETHAX_VEC_MONSTERS=0 NETHAX_MAX_MONSTERS=8 PYTHONPATH=. python -u .test_runs/_bench_monster_step.py \
    --batches 1 --warm 3 --exec-watchdog 2400 \
    --json .test_runs/gpu_mstep_vec0_$SLURM_JOB_ID.json

echo "[sbatch] DONE"
