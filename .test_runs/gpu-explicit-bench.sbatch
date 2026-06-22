#!/bin/bash
# GPU JIT env-scaling bench via the EXPLICIT lower().compile()+invoke path.
#
# Tests the handoff hypothesis (HANDOFF_2026_06_22.md action #1): the cancelled
# Harvard ">56 min compile" (job 23964219) was the implicit pjit-path pathology,
# NOT the fundamental compile. The explicit path compiled in ~40s on Mac CPU;
# expect ~40s here too, and GPU exec should collapse the 159s/step CPU wall
# toward sub-second. Produces the real per-step / env-scaling table.
#
# gpu_test partition: ~1h time cap, --mem < 64G (per handoff). kempner_h100 is
# saturated; gpu_test already gave reset compile 8.3s / warm 3.5s.
#
#SBATCH --job-name=nethax-gpu-bench
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=gpu-explicit-bench-%j.out
#SBATCH --error=gpu-explicit-bench-%j.err

set -euo pipefail
cd /n/netscratch/sompolinsky_lab/Lab/rsiegelmann/nethax_parity

# Per-target compile cache (CUDA HLO != CPU HLO; keep separate from Mac cache).
export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache_gpu
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "[sbatch] node:    $(hostname)"
echo "[sbatch] commit:  $(git rev-parse --short HEAD)"
echo "[sbatch] cache:   $JAX_COMPILATION_CACHE_DIR ($(ls "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null | wc -l) entries)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source .venv/bin/activate

# Full opt (NO XLA_FLAGS, NO disable-opt) — the explicit path is what makes
# full-opt compile terminate. Per-B compile watchdog bounds a stalled shape.
PYTHONPATH=. python -u .test_runs/_bench_jit_explicit.py \
    --env MiniHack-Room-15x15-v0 \
    --batches 1,4,16,64,256 \
    --warm 20 \
    --compile-watchdog 2400 \
    --json .test_runs/gpu_explicit_$SLURM_JOB_ID.json

echo "[sbatch] DONE rc=$?"
