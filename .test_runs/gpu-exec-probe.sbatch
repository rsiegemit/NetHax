#!/bin/bash
# GPU exec-wall diagnostic. Prior job 24036479 proved: explicit compile finishes
# (89.6s on A100), but B=1 first-exec ran >56 min without finishing (killed by
# SLURM wall). So the original GPU "56 min" was EXEC, not compile, and the GPU
# does NOT collapse exec as the handoff hypothesized.
#
# This job isolates WHY exec1 is catastrophic by disabling XLA GPU autotuning.
#   - If exec1 now returns fast  -> first-exec autotune overhead was the cause.
#   - If exec1 still hits the 1500s exec watchdog -> execution is fundamentally
#     slow on this 291k-op serial graph; GPU is not the rescue.
# The exec watchdog ends the job in ~25 min on a hang instead of the SLURM wall.
#
#SBATCH --job-name=nethax-gpu-execprobe
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=gpu-exec-probe-%j.out
#SBATCH --error=gpu-exec-probe-%j.err

set -euo pipefail
cd /n/netscratch/sompolinsky_lab/Lab/rsiegelmann/nethax_parity

export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache_gpu
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Disable GPU autotuning to test whether first-exec autotune is the wall.
export XLA_FLAGS="--xla_gpu_autotune_level=0"

echo "[sbatch] node:    $(hostname)"
echo "[sbatch] commit:  $(git rev-parse --short HEAD)"
echo "[sbatch] XLA_FLAGS=$XLA_FLAGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source .venv/bin/activate

PYTHONPATH=. python -u .test_runs/_bench_jit_explicit.py \
    --env MiniHack-Room-15x15-v0 \
    --batches 1 \
    --warm 10 \
    --compile-watchdog 2400 \
    --exec-watchdog 1500 \
    --json .test_runs/gpu_execprobe_$SLURM_JOB_ID.json

echo "[sbatch] DONE rc=$?"
