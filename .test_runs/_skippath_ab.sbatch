#!/bin/bash
#SBATCH --job-name=nethax-skippath
#SBATCH --partition=gpu_requeue
#SBATCH --account=sompolinsky_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/logs/%x-%j.out
cd /n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/repo || exit 1
export PATH=/n/home02/rsiegelmann/miniconda3/envs/praxis/bin:$PATH
export JAX_COMPILATION_CACHE_DIR=/n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/cache
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "### PATHFIND SHARE of monster-turn: FAST_POST=1 fixed. SKIP_PATHFIND=0 (BFS) vs 1 (greedy). B=512 full fidelity ###"
for SP in 0 1; do
  echo "===== NETHAX_SKIP_PATHFIND=$SP ====="
  NETHAX_SKIP_PATHFIND=$SP NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=64 BATCHES=512 NSTEP=30 \
    PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED|state:"
done
echo ALL_DONE
