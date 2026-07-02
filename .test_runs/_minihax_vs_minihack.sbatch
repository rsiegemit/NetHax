#!/bin/bash
#SBATCH --job-name=nethax-mhvs
#SBATCH --partition=gpu_requeue
#SBATCH --account=sompolinsky_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/logs/%x-%j.out
cd /n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/repo || exit 1
export PATH=/n/home02/rsiegelmann/miniconda3/envs/praxis/bin:$PATH
export JAX_COMPILATION_CACHE_DIR=/n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/cache
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "### MINIHAX vs MINIHACK: reduced config MATCHING MiniHack scope (single level, few monsters). This is the fair comparison — MiniHack is NOT a full dungeon. ###"
echo "--- Minihax @ MiniHack scope: SINGLE_LEVEL=1, MAX_MON=8, train flags + donate ---"
NETHAX_SINGLE_LEVEL=1 NETHAX_MAX_MONSTERS=8 BENCH_DONATE=1 NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=64 BATCHES=1024,4096,16384,65536 NSTEP=30 PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED|state:|device"
echo "--- and MAX_MON=4 (Room-Monster has 1-2 monsters typically) ---"
NETHAX_SINGLE_LEVEL=1 NETHAX_MAX_MONSTERS=4 BENCH_DONATE=1 NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=32 BATCHES=4096,16384,65536 NSTEP=30 PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED|state:"
echo ALL_DONE
