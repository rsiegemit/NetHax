#!/bin/bash
#SBATCH --job-name=nethax-bfshoist
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
echo "### BFS-HOIST + FAST_POST bench: full fidelity, hoist always-on in vec mode. throughput @ B=512 (expect ~2695 vs 1879) + B ceiling (does hoist lift past 1024?) ###"
NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=64 BATCHES=512,1024,2048,4096 NSTEP=30 \
  PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED|state:|device"
echo ALL_DONE
