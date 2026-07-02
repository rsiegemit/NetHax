#!/bin/bash
#SBATCH --job-name=nethax-dirs
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
echo "### DIRECTIONS PROBE: headroom of each NLE-push lever (baseline 3436/s @ B=1024, FAST_POST+vec+BFS) ###"
echo "--- A) VEC_CHUNK ceiling: does chunking the monster vmap fit B=2048? ---"
for CK in 64 128; do echo "== VEC_CHUNK=$CK B=2048 =="; NETHAX_VEC_CHUNK=$CK NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=64 BATCHES=2048 NSTEP=20 PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED"; done
echo "--- B) FOV-free upper bound @ B=1024 ---"
NETHAX_SKIP_FOV=1 NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=64 BATCHES=1024 NSTEP=30 PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED"
echo "--- C) FOV-free + pathfind-free upper bound (full headroom) @ B=1024 ---"
NETHAX_SKIP_FOV=1 NETHAX_SKIP_PATHFIND=1 NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=64 BATCHES=1024 NSTEP=30 PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED"
echo ALL_DONE
