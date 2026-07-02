#!/bin/bash
#SBATCH --job-name=nethax-bnB
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
echo "### BYTE-NEUTRAL B-CEILING: donation (input->output alias) +/- monster-vmap chunking. baseline OOM@2048 ###"
echo "--- 1) DONATE only, B=2048,4096 ---"
BENCH_DONATE=1 NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=64 BATCHES=2048,4096 NSTEP=20 PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED"
echo "--- 2) DONATE + VEC_CHUNK=64, B=2048,4096 ---"
BENCH_DONATE=1 NETHAX_VEC_CHUNK=64 NETHAX_FAST_POST=1 NETHAX_VEC_MONSTERS=1 NETHAX_SPARSE_K=64 BATCHES=2048,4096 NSTEP=20 PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py 2>&1 | grep -E "B=|FAILED"
echo ALL_DONE
