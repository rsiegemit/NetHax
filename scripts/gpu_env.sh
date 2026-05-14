#!/usr/bin/env bash
# Source this before launching any GPU JAX run to get:
#   - persistent on-disk JIT cache (huge win for second+ runs)
#   - async compile across multiple jit'd functions (free-ish multi-core win)
#   - one-GPU pin (don't hog the box)
#
# Usage:
#     source scripts/gpu_env.sh
#     python scripts/ppo_purejax.py ...
#
# Or one-shot:
#     bash -c "source scripts/gpu_env.sh && python scripts/ppo_purejax.py"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Persistent JIT cache.  First compile pays; everything after reads PTX from disk.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax_compile}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# XLA tuning flags.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_async_compile=true --xla_gpu_force_compilation_parallelism=$(nproc)"

# Make jax see both the CPU + CUDA backends (we sometimes query jax.devices('cpu')).
unset JAX_PLATFORMS

echo "gpu_env:" \
     "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
     "cache=$JAX_COMPILATION_CACHE_DIR" \
     "parallel=$(nproc)"
