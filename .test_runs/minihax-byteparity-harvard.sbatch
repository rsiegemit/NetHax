#!/bin/bash
# Harvard kempner_h100 cross-platform verification of MiniHack-Room byte
# parity at seed=0.  Mac currently passes 12/12 at HEAD 5509b35; this run
# confirms whether the JAX wrappers produce identical glyph + inv state on
# Linux + GPU vs Mac CPU.
#
# Pre-flight (run on the cluster head node before sbatch):
#   cd ~/nethax_parity
#   git fetch origin
#   git checkout main && git pull origin main      # picks up byteparity commits
#   git rev-parse --short HEAD                     # expect 5509b35 or newer
#   ls .test_runs/minihax_byteparity.py            # verify the script is present
#
# Submit:
#   sbatch .test_runs/minihax-byteparity-harvard.sbatch
#
# Read alongside .test_runs/vendor_placement_model.md if any seed-0 env
# fails on Linux that passes on Mac — likely indicates a CPU/GPU determinism
# issue or NLE-build flag mismatch, NOT a wrapper bug.
#
#SBATCH --job-name=nethax-minihax-byteparity
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=minihax-byteparity-%j.out
#SBATCH --error=minihax-byteparity-%j.err

set -euo pipefail
cd ~/nethax_parity

echo "[sbatch] node:           $(hostname)"
echo "[sbatch] commit:         $(git rev-parse --short HEAD)"
echo "[sbatch] python:         $(.venv/bin/python -V)"
echo "[sbatch] start:          $(date -u +%FT%TZ)"

# Force CPU + eager so JAX dispatches one-op-at-a-time identically across
# platforms.  This matches how Mac runs the byteparity validator at HEAD.
export JAX_PLATFORMS=cpu
export NETHAX_EAGER=1

source .venv/bin/activate

# Run the full sweep at seed=0 (Mac baseline).  Each env reports PASS/FAIL.
# Exit 0 even on FAIL so the SLURM job succeeds and the log captures the
# divergence rows; the CI/manual reviewer reads the summary line.
.venv/bin/python .test_runs/minihax_byteparity.py --all-rooms --seed 0 || true

echo "[sbatch] end:            $(date -u +%FT%TZ)"
