#!/bin/bash
#SBATCH --job-name=nethax-nlebase
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/logs/%x-%j.out
cd /n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/repo || exit 1
export PATH=/n/home02/rsiegelmann/miniconda3/envs/praxis/bin:$PATH
python /tmp/_nle_baseline.py
