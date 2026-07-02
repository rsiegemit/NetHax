#!/bin/bash
#SBATCH --job-name=nethax-mhbase
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/logs/%x-%j.out
export PATH=/n/home02/rsiegelmann/miniconda3/envs/praxis/bin:$PATH
cd /n/home02/rsiegelmann/nethax 2>/dev/null || cd /n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/repo
python /n/netscratch/sompolinsky_lab/Everyone/rsiegelmann/nethax_run/repo/_minihack_baseline.py
