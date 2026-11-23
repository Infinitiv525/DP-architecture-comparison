#!/bin/bash
#SBATCH --job-name=dp-arch-stats
#SBATCH --account=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
ROOT=/mnt/project/perun2601449/dp_llm_cfd
exec "$ROOT/envs/training-torch-2.12-cu130/bin/python" -B analyze.py --config config.json --input-root "$ROOT" --published-summary frozen_summary55.json --output "$ROOT/results/architecture_statistics_20261123_v1"
