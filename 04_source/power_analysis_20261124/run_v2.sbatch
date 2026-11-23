#!/bin/bash
#SBATCH --job-name=dp-arch-power
#SBATCH --account=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ROOT=/mnt/project/perun2601449/dp_llm_cfd
exec "$ROOT/envs/architecture-pilot-20261110-v1/bin/python" -B power_analysis_v2.py --config config_v2.json --output "$ROOT/results/architecture_power_20261124_v2"
