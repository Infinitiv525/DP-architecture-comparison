#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
exec /mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python -B cpu_audit.py
