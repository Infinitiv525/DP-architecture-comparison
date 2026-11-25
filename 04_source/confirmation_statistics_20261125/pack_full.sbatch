#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=dp-confirm60-full-archive
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/pack-%j.out
#SBATCH --error=/mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/pack-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/pack_full_models.py
