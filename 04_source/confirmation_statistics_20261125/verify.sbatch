#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=dp-confirm60-stats-audit
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/verify-%j.out
#SBATCH --error=/mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/verify-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/verify.py
