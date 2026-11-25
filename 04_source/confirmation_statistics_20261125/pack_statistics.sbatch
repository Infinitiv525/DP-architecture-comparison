#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=dp-confirm60-statistics-archive
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/pack-stats-%j.out
#SBATCH --error=/mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/pack-stats-%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python
"$PY" -B /mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/verify_portable.py /mnt/project/perun2601449/dp_llm_cfd/results/architecture_confirmation_statistics_20261125_v1
"$PY" -B /mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_statistics_20261125_v1/pack_statistics.py
