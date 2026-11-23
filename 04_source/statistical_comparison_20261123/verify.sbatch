#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/project/perun2601449/dp_llm_cfd/envs/training-torch-2.12-cu130/bin/python -B verify_statistics.py /mnt/project/perun2601449/dp_llm_cfd /mnt/project/perun2601449/dp_llm_cfd/results/architecture_statistics_20261123_v1 /mnt/project/perun2601449/dp_llm_cfd/test/architecture_statistics_20261123_v1
