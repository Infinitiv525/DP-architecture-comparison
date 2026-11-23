#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python -B verify_power.py /mnt/project/perun2601449/dp_llm_cfd/results/architecture_power_20261124_v2 /mnt/project/perun2601449/dp_llm_cfd/test/architecture_power_20261124_v1
