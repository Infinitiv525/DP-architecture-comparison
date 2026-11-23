#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/project/perun2601449/dp_llm_cfd/envs/training-torch-2.12-cu130/bin/python -B pack_archive.py
