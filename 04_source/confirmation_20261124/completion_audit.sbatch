#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec /mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python -B completion_audit.py
