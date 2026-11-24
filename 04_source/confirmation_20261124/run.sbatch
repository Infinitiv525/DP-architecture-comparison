#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
ROOT=/mnt/project/perun2601449/dp_llm_cfd
cd "${SLURM_SUBMIT_DIR:?}"
export USER="${SLURM_JOB_USER:-lutoke823}" LOGNAME="${SLURM_JOB_USER:-lutoke823}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
export SLURM_EXPORT_ENV=ALL PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled WANDB_DISABLED=true DO_NOT_TRACK=1 MPLBACKEND=Agg
PY="$ROOT/envs/architecture-pilot-20261110-v1/bin/python"
MODE="$1"; RUN="$2"
set +e
srun --export=ALL --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 "$PY" -B -u launch.py --mode "$MODE" --index "${SLURM_ARRAY_TASK_ID:?}" --root "$RUN"
STEP_EXIT=$?
set -e
"$PY" -B primary/record_allocation.py --root "$RUN" --mode "$MODE" --exit-code "$STEP_EXIT"
exit "$STEP_EXIT"
