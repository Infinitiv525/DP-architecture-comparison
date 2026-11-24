#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:10:00
set -euo pipefail
SOURCE=/mnt/project/perun2601449/dp_llm_cfd/test/architecture_confirmation_20261124_v2
cd "$SOURCE"
export PYTHONDONTWRITEBYTECODE=1
exec /mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python -B "$SOURCE/submit_waves.py" --wave 1
