#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=research-train-audit-v2
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/research_review_20261119_v1/real-train-v2-%j.out
export OMP_NUM_THREADS=4
/mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python /mnt/project/perun2601449/dp_llm_cfd/test/research_review_20261119_v1/real_train_diagnostics_v2.py
