#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --job-name=CasCor-TRAIN-decomposition
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/research_review_20261119_v1/cascor-%j.out
export OMP_NUM_THREADS=4
/mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python /mnt/project/perun2601449/dp_llm_cfd/test/research_review_20261119_v1/cascor_TRAIN_diagnostics_v1.py
