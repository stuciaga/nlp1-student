#!/bin/bash
#SBATCH --job-name=nlp1-predict
#SBATCH --output=logs/predict_%A.out
#SBATCH --error=logs/predict_%A.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --gpus=1


module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/envs/nlp1-2026/bin/activate
cd $HOME/nlp1-student

python -u part2/predict_imdb.py "$@"