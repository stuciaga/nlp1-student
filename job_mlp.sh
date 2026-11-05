#!/bin/bash
#SBATCH --job-name=nlp1-mlp
#SBATCH --output=logs/mlp_%j.out
#SBATCH --error=logs/mlp_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --gpus=1

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/envs/nlp1-2026/bin/activate

mkdir -p logs

cd $HOME/nlp1-student

echo "Starting MLP experiments at $(date)"
python run_mlp.py
echo "Done at $(date)"
