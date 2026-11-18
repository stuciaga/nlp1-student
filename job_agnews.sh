#!/bin/bash
#SBATCH --job-name=nlp1-agnews-ep
#SBATCH --output=logs/agnews_ep_%A.out
#SBATCH --error=logs/agnews_ep_%A.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --gpus=1

set -e  # a crash in any step makes the job FAILED instead of COMPLETED

# Same software setup as job_mlp.sh (part 1)
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/envs/nlp1-2026/bin/activate

mkdir -p logs
cd $HOME/nlp1-student

# Sanity check: fail early (and cheaply) if PyTorch can't see the GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

echo "Starting AG News embed-and-pool CV at $(date)"
python -u part2/run_agnews_embpool.py --emb fasttext --emb-dir /scratch-shared/$USER/nlp1
echo "Done at $(date)"
