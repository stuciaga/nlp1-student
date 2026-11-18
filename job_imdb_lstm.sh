#!/bin/bash
#SBATCH --job-name=nlp1-imdb-lstm
#SBATCH --output=logs/imdb_lstm_%A.out
#SBATCH --error=logs/imdb_lstm_%A.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --gpus=1

# Item 3: embed-and-pool on IMDB with the pruned GloVe from coverage_imdb.py (run job_coverage.sh first).
# Arguments are passed through to run_imdb_embpool.py, for example:
#   sbatch job_imdb_embpool.sh --sample 500 --epochs 2 --folds 2          # quick test
#   sbatch job_imdb_embpool.sh --axis hidden --values 25 50 100 200       # one-axis screen (1 seed, 5 folds)
#   sbatch job_imdb_embpool.sh --seeds 0 3 5 --keep-models                # confirm baseline with 3 seeds

set -e  # a crash in any step makes the job FAILED instead of COMPLETED

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/envs/nlp1-2026/bin/activate

mkdir -p logs
cd $HOME/nlp1-student

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "Starting IMDB embed-and-pool at $(date)"
python -u part2/run_imdb_lstm.py "$@"
echo "Done at $(date)"
