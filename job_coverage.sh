#!/bin/bash
#SBATCH --job-name=nlp1-coverage
#SBATCH --output=logs/coverage_%A.out
#SBATCH --error=logs/coverage_%A.err
#SBATCH --time=00:45:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G

# CPU job (no GPU): tokenizes IMDB under several settings, scans GloVe 840B once, writes coverage tables
# and a pruned embedding file to part2/results/imdb/embeddings/.
#
# Before the first run, on the login node (needs internet), cache IMDB and the NLTK data:
#   source ~/envs/nlp1-2026/bin/activate
#   python -c "from datasets import load_dataset; load_dataset('mteb/imdb')"
#
# Quick test:  sbatch job_coverage.sh --sample 500
# Full run:    sbatch job_coverage.sh

set -e  # a crash in any step makes the job FAILED instead of COMPLETED

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/envs/nlp1-2026/bin/activate

mkdir -p logs
cd $HOME/nlp1-student

echo "Starting coverage analysis at $(date)"
python -u part2/coverage_imdb.py \
    --glove /scratch-shared/$USER/nlp1/glove.840B.300d.zip \
    --workers 16 \
    "$@"
echo "Done at $(date)"
