#!/bin/bash
# Run ON THE LOGIN NODE, once, before `sbatch job_agnews.sh` (downloads need internet; keep it under 15 min).
# Usage: bash prep_agnews.sh
set -e

mkdir -p /scratch-shared/$USER/nlp1
cd /scratch-shared/$USER/nlp1

# Embeddings (fasttext, because stanford.edu was unreachable). Resumes if interrupted.
curl -L -C - -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

# NLTK data used by BasicTokenizer + AG News cached in ~/.cache/huggingface
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/envs/nlp1-2026/bin/activate
python - <<'PY'
import nltk
for r in ['stopwords', 'punkt_tab', 'opinion_lexicon']:
    nltk.download(r)
from datasets import load_dataset
load_dataset("fancyzhx/ag_news")
PY

ls -lh /scratch-shared/$USER/nlp1
