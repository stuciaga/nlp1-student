#!/bin/bash
#SBATCH --job-name=p3-A-pilot
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=part3/logs/%x-%j.out
# Add your own account if needed:  #SBATCH --account=<your-account>
#
# A's first GPU job: fold-0 pilot (0-shot + 1-shot per class), then Yelp (OOD) and the
# contrast stress test, all 0-shot with qwen-0.5b-instruct. Submit from the repo root:
#   mkdir -p part3/logs && sbatch part3/job_A_pilot.sh
#
# Prerequisites (compute nodes may have no internet):
#   1. On the login node, cache the model:  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
#   2. part3/data/ (gitignored) must exist on Snellius: either rsync it from your laptop or run
#        python part3/utils/build_folds.py && build_ood.py && build_stress.py on the login node.
set -e
PROJECT_HOME=${PROJECT_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "${PROJECT_HOME}"
source "${PROJECT_HOME}/.venv/bin/activate"
export HF_HUB_OFFLINE=1   # fail fast if the model was not cached

MODEL=qwen-0.5b-instruct
OUT=part3/predictions/pilot
GEN="python3 part3/utils/generate.py --model_name ${MODEL} --output_path ${OUT}"

date '+[%H:%M:%S] pilot 0-shot'
${GEN} --fold 0 --subsample 500 --instruction_type 0-shot
date '+[%H:%M:%S] pilot few-shot (1 per class)'
${GEN} --fold 0 --subsample 500 --instruction_type few-shot --num_examples_per_class 1

# Guard: if more than 20% of first-pass outputs are unparseable, the prompt/parser needs a look first
python3 - <<'PY'
import json, sys
s = json.load(open("part3/predictions/pilot/qwen-0.5b-instruct_0-shot_0.stats.json"))
print("0-shot pilot:", {k: s[k] for k in ("accuracy", "invalid_first_pass", "invalid_final", "truncated", "gpu_minutes")})
sys.exit(1 if s["invalid_first_pass"] > 0.2 else 0)
PY

for f in part3/data/robustness/yelp.pkl part3/data/robustness/stress_*_k*.pkl; do
  date "+[%H:%M:%S] ${f}"
  ${GEN} --input_path "${f}" --instruction_type 0-shot
done
date '+[%H:%M:%S] done'
