#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

date '+[%H:%M:%S-%d/%m/%y]'

#Change the directories to correspond to the ones you have saved the project and data
PROJECT_HOME=${PROJECT_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}  # submit from repo root
INSTRUCTION_TYPE="0-shot"
MODEL_NAME="qwen-0.5b-instruct"
INPUT_ID_PATH="part3/data/cross_validation_sets.pkl"
OUTPUT_ID_PATH="part3/predictions"

cd "${PROJECT_HOME}"
echo "Activating environment"
#Add the path to the virtual environment you have created
source ${PROJECT_HOME}/.venv/bin/activate

echo "Executing python script"
#Generating the in-distribution predictions
python3 part3/utils/generate.py \
  --instruction_type "${INSTRUCTION_TYPE}" \
  --model_name "${MODEL_NAME}" \
  --input_path "${INPUT_ID_PATH}" \
  --output_path "${OUTPUT_ID_PATH}"

date '+[%H:%M:%S-%d/%m/%y]'
