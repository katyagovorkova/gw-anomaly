#!/bin/bash
#SBATCH --job-name=gw-anomaly-eval
#SBATCH --output=data/job_output_%j.txt
#SBATCH --error=data/job_error_%j.txt
#SBATCH --time=17:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4

# Load modules or source the profile if needed
# module load cuda/12.2
# module load python/3.9.0

# Define the common directory path as a variable
HOME_DIR="/n/home00/emoreno"

# Activate the conda environment
source "${HOME_DIR}/miniconda3/bin/activate" "${HOME_DIR}/miniconda3/envs/gwak"

# Change directory
cd "${HOME_DIR}/gw-anomaly/"
mkdir -p output/logs

eval $(python export_variables.py)

GPU_ID=0
# TIMESLIDE_TOTAL_DURATION=117782754
TIMESLIDE_TOTAL_DURATION=86401
FILES_TO_EVAL=-1
TIMESLIDES_START=1243382418
TIMESLIDES_STOP=1248652818

# Assuming VERSION, TIMESLIDES_START, and TIMESLIDES_STOP are set earlier in the script.
SAVE_EVALS_PATH="output/${VERSION}/${TIMESLIDES_START}_${TIMESLIDES_STOP}_timeslides_GPU${GPU_ID}_duration${TIMESLIDE_TOTAL_DURATION}_files${FILES_TO_EVAL}/"

# The model path should be an array if it is meant to be passed as separate arguments to the Python script
MODEL_PATH=(
  "output/gwak-paper-final-models/trained/models/bbh.pt"
  "output/gwak-paper-final-models/trained/models/sglf.pt"
  "output/gwak-paper-final-models/trained/models/sghf.pt"
  "output/gwak-paper-final-models/trained/models/background.pt"
  "output/gwak-paper-final-models/trained/models/glitches.pt"
)

DATA_PATH="${FILE}"

# Export the variable so it's available to the Python script
export SAVE_EVALS_PATH, GPU_ID, TIMESLIDE_TOTAL_DURATION, FILES_TO_EVAL, MODEL_PATH, DATA_PATH

# Run the Python script with the current file as an argument
mkdir -p "${SAVE_EVALS_PATH}"

# When calling the script, use the ${!var} to dereference the variables and pass them to the script.
# Also, you need to "unwrap" the MODEL_PATH array and pass each item to the script.
{
set -x
python scripts/evaluate_timeslides.py "${MODEL_PATH[@]}" --data-path "${DATA_PATH}" --save-evals-path "${SAVE_EVALS_PATH}"  --gpu "${GPU_ID}" --timeslide-total-duration "${TIMESLIDE_TOTAL_DURATION}" --files-to-eval "${FILES_TO_EVAL}"
set +x
} &> "output/logs/${SLURM_JOB_ID}_evaluate_timeslides.log"