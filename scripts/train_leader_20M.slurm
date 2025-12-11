#!/bin/bash
#SBATCH --job-name=train_leader_20M
#SBATCH --output=train_leader_%j.out
#SBATCH --error=train_leader_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu
#SBATCH --account=PAS2136

# --- Configuration ---
# Adjust these paths and settings as needed
#INPUT_DIR="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/20M"
INPUT_DIR="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/stratified_sample/"
OUTPUT_PATH="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/leader_20M_stratified.index"
INDEX_STRING="IVF65536,PQ16"

# --- Environment Setup ---
# Load modules if required by your cluster
# module load python/3.10
# module load cuda/12.0

# Activate your python environment
# source venv/bin/activate
# conda activate bioclip-env
module load miniconda3/24.1.2-py310
conda activate faiss_env
which python


echo "Starting training job on $(hostname)"
echo "Input Directory: $INPUT_DIR"
echo "Output Path: $OUTPUT_PATH"
echo "Index String: $INDEX_STRING"

# --- Run Training ---
python src/bioclip_vector_db/batch/train_leader.py \
    --input_dir "$INPUT_DIR" \
    --output_path "$OUTPUT_PATH" \
    --index_string "$INDEX_STRING" \
    --mode gpu \
    --gpu_id 0

echo "Job finished at $(date)"
