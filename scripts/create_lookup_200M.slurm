#!/bin/bash
#SBATCH --job-name=create_lookup
#SBATCH --output=create_lookup_%j.out
#SBATCH --error=create_lookup_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=nextgen
#SBATCH --account=PAS2136

# --- Configuration ---
# Adjust these paths as needed
INPUT_DIR="/fs/ess/PAS2136/TreeOfLife/embeddings/model=BioCLIP_2"
OUTPUT_DIR="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/flight_plan"

# --- Environment Setup ---
module load miniconda3/24.1.2-py310
conda activate faiss_env
which python

echo "Starting lookup generation job on $(hostname)"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"

# --- Run Script ---
# Note: passing --workers matching cpus-per-task
python src/bioclip_vector_db/create_lookup.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers 24

echo "Job finished at $(date)"
