#!/bin/bash
#SBATCH --job-name=train_index_200M_normalized
#SBATCH --account=PAS2136
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-7
#SBATCH --output=logs/train_index_200M_normalized%A_%a.out
#SBATCH --error=logs/train_index_200M_normalized%A_%a.err
#SBATCH --partition=nextgen


# --- CONFIGURATION ---
# Total number of parquet files in your manifest (Check Phase 2 output)
TOTAL_FILES=1020 
NUM_TASKS=8 

# --- Configuration ---
OUTPUT_DIR="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified"
MANIFEST="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/flight_plan/manifest.parquet"
LEADER="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/leader_20M_stratified.index"
BATCH_SIZE=3000000  # ~3 million vectors per GPU flush

mkdir -p $OUTPUT_DIR
mkdir -p logs

# --- Environment Setup ---
module load miniconda3/24.1.2-py310
conda activate faiss_env

# Ensure output directory exists (race condition safe-ish if pre-created)
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# --- CALCULATE SLICE ---
# Ceiling division to ensure all files are covered: ceil(TOTAL / TASKS)
CHUNK_SIZE=$(( ($TOTAL_FILES + $NUM_TASKS - 1) / $NUM_TASKS ))

START_IDX=$(( $SLURM_ARRAY_TASK_ID * $CHUNK_SIZE ))
END_IDX=$(( $START_IDX + $CHUNK_SIZE ))

echo "================================================="
echo "SLURM Job: $SLURM_JOB_ID Task: $SLURM_ARRAY_TASK_ID"
echo "Processing Manifest Rows: $START_IDX to $END_IDX"
echo "Batch Size: $BATCH_SIZE"
echo "================================================="

# --- RUN PYTHON WORKER ---
# Note: We rely on os.environ["CUDA_VISIBLE_DEVICES"] inside the script
# typically SLURM sets this automatically to the 1 allocated GPU.


python src/bioclip_vector_db/batch/train_index.py \
    --manifest "$MANIFEST" \
    --leader_index "$LEADER" \
    --output_dir "$OUTPUT_DIR" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --batch_size "$BATCH_SIZE"

echo "Task $SLURM_ARRAY_TASK_ID complete."
