#!/bin/bash
#SBATCH --job-name=merge_index_200M
#SBATCH --account=PAS2136
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/merge_index_200M_%j.out
#SBATCH --error=logs/merge_index_200M_%j.err

# --- Configuration ---
SHARD_DIR="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified"
OUTPUT_PATH="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified_merged.index"

# --- Environment Setup ---
module load miniconda3/24.1.2-py310
conda activate faiss_env

echo "================================================="
echo "SLURM Job: $SLURM_JOB_ID"
echo "Merging shards from: $SHARD_DIR"
echo "Output to: $OUTPUT_PATH"
echo "================================================="

python src/bioclip_vector_db/batch/merge_index_partition.py \
    --shard_dir "$SHARD_DIR" \
    --output_path "$OUTPUT_PATH"

echo "Merge complete."
