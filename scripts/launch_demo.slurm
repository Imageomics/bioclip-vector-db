#!/bin/bash
#SBATCH -A PAS2136
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 4:00:00
#SBATCH --job-name=bioclip_demo
#SBATCH --output=logs/demo_%j.log

# 1. Load Environment
# Adjust these lines if your conda setup is different
module load miniconda3/24.1.2-py310
# conda activate faiss_env

# 2. Define Session Name
SESSION="bioclip_services"

# 3. Start Tmux Session (Detached)
# We start a new session in the background
tmux new-session -d -s $SESSION

# ------------------------------------------------------------------
# Window 0: Image Server
# ------------------------------------------------------------------
tmux rename-window -t $SESSION:0 'image_server'
tmux send-keys -t $SESSION:0 "conda activate faiss_env" C-m
tmux send-keys -t $SESSION:0 "python src/bioclip_vector_db/query/image_server.py \\
    --lookup_path /fs/scratch/PAS2136/TreeOfLife/image_lookup/2024-05-01/hdf5/200M/lookup_temp.db \\
    --port 5002 \\
    --workers 16 \\
    --h5_group images" C-m

# ------------------------------------------------------------------
# Window 1: FAISS Search Server
# ------------------------------------------------------------------
tmux new-window -t $SESSION:1 -n 'faiss_server'
tmux send-keys -t $SESSION:1 "conda activate faiss_env" C-m
tmux send-keys -t $SESSION:1 "python src/bioclip_vector_db/query_monolithic/neighborhood_server.py \\
    --index-path /fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified_merged.index \\
    --id-lookup-path /fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/flight_plan/lookup.db \\
    --nprobes 12 \\
    --port 5001" C-m

# ------------------------------------------------------------------
# Window 2: Gradio App
# ------------------------------------------------------------------
tmux new-window -t $SESSION:2 -n 'app'
# Give servers a moment to start up before launching the app
tmux send-keys -t $SESSION:2 "conda activate faiss_env" C-m
tmux send-keys -t $SESSION:2 "sleep 10; python app_gr_new.py \\
    --neighborhood-server http://localhost:5001 \\
    --image-server http://localhost:5002 \\
    --host 0.0.0.0 \\
    --port 7860 \\
    --model hf-hub:imageomics/bioclip-2 \\
    --disable-export" C-m

# ------------------------------------------------------------------
# Keep Job Alive
# ------------------------------------------------------------------
echo "----------------------------------------------------------------"
echo "Services started in tmux session '$SESSION' on node $(hostname)"
echo "To attach to the session, run:"
echo "  ssh $(hostname)"
echo "  tmux attach -t $SESSION"
echo "----------------------------------------------------------------"

# Sleep infinity to keep the allocation active so tmux doesn't die
sleep infinity
