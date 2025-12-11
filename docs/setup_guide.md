# Setup Guide

This guide demonstrates how to setup the BioCLIP-2 image search application. 

## Prerequisites Checklist

Before running the demo, ensure the following resources are available:

- [ ] **Conda Environment**: `faiss_env`
  - Create if missing: `conda env create -f conda_environment.yml -n faiss_env`
- [ ] **Image Lookup Database**: `/fs/scratch/PAS2136/TreeOfLife/image_lookup/2024-05-01/hdf5/200M/lookup_temp.db`
- [ ] **FAISS Index**: `/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified_merged.index`
- [ ] **ID Lookup Database**: `/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/flight_plan/lookup.db`

## Quick Start (Recommended)

For a quick start, you can use the provided SLURM script to launch all services (Image Server, FAISS Server, and Gradio App) in a single job using `tmux`.

1. **Submit the job:**
   ```bash
   sbatch scripts/launch_demo.slurm
   ```

2. **Check the job status:**
   ```bash
   squeue -u $USER
   ```

3. **Connect to the running node:**
   Once the job is running, ssh into the allocated node (e.g., `cardinal-c001`):
   ```bash
   ssh <node_name>
   ```

4. **Attach to the tmux session:**
   The script starts a tmux session named `bioclip_services`.
   ```bash
   tmux attach -t bioclip_services
   ```

### Tmux Cheatsheet
- **Detach from session:** `Ctrl+b` then `d`
- **Switch windows:** `Ctrl+b` then `n` (next) or `p` (previous), or `0`, `1`, `2`...
- **Scroll mode:** `Ctrl+b` then `[` (use arrow keys/PgUp/PgDn, press `q` to exit)

## Manual Setup (Alternative)

If you prefer to set up each component manually or need to debug specific services, follow the steps below.

### Environment Setup

The supported way to install [`Faiss`](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) is through `conda`. 

``` bash
# module load conda
conda env create -f conda_environment.yml -n faiss_env
conda activate faiss_env
```

## Allocate Compute Resource

Allocate compute node with abundant RAM & CPU cores. 

``` bash
salloc -A PAS2136 -p cpu -N 1 -c 32 -t 2:00:00
```

## Setup Image Server

Spin up a tmux session to provide image retrieval service.

```
Requirements: 
- Images are stored in HDF5
- Lookup table (SQLite with built index): image `uuid` to `h5_file` mapping
```

``` bash
tmux new -s image_server
conda activate faiss_env

python src/bioclip_vector_db/query/image_server.py \
    --lookup_path /fs/scratch/PAS2136/TreeOfLife/image_lookup/2024-05-01/hdf5/200M lookup_temp.db \
    --port 5002 \
    --workers 16 \
    --h5_group images
```

``` bash
curl -s -X GET -H "Content-Type: application/json" http://localhost:5002/health | jq

curl -X POST http://localhost:5002/images \
  -H "Content-Type: application/json" \
  -d '{
    "uuids": [
      "04ca2e57-1821-4bfa-9c0f-ff1058bded28",
      "046d259b-d503-404f-b04b-01afad0139c5"
    ]
  }'
```

## Setup FAISS Search Server

Spin up a tmux session to provide FAISS vector search service.

```
Requirements: 
- a trained, merged FAISS index
- Lookup table (SQLite with built index): image `uuid` to FAISS `id` mapping, and append other metadata as needed
```

``` bash
tmux new -s faiss_server
conda activate faiss_env

python src/bioclip_vector_db/query_monolithic/neighborhood_server.py \
    --index-path /fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified_merged.index \
    --id-lookup-path /fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/flight_plan/lookup.db \
    --nprobes 12 \
    --port 5001
```

``` bash
curl -s -X GET -H "Content-Type: application/json" http://localhost:5001/health | jq
```

## Setup Front-end Application

Spin up a tmux session to host the front-end Gradio application.

``` bash
tmux new -s app
conda activate faiss_env

python app_gr_new.py \
        --neighborhood-server http://localhost:5001 \
        --image-server http://localhost:5002 \
        --host 0.0.0.0 \
        --port 7860 \
        --model hf-hub:imageomics/bioclip-2 \
        --disable-export
```