import faiss
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import argparse
import os
import gc
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="GPU Batch Worker")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.parquet")
    parser.add_argument("--leader_index", type=str, required=True, help="Path to leader.index")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save shards")
    parser.add_argument("--start_idx", type=int, required=True, help="Manifest row index to start at")
    parser.add_argument("--end_idx", type=int, required=True, help="Manifest row index to end at")
    #parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, default=1_000_000, help="Target vectors per GPU flush")
    return parser.parse_args()

def process_batch(gpu_id, leader_path, vector_list, start_global_id, output_dir):
    """
    Helper: Merges buffered vectors, adds to GPU index, saves shard, clears memory.
    """
    if not vector_list:
        return

    t0 = time.time()
    
    # 1. Stack vectors (RAM)
    # vectors are already float32 from the read step
    batch_vectors = np.vstack(vector_list)
    total_batch_rows = batch_vectors.shape[0]
    
    print(f"  [GPU Flush] Processing batch of {total_batch_rows:,} vectors...")
    print(f"  [IDs] {start_global_id} -> {start_global_id + total_batch_rows}")

    # 2. Generate Contiguous IDs
    # Since files are sorted, we start at the first file's ID and increment
    ids = np.arange(start_global_id, start_global_id + total_batch_rows, dtype=np.int64)

    # 3. Load & Transfer Index
    cpu_leader = faiss.read_index(leader_path)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_leader)

    # 4. Add
    # Normalize before adding
    faiss.normalize_L2(batch_vectors)
    gpu_index.add_with_ids(batch_vectors, ids)

    # 5. Save Shard
    # Naming convention: shard_{start_id}.index (Deterministic)
    shard_name = f"shard_{start_global_id}.index"
    out_path = os.path.join(output_dir, shard_name)
    
    cpu_result = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_result, out_path)
    print(f"  [Saved] {out_path} ({time.time() - t0:.2f}s)")

    # 6. Aggressive Cleanup
    del batch_vectors, ids, cpu_leader, gpu_index, cpu_result, res
    gc.collect()



def main():
    args = parse_args()
    # args.gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    # running 1 task per GPU
    args.gpu_id = 0
    print(f"=== GPU Batch Worker {args.gpu_id} ===")
    print(f"Batch Target: {args.batch_size:,} vectors")
    
    # Load and Slice Manifest
    df_manifest = pd.read_parquet(args.manifest)
    max_rows = len(df_manifest)
    end_idx = min(args.end_idx, max_rows)
    my_tasks = df_manifest.iloc[args.start_idx : end_idx]
    
    if my_tasks.empty:
        print("No tasks assigned.")
        sys.exit(0)

    # Buffer State
    vector_buffer = []
    current_buffer_count = 0
    buffer_start_id = -1 

    for idx, task in my_tasks.iterrows():
        file_path = task['file_path']
        file_start_id = task['start_id']
        file_count = task['row_count']

        print(f"Reading: {os.path.basename(file_path)} ({file_count:,} rows)")

        # 1. Read Data
        try:
            table = pq.read_table(file_path, columns=['emb'], use_threads=True)
            chunked_array = table["emb"]
            large_chunks = [chunk.cast(pa.large_list(pa.float32())) for chunk in chunked_array.chunks]
            large_list_array = pa.concat_arrays(large_chunks)
            num_rows = len(large_list_array)
            inner_len = len(large_list_array[0]) # Dimension
            flat_values = large_list_array.values
            np_view = np.frombuffer(flat_values.buffers()[1], dtype=np.float32)[:num_rows * inner_len]
            np_view = np_view.reshape(num_rows, inner_len)
        except Exception as e:
            print(f"SKIP {file_path}: {e}")
            continue

        # 2. Logic: New Buffer or Append?
        if current_buffer_count == 0:
            buffer_start_id = file_start_id

        # 3. Append
        vector_buffer.append(np_view)
        current_buffer_count += file_count

        # 4. Check Threshold
        if current_buffer_count >= args.batch_size:
            # FLUSH
            process_batch(args.gpu_id, args.leader_index, vector_buffer, buffer_start_id, args.output_dir)
            
            # RESET
            vector_buffer = []
            current_buffer_count = 0
            buffer_start_id = -1

    # 5. Process Remainder (Final small batch)
    if current_buffer_count > 0:
        print("Processing final remaining buffer...")
        process_batch(args.gpu_id, args.leader_index, vector_buffer, buffer_start_id, args.output_dir)

    print("All tasks completed.")

if __name__ == "__main__":
    main()