"""
Author: Net Zhang
Executable script for training a FAISS Leader index using PyArrow for efficient zero-copy I/O.
"""

import faiss
import pyarrow as pa
import pyarrow.dataset as ds
import numpy as np
import time
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train FAISS Leader Index with PyArrow Zero-Copy I/O.")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Path to the directory containing Parquet files.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path where the trained index will be saved.")
    parser.add_argument("--index_string", type=str, default="IVF65536,PQ16", 
                        help="FAISS index factory string (default: IVF65536,PQ16)")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="gpu", 
                        help="Training mode: 'gpu' (default) or 'cpu'.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    # running 1 task per GPU
    args.gpu_id = 0
    print(f"=== FAISS Leader Index Training (Arrow Optimized) ===")
    print(f"Mode:         {args.mode.upper()}")
    print(f"Input:        {args.input_dir}")
    print(f"Index:        {args.index_string}")

    # --- Step 1: I/O (PyArrow Zero-Copy) ---
    print(f"\n[1/4] Starting I/O...")
    io_start = time.time()

    # PyArrow Dataset API
    dataset = ds.dataset(args.input_dir, format="parquet")
    
    # Load table with multi-threading
    table = dataset.to_table(columns=["emb"], use_threads=True)
    
    chunked_array = table["emb"]
    
    # Cast to large_list(float32) to ensure compatibility
    # Note: large_list uses 64-bit offsets, allowing >2GB data
    large_chunks = [chunk.cast(pa.large_list(pa.float32())) for chunk in chunked_array.chunks]
    large_list_array = pa.concat_arrays(large_chunks)

    # Zero-Copy View Construction
    num_rows = len(large_list_array)
    inner_len = len(large_list_array[0]) # Dimension
    flat_values = large_list_array.values

    # Direct memory view from PyArrow buffer -> NumPy
    # This avoids copying the data, drastically reducing RAM usage for I/O
    np_view = np.frombuffer(flat_values.buffers()[1], dtype=np.float32)[:num_rows * inner_len]
    np_view = np_view.reshape(num_rows, inner_len)

    n_samples, d = np_view.shape
    print(f"Data Loaded. Shape: ({n_samples:,}, {d})")
    print(f"I/O completed in {time.time() - io_start:.2f} seconds.")

    # --- Step 2: Initialize Index ---
    print(f"\n[2/4] Initializing Index ({args.index_string})...")
    index = faiss.index_factory(d, args.index_string)

    # --- Step 3: GPU Transfer (If enabled) ---
    if args.mode == "gpu":
        print(f"Transferring to GPU {args.gpu_id}...")
        try:
            res = faiss.StandardGpuResources()
            res.noTempMemory()
            config = faiss.GpuIndexIVFPQConfig()
            config.interleavedLayout = True
            assert(config.use_cuvs)
            # Disable temp memory if using RMM, otherwise leave default or set specific size
            # res.noTempMemory() 
            
            # Optional: Configure interleaved layout for performance if supported
            # co = faiss.GpuClonerOptions() 
            # co.useFloat16 = True
            
            index = faiss.index_cpu_to_gpu(res, args.gpu_id, index)
        except Exception as e:
            print(f"GPU Init Error: {e}")
            sys.exit(1)

    # --- Step 4: Training ---
    print(f"\n[3/4] Starting training on {n_samples:,} vectors...")
    train_start = time.time()
    # normalize before training
    faiss.normalize_L2(np_view)
    index.train(np_view)
    
    print(f"Training completed in {time.time() - train_start:.2f} seconds.")

    # --- Step 5: Save ---
    print(f"\n[4/4] Saving index...")
    if args.mode == "gpu":
        index = faiss.index_gpu_to_cpu(index)
        
    faiss.write_index(index, args.output_path)
    print(f"Success! Saved to: {args.output_path}")

if __name__ == "__main__":
    main()