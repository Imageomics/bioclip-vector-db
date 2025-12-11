"""
Author: Net Zhang

Script to create partition manifest and ID-mapped lookup tables from source Parquet files.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor
import time

def list_files(folder_path: str, ext: str) -> List[str]:

    folder = Path(folder_path)
    return [str(p.resolve()) for p in folder.rglob(f"*.{ext}") if p.is_file()]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Partition Manifest & Metadata Lookup")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source Parquet files (with embeddings)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save Manifest and Lookup tables")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers for writing lookup files")
    return parser.parse_args()

def process_lookup_file(args):
    """
    Worker function to read source, strip embeddings, add global ID, and write lookup.
    """
    idx, file_path, start_id, output_dir = args
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"lookup_{filename}")

    # 1. Read everything EXCEPT the embedding column (Speed optimization)
    # Get schema first to exclude 'emb'
    schema = pq.read_schema(file_path)
    cols = [name for name in schema.names if name != 'emb']
    
    table = pq.read_table(file_path, columns=cols)
    row_count = table.num_rows

    # 2. Generate Global Integer IDs
    # Create a range [start_id, start_id + row_count)
    # PyArrow arrays are zero-copy compatible with NumPy
    ids = np.arange(start_id, start_id + row_count, dtype=np.int64)
    pa_ids = pa.array(ids)

    # 3. Prepend 'id' column to table
    # This creates the final schema: id, uuid, kingdom, ...
    table = table.add_column(0, "id", pa_ids)

    # 4. Write Lookup Parquet
    pq.write_table(table, output_path, compression='snappy')
    
    return filename, row_count

def main():
    args = parse_args()
    
    # Setup directories
    lookup_dir = os.path.join(args.output_dir, "lookup")
    os.makedirs(lookup_dir, exist_ok=True)
    
    print(f"=== Partitioning & Lookup Generation ===")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # 1. Scan Files & Calculate Offsets
    print("\n[1/3] Scanning metadata...")
    scan_start = time.time()
    
    # Sort files to ensure deterministic ID assignment every time this runs
    files = sorted(list_files(args.input_dir, "parquet"))
    if not files:
        raise FileNotFoundError("No parquet files found!")

    manifest_data = []
    current_id = 0

    # Fast sequential metadata read (Reading headers is cheap)
    for f in files:
        meta = pq.read_metadata(f)
        count = meta.num_rows
        
        manifest_data.append({
            "file_path": f,
            "row_count": count,
            "start_id": current_id
        })
        current_id += count

    print(f"Total Vectors: {current_id:,}")
    print(f"Scan finished in {time.time() - scan_start:.2f}s")

    # 2. Save Manifest
    print("\n[2/3] Saving Manifest...")
    df_manifest = pd.DataFrame(manifest_data)
    manifest_path = os.path.join(args.output_dir, "manifest.parquet")
    df_manifest.to_parquet(manifest_path, index=False)
    print(f"Manifest saved to: {manifest_path}")

    # 3. Generate Lookup Tables (Parallel)
    print(f"\n[3/3] Generating ID-mapped Lookup tables (Workers: {args.workers})...")
    gen_start = time.time()

    # Prepare arguments for workers
    worker_args = []
    for idx, row in df_manifest.iterrows():
        worker_args.append((idx, row['file_path'], row['start_id'], lookup_dir))

    # Parallel Execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_lookup_file, worker_args))

    print(f"Lookup generation finished in {time.time() - gen_start:.2f}s")
    print(f"Lookup tables saved in: {lookup_dir}")

if __name__ == "__main__":
    main()