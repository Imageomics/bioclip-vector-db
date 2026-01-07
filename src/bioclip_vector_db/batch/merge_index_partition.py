import faiss
import argparse
import os
import sys
import time
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Merge FAISS Index Shards")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing shard_*.index files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged index")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"=== Phase 4: Merging Shards ===")
    print(f"Source: {args.shard_dir}")
    print(f"Target: {args.output_path}")

    # 1. Find all shards
    shard_files = glob.glob(os.path.join(args.shard_dir, "shard_*.index"))
    if not shard_files:
        print("Error: No shard files found!")
        sys.exit(1)

    print(f"Found {len(shard_files)} shards to merge.")

    try:
        # Try to sort by the number in the filename: shard_0.index, shard_1.index, etc.
        # Assuming format shard_X.index
        shard_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    except Exception as e:
        print("Warning: Could not sort filenames numerically, merging in default order.")
        shard_files.sort()

    start_time = time.time()
    first_shard = shard_files[0]
    print(f"Loading master shard: {os.path.basename(first_shard)}...")
    master_index = faiss.read_index(first_shard)

    for i, fname in enumerate(shard_files[1:], 1):
        print(f"[{i}/{len(shard_files)-1}] Merging {os.path.basename(fname)}...")

        # Read the next part
        part_index = faiss.read_index(fname)

        # Merge into master
        master_index.merge_from(part_index, 0)

    print(f"Merge complete. Final Size: {master_index.ntotal:,} vectors.")
    
    print(f"Saving merged index to {args.output_path}...")
    faiss.write_index(master_index, args.output_path)
    
    end_time = time.time()
    print(f"Done in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
