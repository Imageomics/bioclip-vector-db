import argparse
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pathlib import Path
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Spark Stratified Sampler")
    parser.add_argument("--input_path", type=str, required=True, help="Input parquet path (wildcard supported)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the single sample file")
    parser.add_argument("--sample_col", type=str, default="class", help="Column to stratify by")
    parser.add_argument("--target_total", type=int, default=15_000_000, help="Target sample size")
    parser.add_argument("--min_per_group", type=int, default=1000, help="Minimum samples per group (if available)")
    return parser.parse_args()

def list_files(folder_path: str, ext: str) -> List[str]:

    folder = Path(folder_path)
    return [str(p.resolve()) for p in folder.rglob(f"*.{ext}") if p.is_file()]


def init_spark() -> SparkSession:
    N_EXECUTORS=80
    spark = (
        SparkSession.builder
        .appName("StratifiedSampler")
        .config("spark.executor.instances", f"{N_EXECUTORS}")
        .config("spark.executor.memory", "75G")
        .config("spark.executor.cores", "12")
        .config("spark.driver.memory", "64G")
        # Additional Tunning
        .config("spark.sql.shuffle.partitions", "1000")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        #.config("spark.sql.files.maxPartitionBytes", "256MB")
        .config("spark.sql.parquet.enableVectorizedReader", "false") 
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    
    return spark

def main():
    args = parse_args()
    
    spark = init_spark()

    print(f"Input: {args.input_path}")
    print(f"Target Size: {args.target_total:,}")
    print(f"Stratifying by: {args.sample_col}")

    # 1. Read Metadata Only (Fast)
    # We select only the stratification column to speed up the count
    parquet_files = list_files(args.input_path, "parquet")
    print(f"Found {len(parquet_files):,} parquet files.")
    df = spark.read.parquet(*parquet_files)
    
    # Handle nulls in taxonomy
    df = df.fillna({args.sample_col: "Unknown"})

    # 2. Compute Distribution
    print("Computing class distribution...")
    counts_df = df.groupBy(args.sample_col).count().cache()
    
    # Collect to driver (small data: ~100-1000 rows for 'class')
    counts = counts_df.collect()
    total_groups = len(counts)
    total_rows = sum(row['count'] for row in counts)
    
    print(f"Total Rows: {total_rows:,}")
    print(f"Unique Groups: {total_groups}")

    # 3. Calculate Sampling Fractions
    # Strategy: Capped Proportional Sampling
    # Ideally, we want TARGET / GROUPS samples per group
    # But some groups are tiny.
    
    # Simplified Logic:
    # 1. Calculate an "ideal cap" per group to reach target
    #    If we have 50 groups and want 15M samples, cap is 300k.
    # 2. But we allow larger groups to have slightly more presence (log scale or sqrt).
    
    # Let's stick to the "Cap" strategy for robustness
    # Calculate what the "Fair Share" cap would be
    fair_cap = int(args.target_total / total_groups * 2.0) # Allow 2x fair share variance
    if fair_cap < args.min_per_group: fair_cap = args.min_per_group
    
    fractions = {}
    expected_sample_size = 0
    
    for row in counts:
        group_name = row[args.sample_col]
        count = row['count']
        
        # How many do we want from this group?
        # Take ALL if small, otherwise take the CAP
        target_n = min(count, fair_cap)
        
        # Calculate fraction
        frac = target_n / count
        fractions[group_name] = frac
        expected_sample_size += int(count * frac)
        print(f"Group: {group_name}, Count: {count:,}, Target: {target_n:,}, Fraction: {frac:.4f}")

    print(f"Expected Sample Size: {expected_sample_size:,}")
    print(f"Fair Cap per Group: {fair_cap:,}")

    # 4. Perform Stratified Sampling
    # sampleBy is efficient but requires a pass over data
    print("Executing distributed sampling...")
    
    sampled_df = df.sampleBy(args.sample_col, fractions, seed=42)
    
    # 5. Extract Embeddings & Save
    final_df = sampled_df
    
    # Coalesce to 1 partition to generate a SINGLE output file 
    # (Easier for FAISS train_leader.py to consume)
    print(f"Saving to {args.output_path}...")
    
    # Note: repartition(1) is expensive if data is huge, 
    # but for 15M vectors (~60GB RAM needed), ensure driver/executor has memory.
    # If 15M is too big for one file, use repartition(10) and update train_leader to read folder.
    
    final_df.repartition(1000).write.mode("overwrite").parquet(args.output_path)
    
    print("Sampling Complete.")
    spark.stop()

if __name__ == "__main__":
    main()