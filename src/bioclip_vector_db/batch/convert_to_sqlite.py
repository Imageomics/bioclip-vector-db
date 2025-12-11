"""

Convert Parquet dataset to SQLite database in batches.

Usage:
    python src/bioclip_vector_db/batch/convert_to_sqlite.py \
        --parquet_path /fs/scratch/PAS2136/TreeOfLife/image_lookup/2024-05-01/hdf5/200M/lookup_temp \
        --sqlite_path /fs/scratch/PAS2136/TreeOfLife/image_lookup/2024-05-01/hdf5/200M/lookup_temp.db \
        --table_name lookup \
        --index_col uuid
    
    python src/bioclip_vector_db/batch/convert_to_sqlite.py \
        --parquet_path /fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/flight_plan/lookup \
        --sqlite_path /fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/flight_plan/lookup.db \
        --table_name metadata \
        --index_col id
"""
import sqlite3
import pyarrow as pa
import pyarrow.dataset as ds
import argparse
import time
import os

def get_sqlite_type(pa_type):
    """Maps PyArrow types to SQLite types."""
    if pa.types.is_integer(pa_type):
        return "INTEGER"
    elif pa.types.is_floating(pa_type):
        return "REAL"
    else:
        return "TEXT"

def convert_parquet_to_sqlite(parquet_path, sqlite_path, table_name, index_col, batch_size=50000):
    if os.path.exists(sqlite_path):
        print(f"Warning: '{sqlite_path}' already exists. Appending to it.")

    # 1. Inspect Schema
    print(f"Opening Parquet dataset at {parquet_path}...")
    dataset = ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema
    col_names = schema.names

    print(f"Detected columns: {col_names}")

    # Validate Index Column
    if index_col not in col_names:
        # Try to fallback if user didn't specify exact name (e.g. 'uuid' vs 'id')
        if index_col == 'id' and 'uuid' in col_names:
            index_col = 'uuid'
            print(f"Index column 'id' not found. Using 'uuid' instead.")
        elif index_col == 'uuid' and 'id' in col_names:
            index_col = 'id'
            print(f"Index column 'uuid' not found. Using 'id' instead.")
        else:
            raise ValueError(f"Index column '{index_col}' not found in dataset columns: {col_names}")

    # 2. Build Create Table Statement
    columns_sql = []
    for name in col_names:
        dtype = get_sqlite_type(schema.field(name).type)
        columns_sql.append(f'"{name}" {dtype}')
    
    create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns_sql)});"

    # 3. Setup SQLite
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    # Performance settings
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    cursor.execute("PRAGMA cache_size = 100000")
    
    print(f"Creating table '{table_name}'...")
    cursor.execute(create_stmt)
    conn.commit()

    # 4. Stream and Insert
    start_time = time.time()
    total_rows = 0
    
    scanner = dataset.scanner(batch_size=batch_size)
    placeholders = ",".join(["?"] * len(col_names))
    insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
    
    print("Starting data insertion...")
    for batch in scanner.to_batches():
        # Transpose columns to rows efficiently
        columns_data = [batch[name].to_pylist() for name in col_names]
        rows = list(zip(*columns_data))
        
        cursor.executemany(insert_sql, rows)
        total_rows += len(rows)
        
        if total_rows % 1_000_000 == 0:
            print(f"Inserted {total_rows} rows...")
            conn.commit()

    conn.commit()
    print(f"Finished inserting {total_rows} rows in {time.time() - start_time:.2f}s")

    # 5. Create Index
    print(f"Creating index on '{index_col}'...")
    idx_start = time.time()
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{index_col} ON {table_name} (\"{index_col}\")")
    conn.commit()
    print(f"Index created in {time.time() - idx_start:.2f}s")

    conn.close()
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet to SQLite")
    parser.add_argument("--parquet_path", type=str, required=True, help="Input Parquet path")
    parser.add_argument("--sqlite_path", type=str, required=True, help="Output SQLite DB path")
    parser.add_argument("--table_name", type=str, required=True, help="Name of the table to create (e.g., 'lookup' or 'metadata')")
    parser.add_argument("--index_col", type=str, required=True, help="Column to index (e.g., 'uuid' or 'id')")
    
    args = parser.parse_args()
    convert_parquet_to_sqlite(args.parquet_path, args.sqlite_path, args.table_name, args.index_col)