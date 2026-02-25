"""Production data ingestion: append new CSV data to raw dbo.* tables.

This script:
1. Loads new CSV files (clients, policies, vehicles, claims)
2. Validates schema & types against existing raw tables
3. Appends rows to dbo.Clients, dbo.Polices_Assurance, dbo.Vehicules, dbo.Sinistres
4. Logs ingestion summary

Usage:
  python etl/ingest_new_data.py \
    --clients /path/to/clients.csv \
    --policies /path/to/policies.csv \
    --vehicles /path/to/vehicles.csv \
    --claims /path/to/claims.csv \
    --dry-run  # (optional: preview without writing)

Schedule: Run weekly/monthly before running main.py ETL pipeline.
"""
import argparse
import os
import sys
import pandas as pd
from datetime import datetime

# add project root
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from etl.db_connection import get_connection
from etl.extract import extract_table


def validate_and_cast(df_new, df_existing, table_name):
    """Validate schema & cast new data to match existing."""
    # check required columns
    missing = set(df_existing.columns) - set(df_new.columns)
    if missing:
        print(f"[WARN] {table_name}: missing columns {missing}")
        # add missing with NaN
        for col in missing:
            df_new[col] = None

    # drop extra columns in new data
    extra = set(df_new.columns) - set(df_existing.columns)
    if extra:
        print(f"[WARN] {table_name}: dropping extra columns {extra}")
        df_new = df_new.drop(columns=extra)

    # reorder to match existing
    df_new = df_new[df_existing.columns]

    # type casting: try to match dtypes
    for col in df_new.columns:
        if col not in df_existing.columns:
            continue
        src_dtype = df_existing[col].dtype
        try:
            if pd.api.types.is_numeric_dtype(src_dtype):
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
            elif pd.api.types.is_datetime64_any_dtype(src_dtype):
                df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
            elif src_dtype == 'object':
                df_new[col] = df_new[col].astype(str)
        except Exception as e:
            print(f"[WARN] {table_name}/{col}: cast failed: {e}")

    return df_new


def ingest_table(csv_path, dbo_table_name, dry_run=False):
    """Ingest CSV into dbo table via INSERT."""
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {csv_path} not found")
        return 0

    print(f"\n[INGEST] {csv_path} -> {dbo_table_name}")
    df_new = pd.read_csv(csv_path)
    print(f"  New data: {len(df_new)} rows, {len(df_new.columns)} columns")

    # load existing table schema
    df_existing = extract_table(dbo_table_name)
    print(f"  Existing table: {len(df_existing)} rows")

    # validate & cast
    df_new = validate_and_cast(df_new, df_existing, dbo_table_name)

    if dry_run:
        print(f"  [DRY-RUN] Would insert {len(df_new)} rows")
        print(f"  First 3 rows preview:\n{df_new.head(3)}")
        return len(df_new)

    # insert into database
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Use batch inserts for better performance
        cols = ', '.join([f"[{c}]" for c in df_new.columns])
        placeholders = ', '.join(['?' for _ in df_new.columns])
        insert_sql = f"INSERT INTO {dbo_table_name} ({cols}) VALUES ({placeholders})"

        batch_size = 1000  # Insert in batches of 1000
        rows_inserted = 0
        
        for i in range(0, len(df_new), batch_size):
            batch = df_new.iloc[i:i+batch_size]
            batch_data = []
            for idx, row in batch.iterrows():
                vals = tuple(None if pd.isna(v) else v for v in row.values)
                batch_data.append(vals)
            
            try:
                cursor.executemany(insert_sql, batch_data)
                rows_inserted += len(batch_data)
                print(f"  [PROGRESS] Inserted {rows_inserted}/{len(df_new)} rows...")
            except Exception as e:
                print(f"  [WARN] Batch {i//batch_size + 1} failed: {e}")
                # Try individual inserts for failed batch
                for vals in batch_data:
                    try:
                        cursor.execute(insert_sql, vals)
                        rows_inserted += 1
                    except Exception as e2:
                        print(f"  [WARN] Row insert failed: {e2}")
        
        conn.commit()
        print(f"  [SUCCESS] Inserted {rows_inserted} rows")
        return rows_inserted

    except Exception as e:
        print(f"  [ERROR] Insert failed: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        conn.close()


def main():
    p = argparse.ArgumentParser(
        description='Append new CSV data to raw dbo.* tables'
    )
    p.add_argument('--clients', type=str, help='C:/Users/Montassar Tynsa/Downloads/addon_clients.csv')
    p.add_argument('--policies', type=str, help='C:/Users/Montassar Tynsa/Downloads/addon_polices_assurance.csv')
    p.add_argument('--vehicles', type=str, help='C:/Users/Montassar Tynsa/Downloads/addon_vehicules.csv')
    p.add_argument('--claims', type=str, help='C:/Users/Montassar Tynsa/Downloads/addon_sinistres.csv')
    p.add_argument('--dry-run', action='store_true', help='Preview without insert')
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"DATA INGESTION: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    if args.dry_run:
        print("[DRY-RUN MODE] - previewing only, no data will be inserted\n")

    total = 0
    total += ingest_table(args.clients or '', 'dbo.Clients', args.dry_run) if args.clients else 0
    total += ingest_table(args.policies or '', 'dbo.Polices_Assurance', args.dry_run) if args.policies else 0
    total += ingest_table(args.vehicles or '', 'dbo.Vehicules', args.dry_run) if args.vehicles else 0
    total += ingest_table(args.claims or '', 'dbo.Sinistres', args.dry_run) if args.claims else 0

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total} total rows ingested")
    print(f"Next step: run `python main.py` to trigger ETL pipeline")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
