# etl/extract.py
import pandas as pd
from .db_connection import get_connection

def extract_table(table_name: str) -> pd.DataFrame:
    """
    table_name can be:
      - "Clients"
      - "dbo.Clients"
    """
    print(f"[EXTRACT] Reading table: {table_name}")
    conn = get_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()

    print(f"[EXTRACT] {len(df)} rows loaded from {table_name}")
    return df
