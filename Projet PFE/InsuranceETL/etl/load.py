# etl/load.py
from __future__ import annotations

import math
import re
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import pyodbc


# ---------- Identifier safety ----------
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _safe_ident(name: str) -> str:
    """
    Allow only simple identifiers to avoid SQL injection in CREATE statements.
    """
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe identifier: {name}")
    return name

def _q(name: str) -> str:
    # bracket quote
    return f"[{name}]"


# ---------- Schema / table existence ----------
def _ensure_schema(conn: pyodbc.Connection, schema: str) -> None:
    schema = _safe_ident(schema)
    sql = """
    IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = ?)
    EXEC('CREATE SCHEMA %s');
    """ % _q(schema)
    cur = conn.cursor()
    cur.execute(sql, (schema,))
    conn.commit()

def _table_exists(conn: pyodbc.Connection, schema: str, table: str) -> bool:
    schema = _safe_ident(schema)
    table = _safe_ident(table)
    cur = conn.cursor()
    cur.execute("""
        SELECT 1
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE s.name = ? AND t.name = ?
    """, (schema, table))
    return cur.fetchone() is not None


def _get_table_columns(conn: pyodbc.Connection, schema: str, table: str) -> set:
    """Get the set of column names from an existing table."""
    schema = _safe_ident(schema)
    table = _safe_ident(table)
    cur = conn.cursor()
    cur.execute("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
    """, (schema, table))
    return set(row[0].lower() for row in cur.fetchall())


def _drop_table(conn: pyodbc.Connection, schema: str, table: str) -> None:
    """Drop a table if it exists."""
    schema = _safe_ident(schema)
    table = _safe_ident(table)
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {_q(schema)}.{_q(table)}")
    conn.commit()


# ---------- SQL type mapping ----------
def _sql_type_for_series(s: pd.Series) -> str:
    """
    Simple robust mapping.
    """
    if pd.api.types.is_integer_dtype(s):
        # use BIGINT to avoid overflow surprises
        return "BIGINT"
    if pd.api.types.is_float_dtype(s):
        return "FLOAT"
    if pd.api.types.is_bool_dtype(s):
        return "BIT"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "DATETIME2"
    # strings / categories / objects
    # choose NVARCHAR(255) unless longer detected
    max_len = 0
    try:
        # avoid huge cost if big; sample is ok
        sample = s.dropna().astype(str)
        if not sample.empty:
            max_len = int(sample.map(len).max())
    except Exception:
        max_len = 255

    if max_len <= 50:
        return "NVARCHAR(50)"
    if max_len <= 100:
        return "NVARCHAR(100)"
    if max_len <= 255:
        return "NVARCHAR(255)"
    return "NVARCHAR(MAX)"


def _ensure_table_from_df(
    conn: pyodbc.Connection,
    schema: str,
    table: str,
    df: pd.DataFrame,
    primary_key: Optional[str] = None,
) -> None:
    schema = _safe_ident(schema)
    table = _safe_ident(table)

    # Check if table exists and has matching columns
    if _table_exists(conn, schema, table):
        existing_cols = _get_table_columns(conn, schema, table)
        df_cols = set(c.lower() for c in df.columns)
        
        # If columns don't match, drop and recreate
        if existing_cols != df_cols:
            print(f"[SCHEMA MISMATCH] {schema}.{table} - dropping and recreating with new columns")
            _drop_table(conn, schema, table)
        else:
            # Columns match, no need to recreate
            return

    if df is None or df.empty:
        raise RuntimeError(f"Cannot create table {schema}.{table} from empty dataframe.")

    cols_sql = []
    for col in df.columns:
        col_safe = _safe_ident(col)
        sql_type = _sql_type_for_series(df[col])

        # If it is the PK column -> force NOT NULL
        if primary_key and col == primary_key:
            cols_sql.append(f"{_q(col_safe)} {sql_type} NOT NULL")
        else:
            cols_sql.append(f"{_q(col_safe)} {sql_type} NULL")

    pk_sql = ""
    if primary_key:
        pk_safe = _safe_ident(primary_key)
        pk_sql = f", CONSTRAINT PK_{schema}_{table} PRIMARY KEY ({_q(pk_safe)})"

    create_sql = f"""
    CREATE TABLE {_q(schema)}.{_q(table)} (
        {", ".join(cols_sql)}
        {pk_sql}
    );
    """

    cur = conn.cursor()
    cur.execute(create_sql)
    conn.commit()


# ---------- Data cleaning for ODBC ----------
def _to_python_value(v):
    """
    Convert pandas/numpy values into plain Python values,
    and convert NaN/Inf to None (THIS fixes your float RPC error).
    """
    # pandas missing
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # numpy scalar -> python scalar
    if isinstance(v, np.generic):
        v = v.item()

    # datetime
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return None
        return v.to_pydatetime()

    # float NaN / Inf (critical)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)

    return v


def _prep_rows(df: pd.DataFrame) -> Tuple[List[str], List[Tuple]]:
    """
    Prepare column list and rows list for pyodbc.
    Forces every value through _to_python_value to kill NaN/Inf.
    """
    cols = list(df.columns)

    # work in object space
    df2 = df.copy()

    # Replace +/-inf -> NaN first
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].replace([np.inf, -np.inf], np.nan)

    df2 = df2.astype(object)

    rows: List[Tuple] = []
    for r in df2.itertuples(index=False, name=None):
        rows.append(tuple(_to_python_value(x) for x in r))

    return cols, rows


def _truncate_table(conn: pyodbc.Connection, schema: str, table: str) -> None:
    schema = _safe_ident(schema)
    table = _safe_ident(table)
    cur = conn.cursor()
    cur.execute(f"TRUNCATE TABLE {_q(schema)}.{_q(table)};")
    conn.commit()


def _insert_df_fast(
    conn: pyodbc.Connection,
    schema: str,
    table: str,
    df: pd.DataFrame,
    chunksize: int = 2000,
) -> None:
    if df is None or df.empty:
        return

    schema = _safe_ident(schema)
    table = _safe_ident(table)

    cols, rows = _prep_rows(df)

    col_sql = ", ".join(_q(_safe_ident(c)) for c in cols)
    placeholders = ", ".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO {_q(schema)}.{_q(table)} ({col_sql}) VALUES ({placeholders})"

    cur = conn.cursor()

    # fast_executemany can be kept ON, but if your driver is unstable with it,
    # you can switch to False. With our NaN→None fix, ON should work.
    cur.fast_executemany = True

    try:
        for i in range(0, len(rows), chunksize):
            cur.executemany(insert_sql, rows[i:i + chunksize])
        conn.commit()
        return
    except pyodbc.Error as e:
        print(f"\n[LOAD-ERROR] Batch insert failed for table: {table}")
        print("[LOAD-ERROR] Locating failing row...")

        # fallback: find exact failing row
        conn.rollback()
        cur.fast_executemany = False

        for idx, r in enumerate(rows):
            try:
                cur.execute(insert_sql, r)
            except pyodbc.Error as e2:
                print(f"\n[LOAD-ERROR] Insert failed on row index: {idx}")
                print(f"[LOAD-ERROR] Row values: {r}")
                print(f"[LOAD-ERROR] Exception: {e2}")
                raise
        conn.commit()
        raise e


def load_table(
    conn: pyodbc.Connection,
    schema: str,
    table: str,
    df: pd.DataFrame,
    mode: str = "append",   # append | replace
    primary_key: Optional[str] = None,
) -> None:
    _ensure_schema(conn, schema)
    _ensure_table_from_df(conn, schema, table, df, primary_key=primary_key)

    if mode == "replace":
        _truncate_table(conn, schema, table)

    _insert_df_fast(conn, schema, table, df)


def load_all(conn: pyodbc.Connection, out: dict, mode: str = "replace") -> None:
    """
    out is your transform_all output: out["clean_clients"], etc.
    """
    mapping = [
        ("stg", "clean_clients", out.get("clean_clients"), "client_id"),
        ("stg", "clean_policies", out.get("clean_policies"), "contract_id"),
        ("stg", "clean_vehicles", out.get("clean_vehicles"), "vehicle_id"),
        ("stg", "clean_claims", out.get("clean_claims"), "claim_id"),
    
        ("dw", "dim_client", out.get("dim_client"), "client_id"),
        ("dw", "dim_policy", out.get("dim_policy"), "contract_id"),
        ("dw", "dim_vehicle", out.get("dim_vehicle"), "vehicle_id"),
        ("dw", "dim_time", out.get("dim_time"), "date_key"),
        ("dw", "fact_claim", out.get("fact_claim"), "claim_id"),

        ("ml", "ml_claim", out.get("ml_claim"), "claim_id"),
    ]

    for schema, table, df, pk in mapping:
        if df is None:
            continue
        print(f"[LOAD] Loading {schema}.{table} ({len(df)} rows)")
        load_table(conn, schema, table, df, mode=mode, primary_key=pk)
