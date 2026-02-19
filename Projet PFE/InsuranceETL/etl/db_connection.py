# etl/db_connection.py
import pyodbc
from config.db_config import SERVER, DATABASE, DRIVER, TRUSTED_CONNECTION

def get_connection():
    if TRUSTED_CONNECTION:
        conn_str = (
            f"DRIVER={{{DRIVER}}};"
            f"SERVER={SERVER};"
            f"DATABASE={DATABASE};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )
    else:
        # If you later want SQL auth, add UID/PWD here
        raise RuntimeError("Set TRUSTED_CONNECTION=True or implement SQL auth.")
    return pyodbc.connect(conn_str)
