# main.py
from etl.db_connection import get_connection
from etl.extract import extract_table
from etl.transform import transform_all
from etl.load import load_all


def main():
    print("INSURANCE ETL PIPELINE STARTED")

    # Extract (use dbo explicitly)
    clients = extract_table("dbo.Clients")
    policies = extract_table("dbo.Polices_Assurance")
    vehicles = extract_table("dbo.Vehicules")
    claims = extract_table("dbo.Sinistres")

    out = transform_all(clients, policies, vehicles, claims)

    print("[TRANSFORM] Done ✅")
    print({
        "clean_rows": {
            "clients": len(out["clean_clients"]),
            "policies": len(out["clean_policies"]),
            "vehicles": len(out["clean_vehicles"]),
            "claims": len(out["clean_claims"]),
        },
        "dw_rows": {
            "dim_client": len(out["dim_client"]),
            "dim_policy": len(out["dim_policy"]),
            "dim_vehicle": len(out["dim_vehicle"]),
            "dim_time": len(out["dim_time"]),
            "fact_claim": len(out["fact_claim"]),
        },
        "ml_rows": {
            "ml_claim_fraud": len(out["ml_claim_fraud"]),
            "ml_claim_severity": len(out["ml_claim_severity"]),
        }
    })

    # Load using ONE connection
    conn = get_connection()
    try:
        load_all(conn, out, mode="replace")
        print("[LOAD] Done ✅")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
