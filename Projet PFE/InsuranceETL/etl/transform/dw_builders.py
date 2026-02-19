import numpy as np
import pandas as pd
from ..common import clip_outliers

def build_dim_client(clean_clients: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "client_id", "nom", "prenom", "genre", "age", "age_group",
        "niveau_professionnel", "revenu_annuel", "income_band",
        "score_credit", "credit_band",
        "gouvernorat", "ville",
        "driving_risk_score", "financial_stress_score", "responsible_behavior_score",
        "risque_comportemental", "risque_rse", "risque_financier", "risque_fraude", "risque_global",
    ]
    keep = [c for c in keep if c in clean_clients.columns]
    return clean_clients[keep].drop_duplicates(subset=["client_id"]).reset_index(drop=True)

def build_dim_policy(clean_policies: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "contract_id", "client_id",
        "date_debut_contrat", "date_fin_contrat",
        "type_couverture",
        "prime_assurance_annuelle",
        "nb_sinistres_precedents",
        "delai_souscription_sinistre_jours",
        "policy_duration_days", "policy_tenure_bucket",
    ]
    keep = [c for c in keep if c in clean_policies.columns]
    return clean_policies[keep].drop_duplicates(subset=["contract_id"]).reset_index(drop=True)

def build_dim_vehicle(clean_vehicles: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "vehicle_id", "contract_id",
        "type_vehicule", "marque", "modele",
        "annee_modele", "vehicle_age",
        "puissance_fiscale",
        "kilometrage_actuel", "mileage_per_year",
        "valeur_vehicule", "vehicle_value_band",
        "usage_vehicule",
    ]
    keep = [c for c in keep if c in clean_vehicles.columns]
    return clean_vehicles[keep].drop_duplicates(subset=["vehicle_id"]).reset_index(drop=True)

def build_dim_time(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    dates = []
    for col in ["date_debut_contrat", "date_fin_contrat"]:
        if col in policies.columns:
            dates.append(policies[col])
    if "date_sinistre_claim" in claims.columns:
        dates.append(claims["date_sinistre_claim"])

    if not dates:
        return pd.DataFrame(columns=["date_key", "full_date", "year", "month", "day", "quarter", "week_of_year"])

    all_dates = pd.concat(dates, ignore_index=True).dropna().dt.normalize().drop_duplicates().sort_values()
    dim_time = pd.DataFrame({"full_date": all_dates})
    dim_time["date_key"] = dim_time["full_date"].dt.strftime("%Y%m%d").astype(int)
    dim_time["year"] = dim_time["full_date"].dt.year
    dim_time["month"] = dim_time["full_date"].dt.month
    dim_time["day"] = dim_time["full_date"].dt.day
    dim_time["quarter"] = dim_time["full_date"].dt.quarter
    dim_time["week_of_year"] = dim_time["full_date"].dt.isocalendar().week.astype(int)
    return dim_time.reset_index(drop=True)

def build_fact_claim(clean_claims, clean_policies, clean_clients, clean_vehicles) -> pd.DataFrame:
    fact = clean_claims.copy()

    # Policy merge
    pol_cols = [
        "contract_id", "client_id",
        "type_couverture", "prime_assurance_annuelle",
        "nb_sinistres_precedents", "delai_souscription_sinistre_jours",
        "policy_duration_days", "policy_tenure_bucket"
    ]
    pol_cols = [c for c in pol_cols if c in clean_policies.columns]
    pol_df = clean_policies[pol_cols].drop_duplicates(subset=["contract_id"]).copy()

    fact["contract_id"] = fact["contract_id"].astype(str).str.strip().str.upper()
    pol_df["contract_id"] = pol_df["contract_id"].astype(str).str.strip().str.upper()
    fact = fact.merge(pol_df, on="contract_id", how="left")
    # ---- FIX client_id after policy merge (client_id_x / client_id_y) ----
    if "client_id" not in fact.columns:
        if "client_id_x" in fact.columns and "client_id_y" in fact.columns:
            fact["client_id"] = fact["client_id_x"].fillna(fact["client_id_y"])
            fact = fact.drop(columns=["client_id_x", "client_id_y"])
        elif "client_id_x" in fact.columns:
            fact = fact.rename(columns={"client_id_x": "client_id"})
        elif "client_id_y" in fact.columns:
            fact = fact.rename(columns={"client_id_y": "client_id"})

    # Client merge (BULLETPROOF: never loses client_id)
    if "client_id" not in fact.columns:
        raise KeyError(f"fact has no client_id. Columns: {list(fact.columns)[:30]}")
    if "client_id" not in clean_clients.columns:
        raise KeyError(f"clean_clients has no client_id. Columns: {list(clean_clients.columns)[:30]}")

    cli_cols = ["client_id", "risque_global", "driving_risk_score", "financial_stress_score", "responsible_behavior_score"]
    existing = [c for c in cli_cols if c in clean_clients.columns]
    if "client_id" not in existing:
        existing = ["client_id"] + existing  # guarantee

    cli_df = clean_clients[existing].copy()
    cli_df["client_id"] = cli_df["client_id"].astype(str).str.strip().str.upper()
    fact["client_id"] = fact["client_id"].astype(str).str.strip().str.upper()

    cli_df = cli_df.drop_duplicates(subset=["client_id"])
    fact = fact.merge(cli_df, on="client_id", how="left")

    # Vehicle merge
    if "vehicle_id" in fact.columns and "vehicle_id" in clean_vehicles.columns:
        veh_cols = ["vehicle_id", "vehicle_age", "valeur_vehicule", "mileage_per_year", "vehicle_value_band", "usage_vehicule", "type_vehicule"]
        veh_cols = [c for c in veh_cols if c in clean_vehicles.columns]
        veh_df = clean_vehicles[veh_cols].drop_duplicates(subset=["vehicle_id"]).copy()

        fact["vehicle_id"] = fact["vehicle_id"].astype(str).str.strip().str.upper()
        veh_df["vehicle_id"] = veh_df["vehicle_id"].astype(str).str.strip().str.upper()
        fact = fact.merge(veh_df, on="vehicle_id", how="left")

    # Date key
    if "date_sinistre_claim" in fact.columns:
        fact["date_key"] = pd.to_datetime(fact["date_sinistre_claim"], errors="coerce").dt.strftime("%Y%m%d").astype("Int64")

    # KPI: loss ratio
    if "montant_indemnisation_claim" in fact.columns and "prime_assurance_annuelle" in fact.columns:
        denom = fact["prime_assurance_annuelle"].replace(0, np.nan)
        fact["loss_ratio"] = fact["montant_indemnisation_claim"] / denom
        fact["loss_ratio"] = clip_outliers(fact["loss_ratio"])

    return fact
