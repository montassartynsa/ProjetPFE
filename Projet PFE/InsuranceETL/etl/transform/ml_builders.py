import numpy as np
import pandas as pd
from ..common import clip_outliers

def build_ml_claim_dataset(clean_clients, clean_policies, clean_vehicles, clean_claims):
    base = clean_claims.copy()

    # Policy features
    policy_feats = [
        "contract_id", "type_couverture", "prime_assurance_annuelle",
        "nb_sinistres_precedents", "delai_souscription_sinistre_jours",
        "policy_duration_days", "policy_tenure_bucket"
    ]
    policy_feats = [c for c in policy_feats if c in clean_policies.columns]
    base = base.merge(clean_policies[policy_feats].drop_duplicates(subset=["contract_id"]),
                      on="contract_id", how="left")

    # Client features
    client_feats = [
        "client_id", "age", "age_group", "genre",
        "revenu_annuel", "income_band",
        "score_credit", "credit_band",
        "nb_retards_paiement", "nb_infractions_majeures", "points_permis_retires",
        "driving_risk_score", "financial_stress_score", "responsible_behavior_score",
        "risque_comportemental", "risque_rse", "risque_financier", "risque_fraude", "risque_global",
        "changement_frequent_assureur"
    ]
    client_feats = [c for c in client_feats if c in clean_clients.columns]
    base = base.merge(clean_clients[client_feats].drop_duplicates(subset=["client_id"]),
                      on="client_id", how="left")

    # Vehicle features
    vehicle_feats = [
        "vehicle_id", "type_vehicule", "marque", "modele", "usage_vehicule",
        "vehicle_age", "valeur_vehicule", "vehicle_value_band",
        "kilometrage_actuel", "mileage_per_year", "puissance_fiscale"
    ]
    vehicle_feats = [c for c in vehicle_feats if c in clean_vehicles.columns]
    base = base.merge(clean_vehicles[vehicle_feats].drop_duplicates(subset=["vehicle_id"]),
                      on="vehicle_id", how="left")

    # Premium-to-value ratio
    if "prime_assurance_annuelle" in base.columns and "valeur_vehicule" in base.columns:
        denom = base["valeur_vehicule"].replace(0, np.nan)
        base["premium_to_value_ratio"] = base["prime_assurance_annuelle"] / denom
        base["premium_to_value_ratio"] = clip_outliers(base["premium_to_value_ratio"])

    # Date features
    if "date_sinistre_claim" in base.columns:
        dt = pd.to_datetime(base["date_sinistre_claim"], errors="coerce")
        base["claim_year"] = dt.dt.year
        base["claim_month"] = dt.dt.month
        base["claim_quarter"] = dt.dt.quarter
        base["claim_dayofweek"] = dt.dt.dayofweek

    ml_fraud = base.copy()
    if "est_frauduleux_claim" in ml_fraud.columns:
        ml_fraud = ml_fraud[ml_fraud["est_frauduleux_claim"].notna()]
    ml_fraud = ml_fraud.drop(columns=["description_sinistre_claim"], errors="ignore")

    ml_sev = base.copy()
    if "claim_severity_bucket" in ml_sev.columns:
        ml_sev = ml_sev[ml_sev["claim_severity_bucket"].notna()]
    ml_sev = ml_sev.drop(columns=["description_sinistre_claim"], errors="ignore")

    return ml_fraud.reset_index(drop=True), ml_sev.reset_index(drop=True)
