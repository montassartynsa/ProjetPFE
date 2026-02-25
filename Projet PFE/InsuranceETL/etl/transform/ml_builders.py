"""import numpy as np
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
"""

# etl/transform/ml_builders.py
import numpy as np
import pandas as pd
from ..common import clip_outliers

def build_ml_claim_dataset(clean_clients, clean_policies, clean_vehicles, clean_claims):
    base = clean_claims.copy()

    # -------------------------
    # Target for delay model
    # -------------------------
    # Keep ALL claims (active and closed) for the ML dataset
    # Active claims will have is_delayed=NA and can be scored with predictions
    # Closed claims with delay labels can be used for model training
    if "is_delayed" not in base.columns:
        base["is_delayed"] = pd.NA
    
    # Convert to int where available, keep NA for active claims
    base["is_delayed"] = base["is_delayed"].astype("Int64")  # Nullable integer type

    # -------------------------
    # Claim status (for filtering active vs closed claims)
    # -------------------------
    # Keep status column for prediction filtering
    # Active claims: Ouvert, En_cours, En_cours_d_expertise
    # Closed claims: Refusé, Clos, Clos_avec_indemnisation, Clos_sans_indemnisation
    if "statut_sinistre_claim" not in base.columns:
        base["statut_sinistre_claim"] = "Unknown"

    # -------------------------
    # Merge policy features
    # -------------------------
    policy_feats = [
        "contract_id",
        "type_couverture",
        "prime_assurance_annuelle",
        "nb_sinistres_precedents",
        "delai_souscription_sinistre_jours",
        "policy_duration_days",
        "policy_tenure_bucket",
        "date_debut_contrat",
    ]
    policy_feats = [c for c in policy_feats if c in clean_policies.columns]
    if policy_feats:
        base = base.merge(
            clean_policies[policy_feats].drop_duplicates(subset=["contract_id"]),
            on="contract_id", how="left"
        )

    # -------------------------
    # Merge client features
    # -------------------------
    client_feats = [
        "client_id", "age", "age_group", "genre",
        "revenu_annuel", "income_band",
        "score_credit", "credit_band",
        "nb_retards_paiement", "nb_infractions_majeures", "points_permis_retires",
        "driving_risk_score", "financial_stress_score", "responsible_behavior_score",
        "risque_comportemental", "risque_rse", "risque_financier", "risque_fraude", "risque_global",
        "changement_frequent_assureur",
        "nombre_enfants",
    ]
    client_feats = [c for c in client_feats if c in clean_clients.columns]
    if client_feats:
        base = base.merge(
            clean_clients[client_feats].drop_duplicates(subset=["client_id"]),
            on="client_id", how="left"
        )

    # -------------------------
    # Merge vehicle features
    # -------------------------
    vehicle_feats = [
        "vehicle_id", "type_vehicule", "marque", "modele", "usage_vehicule",
        "vehicle_age", "valeur_vehicule", "vehicle_value_band",
        "kilometrage_actuel", "mileage_per_year", "puissance_fiscale",
    ]
    vehicle_feats = [c for c in vehicle_feats if c in clean_vehicles.columns]
    if vehicle_feats:
        base = base.merge(
            clean_vehicles[vehicle_feats].drop_duplicates(subset=["vehicle_id"]),
            on="vehicle_id", how="left"
        )

    # -------------------------
    # Claim-level features
    # -------------------------
    # premium-to-value
    if "prime_assurance_annuelle" in base.columns and "valeur_vehicule" in base.columns:
        denom = base["valeur_vehicule"].replace(0, np.nan)
        base["premium_to_value_ratio"] = base["prime_assurance_annuelle"] / denom
        base["premium_to_value_ratio"] = clip_outliers(base["premium_to_value_ratio"])

    # Date features from claim opening datetime
    if "date_sinistre_claim" in base.columns:
        dt = pd.to_datetime(base["date_sinistre_claim"], errors="coerce")
        base["claim_year"] = dt.dt.year
        base["claim_month"] = dt.dt.month
        base["claim_quarter"] = dt.dt.quarter
        base["claim_dayofweek"] = dt.dt.dayofweek
        base["claim_hour"] = dt.dt.hour

    # -------------------------
    # LEAKAGE CONTROL
    # -------------------------
    # If you want to predict delay at claim creation time,
    # you must drop anything that directly encodes closure/processing duration.
    predict_at_opening = True
    if predict_at_opening:
        leakage_cols = [
            "date_cloture_claim",
            "duree_traitement_jours",
            "duree_traitement_heures",
        ]
        base = base.drop(columns=[c for c in leakage_cols if c in base.columns], errors="ignore")

    # Build unified ML dataset with all features and all three labels
    # Models can filter by non-null labels during training
    ml_claim = base.copy()
    ml_claim = ml_claim.drop(columns=["description_sinistre_claim"], errors="ignore")

    # Ensure critical columns for delay prediction are present
    critical_cols = ["sla_jours", "is_delayed"]
    missing = [col for col in critical_cols if col not in ml_claim.columns]
    if missing:
        print(f"[WARNING] ML dataset missing critical columns: {missing}")
        print(f"[WARNING] Available columns: {list(ml_claim.columns)}")

    return ml_claim.reset_index(drop=True)