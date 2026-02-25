"""import numpy as np
import pandas as pd
from ..common import (
    normalize_columns, require_column, normalize_id,
    safe_to_numeric, safe_to_datetime, safe_to_bool,
    clean_text, clip_outliers
)

def transform_claims(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)

    df = require_column(df, ["claim_id", "claimid", "id_claim", "sinistre_id"], "claim_id")
    df = require_column(df, ["contract_id", "contractid", "id_contract", "contrat_id"], "contract_id")
    df = require_column(df, ["client_id", "clientid", "id_client", "client"], "client_id")
    df = require_column(df, ["vehicle_id", "vehicleid", "id_vehicle", "vehicule_id"], "vehicle_id")

    df["claim_id"] = normalize_id(df["claim_id"])
    df["contract_id"] = normalize_id(df["contract_id"])
    df["client_id"] = normalize_id(df["client_id"])
    df["vehicle_id"] = normalize_id(df["vehicle_id"])

    df["date_sinistre_claim"] = safe_to_datetime(df.get("date_sinistre_claim", pd.Series([pd.NA]*len(df))))

    if "type_sinistre_claim" in df.columns:
        df["type_sinistre_claim"] = clean_text(df["type_sinistre_claim"], mode="title")
    if "statut_sinistre_claim" in df.columns:
        df["statut_sinistre_claim"] = clean_text(df["statut_sinistre_claim"], mode="title")

    if "description_sinistre_claim" in df.columns:
        desc = df["description_sinistre_claim"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df["description_sinistre_claim"] = desc.replace({"nan": pd.NA})

    for c in ["montant_estime_dommage_claim", "montant_indemnisation_claim"]:
        if c in df.columns:
            df[c] = safe_to_numeric(df[c])

    if "est_frauduleux_claim" in df.columns:
        df["est_frauduleux_claim"] = safe_to_bool(df["est_frauduleux_claim"]).astype("Int64")

    for c in ["incoherence_dommages", "nature_sinistre_consistante"]:
        if c in df.columns:
            df[c] = safe_to_bool(df[c])

    if "montant_estime_dommage_claim" in df.columns:
        df.loc[df["montant_estime_dommage_claim"] < 0, "montant_estime_dommage_claim"] = np.nan
        df["montant_estime_dommage_claim"] = clip_outliers(df["montant_estime_dommage_claim"])

    if "montant_indemnisation_claim" in df.columns:
        df.loc[df["montant_indemnisation_claim"] < 0, "montant_indemnisation_claim"] = np.nan
        df["montant_indemnisation_claim"] = clip_outliers(df["montant_indemnisation_claim"])

    if "montant_estime_dommage_claim" in df.columns and "montant_indemnisation_claim" in df.columns:
        df["claim_gap_amount"] = df["montant_indemnisation_claim"] - df["montant_estime_dommage_claim"]
        denom = df["montant_estime_dommage_claim"].replace(0, np.nan)
        df["claim_gap_ratio"] = df["claim_gap_amount"] / denom
        df["claim_gap_ratio"] = clip_outliers(df["claim_gap_ratio"])

    if "montant_indemnisation_claim" in df.columns:
        q = df["montant_indemnisation_claim"].quantile([0.5, 0.8, 0.95]).to_list()
        df["claim_severity_bucket"] = pd.cut(
            df["montant_indemnisation_claim"],
            bins=[-np.inf] + q + [np.inf],
            labels=["low", "medium", "high", "very_high"]
        )

    df = df.drop_duplicates(subset=["claim_id"])
    df = df[df["claim_id"].notna()]

    return df.reset_index(drop=True)
"""
# etl/transform/claims.py
import numpy as np
import pandas as pd

from ..common import (
    normalize_columns, require_column, normalize_id,
    safe_to_numeric, safe_to_datetime, safe_to_bool,
    clean_text, clip_outliers
)

_FINAL_STATUSES = {
    "refuse",
    "clos_avec_indemnisation",
    "clos_sans_indemnisation",
    "clos avec indemnisation",
    "clos sans indemnisation",
    "clos",
}

def _std_status(s: pd.Series) -> pd.Series:
    # normalize variations -> canonical-ish French labels used in your dataset
    x = clean_text(s, mode="lower").astype("string")

    x = x.str.replace(r"\s+", "_", regex=True)
    x = x.str.replace("é", "e").str.replace("è", "e").str.replace("ê", "e")
    x = x.str.replace("à", "a").str.replace("ç", "c")

    mapping = {
        "refuse": "Refusé",
        "refusee": "Refusé",
        "rejet": "Refusé",
        "rejetee": "Refusé",

        "clos": "Clos",
        "cloture": "Clos",
        "cloture_": "Clos",

        "clos_avec_indemnisation": "Clos_avec_indemnisation",
        "clos_avec_indemnisation_": "Clos_avec_indemnisation",
        "clos_avec_indemnisation.": "Clos_avec_indemnisation",
        "clos_avec_indemnisation__": "Clos_avec_indemnisation",

        "clos_sans_indemnisation": "Clos_sans_indemnisation",
        "clos_sans_indemnisation_": "Clos_sans_indemnisation",

        "ouvert": "Ouvert",
        "en_cours": "En_cours",
        "en_cours_dexpertise": "En_cours_d_expertise",
        "en_cours_expertise": "En_cours_d_expertise",
        "en_cours_d_expertise": "En_cours_d_expertise",
    }

    out = x.map(mapping).fillna(clean_text(s, mode="title"))
    return out


def transform_claims(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)

    # Required IDs
    df = require_column(df, ["claim_id", "claimid", "id_claim", "sinistre_id"], "claim_id")
    df = require_column(df, ["contract_id", "contractid", "id_contract", "contrat_id"], "contract_id")
    df = require_column(df, ["client_id", "clientid", "id_client", "client"], "client_id")
    df = require_column(df, ["vehicle_id", "vehicleid", "id_vehicle", "vehicule_id"], "vehicle_id")

    df["claim_id"] = normalize_id(df["claim_id"])
    df["contract_id"] = normalize_id(df["contract_id"])
    df["client_id"] = normalize_id(df["client_id"])
    df["vehicle_id"] = normalize_id(df["vehicle_id"])

    # Datetime (now includes hours)
    if "date_sinistre_claim" in df.columns:
        df["date_sinistre_claim"] = safe_to_datetime(df["date_sinistre_claim"])
    else:
        df["date_sinistre_claim"] = pd.NaT

    # NEW: closing datetime
    if "date_cloture_claim" in df.columns:
        df["date_cloture_claim"] = safe_to_datetime(df["date_cloture_claim"])
    else:
        df["date_cloture_claim"] = pd.NaT

    # Text columns
    if "type_sinistre_claim" in df.columns:
        df["type_sinistre_claim"] = clean_text(df["type_sinistre_claim"], mode="title")

    if "statut_sinistre_claim" in df.columns:
        df["statut_sinistre_claim"] = _std_status(df["statut_sinistre_claim"])

    if "description_sinistre_claim" in df.columns:
        desc = (
            df["description_sinistre_claim"]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["description_sinistre_claim"] = desc.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    # Money columns
    for c in ["montant_estime_dommage_claim", "montant_indemnisation_claim"]:
        if c in df.columns:
            df[c] = safe_to_numeric(df[c])
            df.loc[df[c] < 0, c] = np.nan
            df[c] = clip_outliers(df[c])

    # Fraud + flags
    if "est_frauduleux_claim" in df.columns:
        df["est_frauduleux_claim"] = safe_to_bool(df["est_frauduleux_claim"]).astype("Int64")

    for c in ["incoherence_dommages", "nature_sinistre_consistante"]:
        if c in df.columns:
            df[c] = safe_to_bool(df[c]).astype("boolean")

    # NEW: SLA + durations + delayed
    if "sla_jours" in df.columns:
        df["sla_jours"] = safe_to_numeric(df["sla_jours"])
        df.loc[df["sla_jours"] < 0, "sla_jours"] = np.nan
    else:
        df["sla_jours"] = np.nan

    # If provided durations exist, parse them
    if "duree_traitement_jours" in df.columns:
        df["duree_traitement_jours"] = safe_to_numeric(df["duree_traitement_jours"])
        df.loc[df["duree_traitement_jours"] < 0, "duree_traitement_jours"] = np.nan
    else:
        df["duree_traitement_jours"] = np.nan

    if "duree_traitement_heures" in df.columns:
        df["duree_traitement_heures"] = safe_to_numeric(df["duree_traitement_heures"])
        df.loc[df["duree_traitement_heures"] < 0, "duree_traitement_heures"] = np.nan
    else:
        df["duree_traitement_heures"] = np.nan

    # Recompute duration from dates when possible (source of truth)
    # Only when both datetimes exist and closure >= sinistre
    mask_dt = df["date_sinistre_claim"].notna() & df["date_cloture_claim"].notna()
    if mask_dt.any():
        delta_hours = (df.loc[mask_dt, "date_cloture_claim"] - df.loc[mask_dt, "date_sinistre_claim"]).dt.total_seconds() / 3600.0
        delta_hours = delta_hours.where(delta_hours >= 0, np.nan)

        df.loc[mask_dt, "duree_traitement_heures"] = delta_hours
        df.loc[mask_dt, "duree_traitement_jours"] = delta_hours / 24.0

    # NEW: delayed label
    if "is_delayed" in df.columns:
        df["is_delayed"] = safe_to_bool(df["is_delayed"]).astype("Int64")  # 1/0/NA
    else:
        # derive: duration_days > SLA
        if "duree_traitement_jours" in df.columns and "sla_jours" in df.columns:
            df["is_delayed"] = (df["duree_traitement_jours"] > df["sla_jours"]).astype("Int64")
        else:
            df["is_delayed"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # Extra engineered features useful for ML (kept in clean_claims too)
    if "montant_estime_dommage_claim" in df.columns and "montant_indemnisation_claim" in df.columns:
        df["claim_gap_amount"] = df["montant_indemnisation_claim"] - df["montant_estime_dommage_claim"]
        denom = df["montant_estime_dommage_claim"].replace(0, np.nan)
        df["claim_gap_ratio"] = df["claim_gap_amount"] / denom
        df["claim_gap_ratio"] = clip_outliers(df["claim_gap_ratio"])

    if "montant_indemnisation_claim" in df.columns:
        q = df["montant_indemnisation_claim"].quantile([0.5, 0.8, 0.95]).to_list()
        df["claim_severity_bucket"] = pd.cut(
            df["montant_indemnisation_claim"],
            bins=[-np.inf] + q + [np.inf],
            labels=["low", "medium", "high", "very_high"]
        )

    # Drop duplicates / invalid ids
    df = df[df["claim_id"].notna()].copy()
    df = df.drop_duplicates(subset=["claim_id"], keep="first")

    return df.reset_index(drop=True)