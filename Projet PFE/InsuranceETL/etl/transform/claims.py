import numpy as np
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
