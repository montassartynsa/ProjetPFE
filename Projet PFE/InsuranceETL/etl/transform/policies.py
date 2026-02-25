import numpy as np
import pandas as pd
from ..common import (
    normalize_columns, require_column, normalize_id,
    safe_to_numeric, safe_to_datetime, clean_text, clip_outliers
)

def transform_policies(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)

    df = require_column(df, ["contract_id", "contractid", "id_contract", "contrat_id"], "contract_id")
    df = require_column(df, ["client_id", "clientid", "id_client", "client"], "client_id")

    df["contract_id"] = normalize_id(df["contract_id"])
    df["client_id"] = normalize_id(df["client_id"])

    df["date_debut_contrat"] = safe_to_datetime(df.get("date_debut_contrat", pd.Series([pd.NA]*len(df))))
    df["date_fin_contrat"] = safe_to_datetime(df.get("date_fin_contrat", pd.Series([pd.NA]*len(df))))

    if "type_couverture" in df.columns:
        df["type_couverture"] = clean_text(df["type_couverture"], mode="title")

    for c in ["prime_assurance_annuelle", "nb_sinistres_precedents", "delai_souscription_sinistre_jours"]:
        if c in df.columns:
            df[c] = safe_to_numeric(df[c])

    if "prime_assurance_annuelle" in df.columns:
        df.loc[df["prime_assurance_annuelle"] < 0, "prime_assurance_annuelle"] = np.nan
        df["prime_assurance_annuelle"] = clip_outliers(df["prime_assurance_annuelle"])

    if "nb_sinistres_precedents" in df.columns:
        df.loc[df["nb_sinistres_precedents"] < 0, "nb_sinistres_precedents"] = 0

    if "delai_souscription_sinistre_jours" in df.columns:
        df.loc[df["delai_souscription_sinistre_jours"] < 0, "delai_souscription_sinistre_jours"] = np.nan
        df["delai_souscription_sinistre_jours"] = clip_outliers(df["delai_souscription_sinistre_jours"])

    df["policy_duration_days"] = (df["date_fin_contrat"] - df["date_debut_contrat"]).dt.days
    df.loc[df["policy_duration_days"] < 0, "policy_duration_days"] = np.nan

    df["policy_tenure_bucket"] = pd.cut(
        df["policy_duration_days"],
        bins=[-np.inf, 180, 365, 730, 1095, np.inf],
        labels=["<6m", "6-12m", "1-2y", "2-3y", "3y+"]
    )

    df = df.drop_duplicates(subset=["contract_id"])
    df = df[df["contract_id"].notna()]

    return df.reset_index(drop=True)
