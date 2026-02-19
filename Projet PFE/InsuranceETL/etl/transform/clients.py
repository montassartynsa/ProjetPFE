import re
import numpy as np
import pandas as pd

from ..common import (
    normalize_columns,
    require_column,
    normalize_id,
    clean_text,
    safe_to_bool,
    clip_outliers,
)

# Tokens we want to interpret as missing
_MISSING_TOKENS = {
    "", "nan", "none", "null", "n/a", "na",
    "non renseigne", "non renseigné", "non renseignee", "non renseignée",
    "-", "--", "?", "inconnu", "inconnue"
}

_TRUE_TOKENS = {"1", "true", "vrai", "yes", "oui", "y", "t"}
_FALSE_TOKENS = {"0", "false", "faux", "no", "non", "n", "f"}


def _normalize_missing(s: pd.Series) -> pd.Series:
    """Turn common missing tokens into pd.NA (string-safe)."""
    x = s.astype("string").str.strip()
    xl = x.str.lower()
    return x.mask(xl.isin(_MISSING_TOKENS), pd.NA)


def _to_numeric_flexible(s: pd.Series) -> pd.Series:
    """
    Robust numeric conversion:
    - handles commas as decimal separators (12,5 -> 12.5)
    - strips spaces
    - maps yes/no/true/false to 1/0
    - extracts numbers from messy strings if needed
    - returns float64 (NaN for missing)
    """
    if s is None:
        return pd.Series(dtype="float64")

    x = _normalize_missing(s)

    # map boolean-like tokens
    xl = x.astype("string").str.lower()
    mapped = pd.Series(np.nan, index=x.index, dtype="float64")
    mapped[xl.isin(_TRUE_TOKENS)] = 1.0
    mapped[xl.isin(_FALSE_TOKENS)] = 0.0

    # where we didn't map -> try numeric parse
    # normalize decimal commas and remove spaces
    raw = x.astype("string").str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)

    # if strings contain extra text, extract first numeric chunk
    # examples: "DT: 2" -> "2", "≈3.5" -> "3.5"
    extracted = raw.str.extract(r"([-+]?\d*\.?\d+)", expand=False)

    parsed = pd.to_numeric(extracted, errors="coerce").astype("float64")

    # combine (mapped booleans take precedence)
    out = pd.Series(parsed, index=x.index, dtype="float64")
    out.loc[~np.isnan(mapped)] = mapped.loc[~np.isnan(mapped)]
    return out


def _to_int_count(s: pd.Series, *, allow_negative: bool = False) -> pd.Series:
    """
    Numeric count column -> Int64 (nullable integer).
    Negative values are set to NA unless allow_negative=True.
    """
    x = _to_numeric_flexible(s)
    # round counts sensibly
    x = np.floor(x)  # counts should be integers
    if not allow_negative:
        x = x.mask(x < 0, np.nan)
    return pd.Series(x, index=x.index).astype("Int64")


def transform_clients(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)

    # ---- REQUIRED PK ----
    df = require_column(df, ["client_id", "clientid", "id_client", "client"], "client_id")
    df["client_id"] = normalize_id(df["client_id"])

    # Drop rows where PK is missing (this is the main row-deletion rule)
    df = df[df["client_id"].notna()].copy()

    # ---- TEXT CLEAN ----
    for c in ["nom", "prenom", "gouvernorat", "ville", "niveau_professionnel"]:
        if c in df.columns:
            df[c] = clean_text(_normalize_missing(df[c]), mode="title")

    # ---- GENRE NORMALIZATION (M/F) ----
    if "genre" in df.columns:
        g = clean_text(_normalize_missing(df["genre"]), mode="upper")

        # Normalize typical French values
        g = g.replace({
            "MASCULIN": "M",
            "HOMME": "M",
            "MALE": "M",
            "M": "M",
            "FEMININ": "F",
            "FÉMININ": "F",
            "FEMME": "F",
            "FEMALE": "F",
            "F": "F",
        })
        # Anything else -> NA
        g = g.where(g.isin(["M", "F"]), pd.NA)
        df["genre"] = g

    # ---- NUMERIC COLUMNS (robust parsing) ----
    if "age" in df.columns:
        df["age"] = _to_numeric_flexible(df["age"])
        df.loc[(df["age"] < 16) | (df["age"] > 100), "age"] = np.nan

    if "revenu_annuel" in df.columns:
        df["revenu_annuel"] = _to_numeric_flexible(df["revenu_annuel"])
        df.loc[df["revenu_annuel"] < 0, "revenu_annuel"] = np.nan
        df["revenu_annuel"] = clip_outliers(df["revenu_annuel"])

    if "score_credit" in df.columns:
        df["score_credit"] = _to_numeric_flexible(df["score_credit"])
        df.loc[(df["score_credit"] < 0) | (df["score_credit"] > 1000), "score_credit"] = np.nan
        df["score_credit"] = clip_outliers(df["score_credit"])

    # counts (these are the ones that were becoming all NULL in your previous run)
    for c in [
        "nb_retards_paiement",
        "nb_infractions_majeures",
        "points_permis_retires",
        "incidents_paiement_assureur",
        "historique_dettes",
        "nombre_enfants",
        "num_contracts_target",
    ]:
        if c in df.columns:
            df[c] = _to_int_count(df[c], allow_negative=False)

    # risk scores (float)
    for c in ["risque_comportemental", "risque_rse", "risque_financier", "risque_fraude", "risque_global"]:
        if c in df.columns:
            df[c] = _to_numeric_flexible(df[c])
            df[c] = clip_outliers(df[c])

    # ---- BOOLEAN COLUMNS ----
    for c in [
        "participe_conduite_responsable",
        "entretien_regulier",
        "vehicule_peu_polluant",
        "engagement_securite_routiere",
        "changement_frequent_assureur",
    ]:
        if c in df.columns:
            df[c] = safe_to_bool(df[c]).astype("Int64")  # store as 0/1/NULL

    # ---- CATEGORICAL: driving_behavior MUST stay text ----
    # Your screenshot shows values like "Prudent / Agressif / Normal"
    if "driving_behavior" in df.columns:
        x = clean_text(_normalize_missing(df["driving_behavior"]), mode="title")
        # normalize variants/spelling
        x = x.replace({
            "Aggressive": "Aggressif",
            "Agressif": "Aggressif",
            "Aggressif": "Aggressif",
            "Prudent": "Prudent",
            "Normal": "Normal",
        })
        df["driving_behavior"] = x.where(x.isin(["Prudent", "Normal", "Aggressif"]), pd.NA)

    # ---- ENGINEERED FEATURES ----
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 24, 34, 44, 54, 64, 120],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
            right=True
        )

    if "revenu_annuel" in df.columns and df["revenu_annuel"].notna().any():
        q = df["revenu_annuel"].quantile([0.2, 0.4, 0.6, 0.8]).to_list()
        df["income_band"] = pd.cut(
            df["revenu_annuel"],
            bins=[-np.inf] + q + [np.inf],
            labels=["very_low", "low", "mid", "high", "very_high"]
        )

    if "score_credit" in df.columns:
        df["credit_band"] = pd.cut(
            df["score_credit"],
            bins=[-np.inf, 450, 600, 750, np.inf],
            labels=["poor", "fair", "good", "excellent"]
        )

    # driving_risk_score
    driving_components = []
    if "nb_infractions_majeures" in df.columns:
        driving_components.append(df["nb_infractions_majeures"].fillna(0).astype("float64"))
    if "points_permis_retires" in df.columns:
        driving_components.append(df["points_permis_retires"].fillna(0).astype("float64") / 10.0)
    df["driving_risk_score"] = np.clip(sum(driving_components), 0, None) if driving_components else np.nan

    # financial_stress_score
    fin_components = []
    for c in ["nb_retards_paiement", "incidents_paiement_assureur", "historique_dettes"]:
        if c in df.columns:
            fin_components.append(df[c].astype("float64").fillna(0))
    df["financial_stress_score"] = np.clip(sum(fin_components), 0, None) if fin_components else np.nan

    # responsible_behavior_score: sum of boolean flags (0/1)
    bonus = 0
    for c in ["participe_conduite_responsable", "entretien_regulier", "engagement_securite_routiere"]:
        if c in df.columns:
            bonus += df[c].fillna(0).astype(int)
    df["responsible_behavior_score"] = bonus

    # ---- DEDUPLICATION ----
    df = df.drop_duplicates(subset=["client_id"], keep="first").reset_index(drop=True)

    # ---- DROP UNNECESSARY COLUMNS (WHITELIST) ----
    # Keep only what we actually need downstream
    keep = [
        "client_id",
        "nom", "prenom", "genre", "age", "age_group",
        "niveau_professionnel", "revenu_annuel", "income_band",
        "score_credit", "credit_band",
        "gouvernorat", "ville",
        "nb_retards_paiement", "nb_infractions_majeures", "points_permis_retires",
        "participe_conduite_responsable", "entretien_regulier", "vehicule_peu_polluant",
        "engagement_securite_routiere", "changement_frequent_assureur",
        "incidents_paiement_assureur", "historique_dettes", "nombre_enfants",
        "driving_behavior", "num_contracts_target",
        "risque_comportemental", "risque_rse", "risque_financier", "risque_fraude", "risque_global",
        "driving_risk_score", "financial_stress_score", "responsible_behavior_score",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    return df
