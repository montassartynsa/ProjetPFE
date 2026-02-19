import numpy as np
import pandas as pd

from ..common import (
    normalize_columns, require_column, normalize_id,
    safe_to_numeric, clean_text, clip_outliers
)

# Practical safety bounds to avoid SQL overflow (adjust if your SQL types are different)
_MIN_YEAR = 1970
_MAX_KM = 3_000_000          # very large but still realistic
_MAX_VALUE = 50_000_000      # vehicle value upper bound
_MAX_POWER = 200             # fiscal power upper bound
_MAX_MPY = 300_000           # mileage per year upper bound


def transform_vehicles(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)

    # Required IDs
    df = require_column(df, ["vehicle_id", "vehicleid", "id_vehicle", "vehicule_id"], "vehicle_id")
    df = require_column(df, ["contract_id", "contractid", "id_contract", "contrat_id"], "contract_id")

    df["vehicle_id"] = normalize_id(df["vehicle_id"])
    df["contract_id"] = normalize_id(df["contract_id"])

    # Text cleanup
    for c in ["type_vehicule", "marque", "modele", "usage_vehicule"]:
        if c in df.columns:
            df[c] = clean_text(df[c], mode="title")

    if "immatriculation" in df.columns:
        df["immatriculation"] = (
            clean_text(df["immatriculation"], mode="upper")
            .astype("string")
            .str.replace(" ", "", regex=False)
        )

    # Numeric cleanup
    for c in ["annee_modele", "puissance_fiscale", "kilometrage_actuel", "valeur_vehicule"]:
        if c in df.columns:
            df[c] = safe_to_numeric(df[c])

    # Year + age
    current_year = pd.Timestamp.today().year
    if "annee_modele" in df.columns:
        df.loc[(df["annee_modele"] < _MIN_YEAR) | (df["annee_modele"] > current_year + 1), "annee_modele"] = np.nan

        df["vehicle_age"] = current_year - df["annee_modele"]
        df.loc[(df["vehicle_age"] < 0) | (df["vehicle_age"] > 100), "vehicle_age"] = np.nan
    else:
        df["vehicle_age"] = np.nan

    # puissance_fiscale bounds
    if "puissance_fiscale" in df.columns:
        df.loc[(df["puissance_fiscale"] < 0) | (df["puissance_fiscale"] > _MAX_POWER), "puissance_fiscale"] = np.nan
        df["puissance_fiscale"] = clip_outliers(df["puissance_fiscale"])

    # kilometrage bounds
    if "kilometrage_actuel" in df.columns:
        df.loc[(df["kilometrage_actuel"] < 0) | (df["kilometrage_actuel"] > _MAX_KM), "kilometrage_actuel"] = np.nan
        df["kilometrage_actuel"] = clip_outliers(df["kilometrage_actuel"])

    # value bounds
    if "valeur_vehicule" in df.columns:
        df.loc[(df["valeur_vehicule"] < 0) | (df["valeur_vehicule"] > _MAX_VALUE), "valeur_vehicule"] = np.nan
        df["valeur_vehicule"] = clip_outliers(df["valeur_vehicule"])

    # Mileage per year
    if "kilometrage_actuel" in df.columns:
        denom = df["vehicle_age"].replace(0, np.nan)
        df["mileage_per_year"] = df["kilometrage_actuel"] / denom
        df.loc[(df["mileage_per_year"] < 0) | (df["mileage_per_year"] > _MAX_MPY), "mileage_per_year"] = np.nan
        df["mileage_per_year"] = clip_outliers(df["mileage_per_year"])
    else:
        df["mileage_per_year"] = np.nan

    # Value bands
    if "valeur_vehicule" in df.columns:
        # If all NaN -> keep band NaN
        if df["valeur_vehicule"].notna().any():
            q = df["valeur_vehicule"].quantile([0.2, 0.4, 0.6, 0.8]).to_list()
            df["vehicle_value_band"] = pd.cut(
                df["valeur_vehicule"],
                bins=[-np.inf] + q + [np.inf],
                labels=["very_low", "low", "mid", "high", "very_high"]
            )
        else:
            df["vehicle_value_band"] = pd.Series([pd.NA] * len(df), dtype="string")

    # IMPORTANT: remove duplicates BEFORE load to avoid PK duplicates inside the same batch
    df = df[df["vehicle_id"].notna()].copy()
    df = df.drop_duplicates(subset=["vehicle_id"], keep="first")

    # Ensure we never output pd.NA in numeric cols (keep NaN)
    for c in ["annee_modele", "puissance_fiscale", "kilometrage_actuel", "valeur_vehicule", "vehicle_age", "mileage_per_year"]:
        if c in df.columns:
            df[c] = df[c].astype("float64")

    return df.reset_index(drop=True)
