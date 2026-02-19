import re
import numpy as np
import pandas as pd

_ACCENT_MAP = {
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "à": "a", "â": "a",
    "î": "i", "ï": "i",
    "ô": "o",
    "ù": "u", "û": "u", "ü": "u",
    "ç": "c",
    "É": "E", "È": "E", "Ê": "E", "Ë": "E",
    "À": "A", "Â": "A",
    "Î": "I", "Ï": "I",
    "Ô": "O",
    "Ù": "U", "Û": "U", "Ü": "U",
    "Ç": "C",
}

def _remove_accents(s: str) -> str:
    for k, v in _ACCENT_MAP.items():
        s = s.replace(k, v)
    return s

def to_snake_case(col: str) -> str:
    col = str(col).strip()
    col = _remove_accents(col)
    col = col.replace("/", "_")
    col = re.sub(r"[^\w\s-]", "", col)
    col = col.replace(" ", "_")
    col = re.sub(r"[-]+", "_", col)
    col = re.sub(r"__+", "_", col)
    return col.lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    return df

def normalize_id(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    x = x.replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA})
    return x

def safe_to_numeric(s: pd.Series) -> pd.Series:
    # Convert common comma decimals, strip spaces
    s2 = s.astype("string").str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    out = pd.to_numeric(s2, errors="coerce")
    # Ensure missing is NaN (NOT pd.NA)
    return out.astype("float64")

def safe_to_bool(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "vrai", "yes", "oui", "y", "t"}
    false_set = {"0", "false", "faux", "no", "non", "n", "f"}
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[x.isin(true_set)] = True
    out[x.isin(false_set)] = False
    num = pd.to_numeric(s, errors="coerce")
    out[num == 1] = True
    out[num == 0] = False
    return out

def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def clip_outliers(series: pd.Series, low_q=0.01, high_q=0.99) -> pd.Series:
    x = series.copy()
    if x.dropna().empty:
        return x
    lo = x.quantile(low_q)
    hi = x.quantile(high_q)
    return x.clip(lower=lo, upper=hi)

def clean_text(s: pd.Series, mode: str = "title") -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    if mode == "title":
        return x.str.lower().str.title()
    if mode == "upper":
        return x.str.upper()
    if mode == "lower":
        return x.str.lower()
    return x

def require_column(df: pd.DataFrame, candidates: list, canonical: str) -> pd.DataFrame:
    """
    Ensures df contains `canonical`.
    If one candidate exists, rename it to canonical.
    """
    df = df.copy()
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            if c != canonical:
                df = df.rename(columns={c: canonical})
            return df
    raise KeyError(f"Missing required column '{canonical}'. Tried: {candidates}. Available: {list(df.columns)[:30]}")
