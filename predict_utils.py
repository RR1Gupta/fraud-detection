"""Small helper functions used by the web app (kept separate so they can be unit-tested)."""
import numpy as np
import pandas as pd

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_ROWS = 50_000


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Check an uploaded table and return only the 30 model columns. Raises ValueError on problems."""
    if df is None or df.empty:
        raise ValueError("The file is empty.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Too many rows ({len(df)}). Please upload at most {MAX_ROWS} transactions.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    data = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any():
        raise ValueError("Some values are not numbers or are blank. Please clean the file and try again.")
    return data


def predict_fraud(df: pd.DataFrame, model, threshold: float = 0.5) -> pd.DataFrame:
    """Return a copy of the data with fraud_probability and prediction columns."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    data = validate_dataframe(df)
    proba = model.predict_proba(data)[:, 1]
    out = data.copy()
    out["fraud_probability"] = proba.round(4)
    out["prediction"] = np.where(proba >= threshold, "FRAUD", "Legitimate")
    return out
