"""
feature_engineering.py
----------------------
Chooses the clustering features, encodes categories and scales numbers.

Design rules (each one is a viva answer):
* Only columns that really exist in the data are used (detected dynamically).
* Raw date_of_birth is NOT used, only the derived `age`.
* Numeric columns -> StandardScaler (K-Means / Ward use Euclidean distance, so a column
  measured in dollars would otherwise dominate a column measured in years).
* Binary categories (loan, purpose, client type) -> label encoding to 0/1 (OrdinalEncoder).
  A 0/1 code cannot create a fake order when there are only two categories.
* Multi-class nominal categories (country, referral channel) -> One-Hot Encoding.
  Label encoding would invent an order (e.g. Canada < France < UK) that does not exist.
* Everything lives in one sklearn ColumnTransformer, so the SAME transformation can be
  re-applied to new clients inside the dashboard (no copy-pasted preprocessing code).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Candidate features with a plain-English reason (used in README, report, viva)
# ---------------------------------------------------------------------------
NUMERIC_CANDIDATES: Dict[str, str] = {
    "age": "Life-stage of the buyer. Derived from date_of_birth (age is meaningful, a raw date is not).",
    "satisfaction_score": "Customer experience (1-5). Separates happy from unhappy buyers.",
    "n_units": "Portfolio size: how many units the client bought. Distinguishes bulk buyers from one-off buyers.",
    "avg_price": "Average price paid per unit (USD) = ticket size / budget level.",
    "office_share": "Share of the client's units that are Offices (0-1). Commercial vs residential appetite.",
}
BINARY_CANDIDATES: Dict[str, str] = {
    "acquisition_purpose": "Investment vs Home. The core investment-motivation signal requested in the PRD.",
    "loan_applied": "Financing dependence (Yes/No).",
    "client_type": "Individual vs Company.",
}
NOMINAL_CANDIDATES: Dict[str, str] = {
    "country": "Geographic market. Region is nested inside country, so country is used (see EXCLUDED).",
    "referral_channel": "Acquisition channel (Website / Agency / Client).",
}
EXCLUDED: Dict[str, str] = {
    "region": "Nested inside country (each region belongs to exactly one country) and has 50+ values: "
              "one-hot encoding it would add 50+ columns that mostly repeat country and swamp the other features. "
              "It is used for PROFILING and maps instead.",
    "gender": "Roughly 50/50 and a protected attribute; segmenting/targeting by gender is ethically risky. "
              "Used only to describe clusters.",
    "date_of_birth": "Replaced by age.",
    "client_id": "Identifier, no behavioural meaning.",
    "avg_area": "Correlation with avg_price is ~0.96 (price is almost exactly area x $/sqft); keeping both "
                "would double-count the same information.",
    "total_spend": "= n_units x avg_price, so it is redundant with two features already used.",
    "max_price": "Redundant with avg_price.",
    "avg_price_per_sqft": "Almost constant across clients, adds noise rather than signal.",
    "listing_status / tower / unit_number": "Describe the listing, not the buyer; no leakage-prone fields are used.",
}


def select_features(master: pd.DataFrame) -> Dict:
    """
    Keep only candidate columns that exist AND vary (a constant column carries no information).
    Returns a dict with lists for each type plus the explanation table.
    """
    def usable(cols: Dict[str, str]) -> List[str]:
        return [c for c in cols if c in master.columns and master[c].nunique(dropna=True) > 1]

    numeric = usable(NUMERIC_CANDIDATES)
    binary = usable(BINARY_CANDIDATES)
    nominal = usable(NOMINAL_CANDIDATES)
    if not (numeric or binary or nominal):
        raise ValueError("No usable feature columns were found in the master dataset.")

    reasons = {}
    reasons.update({c: NUMERIC_CANDIDATES[c] for c in numeric})
    reasons.update({c: BINARY_CANDIDATES[c] for c in binary})
    reasons.update({c: NOMINAL_CANDIDATES[c] for c in nominal})
    return {"numeric": numeric, "binary": binary, "nominal": nominal,
            "reasons": reasons, "excluded": EXCLUDED}


def build_preprocessor(numeric: List[str], binary: List[str], nominal: List[str]) -> ColumnTransformer:
    """sklearn ColumnTransformer = scale numbers + label-encode binaries + one-hot nominals."""
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if binary:
        transformers.append(("bin", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), binary))
    if nominal:
        transformers.append(("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_feature_matrix(master: pd.DataFrame, config: Dict):
    """Fit the preprocessor on `master` and return (X, fitted_preprocessor, feature_names)."""
    cols = config["numeric"] + config["binary"] + config["nominal"]
    missing = [c for c in cols if c not in master.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    if master[cols].isna().any().any():
        raise ValueError("Feature columns still contain missing values - run cleaning first.")
    pre = build_preprocessor(config["numeric"], config["binary"], config["nominal"])
    X = pre.fit_transform(master[cols])
    names = list(pre.get_feature_names_out())
    return np.asarray(X, dtype=float), pre, names


def get_scaler(preprocessor: ColumnTransformer) -> StandardScaler:
    """Extract the fitted StandardScaler from inside the ColumnTransformer (saved as scaler.pkl)."""
    return preprocessor.named_transformers_["num"]


def transform_new_clients(preprocessor: ColumnTransformer, config: Dict, df: pd.DataFrame) -> np.ndarray:
    """Apply the SAME fitted preprocessing to new rows (used by the dashboard predictor)."""
    cols = config["numeric"] + config["binary"] + config["nominal"]
    return np.asarray(preprocessor.transform(df[cols]), dtype=float)


def correlation_check(master: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Pearson correlation of candidate numeric columns (used to justify dropping redundant ones)."""
    cols = [c for c in columns if c in master.columns]
    return master[cols].corr().round(3)
