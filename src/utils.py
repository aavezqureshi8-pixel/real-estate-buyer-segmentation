"""
utils.py
--------
Shared paths, constants and small helper functions used by every other module.
Keeping them in one place means a file path or a constant is changed only once.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (all relative to the project root, so the project runs on any machine)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CLIENTS = DATA_DIR / "clients.csv"
RAW_PROPERTIES = DATA_DIR / "properties.csv"

# Files produced by train_pipeline.py and consumed by the dashboard
MASTER_CLIENTS_CSV = PROCESSED_DIR / "master_clients.csv"        # 1 row per client + cluster
PROPERTIES_CLEAN_CSV = PROCESSED_DIR / "properties_clean.csv"    # 1 row per listing
TRANSACTIONS_CSV = PROCESSED_DIR / "transactions.csv"            # sold listing + buyer fields
K_SELECTION_CSV = PROCESSED_DIR / "k_selection.csv"
CLUSTER_PROFILES_CSV = PROCESSED_DIR / "cluster_profiles.csv"
CLUSTER_INSIGHTS_JSON = PROCESSED_DIR / "cluster_insights.json"
RESULTS_JSON = PROCESSED_DIR / "results.json"
CLEANING_LOG_JSON = PROCESSED_DIR / "cleaning_log.json"
DATA_AUDIT_JSON = PROCESSED_DIR / "data_audit.json"

KMEANS_PKL = MODELS_DIR / "kmeans_model.pkl"
SCALER_PKL = MODELS_DIR / "scaler.pkl"
PREPROCESSOR_PKL = MODELS_DIR / "preprocessor.pkl"
PCA_PKL = MODELS_DIR / "pca.pkl"
METADATA_JSON = MODELS_DIR / "metadata.json"

RANDOM_STATE = 42

# Country name -> ISO-3 code (used for the choropleth map in the dashboard)
COUNTRY_ISO3 = {
    "USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU", "France": "FRA",
    "Belgium": "BEL", "Mexico": "MEX", "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK",
}

# US state name -> postal code (for the US-state choropleth)
US_STATE_CODES = {
    "California": "CA", "Nevada": "NV", "Colorado": "CO", "Arizona": "AZ", "Oregon": "OR",
    "Utah": "UT", "Washington": "WA", "Virginia": "VA", "Texas": "TX", "Florida": "FL",
    "New York": "NY", "Georgia": "GA", "Ohio": "OH", "Wyoming": "WY", "Kansas": "KS",
}


def get_logger(name: str = "segmentation") -> logging.Logger:
    """Return a logger that prints simple one-line messages."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs() -> None:
    """Create every output folder if it does not exist yet."""
    for d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    """Save a python object as pretty JSON (numpy / pandas types are converted)."""
    def _convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        raise TypeError(f"Cannot serialise {type(o)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_convert, ensure_ascii=False)


def load_json(path: Path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x: float, digits: int = 1) -> str:
    """Format a 0-1 fraction as a percentage string, e.g. 0.3125 -> '31.2%'."""
    return f"{100 * x:.{digits}f}%"
