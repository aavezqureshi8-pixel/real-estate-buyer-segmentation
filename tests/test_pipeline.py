"""
Basic automated checks. Run with:   python tests/test_pipeline.py     (or: pytest tests)
They verify the cleaning, the join, the feature matrix and the saved models.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from src import utils
from src.clustering import fit_final_kmeans, select_k, sweep_kmeans
from src.data_cleaning import parse_mixed_dates, run_cleaning
from src.feature_engineering import make_feature_matrix, select_features
from src.profiling import name_clusters, profile_clusters

_CACHE = {}


def _clean():
    if "c" not in _CACHE:
        _CACHE["c"] = run_cleaning()
    return _CACHE["c"]


def test_join_key_found_and_valid():
    j = _clean()["audit"]["join_analysis"]
    assert j["found"] and j["primary_key_in_clients"] == "client_id" and j["foreign_key_in_properties"] == "client_ref"
    assert j["share_of_property_refs_matching"] == 1.0


def test_master_is_one_row_per_client_without_missing_values():
    m = _clean()["master"]
    assert m["client_id"].is_unique
    assert m.isna().sum().sum() == 0
    assert (m["n_units"] >= 1).all()


def test_dates_prices_and_age_are_parsed():
    c = _clean()
    assert pd.api.types.is_datetime64_any_dtype(c["properties"]["transaction_date"])
    assert c["properties"]["sale_price_usd"].between(1, 5e6).all()
    assert c["master"]["age"].between(18, 100).all()
    s = pd.Series(["5/28/1976", "05-11-1968", "11/26/1962", "garbage"])
    out = parse_mixed_dates(s, ["%m/%d/%Y", "%m-%d-%Y"])
    assert out.iloc[0] == pd.Timestamp(1976, 5, 28) and out.iloc[1] == pd.Timestamp(1968, 5, 11)
    assert pd.isna(out.iloc[3])


def test_sold_units_total_matches_transactions():
    c = _clean()
    assert int(c["master"]["n_units"].sum()) == len(c["transactions"]) == int((c["properties"]["listing_status"] == "Sold").sum())


def test_feature_matrix_is_scaled_and_finite():
    m = _clean()["master"]
    cfg = select_features(m)
    X, pre, names = make_feature_matrix(m, cfg)
    assert np.isfinite(X).all() and X.shape[1] == len(names)
    for i, col in enumerate(cfg["numeric"]):
        assert abs(X[:, i].mean()) < 1e-8 and abs(X[:, i].std() - 1) < 1e-6
    assert "region" not in cfg["numeric"] + cfg["binary"] + cfg["nominal"]


def test_kmeans_labels_are_reproducible_and_sorted_by_size():
    m = _clean()["master"]
    X, _, _ = make_feature_matrix(m, select_features(m))
    km, lab = fit_final_kmeans(X, 4)
    assert (km.predict(X) == lab).all()
    sizes = np.bincount(lab)
    assert (np.diff(sizes) <= 0).all()                       # cluster 0 is the biggest


def test_k_selection_rule_returns_value_in_range():
    m = _clean()["master"]
    X, _, _ = make_feature_matrix(m, select_features(m))
    sw = sweep_kmeans(X, k_values=range(2, 8), stability_seeds=range(3))
    sel = select_k(sw)
    assert 3 <= sel["k"] <= 6 and sw["silhouette"].between(-1, 1).all()


def test_profile_and_names_consistent():
    p = Path(utils.MASTER_CLIENTS_CSV)
    assert p.exists(), "run train_pipeline.py first"
    m = pd.read_csv(p)
    prof = profile_clusters(m)
    assert prof.loc[[i for i in prof.index if i != "All"], "size"].sum() == len(m)
    assert abs(prof.loc[[i for i in prof.index if i != "All"], "pct_of_clients"].sum() - 1) < 1e-9
    names = name_clusters(m)
    assert len(set(names.values())) == len(names)


def test_saved_models_load_and_agree_with_saved_labels():
    m = pd.read_csv(utils.MASTER_CLIENTS_CSV)
    meta = utils.load_json(utils.METADATA_JSON)
    km, pre = joblib.load(utils.KMEANS_PKL), joblib.load(utils.PREPROCESSOR_PKL)
    joblib.load(utils.SCALER_PKL); joblib.load(utils.PCA_PKL)
    cols = sum(meta["feature_config"].values(), [])
    X = pre.transform(m[cols])
    assert (km.predict(X) == m["cluster"].values).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
