"""
train_pipeline.py
-----------------
Runs the WHOLE project end-to-end with one command:

    python train_pipeline.py

Steps: audit -> clean -> join -> features -> K sweep -> K-Means -> Hierarchical -> PCA ->
       profiling / naming / insights -> save models, tables and figures.

The Streamlit dashboard (app.py) only READS the files this script writes,
so the models are not re-trained every time the dashboard opens.
"""
from __future__ import annotations

import sys
import warnings

import joblib
import pandas as pd

from src import utils
from src.clustering import fit_final_kmeans, fit_pca, run_hierarchical, select_k, sweep_kmeans
from src.data_cleaning import run_cleaning
from src.eda import make_eda_figures, make_model_figures
from src.feature_engineering import correlation_check, get_scaler, make_feature_matrix, select_features
from src.report_builder import build_reports
from src.profiling import (build_cluster_insights, investment_profile, name_clusters, profile_clusters,
                           segment_tests, test_prd_segments, trait_effects)

warnings.filterwarnings("ignore", category=FutureWarning)
log = utils.get_logger()


def run_pipeline(make_figures: bool = True) -> dict:
    utils.ensure_dirs()

    # ---------------------------------------------------------------- 1. clean + join
    log.info("1/7 Auditing and cleaning data ...")
    c = run_cleaning()
    master, props, transactions = c["master"].copy(), c["properties"], c["transactions"]
    utils.save_json(c["audit"], utils.DATA_AUDIT_JSON)
    utils.save_json(c["log"], utils.CLEANING_LOG_JSON)
    log.info("    clients=%d  listings=%d  sold=%d  join key: %s -> %s",
             len(master), len(props), len(transactions),
             c["audit"]["join_analysis"].get("primary_key_in_clients"),
             c["audit"]["join_analysis"].get("foreign_key_in_properties"))

    # ---------------------------------------------------------------- 2. features
    log.info("2/7 Building features (encode + scale) ...")
    config = select_features(master)
    X, preprocessor, feature_names = make_feature_matrix(master, config)
    log.info("    feature matrix: %s rows x %d columns", X.shape[0], X.shape[1])

    # ---------------------------------------------------------------- 3. K sweep + K-Means
    log.info("3/7 K-Means sweep K=2..10 ...")
    sweep = sweep_kmeans(X)
    selection = select_k(sweep)
    k = selection["k"]
    log.info("    %s", selection["rationale"])
    kmeans, labels = fit_final_kmeans(X, k)
    master["cluster"] = labels

    # ---------------------------------------------------------------- 4. hierarchical
    log.info("4/7 Hierarchical clustering (Ward) ...")
    hier = run_hierarchical(X, k, labels)
    master["ward_cluster"] = hier["labels"]
    log.info("    Ward vs K-Means: ARI=%.3f", hier["agreement"]["ARI"])

    # ---------------------------------------------------------------- 5. PCA (visualisation only)
    log.info("5/7 PCA for visualisation ...")
    pca, coords = fit_pca(X)
    master["pca1"], master["pca2"] = coords[:, 0], coords[:, 1]

    # ---------------------------------------------------------------- 6. interpretation
    log.info("6/7 Profiling clusters ...")
    names = name_clusters(master)
    master["cluster_id"] = master["cluster"].map(lambda i: f"C{i + 1}")
    master["segment"] = master["cluster"].map(names)
    profile = profile_clusters(master)
    effects = trait_effects(master)
    prd = test_prd_segments(profile, names)
    tests = segment_tests(master)
    inv = investment_profile(master, profile, names)
    insights = build_cluster_insights(master, profile, names, effects, inv)

    # ---------------------------------------------------------------- 7. save everything
    log.info("7/7 Saving models, tables and figures ...")
    joblib.dump(kmeans, utils.KMEANS_PKL)
    joblib.dump(get_scaler(preprocessor), utils.SCALER_PKL)
    joblib.dump(preprocessor, utils.PREPROCESSOR_PKL)
    joblib.dump(pca, utils.PCA_PKL)

    metadata = {
        "k": k, "feature_config": {kk: config[kk] for kk in ("numeric", "binary", "nominal")},
        "feature_names": feature_names, "cluster_names": {str(i): n for i, n in names.items()},
        "categories": {col: sorted(master[col].unique().tolist()) for col in config["binary"] + config["nominal"]},
        "numeric_ranges": {col: [float(master[col].min()), float(master[col].max()), float(master[col].median())]
                           for col in config["numeric"]},
        "reference_date": str(c["reference_date"].date()),
    }
    utils.save_json(metadata, utils.METADATA_JSON)

    master_out = master.copy()
    for col in ("first_purchase", "last_purchase", "date_of_birth"):
        master_out[col] = pd.to_datetime(master_out[col]).dt.strftime("%Y-%m-%d")
    master_out.to_csv(utils.MASTER_CLIENTS_CSV, index=False)

    props_out = props.copy()
    props_out["transaction_date"] = props_out["transaction_date"].dt.strftime("%Y-%m-%d")
    props_out.to_csv(utils.PROPERTIES_CLEAN_CSV, index=False)

    tx = transactions.merge(master[["client_id", "cluster", "cluster_id", "segment"]], on="client_id", how="left")
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"]).dt.strftime("%Y-%m-%d")
    tx.to_csv(utils.TRANSACTIONS_CSV, index=False)

    sweep.to_csv(utils.K_SELECTION_CSV, index=False)
    profile_out = profile.copy()
    profile_out.insert(0, "segment", [names.get(i, "All clients") for i in profile.index])
    profile_out.to_csv(utils.CLUSTER_PROFILES_CSV)
    utils.save_json({"insights": insights, "investment_profile": inv.to_dict("records"),
                     "prd_segment_test": prd.to_dict("records")}, utils.CLUSTER_INSIGHTS_JSON)

    results = {
        "dataset": {"clients": len(master), "listings": len(props), "sold_listings": len(transactions),
                    "available_listings": int((props["listing_status"] == "Available").sum()),
                    "reference_date": str(c["reference_date"].date())},
        "features": {kk: config[kk] for kk in ("numeric", "binary", "nominal")},
        "n_encoded_columns": X.shape[1],
        "k_selection": selection, "k_sweep": sweep.to_dict("records"),
        "kmeans_final": {"k": k, "inertia": float(kmeans.inertia_),
                         "silhouette": float(sweep.loc[sweep["k"] == k, "silhouette"].iloc[0]),
                         "sizes": {f"C{i+1}": int((labels == i).sum()) for i in range(k)}},
        "hierarchical": {"agreement": hier["agreement"],
                         "linkage_comparison": hier["linkage_comparison"].to_dict("records"),
                         "ward_sweep": hier["ward_sweep"].to_dict("records"),
                         "contingency_kmeans_vs_ward": hier["contingency"].to_dict()},
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "cluster_profiles": profile.to_dict("index"),
        "cluster_names": {str(i): n for i, n in names.items()},
        "trait_effects": effects.to_dict("index"),
        "prd_segment_test": prd.to_dict("records"),
        "statistical_tests": tests.to_dict("records"),
        "outliers": c["outliers"],
        "feature_correlations": correlation_check(master, ["age", "satisfaction_score", "n_units", "avg_price",
                                                           "avg_area", "total_spend", "office_share",
                                                           "avg_price_per_sqft"]).to_dict(),
    }
    utils.save_json(results, utils.RESULTS_JSON)

    if make_figures:
        make_eda_figures(master, props, transactions)
        make_model_figures(sweep, k, hier, master, pca.explained_variance_ratio_, names, effects)

    build_reports()                              # research_report.md + VIVA_QA.md from the real results
    log.info("Done. K=%d | silhouette=%.3f | models in %s", k, results["kmeans_final"]["silhouette"], utils.MODELS_DIR)
    return {"master": master, "props": props, "transactions": tx, "sweep": sweep, "selection": selection,
            "kmeans": kmeans, "hier": hier, "pca": pca, "profile": profile, "names": names, "effects": effects,
            "prd": prd, "tests": tests, "inv": inv, "insights": insights, "results": results,
            "preprocessor": preprocessor, "config": config, "X": X}


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:                       # readable error instead of a long traceback for beginners
        log.error("Pipeline failed: %s", exc)
        raise
    sys.exit(0)
