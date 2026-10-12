"""
clustering.py
-------------
The machine-learning core:

* sweep_kmeans()      K = 2..10 with inertia, silhouette, Davies-Bouldin, Calinski-Harabasz and a stability score
* find_elbow()        automatic "elbow" (point furthest from the straight line joining first and last inertia)
* select_k()          documented, reproducible rule that combines the metrics with a business limit
* fit_final_kmeans()  final model; clusters are re-numbered by size (C1 = largest)
* run_hierarchical()  Ward / Average / Complete linkage, dendrogram data, comparison with K-Means
* fit_pca()           2-D PCA used ONLY to visualise clusters (clustering itself uses all features)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score, silhouette_score)

from .utils import RANDOM_STATE, get_logger

log = get_logger()


# --------------------------------------------------------------------------
# K-Means sweep
# --------------------------------------------------------------------------

def sweep_kmeans(X: np.ndarray, k_values: Iterable[int] = range(2, 11),
                 n_init: int = 20, stability_seeds: Iterable[int] = range(10)) -> pd.DataFrame:
    """
    Fit K-Means for every K and collect quality metrics.

    inertia      : within-cluster sum of squares (lower = tighter, always falls as K grows) -> Elbow method
    silhouette   : -1..1, how well each point fits its own cluster vs the next-nearest one (higher = better)
    davies_bouldin: lower = better separated clusters
    calinski     : higher = better defined clusters
    stability_ari: average Adjusted Rand Index between the reference solution and solutions from other random starts.
                   1.0 = same clusters every time, ~0 = clusters change randomly (not trustworthy).
    """
    rows = []
    seeds = list(stability_seeds)
    for k in k_values:
        ref = KMeans(n_clusters=k, n_init=n_init, random_state=RANDOM_STATE).fit(X)
        aris = []
        for s in seeds:
            other = KMeans(n_clusters=k, n_init=3, random_state=1000 + s).fit(X)
            aris.append(adjusted_rand_score(ref.labels_, other.labels_))
        rows.append({
            "k": k,
            "inertia": float(ref.inertia_),
            "silhouette": float(silhouette_score(X, ref.labels_)),
            "davies_bouldin": float(davies_bouldin_score(X, ref.labels_)),
            "calinski_harabasz": float(calinski_harabasz_score(X, ref.labels_)),
            "stability_ari": float(np.mean(aris)),
            "smallest_cluster_pct": float(100 * np.bincount(ref.labels_).min() / len(X)),
        })
    return pd.DataFrame(rows)


def find_elbow(k_values: List[int], inertia: List[float]) -> int:
    """
    Automatic elbow: normalise both axes to 0-1, draw a line from the first to the last point and
    return the K whose point is furthest from that line (the 'knee' of the curve).
    """
    k = np.asarray(k_values, dtype=float)
    y = np.asarray(inertia, dtype=float)
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([kn[0], yn[0]]), np.array([kn[-1], yn[-1]])
    line = (p2 - p1) / np.linalg.norm(p2 - p1)
    dists = []
    for x_, y_ in zip(kn, yn):
        v = np.array([x_, y_]) - p1
        dists.append(abs(v[0] * line[1] - v[1] * line[0]))        # perpendicular distance to the line
    return int(k[int(np.argmax(dists))])


def select_k(sweep: pd.DataFrame, k_min: int = 3, k_max: int = 6, sil_tolerance: float = 0.01,
             min_stability: float = 0.6) -> Dict:
    """
    Reproducible rule for choosing K (NOT hard-coded):

    1. Business limit : a marketing team can act on only ~3-6 segments -> look at K in [k_min, k_max]
                        (K=2 only splits the clients in two halves and gives no useful targeting).
    2. Silhouette     : keep K values whose silhouette is within `sil_tolerance` of the best one in that range
                        (differences below ~0.01 are noise, so they are treated as ties).
    3. Stability      : keep K values whose clusters survive random restarts (mean ARI >= `min_stability`);
                        a segmentation that changes every time you re-run it cannot be trusted.
    4. Separation     : among what is left choose the lowest Davies-Bouldin index (best-separated clusters).
    The elbow K and the unrestricted best-silhouette K are reported for transparency.
    """
    full_best_sil_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    elbow_k = find_elbow(sweep["k"].tolist(), sweep["inertia"].tolist())

    cand = sweep[(sweep["k"] >= k_min) & (sweep["k"] <= k_max)].copy()
    best_sil = cand["silhouette"].max()
    eligible = cand[cand["silhouette"] >= best_sil - sil_tolerance]
    stable = eligible[eligible["stability_ari"] >= min_stability]
    if stable.empty:                                   # fall back: most stable of the eligible ones
        chosen = eligible.sort_values("stability_ari", ascending=False).iloc[0]
        fallback = True
    else:
        chosen = stable.sort_values("davies_bouldin").iloc[0]
        fallback = False
    k = int(chosen["k"])

    rationale = (
        f"Elbow method suggests K={elbow_k}; the unrestricted silhouette peak is K={full_best_sil_k} "
        f"(a trivial 2-way split). Restricting to {k_min}-{k_max} segments (business limit), K values with "
        f"silhouette within {sil_tolerance} of the best ({best_sil:.3f}): {eligible['k'].astype(int).tolist()}. "
        f"Of these, K values with stability ARI >= {min_stability}: {stable['k'].astype(int).tolist()}. "
        f"K={k} has the lowest Davies-Bouldin index ({chosen['davies_bouldin']:.3f}) among them "
        f"(silhouette={chosen['silhouette']:.3f}, stability ARI={chosen['stability_ari']:.2f})."
        + (" (No K met the stability bar, so the most stable eligible K was used.)" if fallback else "")
    )
    return {"k": k, "elbow_k": elbow_k, "best_silhouette_k_unrestricted": full_best_sil_k,
            "eligible_k": eligible["k"].astype(int).tolist(), "stable_k": stable["k"].astype(int).tolist(),
            "rationale": rationale, "k_min": k_min, "k_max": k_max,
            "sil_tolerance": sil_tolerance, "min_stability": min_stability}


# --------------------------------------------------------------------------
# Final K-Means
# --------------------------------------------------------------------------

def fit_final_kmeans(X: np.ndarray, k: int, n_init: int = 50) -> Tuple[KMeans, np.ndarray]:
    """
    Fit the production K-Means and renumber clusters so that cluster 0 is the largest.
    The cluster centres inside the model are re-ordered too, so model.predict() returns the same ids.
    """
    km = KMeans(n_clusters=k, n_init=n_init, random_state=RANDOM_STATE).fit(X)
    order = np.argsort(-np.bincount(km.labels_, minlength=k))          # biggest first
    km.cluster_centers_ = km.cluster_centers_[order]
    labels = km.predict(X)
    return km, labels


# --------------------------------------------------------------------------
# Hierarchical clustering
# --------------------------------------------------------------------------

def run_hierarchical(X: np.ndarray, k: int, kmeans_labels: np.ndarray,
                     k_values: Iterable[int] = range(2, 11)) -> Dict:
    """
    Agglomerative clustering (bottom-up). Ward linkage merges the pair of clusters that increases
    within-cluster variance the least - the hierarchical twin of K-Means, so the comparison is fair.
    """
    # compare linkage methods at the chosen K
    linkage_rows = []
    for method in ("ward", "average", "complete"):
        lab = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)
        linkage_rows.append({
            "linkage": method,
            "silhouette": float(silhouette_score(X, lab)),
            "davies_bouldin": float(davies_bouldin_score(X, lab)),
            "smallest_cluster_pct": float(100 * np.bincount(lab).min() / len(X)),
            "ARI_vs_kmeans": float(adjusted_rand_score(kmeans_labels, lab)),
        })
    linkage_df = pd.DataFrame(linkage_rows)

    # Ward sweep over K for the same comparison as K-Means
    sweep_rows = []
    for kk in k_values:
        lab = AgglomerativeClustering(n_clusters=kk, linkage="ward").fit_predict(X)
        sweep_rows.append({"k": kk, "silhouette": float(silhouette_score(X, lab)),
                           "davies_bouldin": float(davies_bouldin_score(X, lab))})
    ward_sweep = pd.DataFrame(sweep_rows)

    ward_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    Z = linkage(X, method="ward")                                  # used for the dendrogram
    coph, _ = cophenet(Z, pdist(X))                                # how faithfully the tree preserves distances

    contingency = pd.crosstab(pd.Series(kmeans_labels, name="kmeans"), pd.Series(ward_labels, name="ward"))
    agreement = {
        "ARI": float(adjusted_rand_score(kmeans_labels, ward_labels)),
        "NMI": float(normalized_mutual_info_score(kmeans_labels, ward_labels)),
        "cophenetic_correlation_ward": float(coph),
        "silhouette_ward": float(silhouette_score(X, ward_labels)),
        "silhouette_kmeans": float(silhouette_score(X, kmeans_labels)),
        "davies_bouldin_ward": float(davies_bouldin_score(X, ward_labels)),
        "davies_bouldin_kmeans": float(davies_bouldin_score(X, kmeans_labels)),
    }
    return {"labels": ward_labels, "linkage_matrix": Z, "linkage_comparison": linkage_df,
            "ward_sweep": ward_sweep, "contingency": contingency, "agreement": agreement}


# --------------------------------------------------------------------------
# PCA (visualisation only)
# --------------------------------------------------------------------------

def fit_pca(X: np.ndarray, n_components: int = 2) -> Tuple[PCA, np.ndarray]:
    """PCA squeezes the many encoded features into 2 axes so the clusters can be drawn on a plane."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X)
    return pca, pca.transform(X)
