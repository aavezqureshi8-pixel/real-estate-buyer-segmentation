"""
eda.py
------
Static (matplotlib / seaborn) figures for the report and notebook.
The interactive Plotly versions live in app.py.

Every function receives DataFrames and a `save` path; it returns nothing but writes a PNG.
Only columns that really exist in the data are plotted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")                       # no window needed (works on servers / in tests)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from .utils import FIGURES_DIR, get_logger

log = get_logger()
sns.set_theme(style="whitegrid", palette="deep")
PALETTE = sns.color_palette("deep")


def _save(fig, name: str, folder: Path = FIGURES_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _bar(ax, series: pd.Series, title: str, horizontal: bool = False, color=None):
    """Bar chart with count + percentage labels."""
    total = series.sum()
    colors = color or PALETTE[: len(series)]
    if horizontal:
        bars = ax.barh(series.index.astype(str)[::-1], series.values[::-1], color=colors if color else None)
        for b, v in zip(bars, series.values[::-1]):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v} ({100*v/total:.1f}%)", va="center", fontsize=8)
    else:
        bars = ax.bar(series.index.astype(str), series.values, color=colors)
        for b, v in zip(bars, series.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}\n({100*v/total:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("" if horizontal else "Clients")


def make_eda_figures(master: pd.DataFrame, props: pd.DataFrame, transactions: pd.DataFrame) -> Dict[str, Path]:
    """Create all EDA figures. Returns {figure name: path}."""
    out: Dict[str, Path] = {}

    # 1-6: customer demographics & behaviour (each a small single chart)
    simple = [
        ("01_client_type.png", "client_type", "Customer type distribution"),
        ("03_gender.png", "gender", "Gender distribution"),
        ("04_acquisition_purpose.png", "acquisition_purpose", "Acquisition purpose distribution"),
        ("05_loan_applied.png", "loan_applied", "Loan application distribution"),
        ("06_satisfaction.png", "satisfaction_score", "Satisfaction score distribution"),
        ("09_referral_channel.png", "referral_channel", "Referral channel distribution"),
    ]
    for fname, col, title in simple:
        if col not in master.columns:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4))
        vc = master[col].value_counts()
        if col == "satisfaction_score":
            vc = vc.sort_index()
        _bar(ax, vc, title)
        out[fname] = _save(fig, fname)

    if "age" in master.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sns.histplot(master["age"], bins=25, kde=True, ax=ax, color=PALETTE[0])
        ax.axvline(master["age"].mean(), color="red", ls="--", label=f"mean {master['age'].mean():.1f}")
        ax.set_title("Age distribution", fontweight="bold"); ax.set_xlabel("Age (years)"); ax.legend()
        out["02_age.png"] = _save(fig, "02_age.png")

    if "country" in master.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _bar(ax, master["country"].value_counts(), "Country-wise buyers", horizontal=True)
        out["08_country.png"] = _save(fig, "08_country.png")
    if "region" in master.columns:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        _bar(ax, master["region"].value_counts().head(15), "Region-wise buyers (top 15 of "
             f"{master['region'].nunique()})", horizontal=True)
        out["07_region.png"] = _save(fig, "07_region.png")

    # 10: property statistics panel
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _bar(axes[0], props["unit_category"].value_counts(), "Listings by unit category")
    axes[0].set_ylabel("Listings")
    _bar(axes[1], props["listing_status"].value_counts(), "Listing status")
    axes[1].set_ylabel("Listings")
    tw = props.groupby("tower_number").size()
    axes[2].bar(tw.index.astype(str), tw.values, color=PALETTE[2])
    axes[2].set_title("Listings per tower", fontweight="bold"); axes[2].set_xlabel("Tower")
    axes[2].tick_params(axis="x", labelsize=7)
    out["10_property_stats.png"] = _save(fig, "10_property_stats.png")

    # 11: price distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=props, x="sale_price_usd", hue="unit_category", bins=40, kde=True, ax=axes[0], element="step")
    axes[0].set_title("Sale price distribution by unit category", fontweight="bold"); axes[0].set_xlabel("Sale price (USD)")
    sns.boxplot(data=props, x="unit_category", y="sale_price_usd", ax=axes[1])
    axes[1].set_title("Price spread (no IQR outliers detected)" if props["is_price_outlier"].sum() == 0
                      else "Price spread", fontweight="bold")
    out["11_price_distribution.png"] = _save(fig, "11_price_distribution.png")

    # 12: price vs area
    fig, ax = plt.subplots(figsize=(6.5, 5))
    s = props.sample(min(3000, len(props)), random_state=1)
    sns.scatterplot(data=s, x="floor_area_sqft", y="sale_price_usd", hue="unit_category", alpha=0.5, s=14, ax=ax)
    ax.set_title(f"Price vs floor area (correlation {props['floor_area_sqft'].corr(props['sale_price_usd']):.2f})",
                 fontweight="bold")
    out["12_price_vs_area.png"] = _save(fig, "12_price_vs_area.png")

    # 13: monthly sales
    sold = props[props["listing_status"] == "Sold"]
    monthly = sold.groupby("transaction_month").agg(units=("listing_id", "count"), value=("sale_price_usd", "sum"))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(monthly.index, monthly["units"], color=PALETTE[0])
    ax.set_title("Units sold per month", fontweight="bold"); ax.tick_params(axis="x", rotation=60, labelsize=8)
    out["13_monthly_sales.png"] = _save(fig, "13_monthly_sales.png")

    # 14: relationship plots - purpose vs loan / client type / channel
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["loan_applied", "client_type", "referral_channel"]):
        ct = pd.crosstab(master[col], master["acquisition_purpose"], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[0], PALETTE[3]], rot=0)
        ax.set_title(f"Purpose by {col}", fontweight="bold"); ax.set_ylabel("% of clients"); ax.legend(title="")
    out["14_purpose_relationships.png"] = _save(fig, "14_purpose_relationships.png")

    # 15: correlation heatmap
    num = [c for c in ["age", "satisfaction_score", "n_units", "avg_price", "avg_area", "total_spend", "office_share",
                       "is_investment", "loan_flag", "is_company", "is_international"] if c in master.columns]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(master[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation of client-level variables", fontweight="bold")
    out["15_correlation.png"] = _save(fig, "15_correlation.png")

    # 16: units per client and spend by purpose
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vc = master["n_units"].value_counts().sort_index()
    axes[0].bar(vc.index.astype(str), vc.values, color=PALETTE[4])
    axes[0].set_title("Units owned per client", fontweight="bold"); axes[0].set_xlabel("Units")
    sns.boxplot(data=master, x="acquisition_purpose", y="total_spend", ax=axes[1])
    axes[1].set_title("Total spend by acquisition purpose", fontweight="bold")
    out["16_units_and_spend.png"] = _save(fig, "16_units_and_spend.png")

    # 17: satisfaction by age group & purpose rate by country
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    order = ["<35", "35-44", "45-54", "55-64", "65-74", "75+"]
    sns.barplot(data=master, x="age_group", y="satisfaction_score", order=order, ax=axes[0], color=PALETTE[0])
    axes[0].set_title("Average satisfaction by age group", fontweight="bold")
    rate = master.groupby("country")["is_investment"].mean().sort_values(ascending=False) * 100
    axes[1].bar(rate.index, rate.values, color=PALETTE[3])
    axes[1].axhline(master["is_investment"].mean() * 100, color="k", ls="--", label="overall")
    axes[1].set_title("% investment buyers by country", fontweight="bold"); axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()
    out["17_age_sat_country_invest.png"] = _save(fig, "17_age_sat_country_invest.png")
    return out


def make_model_figures(sweep: pd.DataFrame, chosen_k: int, hier: Dict, master: pd.DataFrame,
                       pca_var: np.ndarray, names: Dict[int, str], effects: pd.DataFrame) -> Dict[str, Path]:
    """Elbow, silhouette, stability, dendrogram, PCA scatter, cluster heat-map."""
    out: Dict[str, Path] = {}
    ks = sweep["k"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, sweep["inertia"], "o-", color=PALETTE[0]); ax.axvline(chosen_k, color="red", ls="--", label=f"chosen K={chosen_k}")
    ax.set_title("Elbow method (inertia)", fontweight="bold"); ax.set_xlabel("K"); ax.set_ylabel("Inertia"); ax.legend()
    out["20_elbow.png"] = _save(fig, "20_elbow.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, sweep["silhouette"], "o-", color=PALETTE[2]); ax.axvline(chosen_k, color="red", ls="--", label=f"chosen K={chosen_k}")
    ax.set_title("Silhouette score vs K", fontweight="bold"); ax.set_xlabel("K"); ax.set_ylabel("Silhouette"); ax.legend()
    out["21_silhouette.png"] = _save(fig, "21_silhouette.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, sweep["davies_bouldin"], "o-", color=PALETTE[3]); axes[0].set_title("Davies-Bouldin (lower = better)", fontweight="bold")
    axes[1].plot(ks, sweep["stability_ari"], "o-", color=PALETTE[4]); axes[1].axhline(0.6, color="grey", ls=":")
    axes[1].set_title("Stability: ARI across random restarts", fontweight="bold")
    for ax in axes:
        ax.axvline(chosen_k, color="red", ls="--"); ax.set_xlabel("K")
    out["22_db_stability.png"] = _save(fig, "22_db_stability.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(hier["linkage_matrix"], truncate_mode="lastp", p=30, ax=ax, leaf_rotation=90, leaf_font_size=8,
               show_contracted=True)
    ax.set_title("Dendrogram (Ward linkage, last 30 merges; bracket numbers = clients merged)", fontweight="bold")
    ax.set_ylabel("Ward distance")
    out["23_dendrogram.png"] = _save(fig, "23_dendrogram.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for cid, grp in master.groupby("cluster"):
        ax.scatter(grp["pca1"], grp["pca2"], s=10, alpha=0.6, label=f"C{cid+1} {names[cid]}")
    ax.set_xlabel(f"PC1 ({100*pca_var[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({100*pca_var[1]:.1f}% variance)")
    ax.set_title("PCA projection of K-Means clusters (visualisation only)", fontweight="bold"); ax.legend(fontsize=7, markerscale=2)
    out["24_pca_clusters.png"] = _save(fig, "24_pca_clusters.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    e = effects.copy(); e.index = [f"C{i+1} {names[i]}" for i in e.index]
    e.columns = [c.replace("_", " ") for c in e.columns]
    sns.heatmap(e, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("What makes each cluster different (effect size vs overall mean, in std. deviations)", fontweight="bold", fontsize=10)
    out["25_cluster_effects_heatmap.png"] = _save(fig, "25_cluster_effects_heatmap.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    vc = master["cluster"].value_counts().sort_index()
    _bar(ax, pd.Series(vc.values, index=[f"C{i+1}" for i in vc.index]), "Cluster sizes")
    out["26_cluster_sizes.png"] = _save(fig, "26_cluster_sizes.png")
    return out
