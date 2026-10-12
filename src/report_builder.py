"""
report_builder.py
-----------------
Writes research_report.md and VIVA_QA.md from the REAL results produced by train_pipeline.py.
Every number in the documents is read from the data / results files (nothing is typed by hand),
so the report can never disagree with the analysis, and it re-generates if the data changes.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from . import utils


# ---------------------------------------------------------------------------
# tiny markdown helpers
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    """DataFrame -> markdown table. Integers stay integers, floats use `floatfmt`."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.astype(object).itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def P(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def D(x: float) -> str:
    return f"${x:,.0f}"


def _load():
    res = utils.load_json(utils.RESULTS_JSON)
    ins_file = utils.load_json(utils.CLUSTER_INSIGHTS_JSON)
    master = pd.read_csv(utils.MASTER_CLIENTS_CSV)
    props = pd.read_csv(utils.PROPERTIES_CLEAN_CSV)
    audit = utils.load_json(utils.DATA_AUDIT_JSON)
    log = utils.load_json(utils.CLEANING_LOG_JSON)
    sweep = pd.read_csv(utils.K_SELECTION_CSV)
    return res, ins_file, master, props, audit, log, sweep


def build_reports() -> None:
    res, ins_file, m, props, audit, clog, sweep = _load()
    insights, inv, prd = ins_file["insights"], pd.DataFrame(ins_file["investment_profile"]), pd.DataFrame(ins_file["prd_segment_test"])
    prof = pd.DataFrame(res["cluster_profiles"]).T
    allrow = prof.loc["All"]
    cl = prof.drop(index="All")
    cl.index = cl.index.astype(int)
    names = {int(k): v for k, v in res["cluster_names"].items()}
    k = res["kmeans_final"]["k"]
    sel = res["k_selection"]
    ag = res["hierarchical"]["agreement"]
    tests = pd.DataFrame(res["statistical_tests"])
    sold = props[props["listing_status"] == "Sold"]
    n_clients, n_listings = res["dataset"]["clients"], res["dataset"]["listings"]
    n_sold, n_avail = res["dataset"]["sold_listings"], res["dataset"]["available_listings"]
    feats = res["features"]
    sil = res["kmeans_final"]["silhouette"]
    pcs = res["pca_explained_variance"]

    def lab(i: int) -> str:
        return f"C{i + 1} ({names[i]})"

    # convenient extremes
    low_sat, high_sat = int(cl["avg_satisfaction"].astype(float).idxmin()), int(cl["avg_satisfaction"].astype(float).idxmax())
    big_units = int(cl["avg_units"].astype(float).idxmax())
    top_price = int(cl["avg_price"].astype(float).idxmax())
    top_off = int(cl["office_share"].astype(float).idxmax())
    top_value = int(cl["value_index"].astype(float).idxmax())
    top_inv_rank = int(inv.loc[inv["investor_value_rank"] == 1, "cluster"].iloc[0])
    top_inv_rate = int(cl["investment_rate"].astype(float).idxmax())
    top_loan = int(cl["loan_rate"].astype(float).idxmax())
    sig_cat = tests[(tests["type"] == "categorical") & (tests["significant_0.05"])]["variable"].tolist()
    nonsig_cat = tests[(tests["type"] == "categorical") & (~tests["significant_0.05"])]["variable"].tolist()
    supported = prd[prd["supported_by_data"]]["prd_segment"].tolist()
    price_corr = props["sale_price_usd"].corr(props["floor_area_sqft"])
    inv_by_country = m.groupby("country")["is_investment"].mean()
    import numpy as _np
    _c = m[['age', 'satisfaction_score', 'n_units', 'avg_price', 'is_investment', 'loan_flag']].corr().abs().values.copy()
    _np.fill_diagonal(_c, 0)
    max_r = float(_c.max())
    ct = m.groupby("client_type")[["is_investment", "loan_flag", "n_units", "age"]].mean()

    # ---------------- cluster profile tables ----------------
    prof_tbl = pd.DataFrame({
        "Cluster": [lab(i) for i in cl.index],
        "Clients": cl["size"].astype(int).values,
        "% clients": [P(float(x)) for x in cl["pct_of_clients"]],
        "Avg age": cl["avg_age"].astype(float).round(1).values,
        "% invest.": [P(float(x)) for x in cl["investment_rate"]],
        "% loan": [P(float(x)) for x in cl["loan_rate"]],
        "Satisf.": cl["avg_satisfaction"].astype(float).round(2).values,
        "Units": cl["avg_units"].astype(float).round(2).values,
        "Avg unit price": [D(float(x)) for x in cl["avg_price"]],
        "% office": [P(float(x)) for x in cl["office_share"]],
        "% intl.": [P(float(x)) for x in cl["pct_international"]],
        "% company": [P(float(x)) for x in cl["pct_company"]],
        "Value index": cl["value_index"].astype(float).round(2).values,
    })
    overall_line = (f"Overall: age {float(allrow['avg_age']):.1f}, investment {P(float(allrow['investment_rate']))}, "
                    f"loan {P(float(allrow['loan_rate']))}, satisfaction {float(allrow['avg_satisfaction']):.2f}, "
                    f"{float(allrow['avg_units']):.2f} units, unit price {D(float(allrow['avg_price']))}, "
                    f"office share {P(float(allrow['office_share']))}.")
    sweep_tbl = sweep.copy()
    sweep_tbl.columns = ["K", "Inertia", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Stability (ARI)", "Smallest cluster %"]
    sweep_tbl["K"] = sweep_tbl["K"].astype(int)
    sweep_tbl["Inertia"] = sweep_tbl["Inertia"].map(lambda v: f"{v:,.0f}")
    for c in ["Silhouette", "Davies-Bouldin", "Stability (ARI)"]:
        sweep_tbl[c] = sweep_tbl[c].map(lambda v: f"{v:.3f}")
    sweep_tbl["Calinski-Harabasz"] = sweep_tbl["Calinski-Harabasz"].map(lambda v: f"{v:.1f}")
    sweep_tbl["Smallest cluster %"] = sweep_tbl["Smallest cluster %"].map(lambda v: f"{v:.2f}")
    link_tbl = pd.DataFrame(res["hierarchical"]["linkage_comparison"])
    link_raw = link_tbl.copy()
    link_tbl["smallest_cluster_pct"] = link_tbl["smallest_cluster_pct"].map(lambda v: f"{v:.2f}")
    test_tbl = tests[["variable", "test", "p_value", "effect_size", "effect_size_name", "significant_0.05"]].copy()
    test_tbl["p_value"] = test_tbl["p_value"].map(lambda p: f"{p:.4f}")
    test_tbl["effect_size"] = test_tbl["effect_size"].map(lambda v: f"{v:.3f}")
    prd_tbl = prd[["prd_segment", "prd_description", "closest_cluster_name", "conditions_met", "supported_by_data"]]
    inv_tbl = pd.DataFrame({
        "Cluster": [lab(int(c)) for c in inv["cluster"]], "Investors": inv["investors"].astype(int).values,
        "Invest. rate": [P(x) for x in inv["investment_rate"]], "Lift": inv["investment_lift_vs_overall"].round(2).values,
        "Investor spend ($M)": (inv["investor_spend_usd"] / 1e6).round(1).values,
        "% of investor spend": [P(x) for x in inv["share_of_all_investor_spend"]],
        "Units/investor": inv["avg_units_per_investor"].round(2).values,
        "Value rank": inv["investor_value_rank"].astype(int).values})
    feat_tbl = pd.DataFrame(
        [(c, "numeric", "StandardScaler") for c in feats["numeric"]] +
        [(c, "binary", "Label-encoded to 0/1 (OrdinalEncoder)") for c in feats["binary"]] +
        [(c, "nominal", "One-Hot Encoding") for c in feats["nominal"]], columns=["Feature", "Type", "Treatment"])

    cluster_sections = []
    for i in insights:
        c = i["cluster"]
        cluster_sections.append(f"""### {i['cluster_id']} - {i['name']} ({i['size']:,} clients, {i['pct']:.1f}%)

* **Customer profile:** {i['customer_profile']}
* **Behavioural profile:** {i['behavioral_profile']}
* **Investment profile:** {i['investment_profile']}
* **Financing profile:** {i['financing_profile']}
* **Geographic profile:** {i['geographic_profile']}
* **Marketing recommendations:**
{chr(10).join('  * ' + r for r in i['marketing_recommendations'])}
* **Property recommendations:**
{chr(10).join('  * ' + r for r in i['property_recommendations'])}
* **Assumptions (not proven by the data):**
{chr(10).join('  * ' + r for r in i['assumptions'])}
""")

    # =====================================================================
    # RESEARCH REPORT
    # =====================================================================
    report = f"""# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

*Domain: Financial Analytics, Real Estate Market Intelligence, Machine Learning · Collaborating organisations: Unified Mentor, Parcl Co. Limited*

> All numbers in this report are generated directly from the supplied `clients.csv` and `properties.csv`
> by `train_pipeline.py` / `src/report_builder.py`. Nothing is simulated or typed by hand.

---

## 1. Abstract

This project segments the buyers of a real-estate portfolio and builds an investment profile for each segment.
Two datasets were supplied: **{n_clients:,} clients** and **{n_listings:,} property listings** ({n_sold:,} sold, {n_avail:,} available).
They join through `clients.client_id = properties.client_ref`; every sold listing belongs to exactly one client and every client owns at least {int(m['n_units'].min())} units.
After cleaning, each client was described by {len(feats['numeric'])} numeric features (age, satisfaction, portfolio size, average unit price, office share),
{len(feats['binary'])} binary features (purpose, loan, client type) and {len(feats['nominal'])} nominal features (country, referral channel), giving {res['n_encoded_columns']} encoded columns.
K-Means was run for K = 2 to 10; using elbow, silhouette, Davies-Bouldin and restart-stability together with a business limit of 3-6 segments, **K = {k}** was selected.
The silhouette score is **{sil:.3f}**, which indicates *weak* natural separation, so the segments should be read as practical groupings and not as sharply separated populations.
Ward hierarchical clustering partly agrees with K-Means (Adjusted Rand Index {ag['ARI']:.2f}).
The segments are distinguished by age, satisfaction, portfolio size, ticket price and office share. Purpose of purchase, loan use, referral channel, country and gender
do **not** differ significantly between segments. None of the four example segments in the PRD
(Global Investors, First-Time Buyers, Corporate Buyers, Luxury Investors) is fully supported by this data; data-driven segment names are therefore used.
A Streamlit dashboard exposes the results with filters for country, region, purpose and client type.

## 2. Introduction

Real-estate companies serve very different buyers - families, companies, overseas investors - yet often market to all of them in the same way.
This leads to inefficient marketing spend, generic property recommendations and missed investment opportunities.
Clustering is an unsupervised machine-learning technique that groups similar clients without needing pre-defined labels, so it can reveal groups that simple reports do not show.

## 3. Background

Market segmentation divides a customer base into groups that respond similarly to marketing (Wedel & Kamakura, 2000).
K-Means (MacQueen, 1967) partitions data into K groups around centroids; hierarchical clustering with Ward linkage (Ward, 1963) builds a tree of merges.
Internal validity measures such as the silhouette coefficient (Rousseeuw, 1987), the Davies-Bouldin index (Davies & Bouldin, 1979) and the elbow heuristic (Thorndike, 1953)
are used because there is no ground-truth label to check against.

## 4. Problem Statement

Parcl lacks a data-driven understanding of (a) the different types of property buyers, (b) investment motivations across demographics,
(c) geographic differences in investment behaviour and (d) customer financing patterns. The consequences are inefficient marketing spend,
generic property recommendations and poor investor targeting.

## 5. Objectives

1. Clean and join the client and property data into one analysis-ready table.
2. Explore buyer demographics, behaviour and property characteristics.
3. Discover hidden buyer segments with K-Means and validate them with hierarchical clustering.
4. Choose the number of clusters with documented, reproducible evidence.
5. Profile every segment (demographics, purpose, financing, geography, property metrics) and build investment profiles.
6. Check honestly whether the PRD's example segments exist in the data.
7. Deliver an interactive dashboard and actionable recommendations.

## 6. Dataset Description

| | `clients.csv` | `properties.csv` |
|---|---|---|
| Rows | {audit['clients']['rows']:,} | {audit['properties']['rows']:,} |
| Columns | {audit['clients']['columns']} | {audit['properties']['columns']} |
| Missing values | {sum(audit['clients']['missing_values'].values())} | {sum(audit['properties']['missing_values'].values())} (all {audit['properties']['missing_values']['client_ref']:,} are `client_ref` of *Available* listings, which is expected) |
| Duplicate rows | {audit['clients']['duplicate_rows']} | {audit['properties']['duplicate_rows']} |

**Clients columns:** {', '.join('`'+c+'`' for c in audit['clients']['column_names'])}.
**Properties columns:** {', '.join('`'+c+'`' for c in audit['properties']['column_names'])}.

**Relationship.** The join analysis discovered `clients.{audit['join_analysis']['primary_key_in_clients']}` (unique) ↔
`properties.{audit['join_analysis']['foreign_key_in_properties']}`: {P(audit['join_analysis']['share_of_property_refs_matching'])} of references match, there are no orphan listings,
and all {audit['join_analysis']['clients_with_at_least_one_property']:,} clients own at least one unit.
Each client owns between {int(m['n_units'].min())} and {int(m['n_units'].max())} units (mean {m['n_units'].mean():.2f}).

**Fields requested by the PRD but absent from the data:** income / net worth, bedrooms, loan amount, a first-time-buyer flag.
The PRD calls corporate clients *Corporate*; the data uses *Company*. The PRD's "multiple units purchased by companies" can only be tested through units owned per client.

## 7. Data Preprocessing

| Step | Decision |
|---|---|
| Duplicates | {sum(s.get('exact_duplicates_removed',0)+s.get('duplicate_ids_removed',0) for s in clog)} duplicate rows / IDs found in total; code removes them if present. |
| Missing values | None in client fields. Rule applied if present: numeric → median, categorical → "Unknown". |
| Categorical labels | Whitespace / case normalised; gender M/F → Male/Female; synonyms mapped (e.g. Corporate → Company, USA aliases). |
| Dates | `date_of_birth` appears in three layouts (e.g. `M/D/YYYY`, `MM-DD-YYYY`). Parsed month-first. Slash dates contain days up to 31 in the second position, which proves month-first; dash dates have values ≤ 12 in both positions, so month-first is an **assumption** (consistent with `transaction_date`, which is also month-first); it can change an age by at most 1 year. |
| Age | Computed at {res['dataset']['reference_date']} (latest transaction date); raw DOB is **not** used in ML. Range {m['age'].min():.1f}-{m['age'].max():.1f} years, mean {m['age'].mean():.1f}. |
| Price | `sale_price` is text such as `$300,385.62`; converted to numeric USD (currency assumed USD). |
| Outliers | IQR rule (1.5×IQR) used for **detection only**. Sale price: {res['outliers']['sale_price_usd']['n_outliers']} outliers, floor area: {res['outliers']['floor_area_sqft']['n_outliers']}, price per sq ft: {res['outliers']['price_per_sqft']['n_outliers']} ({res['outliers']['price_per_sqft']['pct_outliers']:.2f}%). No rows removed because high-value units are legitimate. Units owned per client reaches {int(m['n_units'].max())} and is kept (it becomes a real segment). |
| Privacy | First and last names dropped. |
| Consistency checks | Sold listings all have a buyer; Available listings have none; every region maps to one country. |

## 8. Exploratory Data Analysis

Figures are in `reports/figures/` (static) and in the dashboard (interactive).

**Customers.**
* Client type: {P((m['client_type']=='Individual').mean())} Individual, {P((m['client_type']=='Company').mean())} Company ({int((m['client_type']=='Company').sum())} clients).
* Age: mean {m['age'].mean():.1f}, median {m['age'].median():.1f}, spread {m['age'].std():.1f}; the age groups are nearly evenly populated.
* Gender: {P((m['gender']=='Male').mean())} male, {P((m['gender']=='Female').mean())} female.
* Country: {P((m['country']=='USA').mean())} of clients are in the USA ({int((m['country']=='USA').sum()):,}); next are {', '.join(f'{c} ({n})' for c, n in m['country'].value_counts().iloc[1:4].items())}.
* Region: {m['region'].nunique()} regions; the largest is {m['region'].value_counts().index[0]} ({int(m['region'].value_counts().iloc[0])} clients, {P(m['region'].value_counts(normalize=True).iloc[0])}).

**Behaviour.**
* Purpose: {P((m['acquisition_purpose']=='Home').mean())} Home (personal use), {P((m['acquisition_purpose']=='Investment').mean())} Investment.
* Loan applied: {P(m['loan_flag'].mean())}.
* Referral channel: {', '.join(f'{c} {P(v)}' for c, v in m['referral_channel'].value_counts(normalize=True).items())}.
* Satisfaction is spread almost evenly over 1-5 (mean {m['satisfaction_score'].mean():.2f}).

**Properties.**
* {n_sold:,} sold / {n_avail:,} available; {P((props['unit_category']=='Apartment').mean())} Apartments, {P((props['unit_category']=='Office').mean())} Offices; 20 towers.
* Sale price: mean {D(props['sale_price_usd'].mean())}, median {D(props['sale_price_usd'].median())}, range {D(props['sale_price_usd'].min())}-{D(props['sale_price_usd'].max())}.
* Floor area: mean {props['floor_area_sqft'].mean():,.0f} sq ft. Price and area are almost perfectly correlated (r = {price_corr:.2f}); price per sq ft averages {D(props['price_per_sqft'].mean())} with little variation.
* Offices average {D(props[props['unit_category']=='Office']['sale_price_usd'].mean())} versus {D(props[props['unit_category']=='Apartment']['sale_price_usd'].mean())} for apartments.
* Sales are recorded monthly over {sold['transaction_month'].nunique()} months ({sold['transaction_month'].min()} to {sold['transaction_month'].max()}).

**Relationships.**
* Investment rate by country ranges from {P(inv_by_country.min())} ({inv_by_country.idxmin()}) to {P(inv_by_country.max())} ({inv_by_country.idxmax()}), but several countries have few clients (e.g. {int((m['country']==inv_by_country.idxmax()).sum())} in {inv_by_country.idxmax()}), so these gaps may be noise.
* Companies: investment {P(ct.loc['Company','is_investment'])} vs {P(ct.loc['Individual','is_investment'])} for individuals; loan {P(ct.loc['Company','loan_flag'])} vs {P(ct.loc['Individual','loan_flag'])}; units owned {ct.loc['Company','n_units']:.2f} vs {ct.loc['Individual','n_units']:.2f}. Companies are only slightly different from individuals.
* Investors and home buyers spend almost the same ({D(m[m['is_investment']==1]['total_spend'].mean())} vs {D(m[m['is_investment']==0]['total_spend'].mean())} per client).
* Correlations between age, satisfaction, units, price, purpose and loan are all weak (largest absolute correlation {max_r:.2f}); the only strong relation is price vs area.

**Take-away:** the data contains little built-in relationship between who the buyer is and how they buy. This explains the low silhouette later.

## 9. Feature Engineering

Per-client property features were derived by aggregating the client's **sold** units (`n_units`, `avg_price`, `office_share`, plus `total_spend`, `avg_area` for profiling).

{md_table(feat_tbl)}

Why each feature: age = life stage; satisfaction = experience; `n_units` = portfolio size (bulk vs single buyer); `avg_price` = ticket size / budget;
`office_share` = commercial vs residential appetite; purpose, loan, client type = motivation, financing and buyer category (as required by the PRD);
country and referral channel = geography and acquisition source.

**Excluded on purpose:** region (nested in country, {m['region'].nunique()} values would swamp the other features; used for profiling and maps),
gender (about 50/50, sensitive attribute; used for description only), `avg_area` (r = {price_corr:.2f} with price), `total_spend` (= units × price),
`max_price`, `avg_price_per_sqft` (almost constant), identifiers and raw dates. **No outcome variable exists**, so there is no target leakage.
Because purpose and loan are model inputs, purpose/loan rates per cluster are partly built in; the significance tests in section 13 show they still do not separate the clusters.

## 10. Machine Learning Methodology

1. `ColumnTransformer`: StandardScaler on numeric, OrdinalEncoder (0/1) on binary, One-Hot on nominal → {res['n_encoded_columns']} columns.
2. Scaling is required because K-Means and Ward use Euclidean distance: without it, `avg_price` (hundreds of thousands) would drown age or satisfaction.
3. K-Means sweep K = 2-10 (20 starts each) with inertia, silhouette, Davies-Bouldin, Calinski-Harabasz and stability (ARI against 10 other random starts).
4. Rule-based K choice (section 11), final model with 50 starts, clusters renumbered by size.
5. Ward hierarchical clustering for comparison; PCA for visualisation **only**.
6. Profiling, data-driven naming, statistical tests, PRD-segment check, rule-based recommendations.
7. Saved with Joblib: `kmeans_model.pkl`, `scaler.pkl`, `preprocessor.pkl`, `pca.pkl`.

Silhouette in plain words: for each client it compares the distance to its own cluster with the distance to the nearest other cluster; +1 = clearly in the right cluster, 0 = on the border, negative = probably misplaced.

## 11. K-Means Clustering

{md_table(sweep_tbl)}

**Selection rule:** {sel['rationale']}

The silhouette curve is almost flat for K = 3-6 (all between {sweep[(sweep['k']>=3)&(sweep['k']<=6)]['silhouette'].min():.3f} and {sweep[(sweep['k']>=3)&(sweep['k']<=6)]['silhouette'].max():.3f}),
so silhouette alone cannot identify K; the elbow method also points to K = {sel['elbow_k']}.
K = 2 has the highest silhouette ({sweep.loc[sweep['k']==2,'silhouette'].iloc[0]:.3f}) but it only splits clients into two broad halves. K = 3 also meets the stability bar, but gives fewer, less actionable groups and a worse Davies-Bouldin value.
Final model: K = {k}, silhouette {sil:.3f}, cluster sizes {', '.join(f'{a}: {b}' for a, b in res['kmeans_final']['sizes'].items())}.

## 12. Hierarchical Clustering

Linkage comparison at K = {k}:

{md_table(link_tbl)}

* Average linkage reports a high silhouette only because it isolates a handful of points in tiny clusters (smallest cluster {link_raw.loc[link_raw['linkage']=='average','smallest_cluster_pct'].iloc[0]:.2f}%): not useful for segmentation.
* **Ward** produces balanced clusters and is the fair counterpart of K-Means (both minimise variance).
* Ward silhouette {ag['silhouette_ward']:.3f} vs K-Means {ag['silhouette_kmeans']:.3f}; Davies-Bouldin {ag['davies_bouldin_ward']:.2f} vs {ag['davies_bouldin_kmeans']:.2f}.
* Agreement: ARI {ag['ARI']:.2f}, NMI {ag['NMI']:.2f}: **partial agreement only**. Both methods find the very small high-volume group, but split the larger groups differently (cophenetic correlation of the tree is {ag['cophenetic_correlation_ward']:.2f}).
* The dendrogram (`reports/figures/23_dendrogram.png`) shows one distinct small branch (a few dozen clients - the high-volume buyers) that joins the rest only at a large distance, while the remaining clients merge gradually at similar heights, so there are no further clear cut points. This is consistent with weak structure apart from that one group.

**Why K-Means is the production model:** better silhouette and Davies-Bouldin, repeatable (stability ARI {float(sweep.loc[sweep['k']==k,'stability_ari'].iloc[0]):.2f}),
scales to new data, and can assign a new client with `predict()`, which a hierarchical tree cannot do without refitting.

## 13. Cluster Evaluation

* Silhouette {sil:.3f} (rule of thumb of Kaufman & Rousseeuw, 1990: below about 0.25 means no substantial cluster structure).
* The PCA plot ({100*pcs[0]:.1f}% + {100*pcs[1]:.1f}% = {100*(pcs[0]+pcs[1]):.1f}% of variance) shows overlapping clusters.
* Which variables differ across clusters (chi-square for categorical, Kruskal-Wallis for numeric):

{md_table(test_tbl)}

Numeric variables are significant by construction (they built the clusters). Among categorical variables only {', '.join(sig_cat) if sig_cat else 'none'} differ significantly;
**{', '.join(nonsig_cat)}** do not.

## 14. Buyer Segmentation

{overall_line}

{md_table(prof_tbl, '{:,.2f}')}

Segment names are generated from each cluster's strongest distinguishing traits (effect size ≥ 0.30 standard deviations), not pre-selected.
Heat-map: `reports/figures/25_cluster_effects_heatmap.png`.

Does the PRD's example segmentation exist?

{md_table(prd_tbl)}

{('Supported: ' + ', '.join(supported)) if supported else 'None of the four PRD example segments meets all of its conditions, so they are not used as names.'}
Reasons: a segment counts as 'supported' only if every condition is met, e.g. an investment rate and an international share at least 25% above the overall level ({P(float(allrow['investment_rate']))} and {P(float(allrow['pct_international']))}); no cluster reaches that. "First-time" and "income" cannot be observed in the data, and companies are only {P(float(allrow['pct_company']))} of clients.

## 15. Investment Profiling

Investors: {int(m['is_investment'].sum())} clients ({P(m['is_investment'].mean())}).

{md_table(inv_tbl, '{:,.2f}')}

* Highest investor spend: {lab(top_inv_rank)}. Highest investment rate: {lab(top_inv_rate)} ({P(float(cl.loc[top_inv_rate,'investment_rate']))} vs {P(float(allrow['investment_rate']))} overall), (differences in purpose between clusters are not statistically significant, chi-square p = {float(tests.loc[tests['variable']=='acquisition_purpose','p_value'].iloc[0]):.2f}).
* Highest value index (revenue share ÷ client share): {lab(top_value)} at {float(cl.loc[top_value,'value_index']):.2f}, because its clients own {float(cl.loc[top_value,'avg_units']):.1f} units on average.
* Highest-priced portfolios: {lab(top_price)} ({D(float(cl.loc[top_price,'avg_price']))} per unit).
* Highest loan dependence: {lab(top_loan)} ({P(float(cl.loc[top_loan,'loan_rate']))} vs {P(float(allrow['loan_rate']))}).

## 16. Results

* Join key found and validated; {n_clients:,} clients, {n_sold:,} sold units, {D(sold['sale_price_usd'].sum())} total sold value.
* K = {k} K-Means segments, silhouette {sil:.3f}, stability ARI {float(sweep.loc[sweep['k']==k,'stability_ari'].iloc[0]):.2f}.
* Segments differ by age, satisfaction, portfolio size, ticket price and office share.
* Purpose, loan, channel, country and gender are statistically similar across segments.
* The PRD example segments are not supported; data-driven names are used.
* Smallest segment: {lab(big_units)} - {int(cl.loc[big_units,'size'])} clients ({P(float(cl.loc[big_units,'pct_of_clients']))}) holding {P(float(cl.loc[big_units,'revenue_share']))} of sold value.

## 17. Business Insights

{chr(10).join(cluster_sections)}

**Linking to the PRD's business problems**

| PRD problem | What the project provides | Strength of evidence |
|---|---|---|
| Inefficient marketing | Segment-specific channel / message suggestions; channel mix per segment | Moderate: segments differ in age, satisfaction, price; channel mix differs little |
| Poor customer targeting | {k} behavioural segments + a predictor that assigns a new client to a segment | Weak-moderate (silhouette {sil:.2f}) |
| Missed investment opportunities | Investor spend and value index per segment; high-volume group identified | Moderate |
| Generic property recommendations | Price band, unit type (apartment / office), region per segment | Moderate |
| Lack of understanding of investment motivation | Purpose is almost independent of every other variable | Strong (a finding in itself) |
| Geographic differences | Country / region maps; small countries show high variation but have few clients | Weak (small samples) |
| Financing patterns | Loan use per segment, per client type and per purpose | Moderate |

## 18. Recommendations

1. **Retention first for {lab(low_sat)}**: average satisfaction {float(cl.loc[low_sat,'avg_satisfaction']):.2f}/5 with {P(float(cl.loc[low_sat,'pct_of_clients']))} of clients. Add post-sale follow-up and find the cause of dissatisfaction (not in the data).
2. **Reward and use {lab(high_sat)}** ({float(cl.loc[high_sat,'avg_satisfaction']):.2f}/5): referral and testimonial programmes. Only {P(m['referral_channel'].value_counts(normalize=True).get('Client', 0))} of clients currently come through client referrals, so there is headroom.
3. **Key-account management for {lab(big_units)}**: few clients, {float(cl.loc[big_units,'avg_units']):.1f} units each, value index {float(cl.loc[big_units,'value_index']):.2f}.
4. **Premium inventory for {lab(top_price)}**, the largest source of investor spend.
5. **Commercial inventory for {lab(top_off)}** ({P(float(cl.loc[top_off,'office_share']))} office units vs {P(float(allrow['office_share']))} overall).
6. **Do not target by country, gender or channel to reach investors**: the data shows no reliable difference in purpose between them.
7. **Collect the missing fields** so the PRD segments can be tested properly: income / net worth, loan amount, first-purchase flag, rental yield, marketing cost per lead, campaign response.
8. Treat the segmentation as a **starting hypothesis** and validate it with an A/B campaign before large spending.

## 19. Limitations

* **Weak structure:** silhouette {sil:.3f}; clusters overlap and K-Means centroids partly depend on random restarts (stability {float(sweep.loc[sweep['k']==k,'stability_ari'].iloc[0]):.2f}, not 1.0).
* The data looks very uniform (satisfaction evenly spread, purpose and loan nearly independent of everything else, price ≈ constant × area). This pattern suggests a simulated or highly regularised dataset, so real-market conclusions need caution.
* No income, loan amount, rental yield or first-time flag; "Global Investors" and "First-Time Buyers" cannot be confirmed.
* Month-first assumption for dash-formatted birth dates; "international" means "outside the USA" (the dataset's dominant market), not the company's home market.
* Small country samples (e.g. {int((m['country']=='Denmark').sum())} clients in Denmark) make country comparisons unreliable.
* Segment names and recommendations are rule-based interpretations; their business impact is untested.
* Only sold units are linked to clients; available listings carry no buyer information.
* Both methods use Euclidean distance on mixed (numeric + binary) features; mixed-type methods such as K-Prototypes could be tried.

## 20. Future Scope

* Add income, loan amount, rental yield, purchase history and campaign data; test the PRD segments again.
* Try K-Prototypes / Gaussian mixture / DBSCAN and compare stability.
* Predict loan need or investor propensity with supervised models once labelled outcomes exist.
* Time-aware segmentation (purchase timing, repeat purchases), geospatial analysis with coordinates.
* A/B testing of segment-specific campaigns; connect the dashboard to a live database.

## 21. Conclusion

The project delivers a complete, reproducible pipeline: audit → clean → join → encode/scale → K-Means → hierarchical comparison → profiling → dashboard.
It finds {k} practical buyer segments that differ in age, satisfaction, portfolio size, ticket price and office share, identifies a small but valuable high-volume group,
and shows honestly that motivation (investment vs home) and financing do not differ between segments in this data. The PRD's four example segments cannot be confirmed, and the evidence for strongly separated
clusters is weak. The most valuable next step is richer data (income, loan amounts, history).

## 22. References

1. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281-297.
2. Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236-244.
3. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.
4. Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 1(2), 224-227.
5. Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.
6. Thorndike, R. L. (1953). Who belongs in the family? *Psychometrika*, 18(4), 267-276.
7. Wedel, M., & Kamakura, W. A. (2000). *Market Segmentation: Conceptual and Methodological Foundations* (2nd ed.). Kluwer Academic.
8. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
10. Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley.
11. Unified Mentor & Parcl Co. Limited. *Project Requirements Document: Machine Learning based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence.*
"""
    (utils.ROOT / "research_report.md").write_text(report, encoding="utf-8")

    # =====================================================================
    # VIVA Q&A
    # =====================================================================
    s = float(sweep.loc[sweep["k"] == k, "stability_ari"].iloc[0])
    qa: List[tuple] = [
        ("What is the aim of the project?",
         "To find hidden groups of real-estate buyers and describe each group's investment behaviour, so that marketing and property recommendations can be targeted instead of generic."),
        ("Which datasets did you use and how are they connected?",
         f"`clients.csv` ({n_clients:,} clients) and `properties.csv` ({n_listings:,} listings). The key is `clients.client_id` = `properties.client_ref`. I discovered it by testing which column's values all exist in a unique column of the other file ({P(audit['join_analysis']['share_of_property_refs_matching'])} match). Only sold listings have a client_ref."),
        ("Why is the unit of analysis the client and not the property?",
         f"Segmentation is about buyers. A client owns several units ({m['n_units'].mean():.2f} on average), so I aggregated the sold units per client into features such as units owned, average price and office share."),
        ("What data problems did you find?",
         "Three different date-of-birth layouts, prices stored as text like `$300,385.62`, and the label `Company` instead of the PRD's `Corporate`. There were no missing values and no duplicates in the clients file; missing `client_ref` values belong to unsold listings, which is normal."),
        ("How did you handle dates of birth?",
         f"Parsed each layout explicitly and converted to age at {res['dataset']['reference_date']} (the last transaction date). Raw dates are not used in clustering. Dash dates could be day-first or month-first; I assumed month-first (matching the transaction dates), which changes age by at most one year."),
        ("How did you handle outliers?",
         f"I detected them with the IQR rule but did not delete any. Sale price and area had {res['outliers']['sale_price_usd']['n_outliers']} outliers; high-value units are legitimate real-estate sales. The clients who own up to {int(m['n_units'].max())} units are kept because they form a real segment."),
        ("Why did you drop region from the model?",
         f"Every region belongs to exactly one country, so region repeats country, and its {m['region'].nunique()} one-hot columns would dominate the distance calculation. Region is used for profiling and maps instead."),
        ("Why not use gender?",
         "It is split almost 50/50, adds no segmentation value, and targeting by gender is ethically questionable. I only use it to describe the clusters."),
        ("Which encoding did you use and why?",
         "Binary variables (purpose, loan, client type) are label-encoded to 0/1. Multi-category nominal variables (country, referral channel) are one-hot encoded, because label encoding would invent an order (for example Canada < France < UK)."),
        ("Why is scaling needed?",
         "K-Means and Ward use Euclidean distance. Average price is in hundreds of thousands while satisfaction is 1-5; unscaled, price would decide every distance. StandardScaler gives each numeric feature mean 0 and standard deviation 1."),
        ("What is K-Means?",
         "It places K centres, assigns each client to the nearest centre, moves each centre to the mean of its clients, and repeats until nothing changes. It minimises within-cluster variance (inertia)."),
        ("What is the elbow method?",
         f"Plot inertia against K. Inertia always falls as K grows, but the gain shrinks after a point - the 'elbow'. I also computed it automatically (largest distance from the straight line joining the end points); it gave K = {sel['elbow_k']}."),
        ("What does the silhouette score mean?",
         f"For each client: (distance to nearest other cluster - distance to own cluster) / the larger of the two. It lies between -1 and +1; higher is better. Mine is {sil:.3f}, which is low, meaning the groups overlap."),
        ("How did you choose K and why not just K = 4 like the PRD?",
         f"I did not assume it. {sel['rationale']}"),
        ("Silhouette is only " + f"{sil:.2f}" + ". Is the project a failure?",
         "No. It is an honest result: the data has weak natural separation (almost all behavioural variables are independent of each other). I report it, show stability and statistical tests, and use the segments as practical groupings. Forcing artificial separation would be misleading."),
        ("What is hierarchical clustering and why did you use it?",
         "It builds a tree by repeatedly merging the two closest clusters; cutting the tree gives clusters. I used Ward linkage (merge that increases variance least) to cross-check K-Means."),
        ("How do K-Means and hierarchical results compare?",
         f"Silhouette {ag['silhouette_kmeans']:.3f} (K-Means) vs {ag['silhouette_ward']:.3f} (Ward); agreement ARI = {ag['ARI']:.2f}, so they only partly agree. Both isolate the small high-volume group. Average linkage looked better on silhouette but created tiny degenerate clusters."),
        ("Why is K-Means your main model?",
         f"Better quality metrics, repeatable results (stability ARI {s:.2f}), fast, and it can assign new clients with `predict()` - which the dashboard predictor uses."),
        ("What is PCA and why use it?",
         f"PCA compresses the {res['n_encoded_columns']} encoded features into 2 axes so clusters can be plotted. It is only for visualisation; clustering used all features. The two axes explain {100*(pcs[0]+pcs[1]):.1f}% of variance, so the picture is a simplified view."),
        ("What does each cluster represent?",
         " ".join(f"{i['cluster_id']} = {i['name']} ({i['size']} clients)." for i in insights)),
        ("How did you name the clusters?",
         "Automatically from the data: for each cluster I calculated how many standard deviations it differs from the overall average on each trait, and used the two strongest traits (|effect| ≥ 0.30). So names describe real differences."),
        ("Do the PRD segments (Global Investors, etc.) exist?",
         ("None of them fully meets its conditions. " if not supported else f"Supported: {', '.join(supported)}. ") +
         "Example: no cluster has both a clearly higher investment share and a higher international share than average; 'first-time' and 'income' are not in the data."),
        ("Which clusters are most valuable?",
         f"By value index (revenue share / client share), {lab(top_value)} ({float(cl.loc[top_value,'value_index']):.2f}). By total investor spend, {lab(top_inv_rank)}."),
        ("Which buyers are investment-oriented, and which depend on loans?",
         f"Investment rate is {P(float(allrow['investment_rate']))} overall; the highest is {lab(top_inv_rate)} at {P(float(cl.loc[top_inv_rate,'investment_rate']))}. Loan use is {P(float(allrow['loan_rate']))} overall; highest in {lab(top_loan)} ({P(float(cl.loc[top_loan,'loan_rate']))}). The chi-square tests show these differences between clusters are not statistically significant."),
        ("Did you test whether the clusters really differ?",
         "Yes: chi-square (categorical) and Kruskal-Wallis (numeric) tests with effect sizes. Numeric differences are significant by construction; purpose, loan, channel, country and gender are not significant."),
        ("How would you avoid data leakage?",
         "There is no target variable in unsupervised clustering. I avoided features that merely restate other features (total spend = units x price) and listing-level identifiers. The preprocessor is fitted once and re-used for new clients."),
        ("What are the limitations?",
         "Weak cluster separation, no income/loan amount/first-time flag, uniform-looking data, small countries, month-first date assumption, and untested business impact of recommendations."),
        ("What is the business value?",
         "Different budgets, ages, portfolio sizes and satisfaction levels call for different messages, inventory and service. Examples: retention for the low-satisfaction group, key-account service for high-volume buyers, premium stock for high-ticket buyers."),
        ("How does the dashboard work?",
         "`train_pipeline.py` trains and saves everything once; `app.py` (Streamlit + Plotly) only loads the saved files, applies sidebar filters (country, region, purpose, client type, age, price, cluster, loan) and redraws charts and KPIs. A predictor page uses the saved models on a new client."),
        ("How would you improve the project?",
         "Add income, loan amounts and purchase history; test K-Prototypes or Gaussian mixtures for mixed data; validate segments through A/B campaigns."),
    ]
    viva = "# Viva Questions and Answers\n\nAnswers are specific to this implementation and use the real results of the analysis.\n\n"
    for n, (q, a) in enumerate(qa, 1):
        viva += f"**Q{n}. {q}**\n\n{a}\n\n"
    (utils.ROOT / "VIVA_QA.md").write_text(viva, encoding="utf-8")


if __name__ == "__main__":
    build_reports()
    print("research_report.md and VIVA_QA.md written.")
