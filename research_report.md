# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

*Domain: Financial Analytics, Real Estate Market Intelligence, Machine Learning · Collaborating organisations: Unified Mentor, Parcl Co. Limited*

> All numbers in this report are generated directly from the supplied `clients.csv` and `properties.csv`
> by `train_pipeline.py` / `src/report_builder.py`. Nothing is simulated or typed by hand.

---

## 1. Abstract

This project segments the buyers of a real-estate portfolio and builds an investment profile for each segment.
Two datasets were supplied: **2,000 clients** and **10,000 property listings** (7,305 sold, 2,695 available).
They join through `clients.client_id = properties.client_ref`; every sold listing belongs to exactly one client and every client owns at least 3 units.
After cleaning, each client was described by 5 numeric features (age, satisfaction, portfolio size, average unit price, office share),
3 binary features (purpose, loan, client type) and 2 nominal features (country, referral channel), giving 21 encoded columns.
K-Means was run for K = 2 to 10; using elbow, silhouette, Davies-Bouldin and restart-stability together with a business limit of 3-6 segments, **K = 6** was selected.
The silhouette score is **0.123**, which indicates *weak* natural separation, so the segments should be read as practical groupings and not as sharply separated populations.
Ward hierarchical clustering partly agrees with K-Means (Adjusted Rand Index 0.25).
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
| Rows | 2,000 | 10,000 |
| Columns | 12 | 9 |
| Missing values | 0 | 2695 (all 2,695 are `client_ref` of *Available* listings, which is expected) |
| Duplicate rows | 0 | 0 |

**Clients columns:** `client_id`, `client_type`, `first_name`, `last_name`, `date_of_birth`, `gender`, `country`, `region`, `acquisition_purpose`, `satisfaction_score`, `loan_applied`, `referral_channel`.
**Properties columns:** `listing_id`, `tower_number`, `transaction_date`, `unit_category`, `unit_number`, `floor_area_sqft`, `sale_price`, `listing_status`, `client_ref`.

**Relationship.** The join analysis discovered `clients.client_id` (unique) ↔
`properties.client_ref`: 100.0% of references match, there are no orphan listings,
and all 2,000 clients own at least one unit.
Each client owns between 3 and 13 units (mean 3.65).

**Fields requested by the PRD but absent from the data:** income / net worth, bedrooms, loan amount, a first-time-buyer flag.
The PRD calls corporate clients *Corporate*; the data uses *Company*. The PRD's "multiple units purchased by companies" can only be tested through units owned per client.

## 7. Data Preprocessing

| Step | Decision |
|---|---|
| Duplicates | 0 duplicate rows / IDs found in total; code removes them if present. |
| Missing values | None in client fields. Rule applied if present: numeric → median, categorical → "Unknown". |
| Categorical labels | Whitespace / case normalised; gender M/F → Male/Female; synonyms mapped (e.g. Corporate → Company, USA aliases). |
| Dates | `date_of_birth` appears in three layouts (e.g. `M/D/YYYY`, `MM-DD-YYYY`). Parsed month-first. Slash dates contain days up to 31 in the second position, which proves month-first; dash dates have values ≤ 12 in both positions, so month-first is an **assumption** (consistent with `transaction_date`, which is also month-first); it can change an age by at most 1 year. |
| Age | Computed at 2025-12-01 (latest transaction date); raw DOB is **not** used in ML. Range 25.0-94.8 years, mean 55.5. |
| Price | `sale_price` is text such as `$300,385.62`; converted to numeric USD (currency assumed USD). |
| Outliers | IQR rule (1.5×IQR) used for **detection only**. Sale price: 0 outliers, floor area: 0, price per sq ft: 94 (0.94%). No rows removed because high-value units are legitimate. Units owned per client reaches 13 and is kept (it becomes a real segment). |
| Privacy | First and last names dropped. |
| Consistency checks | Sold listings all have a buyer; Available listings have none; every region maps to one country. |

## 8. Exploratory Data Analysis

Figures are in `reports/figures/` (static) and in the dashboard (interactive).

**Customers.**
* Client type: 94.8% Individual, 5.1% Company (103 clients).
* Age: mean 55.5, median 56.1, spread 17.3; the age groups are nearly evenly populated.
* Gender: 50.6% male, 49.4% female.
* Country: 76.9% of clients are in the USA (1,538); next are UK (95), Canada (85), Germany (56).
* Region: 57 regions; the largest is California (633 clients, 31.6%).

**Behaviour.**
* Purpose: 69.2% Home (personal use), 30.8% Investment.
* Loan applied: 36.8%.
* Referral channel: Website 55.1%, Agency 35.2%, Client 9.6%.
* Satisfaction is spread almost evenly over 1-5 (mean 3.03).

**Properties.**
* 7,305 sold / 2,695 available; 85.5% Apartments, 14.5% Offices; 20 towers.
* Sale price: mean $344,375, median $330,681, range $97,403-$736,652.
* Floor area: mean 1,140 sq ft. Price and area are almost perfectly correlated (r = 0.96); price per sq ft averages $302 with little variation.
* Offices average $362,465 versus $341,299 for apartments.
* Sales are recorded monthly over 24 months (2024-01 to 2025-12).

**Relationships.**
* Investment rate by country ranges from 20.5% (Australia) to 38.8% (Canada), but several countries have few clients (e.g. 85 in Canada), so these gaps may be noise.
* Companies: investment 35.0% vs 30.5% for individuals; loan 41.7% vs 36.5%; units owned 3.70 vs 3.65. Companies are only slightly different from individuals.
* Investors and home buyers spend almost the same ($1,253,142 vs $1,263,588 per client).
* Correlations between age, satisfaction, units, price, purpose and loan are all weak (largest absolute correlation 0.13); the only strong relation is price vs area.

**Take-away:** the data contains little built-in relationship between who the buyer is and how they buy. This explains the low silhouette later.

## 9. Feature Engineering

Per-client property features were derived by aggregating the client's **sold** units (`n_units`, `avg_price`, `office_share`, plus `total_spend`, `avg_area` for profiling).

| Feature | Type | Treatment |
|---|---|---|
| age | numeric | StandardScaler |
| satisfaction_score | numeric | StandardScaler |
| n_units | numeric | StandardScaler |
| avg_price | numeric | StandardScaler |
| office_share | numeric | StandardScaler |
| acquisition_purpose | binary | Label-encoded to 0/1 (OrdinalEncoder) |
| loan_applied | binary | Label-encoded to 0/1 (OrdinalEncoder) |
| client_type | binary | Label-encoded to 0/1 (OrdinalEncoder) |
| country | nominal | One-Hot Encoding |
| referral_channel | nominal | One-Hot Encoding |

Why each feature: age = life stage; satisfaction = experience; `n_units` = portfolio size (bulk vs single buyer); `avg_price` = ticket size / budget;
`office_share` = commercial vs residential appetite; purpose, loan, client type = motivation, financing and buyer category (as required by the PRD);
country and referral channel = geography and acquisition source.

**Excluded on purpose:** region (nested in country, 57 values would swamp the other features; used for profiling and maps),
gender (about 50/50, sensitive attribute; used for description only), `avg_area` (r = 0.96 with price), `total_spend` (= units × price),
`max_price`, `avg_price_per_sqft` (almost constant), identifiers and raw dates. **No outcome variable exists**, so there is no target leakage.
Because purpose and loan are model inputs, purpose/loan rates per cluster are partly built in; the significance tests in section 13 show they still do not separate the clusters.

## 10. Machine Learning Methodology

1. `ColumnTransformer`: StandardScaler on numeric, OrdinalEncoder (0/1) on binary, One-Hot on nominal → 21 columns.
2. Scaling is required because K-Means and Ward use Euclidean distance: without it, `avg_price` (hundreds of thousands) would drown age or satisfaction.
3. K-Means sweep K = 2-10 (20 starts each) with inertia, silhouette, Davies-Bouldin, Calinski-Harabasz and stability (ARI against 10 other random starts).
4. Rule-based K choice (section 11), final model with 50 starts, clusters renumbered by size.
5. Ward hierarchical clustering for comparison; PCA for visualisation **only**.
6. Profiling, data-driven naming, statistical tests, PRD-segment check, rule-based recommendations.
7. Saved with Joblib: `kmeans_model.pkl`, `scaler.pkl`, `preprocessor.pkl`, `pca.pkl`.

Silhouette in plain words: for each client it compares the distance to its own cluster with the distance to the nearest other cluster; +1 = clearly in the right cluster, 0 = on the border, negative = probably misplaced.

## 11. K-Means Clustering

| K | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Stability (ARI) | Smallest cluster % |
|---|---|---|---|---|---|---|
| 2 | 11,296 | 0.133 | 2.527 | 286.7 | 0.212 | 44.85 |
| 3 | 10,290 | 0.117 | 2.263 | 254.8 | 0.769 | 27.70 |
| 4 | 9,389 | 0.124 | 1.874 | 249.9 | 0.431 | 2.50 |
| 5 | 8,679 | 0.124 | 1.786 | 243.5 | 0.391 | 2.55 |
| 6 | 8,087 | 0.123 | 1.696 | 238.2 | 0.684 | 2.55 |
| 7 | 7,694 | 0.121 | 1.770 | 225.5 | 0.679 | 2.55 |
| 8 | 7,364 | 0.118 | 1.750 | 214.6 | 0.746 | 2.55 |
| 9 | 7,082 | 0.120 | 1.789 | 205.0 | 0.711 | 2.50 |
| 10 | 6,861 | 0.121 | 1.793 | 195.1 | 0.643 | 2.35 |

**Selection rule:** Elbow method suggests K=6; the unrestricted silhouette peak is K=2 (a trivial 2-way split). Restricting to 3-6 segments (business limit), K values with silhouette within 0.01 of the best (0.124): [3, 4, 5, 6]. Of these, K values with stability ARI >= 0.6: [3, 6]. K=6 has the lowest Davies-Bouldin index (1.696) among them (silhouette=0.123, stability ARI=0.68).

The silhouette curve is almost flat for K = 3-6 (all between 0.117 and 0.124),
so silhouette alone cannot identify K; the elbow method also points to K = 6.
K = 2 has the highest silhouette (0.133) but it only splits clients into two broad halves. K = 3 also meets the stability bar, but gives fewer, less actionable groups and a worse Davies-Bouldin value.
Final model: K = 6, silhouette 0.123, cluster sizes C1: 437, C2: 415, C3: 397, C4: 387, C5: 313, C6: 51.

## 12. Hierarchical Clustering

Linkage comparison at K = 6:

| linkage | silhouette | davies_bouldin | smallest_cluster_pct | ARI_vs_kmeans |
|---|---|---|---|---|
| ward | 0.090 | 2.207 | 2.00 | 0.245 |
| average | 0.182 | 1.053 | 0.05 | 0.017 |
| complete | 0.062 | 2.002 | 0.15 | 0.140 |

* Average linkage reports a high silhouette only because it isolates a handful of points in tiny clusters (smallest cluster 0.05%): not useful for segmentation.
* **Ward** produces balanced clusters and is the fair counterpart of K-Means (both minimise variance).
* Ward silhouette 0.090 vs K-Means 0.123; Davies-Bouldin 2.21 vs 1.69.
* Agreement: ARI 0.25, NMI 0.36: **partial agreement only**. Both methods find the very small high-volume group, but split the larger groups differently (cophenetic correlation of the tree is 0.34).
* The dendrogram (`reports/figures/23_dendrogram.png`) shows one distinct small branch (a few dozen clients - the high-volume buyers) that joins the rest only at a large distance, while the remaining clients merge gradually at similar heights, so there are no further clear cut points. This is consistent with weak structure apart from that one group.

**Why K-Means is the production model:** better silhouette and Davies-Bouldin, repeatable (stability ARI 0.68),
scales to new data, and can assign a new client with `predict()`, which a hierarchical tree cannot do without refitting.

## 13. Cluster Evaluation

* Silhouette 0.123 (rule of thumb of Kaufman & Rousseeuw, 1990: below about 0.25 means no substantial cluster structure).
* The PCA plot (18.4% + 16.0% = 34.3% of variance) shows overlapping clusters.
* Which variables differ across clusters (chi-square for categorical, Kruskal-Wallis for numeric):

| variable | test | p_value | effect_size | effect_size_name | significant_0.05 |
|---|---|---|---|---|---|
| acquisition_purpose | chi-square | 0.3447 | 0.053 | Cramer's V | False |
| loan_applied | chi-square | 0.1910 | 0.061 | Cramer's V | False |
| client_type | chi-square | 0.0001 | 0.116 | Cramer's V | True |
| referral_channel | chi-square | 0.1183 | 0.062 | Cramer's V | False |
| country | chi-square | 0.7752 | 0.061 | Cramer's V | False |
| gender | chi-square | 0.5829 | 0.043 | Cramer's V | False |
| age | Kruskal-Wallis | 0.0000 | 0.519 | eta-squared | True |
| satisfaction_score | Kruskal-Wallis | 0.0000 | 0.426 | eta-squared | True |
| n_units | Kruskal-Wallis | 0.0000 | 0.512 | eta-squared | True |
| avg_price | Kruskal-Wallis | 0.0000 | 0.440 | eta-squared | True |
| office_share | Kruskal-Wallis | 0.0000 | 0.512 | eta-squared | True |

Numeric variables are significant by construction (they built the clusters). Among categorical variables only client_type differ significantly;
**acquisition_purpose, loan_applied, referral_channel, country, gender** do not.

## 14. Buyer Segmentation

Overall: age 55.5, investment 30.8%, loan 36.8%, satisfaction 3.03, 3.65 units, unit price $347,090, office share 14.9%.

| Cluster | Clients | % clients | Avg age | % invest. | % loan | Satisf. | Units | Avg unit price | % office | % intl. | % company | Value index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 (Office-Focused Senior Buyers) | 437 | 21.9% | 65.90 | 31.1% | 34.8% | 2.86 | 3.58 | $331,202 | 37.6% | 23.3% | 2.5% | 0.94 |
| C2 (Senior Buyers) | 415 | 20.8% | 72.40 | 28.4% | 33.7% | 2.75 | 3.56 | $343,253 | 0.0% | 21.7% | 2.2% | 0.97 |
| C3 (Highly Satisfied Younger Buyers) | 397 | 19.9% | 46.30 | 27.7% | 41.6% | 4.53 | 3.69 | $311,773 | 7.4% | 25.2% | 6.8% | 0.91 |
| C4 (Dissatisfied Younger Buyers) | 387 | 19.4% | 40.00 | 32.6% | 36.7% | 1.67 | 3.60 | $322,880 | 10.3% | 22.0% | 8.5% | 0.92 |
| C5 (Premium-Ticket Younger Buyers) | 313 | 15.7% | 48.10 | 34.2% | 38.7% | 3.34 | 3.30 | $451,715 | 18.1% | 24.9% | 6.7% | 1.18 |
| C6 (High-Volume Senior Buyers) | 51 | 2.5% | 64.30 | 35.3% | 31.4% | 3.49 | 7.29 | $330,963 | 13.9% | 13.7% | 3.9% | 1.90 |

Segment names are generated from each cluster's strongest distinguishing traits (effect size ≥ 0.30 standard deviations), not pre-selected.
Heat-map: `reports/figures/25_cluster_effects_heatmap.png`.

Does the PRD's example segmentation exist?

| prd_segment | prd_description | closest_cluster_name | conditions_met | supported_by_data |
|---|---|---|---|---|
| Global Investors | High income, investment purchases (international) | Premium-Ticket Younger Buyers | 0/2 | False |
| First-Time Buyers | Younger, loan dependent | Highly Satisfied Younger Buyers | 1/3 | False |
| Corporate Buyers | Companies purchasing multiple units | High-Volume Senior Buyers | 1/2 | False |
| Luxury Investors | High satisfaction, large investments | Premium-Ticket Younger Buyers | 2/3 | False |

None of the four PRD example segments meets all of its conditions, so they are not used as names.
Reasons: a segment counts as 'supported' only if every condition is met, e.g. an investment rate and an international share at least 25% above the overall level (30.8% and 23.1%); no cluster reaches that. "First-time" and "income" cannot be observed in the data, and companies are only 5.1% of clients.

## 15. Investment Profiling

Investors: 615 clients (30.8%).

| Cluster | Investors | Invest. rate | Lift | Investor spend ($M) | % of investor spend | Units/investor | Value rank |
|---|---|---|---|---|---|---|---|
| C1 (Office-Focused Senior Buyers) | 136 | 31.1% | 1.01 | 157.00 | 20.4% | 3.54 | 2 |
| C2 (Senior Buyers) | 118 | 28.4% | 0.92 | 138.00 | 17.9% | 3.47 | 4 |
| C3 (Highly Satisfied Younger Buyers) | 110 | 27.7% | 0.90 | 126.90 | 16.5% | 3.70 | 5 |
| C4 (Dissatisfied Younger Buyers) | 126 | 32.6% | 1.06 | 142.70 | 18.5% | 3.57 | 3 |
| C5 (Premium-Ticket Younger Buyers) | 107 | 34.2% | 1.11 | 160.60 | 20.8% | 3.36 | 1 |
| C6 (High-Volume Senior Buyers) | 18 | 35.3% | 1.15 | 45.40 | 5.9% | 7.67 | 6 |

* Highest investor spend: C5 (Premium-Ticket Younger Buyers). Highest investment rate: C6 (High-Volume Senior Buyers) (35.3% vs 30.8% overall), (differences in purpose between clusters are not statistically significant, chi-square p = 0.34).
* Highest value index (revenue share ÷ client share): C6 (High-Volume Senior Buyers) at 1.90, because its clients own 7.3 units on average.
* Highest-priced portfolios: C5 (Premium-Ticket Younger Buyers) ($451,715 per unit).
* Highest loan dependence: C3 (Highly Satisfied Younger Buyers) (41.6% vs 36.8%).

## 16. Results

* Join key found and validated; 2,000 clients, 7,305 sold units, $2,520,750,961 total sold value.
* K = 6 K-Means segments, silhouette 0.123, stability ARI 0.68.
* Segments differ by age, satisfaction, portfolio size, ticket price and office share.
* Purpose, loan, channel, country and gender are statistically similar across segments.
* The PRD example segments are not supported; data-driven names are used.
* Smallest segment: C6 (High-Volume Senior Buyers) - 51 clients (2.5%) holding 4.9% of sold value.

## 17. Business Insights

### C1 - Office-Focused Senior Buyers (437 clients, 21.9%)

* **Customer profile:** Office-Focused Senior Buyers: average age 66, 50% female, 2.5% companies, 23% international.
* **Behavioural profile:** Average satisfaction 2.86/5 (below the overall 3.03); owns 3.6 units on average; most common channel is Website (58%).
* **Investment profile:** 31.1% buy for investment (lift 1.01x vs overall); this cluster holds 20.4% of all investor spend (136 investors, rank 2 of 6). Average unit price $331,202.
* **Financing profile:** 34.8% applied for a loan (in line with the overall 36.8%).
* **Geographic profile:** USA is the main country (77% of the cluster); main region California (30%).
* **Marketing recommendations:**
  * Advisor-led, relationship-based outreach (events, phone/in-person) for an older audience.
* **Property recommendations:**
  * Office / commercial units: 38% of their purchases are offices (overall 15%).
  * Geography does not set this segment apart (country mix is close to the overall mix), so use the standard regional inventory.
* **Assumptions (not proven by the data):**
  * Marketing recommendations are rule-based interpretations of the measured traits; their effect on conversion has not been tested (no campaign data available).

### C2 - Senior Buyers (415 clients, 20.8%)

* **Customer profile:** Senior Buyers: average age 72, 50% female, 2.2% companies, 22% international.
* **Behavioural profile:** Average satisfaction 2.75/5 (below the overall 3.03); owns 3.6 units on average; most common channel is Website (55%).
* **Investment profile:** 28.4% buy for investment (lift 0.92x vs overall); this cluster holds 17.9% of all investor spend (118 investors, rank 4 of 6). Average unit price $343,253.
* **Financing profile:** 33.7% applied for a loan (in line with the overall 36.8%).
* **Geographic profile:** USA is the main country (78% of the cluster); main region California (32%).
* **Marketing recommendations:**
  * Advisor-led, relationship-based outreach (events, phone/in-person) for an older audience.
* **Property recommendations:**
  * Residential apartments: this cluster essentially buys no offices.
  * Geography does not set this segment apart (country mix is close to the overall mix), so use the standard regional inventory.
* **Assumptions (not proven by the data):**
  * Marketing recommendations are rule-based interpretations of the measured traits; their effect on conversion has not been tested (no campaign data available).

### C3 - Highly Satisfied Younger Buyers (397 clients, 19.9%)

* **Customer profile:** Highly Satisfied Younger Buyers: average age 46, 51% female, 6.8% companies, 25% international.
* **Behavioural profile:** Average satisfaction 4.53/5 (above the overall 3.03); owns 3.7 units on average; most common channel is Website (50%).
* **Investment profile:** 27.7% buy for investment (lift 0.90x vs overall); this cluster holds 16.5% of all investor spend (110 investors, rank 5 of 6). Average unit price $311,773.
* **Financing profile:** 41.6% applied for a loan (above the overall 36.8%).
* **Geographic profile:** USA is the main country (75% of the cluster); main region California (32%).
* **Marketing recommendations:**
  * Referral / ambassador programme and testimonials: these are the happiest buyers (4.53/5).
  * Digital-first campaigns (social media, online calculators, virtual tours) for a younger audience.
* **Property recommendations:**
  * Value units: average price paid is $311,773 (overall $347,090) - lead with entry-price apartments.
  * Geography does not set this segment apart (country mix is close to the overall mix), so use the standard regional inventory.
* **Assumptions (not proven by the data):**
  * Marketing recommendations are rule-based interpretations of the measured traits; their effect on conversion has not been tested (no campaign data available).

### C4 - Dissatisfied Younger Buyers (387 clients, 19.4%)

* **Customer profile:** Dissatisfied Younger Buyers: average age 40, 48% female, 8.5% companies, 22% international.
* **Behavioural profile:** Average satisfaction 1.67/5 (below the overall 3.03); owns 3.6 units on average; most common channel is Website (55%).
* **Investment profile:** 32.6% buy for investment (lift 1.06x vs overall); this cluster holds 18.5% of all investor spend (126 investors, rank 3 of 6). Average unit price $322,880.
* **Financing profile:** 36.7% applied for a loan (in line with the overall 36.8%).
* **Geographic profile:** USA is the main country (78% of the cluster); main region California (31%).
* **Marketing recommendations:**
  * Service-recovery and retention programme (follow-up calls, post-sale support): satisfaction is only 1.67/5.
  * Digital-first campaigns (social media, online calculators, virtual tours) for a younger audience.
* **Property recommendations:**
  * Geography does not set this segment apart (country mix is close to the overall mix), so use the standard regional inventory.
* **Assumptions (not proven by the data):**
  * The reason for low satisfaction is not in the data (could be price, service or delivery).
  * Marketing recommendations are rule-based interpretations of the measured traits; their effect on conversion has not been tested (no campaign data available).

### C5 - Premium-Ticket Younger Buyers (313 clients, 15.7%)

* **Customer profile:** Premium-Ticket Younger Buyers: average age 48, 49% female, 6.7% companies, 25% international.
* **Behavioural profile:** Average satisfaction 3.34/5 (above the overall 3.03); owns 3.3 units on average; most common channel is Website (55%).
* **Investment profile:** 34.2% buy for investment (lift 1.11x vs overall); this cluster holds 20.8% of all investor spend (107 investors, rank 1 of 6). Average unit price $451,715.
* **Financing profile:** 38.7% applied for a loan (in line with the overall 36.8%).
* **Geographic profile:** USA is the main country (75% of the cluster); main region California (31%).
* **Marketing recommendations:**
  * Premium positioning: private viewings and a dedicated advisor for high-ticket buyers (they pay $451,715 per unit on average).
  * Digital-first campaigns (social media, online calculators, virtual tours) for a younger audience.
* **Property recommendations:**
  * Premium units: this cluster pays $451,715 per unit on average (overall $347,090) - show the upper price band first.
  * Geography does not set this segment apart (country mix is close to the overall mix), so use the standard regional inventory.
* **Assumptions (not proven by the data):**
  * Marketing recommendations are rule-based interpretations of the measured traits; their effect on conversion has not been tested (no campaign data available).

### C6 - High-Volume Senior Buyers (51 clients, 2.5%)

* **Customer profile:** High-Volume Senior Buyers: average age 64, 37% female, 3.9% companies, 14% international.
* **Behavioural profile:** Average satisfaction 3.49/5 (above the overall 3.03); owns 7.3 units on average; most common channel is Website (71%).
* **Investment profile:** 35.3% buy for investment (lift 1.15x vs overall); this cluster holds 5.9% of all investor spend (18 investors, rank 6 of 6). Average unit price $330,963.
* **Financing profile:** 31.4% applied for a loan (below the overall 36.8%).
* **Geographic profile:** USA is the main country (86% of the cluster); main region California (47%).
* **Marketing recommendations:**
  * Dedicated relationship manager and multi-unit / bulk-purchase packages: clients here buy 7.3 units on average.
  * Advisor-led, relationship-based outreach (events, phone/in-person) for an older audience.
  * Prioritise the Website channel: 71% of this cluster comes from it (overall 55%).
* **Property recommendations:**
  * Offer several units in the same tower or adjacent towers as bundles.
  * Over-represented market: USA (86% vs 77% overall).
* **Assumptions (not proven by the data):**
  * Large portfolios suggest portfolio-building behaviour, but the data does not show a higher 'Investment' share (35% vs 31% overall).
  * Marketing recommendations are rule-based interpretations of the measured traits; their effect on conversion has not been tested (no campaign data available).


**Linking to the PRD's business problems**

| PRD problem | What the project provides | Strength of evidence |
|---|---|---|
| Inefficient marketing | Segment-specific channel / message suggestions; channel mix per segment | Moderate: segments differ in age, satisfaction, price; channel mix differs little |
| Poor customer targeting | 6 behavioural segments + a predictor that assigns a new client to a segment | Weak-moderate (silhouette 0.12) |
| Missed investment opportunities | Investor spend and value index per segment; high-volume group identified | Moderate |
| Generic property recommendations | Price band, unit type (apartment / office), region per segment | Moderate |
| Lack of understanding of investment motivation | Purpose is almost independent of every other variable | Strong (a finding in itself) |
| Geographic differences | Country / region maps; small countries show high variation but have few clients | Weak (small samples) |
| Financing patterns | Loan use per segment, per client type and per purpose | Moderate |

## 18. Recommendations

1. **Retention first for C4 (Dissatisfied Younger Buyers)**: average satisfaction 1.67/5 with 19.4% of clients. Add post-sale follow-up and find the cause of dissatisfaction (not in the data).
2. **Reward and use C3 (Highly Satisfied Younger Buyers)** (4.53/5): referral and testimonial programmes. Only 9.6% of clients currently come through client referrals, so there is headroom.
3. **Key-account management for C6 (High-Volume Senior Buyers)**: few clients, 7.3 units each, value index 1.90.
4. **Premium inventory for C5 (Premium-Ticket Younger Buyers)**, the largest source of investor spend.
5. **Commercial inventory for C1 (Office-Focused Senior Buyers)** (37.6% office units vs 14.9% overall).
6. **Do not target by country, gender or channel to reach investors**: the data shows no reliable difference in purpose between them.
7. **Collect the missing fields** so the PRD segments can be tested properly: income / net worth, loan amount, first-purchase flag, rental yield, marketing cost per lead, campaign response.
8. Treat the segmentation as a **starting hypothesis** and validate it with an A/B campaign before large spending.

## 19. Limitations

* **Weak structure:** silhouette 0.123; clusters overlap and K-Means centroids partly depend on random restarts (stability 0.68, not 1.0).
* The data looks very uniform (satisfaction evenly spread, purpose and loan nearly independent of everything else, price ≈ constant × area). This pattern suggests a simulated or highly regularised dataset, so real-market conclusions need caution.
* No income, loan amount, rental yield or first-time flag; "Global Investors" and "First-Time Buyers" cannot be confirmed.
* Month-first assumption for dash-formatted birth dates; "international" means "outside the USA" (the dataset's dominant market), not the company's home market.
* Small country samples (e.g. 15 clients in Denmark) make country comparisons unreliable.
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
It finds 6 practical buyer segments that differ in age, satisfaction, portfolio size, ticket price and office share, identifies a small but valuable high-volume group,
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
