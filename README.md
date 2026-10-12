# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

**Domain:** Financial Analytics · Real Estate Market Intelligence · Machine Learning
**Collaborating organisations:** Unified Mentor · Parcl Co. Limited

---

## 1. Project overview
An end-to-end machine-learning system that discovers hidden buyer segments in a real-estate portfolio, builds an
investment profile for each segment, and presents everything in an interactive Streamlit dashboard.
It uses the two supplied files: `data/clients.csv` (2,000 clients) and `data/properties.csv` (10,000 listings).

## 2. Problem statement
The business lacks a data-driven view of (a) the types of property buyers, (b) investment motivations,
(c) geographic differences and (d) financing patterns, causing inefficient marketing, generic property recommendations and poor investor targeting.

## 3. Objectives
Clean and join the data → explore it → cluster buyers (K-Means, validated with hierarchical clustering) → choose K with evidence →
profile each segment → test whether the PRD's example segments really exist → deliver a dashboard, a research report and recommendations.

## 4. Dataset description (discovered from the files, not assumed)
| File | Rows × Cols | Key fields |
|---|---|---|
| `clients.csv` | 2,000 × 12 | client_id, client_type (Individual/Company), date_of_birth, gender, country (10), region (57), acquisition_purpose (Home/Investment), satisfaction_score (1-5), loan_applied, referral_channel |
| `properties.csv` | 10,000 × 9 | listing_id, tower_number (20), transaction_date, unit_category (Apartment/Office), floor_area_sqft, sale_price (text, `$…`), listing_status (Sold/Available), client_ref |

**Relationship:** `properties.client_ref` → `clients.client_id` (100% of references match; only *Sold* listings have a client).
7,305 listings are sold; each client owns 3-13 units.

**Not in the data (but mentioned in the PRD):** income, bedrooms, loan amount, a first-time-buyer flag. The data says `Company`, the PRD says `Corporate`.

## 5. Technologies
Python · Pandas · NumPy · SciPy · Scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit · Joblib

## 6. Methodology (pipeline)
`audit → clean → join → feature engineering → encode + scale → K-Means sweep → choose K → final K-Means → hierarchical comparison → PCA → profiling → naming → tests → insights → save models → dashboard`

## 7. Data preprocessing
* Three mixed date-of-birth layouts parsed explicitly → **age** (raw DOB is not used in ML). Dash-formatted dates are assumed month-first (see report).
* `sale_price` text (`$300,385.62`) → numeric. `transaction_date` parsed (monthly data).
* Labels normalised (case/whitespace/synonyms), duplicates and missing-value rules implemented (the supplied data had none to fix).
* Outliers detected with the IQR rule and **kept** (high-value units are legitimate); names dropped for privacy.
* Sold units aggregated per client: `n_units`, `avg_price`, `office_share` (and `total_spend`, `avg_area` for profiling).

## 8. EDA
24 figures in `reports/figures/` (customer type, age, gender, purpose, loan, satisfaction, region, country, referral channel, property statistics,
price distribution, price vs area, monthly sales, relationship plots, correlation heat-map …) plus interactive Plotly versions in the dashboard.

## 9. Machine-learning methodology
One `ColumnTransformer`: **StandardScaler** (age, satisfaction, units, price, office share) · **0/1 label encoding** for binary variables
(purpose, loan, client type) · **One-Hot** for country and referral channel → 21 columns. Region and gender are *not* model inputs (reasons in the report).

## 10. K-Means
K = 2…10 evaluated with inertia, silhouette, Davies-Bouldin, Calinski-Harabasz and restart-stability. The K rule is documented in `src/clustering.py::select_k`
(business limit 3-6, silhouette within 0.01 of best, stability ≥ 0.6, then lowest Davies-Bouldin) → **K = 6**.

## 11. Hierarchical clustering
Ward / average / complete linkage compared; dendrogram saved; Ward agrees only partly with K-Means (ARI 0.25). K-Means is the production model (better metrics, repeatable, can score new clients).

## 12. Cluster evaluation (honest summary)
Silhouette = **0.123** → weak natural separation; clusters overlap. The clusters differ by **age, satisfaction, portfolio size, ticket price and office share**.
Purpose, loan use, referral channel, country and gender do **not** differ significantly between clusters (chi-square tests).

## 13. Segments found (names are generated from the data)
| Segment | Clients |
|---|---|
| C1 Office-Focused Senior Buyers | 437 |
| C2 Senior Buyers | 415 |
| C3 Highly Satisfied Younger Buyers | 397 |
| C4 Dissatisfied Younger Buyers | 387 |
| C5 Premium-Ticket Younger Buyers | 313 |
| C6 High-Volume Senior Buyers | 51 |

"Younger/Senior" are relative to the average client age (55.5). **None of the PRD's four example segments (Global Investors, First-Time Buyers, Corporate Buyers, Luxury Investors) is supported by this data**; the test is in `src/profiling.py::test_prd_segments` and shown in the dashboard.

## 14. Dashboard features
Executive Overview (7 KPI cards) · Buyer Segmentation (cluster sizes, profile table, heat-map, PCA scatter) · Investor Behaviour ·
Geographic Analysis (world / US-state maps, region charts) · Segment Insights (per-cluster profiles + recommendations, data-supported vs assumptions) ·
Data Explorer (CSV downloads) · Model Evaluation (elbow, silhouette, linkage comparison, PRD check, tests) · Segment Predictor (assigns a new client using the saved models).
Sidebar filters: country, region, acquisition purpose, client type, loan status, cluster, age range, unit-price range - all update the charts and KPIs.

## 15. Installation
```bash
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```
(Optional, only to open the notebook: `pip install jupyter`.)

## 16. How to run
```bash
python train_pipeline.py      # re-creates models, tables, figures, report (about 10-30 seconds)
streamlit run app.py          # opens http://localhost:8501
python tests/test_pipeline.py # optional: 9 automated checks
```
The generated files are already included, so `streamlit run app.py` works immediately. Run `python train_pipeline.py` once if you get a
scikit-learn "version" warning when loading the models, because pickles are tied to the library version.

## 17. Project structure
```
real_estate_buyer_segmentation/
├── data/                 clients.csv, properties.csv  (+ processed/ = generated tables & results)
├── models/               kmeans_model.pkl, scaler.pkl, preprocessor.pkl, pca.pkl, metadata.json
├── notebooks/            real_estate_segmentation.ipynb  (step-by-step, outputs included)
├── src/                  data_cleaning.py, feature_engineering.py, clustering.py, profiling.py,
│                         eda.py, report_builder.py, utils.py
├── reports/figures/      24 PNG charts
├── tests/                test_pipeline.py
├── train_pipeline.py     one-command pipeline
├── app.py                Streamlit dashboard
├── requirements.txt · README.md · research_report.md · VIVA_QA.md
```

## 18. Results
See `research_report.md` (all numbers are generated from the data by `src/report_builder.py`). Headlines: 6 segments; silhouette 0.123 (weak); a small
51-client high-volume group holds about 4.9% of sold value; no significant link between segment and purpose/loan/country/channel; PRD segments not supported.

## 19. Business insights
Retention programme for the low-satisfaction group · key-account management for high-volume buyers · premium inventory for the high-ticket group ·
commercial inventory for the office-focused group · don't expect country/gender/channel to predict investment purpose · collect income, loan amount and
purchase history to test the PRD segments properly. Details and assumptions per segment: report section 17 and the dashboard.

## 20. Future scope
Richer data (income, loan amount, history) · K-Prototypes / Gaussian mixtures · supervised investor-propensity model · A/B tests of segment campaigns · live database connection.

---
**Viva preparation:** see `VIVA_QA.md` (30 questions with answers based on this implementation).
