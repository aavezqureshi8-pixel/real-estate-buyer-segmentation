"""
data_cleaning.py
----------------
Everything that happens BEFORE machine learning:

1. inspect_datasets()      -> audit of both raw CSV files (rows, types, nulls, keys ...)
2. clean_clients()         -> fix labels, dates, missing values, duplicates, compute age
3. clean_properties()      -> parse price/date, validate status vs. buyer, flag outliers
4. build_client_features() -> aggregate what each client BOUGHT (property-side features)
5. build_master_dataset()  -> one clean row per client (the table used for clustering)

Every decision is written to a "cleaning log" so it can be shown in the report / viva.
Nothing here is hard-coded to row counts; columns are detected where reasonable.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import RAW_CLIENTS, RAW_PROPERTIES, get_logger

log = get_logger()

# --------------------------------------------------------------------------
# 0. Loading
# --------------------------------------------------------------------------

def load_raw_data(clients_path=RAW_CLIENTS, properties_path=RAW_PROPERTIES) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw CSV files exactly as provided (no cleaning yet)."""
    try:
        clients = pd.read_csv(clients_path, dtype=str, keep_default_na=True)
        properties = pd.read_csv(properties_path, dtype=str, keep_default_na=True)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Could not find the data file: {err.filename}. "
            "Place clients.csv and properties.csv inside the data/ folder."
        ) from err
    return clients, properties


# --------------------------------------------------------------------------
# 1. Dataset audit (STEP 1 of the workflow)
# --------------------------------------------------------------------------

def _guess_key_columns(clients: pd.DataFrame, properties: pd.DataFrame) -> Dict:
    """
    Discover a join key instead of assuming one.
    A column in `properties` is a candidate foreign key when (almost) all of its
    non-null values exist inside a unique column of `clients`.
    """
    best = {"found": False}
    for c_col in clients.columns:
        c_vals = clients[c_col].dropna()
        if not c_vals.is_unique:
            continue                                   # a primary key must be unique
        c_set = set(c_vals.astype(str))
        for p_col in properties.columns:
            p_vals = properties[p_col].dropna().astype(str)
            if len(p_vals) == 0:
                continue
            overlap = p_vals.isin(c_set).mean()
            if overlap >= 0.99 and p_vals.nunique() > 1:
                cand = {
                    "found": True,
                    "primary_key_in_clients": c_col,
                    "foreign_key_in_properties": p_col,
                    "share_of_property_refs_matching": float(overlap),
                    "clients_with_at_least_one_property": int(pd.Series(list(c_set)).isin(set(p_vals)).sum()),
                    "total_clients": int(len(c_set)),
                    "properties_with_null_reference": int(properties[p_col].isna().sum()),
                }
                if not best["found"] or overlap > best["share_of_property_refs_matching"]:
                    best = cand
    return best


def inspect_datasets(clients: pd.DataFrame, properties: pd.DataFrame, verbose: bool = True) -> Dict:
    """Profile both raw files. Returns a dictionary (also saved by the pipeline)."""
    audit: Dict = {}
    for name, df in (("clients", clients), ("properties", properties)):
        audit[name] = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
            "duplicate_rows": int(df.duplicated().sum()),
            "unique_values": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
            # small categorical columns: show distribution
            "categorical_distributions": {
                c: df[c].value_counts(dropna=False).head(15).to_dict()
                for c in df.columns if df[c].nunique(dropna=True) <= 12
            },
        }
    audit["clients"]["duplicate_client_id"] = int(clients["client_id"].duplicated().sum()) \
        if "client_id" in clients.columns else None
    audit["properties"]["duplicate_listing_id"] = int(properties["listing_id"].duplicated().sum()) \
        if "listing_id" in properties.columns else None

    # Detect the format of text-encoded date / money columns
    def shape_of(series: pd.Series) -> Dict:
        return series.dropna().astype(str).str.replace(r"\d", "9", regex=True).value_counts().head(6).to_dict()

    audit["format_patterns"] = {
        "clients.date_of_birth": shape_of(clients["date_of_birth"]) if "date_of_birth" in clients else {},
        "properties.transaction_date": shape_of(properties["transaction_date"]) if "transaction_date" in properties else {},
        "properties.sale_price": shape_of(properties["sale_price"]) if "sale_price" in properties else {},
    }
    audit["join_analysis"] = _guess_key_columns(clients, properties)

    # What the PRD asks for but the data does NOT contain
    audit["fields_missing_vs_prd"] = {
        "income / net worth": "not present (PRD mentions 'high income')",
        "bedrooms / property type beyond Apartment-Office": "not present",
        "loan amount": "not present (only a Yes/No flag)",
        "first-time-buyer flag": "not present (every client owns 3+ units)",
        "client_type value 'Corporate'": "dataset uses 'Company'",
    }
    if verbose:
        print("=" * 70)
        for name in ("clients", "properties"):
            a = audit[name]
            print(f"{name.upper()}: {a['rows']} rows x {a['columns']} columns")
            print("  columns:", a["column_names"])
            print("  missing values:", sum(a["missing_values"].values()),
                  "| duplicate rows:", a["duplicate_rows"])
        print("JOIN:", audit["join_analysis"])
        print("=" * 70)
    return audit


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def parse_mixed_dates(series: pd.Series, formats: List[str]) -> pd.Series:
    """
    Parse a text column that mixes several date layouts.
    Formats are tried in order; each value keeps the first format that works.
    Unparseable values become NaT (never silently wrong).
    """
    s = series.astype("object").where(series.notna(), None)
    s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in formats:
        todo = result.isna() & s.notna()
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        result = result.combine_first(parsed.astype("datetime64[ns]"))
    return result


def iqr_outlier_report(df: pd.DataFrame, columns: List[str], k: float = 1.5) -> Dict:
    """Tukey IQR rule: flag values outside [Q1 - k*IQR, Q3 + k*IQR]. Only REPORTS - never deletes."""
    report = {}
    for col in columns:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        report[col] = {"lower_bound": float(lo), "upper_bound": float(hi),
                       "n_outliers": n_out, "pct_outliers": float(100 * n_out / max(len(x), 1)),
                       "min": float(x.min()), "max": float(x.max())}
    return report


def _clean_text(s: pd.Series) -> pd.Series:
    """Strip whitespace and collapse repeated spaces; keep NaN as NaN."""
    return s.astype("object").map(lambda v: " ".join(v.split()) if isinstance(v, str) else v)


# --------------------------------------------------------------------------
# 2. Cleaning clients
# --------------------------------------------------------------------------

def clean_clients(raw: pd.DataFrame, reference_date: pd.Timestamp) -> Tuple[pd.DataFrame, List[Dict]]:
    """Clean the clients table. Returns (clean_df, cleaning_log)."""
    df = raw.copy()
    steps: List[Dict] = []

    def note(step: str, detail: str, **numbers):
        steps.append({"table": "clients", "step": step, "detail": detail, **numbers})

    n0 = len(df)
    df.columns = [c.strip().lower() for c in df.columns]

    # ---- duplicates -------------------------------------------------------
    # (a) exact duplicate rows, (b) the same client_id repeated -> keep first
    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates()
    n_id = int(df["client_id"].duplicated().sum())
    df = df.drop_duplicates(subset="client_id", keep="first")
    note("duplicates", "Removed exact duplicate rows and repeated client_id values (kept first)",
         exact_duplicates_removed=n_exact, duplicate_ids_removed=n_id, rows_before=n0, rows_after=len(df))

    # ---- text normalisation ----------------------------------------------
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = _clean_text(df[col])
    df["client_id"] = df["client_id"].str.upper()

    # client_type : Individual / Company  (PRD calls it Corporate)
    ct = df["client_type"].str.title().replace(
        {"Corporate": "Company", "Corp": "Company", "Business": "Company", "Firm": "Company",
         "Person": "Individual", "Private": "Individual"})
    before = df["client_type"].nunique()
    df["client_type"] = ct
    note("normalise client_type", "Case/whitespace fixed, synonyms of Corporate mapped to 'Company' (label used in the data)",
         unique_before=int(before), unique_after=int(df["client_type"].nunique()))

    # gender
    g = df["gender"].str.strip().str.upper().str[0]
    df["gender"] = g.map({"M": "Male", "F": "Female"}).fillna("Unknown")
    note("normalise gender", "M/F -> Male/Female; anything else -> 'Unknown'")

    # acquisition_purpose : Home (personal use) / Investment
    ap = df["acquisition_purpose"].str.title().replace(
        {"Personal Use": "Home", "Personal": "Home", "Residential": "Home", "Own Use": "Home",
         "Invest": "Investment"})
    df["acquisition_purpose"] = ap
    note("normalise acquisition_purpose", "Case fixed; 'Personal use' style labels mapped to 'Home'")

    # loan_applied : Yes / No
    la = df["loan_applied"].str.strip().str.lower().map(
        {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "no": "No", "n": "No", "false": "No", "0": "No"})
    df["loan_applied"] = la
    note("normalise loan_applied", "yes/y/true/1 -> Yes ; no/n/false/0 -> No")

    # country / region / channel
    df["country"] = df["country"].str.strip().replace(
        {"United States": "USA", "US": "USA", "U.S.A.": "USA", "United States of America": "USA",
         "United Kingdom": "UK", "Great Britain": "UK", "GB": "UK"})
    df["region"] = df["region"].str.strip()
    df["referral_channel"] = df["referral_channel"].str.strip().str.title()
    note("normalise country/region/referral_channel", "Whitespace/case fixed; USA & UK aliases unified")

    # check geography consistency: each region should belong to ONE country
    reg_country = df.groupby("region")["country"].nunique()
    multi = reg_country[reg_country > 1].index.tolist()
    note("geography consistency", "Every region should map to exactly one country",
         regions_in_more_than_one_country=len(multi))

    # ---- satisfaction score (numeric, 1-5) --------------------------------
    sat = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    n_bad = int(((sat < 1) | (sat > 5)).sum())
    sat = sat.where((sat >= 1) & (sat <= 5))          # impossible ratings -> missing
    df["satisfaction_score"] = sat
    note("validate satisfaction_score", "Scores must be within 1-5; others set to missing", invalid_values=n_bad)

    # ---- date of birth -> age --------------------------------------------
    # The file mixes layouts: 'M/D/YYYY' (slashes) and 'MM-DD-YYYY' (dashes).
    # Slash dates contain days up to 31 in the 2nd position => month-first.
    # Dash dates only contain numbers <= 12 in BOTH positions, so day/month cannot be proven;
    # month-first is assumed (the transaction_date column in properties.csv is also month-first).
    # Age in whole years is affected by at most 1 year if this assumption were wrong.
    dob = parse_mixed_dates(df["date_of_birth"], ["%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"])
    n_unparsed = int(dob.isna().sum())
    df["date_of_birth"] = dob
    df["age"] = (reference_date - dob).dt.days / 365.25
    # Implausible ages (<18 or >100) are treated as errors
    n_implausible = int(((df["age"] < 18) | (df["age"] > 100)).sum())
    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan
    note("date_of_birth -> age",
         f"Parsed mixed date layouts; age = (reference date {reference_date.date()} - DOB)/365.25. "
         "Raw DOB is NOT used in ML, only age.",
         unparseable_dob=n_unparsed, implausible_ages=n_implausible)
    df["age_group"] = pd.cut(df["age"], bins=[0, 35, 45, 55, 65, 75, 120],
                             labels=["<35", "35-44", "45-54", "55-64", "65-74", "75+"], right=False).astype(object)

    # ---- missing values ---------------------------------------------------
    # numeric -> median ; categorical -> 'Unknown'
    num_cols = [c for c in ["age", "satisfaction_score"] if c in df.columns]
    cat_cols = [c for c in ["client_type", "gender", "country", "region", "acquisition_purpose",
                            "loan_applied", "referral_channel"] if c in df.columns]
    imputed = {}
    for c in num_cols:
        n_missing = int(df[c].isna().sum())
        if n_missing:
            df[c] = df[c].fillna(df[c].median())
        imputed[c] = n_missing
    for c in cat_cols:
        n_missing = int(df[c].isna().sum())
        if n_missing:
            df[c] = df[c].fillna("Unknown")
        imputed[c] = n_missing
    df["age_group"] = df["age_group"].fillna("Unknown")
    note("missing values", "Numeric -> median, categorical -> 'Unknown' (counts below are values that needed filling)",
         **{f"filled_{k}": v for k, v in imputed.items()})

    # ---- helper flags (0/1) for easy averaging ---------------------------
    df["is_investment"] = (df["acquisition_purpose"] == "Investment").astype(int)
    df["loan_flag"] = (df["loan_applied"] == "Yes").astype(int)
    df["is_company"] = (df["client_type"] == "Company").astype(int)
    # 'International' = lives outside the dominant home market of the dataset.
    home_market = df["country"].mode().iloc[0]
    df["is_international"] = (df["country"] != home_market).astype(int)
    note("derived flags",
         f"is_investment, loan_flag, is_company, is_international (country != most common country '{home_market}')")

    # PII is not needed for analysis -> drop names
    df = df.drop(columns=[c for c in ["first_name", "last_name"] if c in df.columns])
    note("privacy", "first_name / last_name dropped (personal identifiers, no analytical value)")
    return df.reset_index(drop=True), steps


# --------------------------------------------------------------------------
# 3. Cleaning properties
# --------------------------------------------------------------------------

def clean_properties(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """Clean the properties table. Returns (clean_df, log, outlier_report)."""
    df = raw.copy()
    steps: List[Dict] = []

    def note(step: str, detail: str, **numbers):
        steps.append({"table": "properties", "step": step, "detail": detail, **numbers})

    n0 = len(df)
    df.columns = [c.strip().lower() for c in df.columns]

    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates()
    n_id = int(df["listing_id"].duplicated().sum())
    df = df.drop_duplicates(subset="listing_id", keep="first")
    note("duplicates", "Removed exact duplicates and repeated listing_id values",
         exact_duplicates_removed=n_exact, duplicate_ids_removed=n_id, rows_before=n0, rows_after=len(df))

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = _clean_text(df[col])

    # numeric columns stored as text
    for col in ["listing_id", "tower_number", "unit_number"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["floor_area_sqft"] = pd.to_numeric(df["floor_area_sqft"], errors="coerce")

    # price is text like "$300,385.62" -> numeric (currency symbol "$" assumed to be USD)
    df["sale_price_usd"] = pd.to_numeric(
        df["sale_price"].astype("object").str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
    df = df.drop(columns=["sale_price"])
    note("sale_price -> numeric", "Removed '$' and ',' and converted to float (assumed USD)",
         unparseable=int(df["sale_price_usd"].isna().sum()))

    # transaction_date: all values look like 'MM-01-YYYY' (first day of the month, month-first)
    df["transaction_date"] = parse_mixed_dates(df["transaction_date"], ["%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d"])
    df["transaction_month"] = df["transaction_date"].dt.to_period("M").astype(str)
    note("transaction_date", "Parsed as month-first; every value is the 1st of a month, so data is monthly",
         unparseable=int(df["transaction_date"].isna().sum()),
         first_date=str(df["transaction_date"].min().date()), last_date=str(df["transaction_date"].max().date()))

    df["unit_category"] = df["unit_category"].str.title()
    df["listing_status"] = df["listing_status"].str.title()
    df["client_ref"] = df["client_ref"].str.upper().replace({"": np.nan})

    # invalid numbers -> missing -> median
    for col in ["floor_area_sqft", "sale_price_usd"]:
        bad = int((df[col] <= 0).sum())
        df.loc[df[col] <= 0, col] = np.nan
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna(df[col].median())
        note(f"validate {col}", "Non-positive values treated as missing and filled with median",
             non_positive=bad, filled=n_missing)
    for col in ["unit_category", "listing_status"]:
        df[col] = df[col].fillna("Unknown")

    df["price_per_sqft"] = df["sale_price_usd"] / df["floor_area_sqft"]

    # consistency: Sold listings should have a buyer; Available ones should not
    sold_no_buyer = int(((df["listing_status"] == "Sold") & df["client_ref"].isna()).sum())
    avail_with_buyer = int(((df["listing_status"] == "Available") & df["client_ref"].notna()).sum())
    note("status vs buyer consistency", "Sold listings must have client_ref, Available must not",
         sold_without_buyer=sold_no_buyer, available_with_buyer=avail_with_buyer)

    # outliers: REPORT only. High-value units are legitimate real-estate sales.
    outliers = iqr_outlier_report(df, ["sale_price_usd", "floor_area_sqft", "price_per_sqft"])
    note("outliers", "IQR (1.5xIQR) rule used for detection only; no rows removed because extreme values "
                     "are valid high-value units, not errors",
         **{f"outliers_{k}": v["n_outliers"] for k, v in outliers.items()})
    df["is_price_outlier"] = ((df["sale_price_usd"] < outliers["sale_price_usd"]["lower_bound"]) |
                              (df["sale_price_usd"] > outliers["sale_price_usd"]["upper_bound"])).astype(int)
    return df.reset_index(drop=True), steps, outliers


# --------------------------------------------------------------------------
# 4/5. Joining: property-side features per client + master table
# --------------------------------------------------------------------------

def build_client_features(properties: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SOLD listings per client (client_ref).
    Only sold units are used because only they belong to a buyer.
    """
    sold = properties[(properties["listing_status"] == "Sold") & properties["client_ref"].notna()]
    grp = sold.groupby("client_ref")
    feats = pd.DataFrame({
        "n_units": grp["listing_id"].count(),
        "total_spend": grp["sale_price_usd"].sum(),
        "avg_price": grp["sale_price_usd"].mean(),
        "max_price": grp["sale_price_usd"].max(),
        "avg_area": grp["floor_area_sqft"].mean(),
        "avg_price_per_sqft": grp["price_per_sqft"].mean(),
        "office_share": grp["unit_category"].apply(lambda x: float((x == "Office").mean())),
        "n_towers": grp["tower_number"].nunique(),
        "first_purchase": grp["transaction_date"].min(),
        "last_purchase": grp["transaction_date"].max(),
    })
    feats.index.name = "client_id"
    return feats.reset_index()


def build_master_dataset(clients: pd.DataFrame, properties: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Join clients with the per-client property features (LEFT join on client_id = client_ref).
    Returns (master_clients, transactions, log).
    """
    steps: List[Dict] = []
    feats = build_client_features(properties)
    master = clients.merge(feats, on="client_id", how="left", validate="one_to_one")

    n_no_purchase = int(master["n_units"].isna().sum())
    # clients without any purchase: counts are 0, averages -> median
    for col in ["n_units", "total_spend", "office_share", "n_towers"]:
        master[col] = master[col].fillna(0)
    for col in ["avg_price", "max_price", "avg_area", "avg_price_per_sqft"]:
        master[col] = master[col].fillna(master[col].median())
    # plain numpy dtypes (scikit-learn does not like pandas nullable integers)
    master["n_units"] = master["n_units"].astype(int)
    master["n_towers"] = master["n_towers"].astype(int)
    for col in ["total_spend", "office_share", "avg_price", "max_price", "avg_area", "avg_price_per_sqft"]:
        master[col] = master[col].astype(float)
    steps.append({"table": "master", "step": "join clients + properties",
                  "detail": "LEFT JOIN clients.client_id = properties.client_ref on SOLD listings aggregated per client",
                  "clients": int(len(master)), "clients_without_purchase": n_no_purchase})

    # transaction-level table (one row per sold listing with buyer attributes)
    buyer_cols = ["client_id", "client_type", "gender", "country", "region", "acquisition_purpose",
                  "loan_applied", "referral_channel", "satisfaction_score", "age", "age_group"]
    sold = properties[(properties["listing_status"] == "Sold") & properties["client_ref"].notna()]
    transactions = sold.merge(master[buyer_cols], left_on="client_ref", right_on="client_id", how="left")
    unmatched = int(transactions["client_id"].isna().sum())
    steps.append({"table": "master", "step": "transaction table",
                  "detail": "Sold listings joined to buyer attributes", "rows": int(len(transactions)),
                  "orphan_listings_without_matching_client": unmatched})
    return master, transactions, steps


def run_cleaning(clients_path=RAW_CLIENTS, properties_path=RAW_PROPERTIES):
    """
    One call that does the whole cleaning stage.
    Returns dict with: raw_clients, raw_properties, audit, clients, properties,
    master, transactions, log, outliers, reference_date
    """
    raw_c, raw_p = load_raw_data(clients_path, properties_path)
    audit = inspect_datasets(raw_c, raw_p, verbose=False)

    # clean properties FIRST: its latest transaction date is the reference date for age
    props, p_log, outliers = clean_properties(raw_p)
    reference_date = props["transaction_date"].max()
    clients, c_log = clean_clients(raw_c, reference_date)
    master, transactions, m_log = build_master_dataset(clients, props)
    return {
        "raw_clients": raw_c, "raw_properties": raw_p, "audit": audit,
        "clients": clients, "properties": props, "master": master, "transactions": transactions,
        "log": c_log + p_log + m_log, "outliers": outliers, "reference_date": reference_date,
    }
