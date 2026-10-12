"""
app.py - Buyer Segmentation & Investment Profiling dashboard (Streamlit + Plotly)

Run with:   streamlit run app.py

The app does NOT retrain anything. It reads the files written by `python train_pipeline.py`
(cleaned data with cluster labels, cluster insights, saved K-Means / preprocessing objects).
"""
from __future__ import annotations

import inspect
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import utils
from src.feature_engineering import transform_new_clients
from src.profiling import investment_profile, profile_clusters, trait_effects

st.set_page_config(page_title="Buyer Segmentation | Real Estate Intelligence", page_icon="🏢",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg,#0f2a43 0%,#1d5a8a 60%,#2a8bb6 100%); color:#fff;
       padding: 1.1rem 1.5rem; border-radius: 14px; margin-bottom: 1.1rem;}
.hero h1 {color:#fff; font-size: 1.65rem; margin:0; padding:0;}
.hero p {color:#d6e8f5; margin:.25rem 0 0 0; font-size:.95rem;}
.kpi {background:#ffffff; border:1px solid #e4e9ef; border-left:5px solid #1d5a8a; border-radius:12px;
      padding:.8rem 1rem; box-shadow:0 1px 3px rgba(15,42,67,.08); min-height: 88px;}
.kpi .label {font-size:.78rem; color:#5b6b7b; text-transform:uppercase; letter-spacing:.04em;}
.kpi .value {font-size:1.6rem; font-weight:700; color:#0f2a43; line-height:1.25;}
.kpi .sub {font-size:.75rem; color:#7a8794;}
.note {background:#f3f8fc; border-left:4px solid #2a8bb6; padding:.7rem 1rem; border-radius:8px; font-size:.9rem;}
.warn {background:#fff8ec; border-left:4px solid #e0a030; padding:.7rem 1rem; border-radius:8px; font-size:.9rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SEG_COLORS = ["#1d5a8a", "#e07a2f", "#3a9e6f", "#c0444f", "#7b5ea7", "#8c6d4f", "#d1a300", "#4aa3c7"]
PLOT_LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=45, b=10), font=dict(size=12))


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _accepts_width(func) -> bool:
    """True if this Streamlit version supports width='stretch' (newer) instead of use_container_width."""
    try:
        return "width" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def show_plot(fig, key: str | None = None) -> None:
    """Display a Plotly figure full width, compatible with old and new Streamlit versions."""
    fig.update_layout(**PLOT_LAYOUT)
    if _accepts_width(st.plotly_chart):
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df: pd.DataFrame, height: int | None = None) -> None:
    """Display a dataframe full width (old/new Streamlit)."""
    kwargs = {"height": height} if height else {}
    if _accepts_width(st.dataframe):
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def csv_button(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key)


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data ...")
def load_data() -> Dict:
    needed = [utils.MASTER_CLIENTS_CSV, utils.PROPERTIES_CLEAN_CSV, utils.TRANSACTIONS_CSV,
              utils.CLUSTER_INSIGHTS_JSON, utils.RESULTS_JSON, utils.K_SELECTION_CSV]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    master = pd.read_csv(utils.MASTER_CLIENTS_CSV)
    master["segment_label"] = master["cluster_id"] + " · " + master["segment"]
    props = pd.read_csv(utils.PROPERTIES_CLEAN_CSV)
    tx = pd.read_csv(utils.TRANSACTIONS_CSV)
    tx["segment_label"] = tx["cluster_id"] + " · " + tx["segment"]
    insights_file = utils.load_json(utils.CLUSTER_INSIGHTS_JSON)
    return {"master": master, "props": props, "tx": tx,
            "insights": insights_file["insights"], "prd": insights_file["prd_segment_test"],
            "results": utils.load_json(utils.RESULTS_JSON), "sweep": pd.read_csv(utils.K_SELECTION_CSV)}


@st.cache_resource(show_spinner=False)
def load_models():
    """Load the saved model objects (returns None if they cannot be loaded, e.g. version mismatch)."""
    try:
        return {"kmeans": joblib.load(utils.KMEANS_PKL), "preprocessor": joblib.load(utils.PREPROCESSOR_PKL),
                "scaler": joblib.load(utils.SCALER_PKL), "pca": joblib.load(utils.PCA_PKL),
                "meta": utils.load_json(utils.METADATA_JSON)}
    except Exception as exc:                                  # noqa: BLE001
        return {"error": str(exc)}


def seg_color_map(master: pd.DataFrame) -> Dict[str, str]:
    labels = master.sort_values("cluster")["segment_label"].unique().tolist()
    return {lab: SEG_COLORS[i % len(SEG_COLORS)] for i, lab in enumerate(labels)}


# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
def apply_in(df: pd.DataFrame, col: str, selected: List) -> pd.DataFrame:
    """Keep rows whose `col` is in `selected`; an empty selection means 'no filter'."""
    return df[df[col].isin(selected)] if selected else df


def sidebar_filters(master: pd.DataFrame, tx: pd.DataFrame):
    sb = st.sidebar
    sb.markdown("### 🔎 Filters")
    sb.caption("Leave a box empty to include everything.")
    countries = sb.multiselect("Country", sorted(master["country"].unique()))
    region_pool = master[master["country"].isin(countries)] if countries else master
    regions = sb.multiselect("Region", sorted(region_pool["region"].unique()))
    purposes = sb.multiselect("Acquisition purpose", sorted(master["acquisition_purpose"].unique()))
    types = sb.multiselect("Client type", sorted(master["client_type"].unique()))
    loans = sb.multiselect("Loan applied", sorted(master["loan_applied"].unique()))
    seg_labels = master.sort_values("cluster")["segment_label"].unique().tolist()
    segs = sb.multiselect("Cluster / segment", seg_labels)
    a_min, a_max = int(np.floor(master["age"].min())), int(np.ceil(master["age"].max()))
    ages = sb.slider("Age range", a_min, a_max, (a_min, a_max))
    p_min = int(np.floor(master["avg_price"].min() / 1000) * 1000)
    p_max = int(np.ceil(master["avg_price"].max() / 1000) * 1000)
    prices = sb.slider("Average unit price paid (USD)", p_min, p_max, (p_min, p_max), step=1000)

    fm = master
    for col, sel in [("country", countries), ("region", regions), ("acquisition_purpose", purposes),
                     ("client_type", types), ("loan_applied", loans), ("segment_label", segs)]:
        fm = apply_in(fm, col, sel)
    fm = fm[(fm["age"] >= ages[0]) & (fm["age"] <= ages[1]) &
            (fm["avg_price"] >= prices[0]) & (fm["avg_price"] <= prices[1])]
    ftx = tx[tx["client_id"].isin(fm["client_id"])]
    sb.markdown("---")
    sb.metric("Buyers after filters", f"{len(fm):,} / {len(master):,}")
    return fm, ftx


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def kpi_row(fm: pd.DataFrame, ftx: pd.DataFrame, ctx: Dict) -> None:
    cols = st.columns(7)
    kpi(cols[0], "Total buyers", f"{len(fm):,}", f"of {len(ctx['master']):,} clients")
    kpi(cols[1], "Properties owned", f"{len(ftx):,}", f"sold units; {len(ctx['props']):,} listings in total")
    kpi(cols[2], "Clusters", f"{fm['cluster'].nunique()}", "K-Means segments")
    kpi(cols[3], "Investment buyers", f"{100 * fm['is_investment'].mean():.1f}%", "purpose = Investment")
    kpi(cols[4], "Loan users", f"{100 * fm['loan_flag'].mean():.1f}%", "applied for a loan")
    kpi(cols[5], "Avg satisfaction", f"{fm['satisfaction_score'].mean():.2f}", "scale 1-5")
    kpi(cols[6], "Avg property value", fmt_money(ftx["sale_price_usd"].mean()) if len(ftx) else "n/a", "mean sale price (USD)")


def page_overview(fm, ftx, ctx):
    kpi_row(fm, ftx, ctx)
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        d = fm["client_type"].value_counts().reset_index()
        d.columns = ["client_type", "buyers"]
        show_plot(px.pie(d, names="client_type", values="buyers", hole=0.55, title="Customer type"), "ov_type")
    with c2:
        d = fm["acquisition_purpose"].value_counts().reset_index()
        d.columns = ["purpose", "buyers"]
        show_plot(px.pie(d, names="purpose", values="buyers", hole=0.55, title="Acquisition purpose"), "ov_purpose")
    with c3:
        d = fm["loan_applied"].value_counts().reset_index()
        d.columns = ["loan", "buyers"]
        show_plot(px.pie(d, names="loan", values="buyers", hole=0.55, title="Loan application"), "ov_loan")

    c1, c2 = st.columns(2)
    with c1:
        show_plot(px.histogram(fm, x="age", nbins=25, title="Age distribution", color_discrete_sequence=SEG_COLORS), "ov_age")
    with c2:
        d = fm["satisfaction_score"].value_counts().sort_index().reset_index()
        d.columns = ["score", "buyers"]
        show_plot(px.bar(d, x="score", y="buyers", text="buyers", title="Satisfaction score distribution",
                         color_discrete_sequence=SEG_COLORS), "ov_sat")
    c1, c2 = st.columns(2)
    with c1:
        d = fm["country"].value_counts().reset_index()
        d.columns = ["country", "buyers"]
        show_plot(px.bar(d, x="buyers", y="country", orientation="h", title="Buyers by country",
                         color_discrete_sequence=SEG_COLORS).update_yaxes(autorange="reversed"), "ov_country")
    with c2:
        d = fm["referral_channel"].value_counts().reset_index()
        d.columns = ["channel", "buyers"]
        show_plot(px.bar(d, x="channel", y="buyers", text="buyers", title="Referral channel",
                         color_discrete_sequence=SEG_COLORS), "ov_channel")
    if len(ftx):
        m = ftx.groupby("transaction_month").agg(units=("listing_id", "count"),
                                                 value=("sale_price_usd", "sum")).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            show_plot(px.line(m, x="transaction_month", y="units", markers=True, title="Units sold per month"), "ov_m1")
        with c2:
            show_plot(px.histogram(ftx, x="sale_price_usd", color="unit_category", nbins=40, barmode="overlay",
                                   opacity=0.65, title="Sale price distribution by unit type"), "ov_price")
    st.markdown('<div class="note"><b>Reading the data honestly:</b> purpose, loan use and spending are very similar '
                'across countries, channels and client types. The clusters are driven mainly by age, satisfaction, '
                'portfolio size, ticket price and office share (see <i>Model Evaluation</i>).</div>',
                unsafe_allow_html=True)


def page_segmentation(fm, ftx, ctx):
    kpi_row(fm, ftx, ctx)
    st.write("")
    cmap = seg_color_map(ctx["master"])
    sizes = fm.groupby(["cluster", "segment_label"]).size().reset_index(name="buyers").sort_values("cluster")
    sizes["percent"] = (100 * sizes["buyers"] / sizes["buyers"].sum()).round(1)
    c1, c2 = st.columns(2)
    with c1:
        show_plot(px.bar(sizes, x="segment_label", y="buyers", color="segment_label", text="buyers",
                         color_discrete_map=cmap, title="Cluster distribution (buyers)").update_layout(showlegend=False,
                         xaxis_title=""), "sg_bar")
    with c2:
        show_plot(px.pie(sizes, names="segment_label", values="buyers", hole=0.5, color="segment_label",
                         color_discrete_map=cmap, title="Cluster share (%)"), "sg_pie")

    st.subheader("Cluster profiles")
    prof = profile_clusters(fm).drop(index="All")
    names = fm.drop_duplicates("cluster").set_index("cluster")["segment_label"]
    table = pd.DataFrame({
        "Segment": [names[i] for i in prof.index], "Buyers": prof["size"].astype(int).values,
        "% of buyers": (100 * prof["pct_of_clients"]).round(1).values,
        "Avg age": prof["avg_age"].round(1).values, "% company": (100 * prof["pct_company"]).round(1).values,
        "% investment": (100 * prof["investment_rate"]).round(1).values,
        "% loan": (100 * prof["loan_rate"]).round(1).values,
        "Avg satisfaction": prof["avg_satisfaction"].round(2).values,
        "Avg units": prof["avg_units"].round(2).values, "Avg unit price ($)": prof["avg_price"].round(0).values,
        "% office units": (100 * prof["office_share"]).round(1).values,
        "% international": (100 * prof["pct_international"]).round(1).values,
        "Top country": prof["top_country"].values, "Top region": prof["top_region"].values,
        "Value index": prof["value_index"].astype(float).round(2).values,
    })
    show_table(table)
    csv_button(table, "⬇ Download cluster profiles (CSV)", "cluster_profiles_filtered.csv", "dl_prof")
    st.caption("Value index = share of total sold value ÷ share of buyers (above 1 = brings more value than its size).")

    st.subheader("What makes each cluster different?")
    eff = trait_effects(fm)
    eff.index = [names[i] for i in eff.index]
    eff.columns = [c.replace("_", " ") for c in eff.columns]
    show_plot(px.imshow(eff, text_auto=".2f", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                        aspect="auto", title="Difference vs overall average (in standard deviations)"), "sg_heat")

    st.subheader("PCA view of the clusters")
    st.markdown('<div class="note">PCA squeezes the encoded features into 2 axes <b>only to draw the clusters</b>. '
                'The clustering itself used all features. Overlap here reflects the weak natural separation '
                'in this dataset (silhouette ≈ 0.12).</div>', unsafe_allow_html=True)
    pcs = ctx["results"]["pca_explained_variance"]
    fig = px.scatter(fm, x="pca1", y="pca2", color="segment_label", color_discrete_map=cmap, opacity=0.65,
                     hover_name="client_id",
                     hover_data=["segment", "age", "satisfaction_score", "n_units", "acquisition_purpose", "country"],
                     labels={"pca1": f"Principal Component 1 ({100 * pcs[0]:.1f}% variance)",
                             "pca2": f"Principal Component 2 ({100 * pcs[1]:.1f}% variance)"},
                     title="Buyer clusters in 2-D (PCA)")
    fig.update_traces(marker=dict(size=6))
    show_plot(fig, "sg_pca")


def page_investor(fm, ftx, ctx):
    kpi_row(fm, ftx, ctx)
    st.write("")
    cmap = seg_color_map(ctx["master"])
    g = fm.groupby(["cluster", "segment_label"]).agg(
        buyers=("client_id", "count"), investment=("is_investment", "mean"), loan=("loan_flag", "mean"),
        satisfaction=("satisfaction_score", "mean"), age=("age", "mean")).reset_index().sort_values("cluster")
    g["loan_pct"] = 100 * g["loan"]
    c1, c2 = st.columns(2)
    with c1:
        d = fm.groupby(["segment_label", "acquisition_purpose"]).size().reset_index(name="buyers")
        d["percent"] = 100 * d["buyers"] / d.groupby("segment_label")["buyers"].transform("sum")
        show_plot(px.bar(d, x="segment_label", y="percent", color="acquisition_purpose", barmode="stack",
                         title="Investment vs personal use (Home) by cluster (%)").update_layout(xaxis_title=""), "iv_purpose")
    with c2:
        show_plot(px.bar(g, x="segment_label", y="loan_pct", color="segment_label", color_discrete_map=cmap,
                         title="Loan usage by cluster (%)", labels={"loan_pct": "% applied for loan"}
                         ).update_layout(showlegend=False, xaxis_title=""), "iv_loan")
    c1, c2 = st.columns(2)
    with c1:
        show_plot(px.bar(g, x="segment_label", y="satisfaction", color="segment_label", color_discrete_map=cmap,
                         title="Average satisfaction by cluster", range_y=[0, 5]
                         ).update_layout(showlegend=False, xaxis_title=""), "iv_sat")
    with c2:
        show_plot(px.box(fm, x="segment_label", y="age", color="segment_label", color_discrete_map=cmap,
                         title="Age by cluster").update_layout(showlegend=False, xaxis_title=""), "iv_age")
    c1, c2 = st.columns(2)
    with c1:
        if len(ftx):
            show_plot(px.box(ftx, x="segment_label", y="sale_price_usd", color="segment_label", color_discrete_map=cmap,
                             title="Property value by cluster (sold units)").update_layout(showlegend=False, xaxis_title=""), "iv_val")
    with c2:
        d = fm.groupby(["client_type", "acquisition_purpose"]).size().reset_index(name="buyers")
        d["percent"] = 100 * d["buyers"] / d.groupby("client_type")["buyers"].transform("sum")
        show_plot(px.bar(d, x="client_type", y="percent", color="acquisition_purpose", barmode="group", text_auto=".1f",
                         title="Investment behaviour by client type (%)"), "iv_type")

    st.subheader("Where is the investor value?")
    names = fm.drop_duplicates("cluster").set_index("cluster")["segment"].to_dict()
    inv = investment_profile(fm, profile_clusters(fm), names)
    show_plot(px.bar(inv, x="segment", y="investor_spend_usd", color="investment_rate", color_continuous_scale="Blues",
                     title="Total spend by investment-purpose buyers (colour = investment rate)",
                     labels={"investor_spend_usd": "Investor spend (USD)", "segment": ""}), "iv_value")
    show_table(pd.DataFrame({
        "Segment": inv["segment"], "Investors": inv["investors"], "Investment rate %": (100 * inv["investment_rate"]).round(1),
        "Lift vs overall": inv["investment_lift_vs_overall"].round(2),
        "Investor spend ($M)": (inv["investor_spend_usd"] / 1e6).round(1),
        "% of all investor spend": (100 * inv["share_of_all_investor_spend"]).round(1),
        "Avg units / investor": inv["avg_units_per_investor"].round(2),
        "Investor loan %": (100 * inv["investor_loan_rate"]).round(1), "Value rank": inv["investor_value_rank"]}))


def page_geography(fm, ftx, ctx):
    kpi_row(fm, ftx, ctx)
    st.write("")
    cmap = seg_color_map(ctx["master"])
    metric = st.selectbox("Map metric", ["Buyers", "Investment rate (%)", "Loan rate (%)", "Average satisfaction"])
    agg = fm.groupby("country").agg(buyers=("client_id", "count"), inv=("is_investment", "mean"),
                                    loan=("loan_flag", "mean"), sat=("satisfaction_score", "mean")).reset_index()
    agg["iso3"] = agg["country"].map(utils.COUNTRY_ISO3)
    agg["Investment rate (%)"] = 100 * agg["inv"]
    agg["Loan rate (%)"] = 100 * agg["loan"]
    agg["Average satisfaction"] = agg["sat"]
    agg["Buyers"] = agg["buyers"]
    c1, c2 = st.columns([3, 2])
    with c1:
        mp = agg.dropna(subset=["iso3"])
        if len(mp):
            show_plot(px.choropleth(mp, locations="iso3", color=metric, hover_name="country",
                                    hover_data={"iso3": False, "buyers": True},
                                    color_continuous_scale="Blues", title=f"{metric} by country"), "geo_world")
    with c2:
        d = agg.sort_values("buyers", ascending=False)
        show_plot(px.bar(d, x="buyers", y="country", orientation="h", title="Country-wise buyers",
                         color_discrete_sequence=SEG_COLORS).update_yaxes(autorange="reversed"), "geo_country")

    us = fm[fm["country"] == "USA"].groupby("region").agg(buyers=("client_id", "count"),
                                                         inv=("is_investment", "mean")).reset_index()
    us["code"] = us["region"].map(utils.US_STATE_CODES)
    us = us.dropna(subset=["code"])
    c1, c2 = st.columns(2)
    with c1:
        if len(us):
            us["Investment rate (%)"] = 100 * us["inv"]
            show_plot(px.choropleth(us, locations="code", locationmode="USA-states", scope="usa", color="buyers",
                                    hover_name="region", hover_data={"code": False, "Investment rate (%)": ":.1f"},
                                    color_continuous_scale="Blues", title="US buyers by state"), "geo_us")
    with c2:
        top = fm["region"].value_counts().head(15).reset_index()
        top.columns = ["region", "buyers"]
        show_plot(px.bar(top, x="buyers", y="region", orientation="h", title="Region-wise buyers (top 15)",
                         color_discrete_sequence=SEG_COLORS).update_yaxes(autorange="reversed"), "geo_region")

    top_regions = fm["region"].value_counts().head(12).index
    d = fm[fm["region"].isin(top_regions)].groupby(["region", "segment_label"]).size().reset_index(name="buyers")
    show_plot(px.bar(d, x="region", y="buyers", color="segment_label", color_discrete_map=cmap, barmode="stack",
                     title="Cluster distribution by region (top 12 regions)").update_layout(xaxis_title=""), "geo_cluster_region")
    d = fm.groupby(["country", "region"]).size().reset_index(name="buyers")
    show_plot(px.treemap(d, path=["country", "region"], values="buyers", title="Country → region hierarchy"), "geo_tree")
    st.caption("Maps need an internet connection in the browser (Plotly loads the map shapes online).")


def page_insights(fm, ftx, ctx):
    insights = ctx["insights"]
    options = {f"{i['cluster_id']} · {i['name']}": i for i in insights}
    choice = st.selectbox("Choose a segment", list(options.keys()))
    ins = options[choice]
    st.markdown(f"## {ins['name']}")
    st.caption(f"Segment {ins['cluster_id']} · {ins['size']:,} buyers ({ins['pct']:.1f}% of all buyers). "
               "Text below describes the full dataset; the metric cards respond to your sidebar filters.")
    sub = fm[fm["cluster"] == ins["cluster"]]
    subtx = ftx[ftx["cluster"] == ins["cluster"]]
    if sub.empty:
        st.warning("No buyers of this segment match the current filters.")
    else:
        c = st.columns(6)
        kpi(c[0], "Buyers (filtered)", f"{len(sub):,}", f"{100 * len(sub) / len(fm):.1f}% of filtered")
        kpi(c[1], "Avg age", f"{sub['age'].mean():.1f}", "years")
        kpi(c[2], "Investment", f"{100 * sub['is_investment'].mean():.1f}%", "purpose = Investment")
        kpi(c[3], "Loan", f"{100 * sub['loan_flag'].mean():.1f}%", "applied")
        kpi(c[4], "Satisfaction", f"{sub['satisfaction_score'].mean():.2f}", "1-5")
        kpi(c[5], "Avg unit price", fmt_money(subtx["sale_price_usd"].mean()) if len(subtx) else "n/a",
            f"{sub['n_units'].mean():.1f} units per buyer")
        st.write("")
        c1, c2, c3 = st.columns(3)
        with c1:
            d = sub["country"].value_counts().reset_index()
            d.columns = ["country", "buyers"]
            show_plot(px.bar(d, x="country", y="buyers", title="Geography", color_discrete_sequence=SEG_COLORS), "si_geo")
        with c2:
            d = sub["referral_channel"].value_counts().reset_index()
            d.columns = ["channel", "buyers"]
            show_plot(px.pie(d, names="channel", values="buyers", hole=0.5, title="Referral channel"), "si_ch")
        with c3:
            d = sub["age_group"].value_counts().reset_index()
            d.columns = ["age_group", "buyers"]
            show_plot(px.bar(d, x="age_group", y="buyers", title="Age groups", color_discrete_sequence=SEG_COLORS), "si_age")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 👤 Customer profile");   st.write(ins["customer_profile"])
        st.markdown("#### 🧭 Behavioural profile"); st.write(ins["behavioral_profile"])
        st.markdown("#### 📈 Investment profile");  st.write(ins["investment_profile"])
        st.markdown("#### 🏦 Financing profile");   st.write(ins["financing_profile"])
        st.markdown("#### 🌍 Geographic profile");  st.write(ins["geographic_profile"])
    with c2:
        st.markdown("#### 📣 Marketing recommendations")
        for r in ins["marketing_recommendations"]:
            st.markdown(f"- {r}")
        st.markdown("#### 🏢 Property recommendations")
        for r in ins["property_recommendations"]:
            st.markdown(f"- {r}")
    with st.expander("✅ Data-supported findings (measured)"):
        for f in ins["data_supported_findings"]:
            st.markdown(f"- {f}")
    with st.expander("⚠️ Assumptions (NOT proven by the data)"):
        for a in ins["assumptions"]:
            st.markdown(f"- {a}")


def page_explorer(fm, ftx, ctx):
    st.markdown('<div class="note">All tables below respect the sidebar filters. Use the buttons to download CSV files.</div>',
                unsafe_allow_html=True)
    tabs = st.tabs(["Clients + clusters", "Properties (clean)", "Transactions (sold units)", "Cluster assignments"])
    with tabs[0]:
        show_table(fm.drop(columns=["segment_label"]), height=420)
        csv_button(fm.drop(columns=["segment_label"]), "⬇ Download filtered clients", "clients_filtered.csv", "dl_clients")
    with tabs[1]:
        owned = ctx["props"]
        flt = owned[owned["client_ref"].isin(fm["client_id"]) | (owned["listing_status"] == "Available")]
        st.caption("Includes all 'Available' listings plus units owned by the filtered buyers.")
        show_table(flt, height=420)
        csv_button(flt, "⬇ Download properties", "properties_filtered.csv", "dl_props")
    with tabs[2]:
        show_table(ftx.drop(columns=["segment_label"]), height=420)
        csv_button(ftx.drop(columns=["segment_label"]), "⬇ Download transactions", "transactions_filtered.csv", "dl_tx")
    with tabs[3]:
        a = fm[["client_id", "cluster_id", "segment", "ward_cluster", "pca1", "pca2"]]
        show_table(a, height=420)
        csv_button(a, "⬇ Download cluster assignments", "cluster_assignments.csv", "dl_assign")


def page_evaluation(fm, ftx, ctx):
    res, sweep = ctx["results"], ctx["sweep"]
    k = res["kmeans_final"]["k"]
    st.markdown(f'<div class="note"><b>Chosen K = {k}.</b> {res["k_selection"]["rationale"]}</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(sweep, x="k", y="inertia", markers=True, title="Elbow method (inertia)")
        fig.add_vline(x=k, line_dash="dash", line_color="red")
        show_plot(fig, "ev_elbow")
    with c2:
        fig = px.line(sweep, x="k", y="silhouette", markers=True, title="Silhouette score")
        fig.add_vline(x=k, line_dash="dash", line_color="red")
        show_plot(fig, "ev_sil")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(sweep, x="k", y="davies_bouldin", markers=True, title="Davies-Bouldin index (lower = better)")
        fig.add_vline(x=k, line_dash="dash", line_color="red")
        show_plot(fig, "ev_db")
    with c2:
        fig = px.line(sweep, x="k", y="stability_ari", markers=True, title="Stability (ARI across random restarts)")
        fig.add_vline(x=k, line_dash="dash", line_color="red")
        show_plot(fig, "ev_stab")
    st.subheader("K-Means vs Hierarchical (Ward)")
    ag = res["hierarchical"]["agreement"]
    c = st.columns(4)
    kpi(c[0], "Silhouette K-Means", f"{ag['silhouette_kmeans']:.3f}", "higher = better")
    kpi(c[1], "Silhouette Ward", f"{ag['silhouette_ward']:.3f}", "higher = better")
    kpi(c[2], "Agreement (ARI)", f"{ag['ARI']:.2f}", "1 = identical clusters")
    kpi(c[3], "Cophenetic corr.", f"{ag['cophenetic_correlation_ward']:.2f}", "tree vs real distances")
    st.write("")
    show_table(pd.DataFrame(res["hierarchical"]["linkage_comparison"]).round(3))
    st.caption("'Average' linkage gets a high silhouette only because it isolates a handful of points in tiny clusters "
               "(see smallest_cluster_pct), which makes it useless for segmentation.")
    st.subheader("Do the PRD example segments exist in this data?")
    show_table(pd.DataFrame(ctx["prd"])[["prd_segment", "prd_description", "closest_cluster_name",
                                         "conditions_met", "supported_by_data", "limitation"]])
    st.subheader("Which variables really differ between clusters?")
    t = pd.DataFrame(res["statistical_tests"])
    t["p_value"] = t["p_value"].map(lambda p: f"{p:.4f}")
    show_table(t[["variable", "test", "p_value", "effect_size", "effect_size_name", "significant_0.05"]].round(3))
    st.caption("Numeric variables are significant by construction (they were used to build the clusters). "
               "Purpose, loan, channel, country and gender do NOT differ significantly between clusters.")


def page_predictor(fm, ftx, ctx):
    models = load_models()
    if "error" in models:
        st.error("Saved models could not be loaded (often a scikit-learn version difference). "
                 "Run `python train_pipeline.py` once to re-create them. Details: " + models["error"])
        return
    meta, kmeans, pre = models["meta"], models["kmeans"], models["preprocessor"]
    cfg = meta["feature_config"]
    st.markdown('<div class="note">Describe a (new) buyer and the saved K-Means model will assign the closest segment. '
                'This uses <code>preprocessor.pkl</code> and <code>kmeans_model.pkl</code> - nothing is retrained.</div>',
                unsafe_allow_html=True)
    st.write("")
    row = {}
    cols = st.columns(3)
    for i, col in enumerate(cfg["numeric"]):
        lo, hi, med = meta["numeric_ranges"][col]
        with cols[i % 3]:
            if col in ("n_units",):
                row[col] = st.number_input("Units owned", int(lo), int(max(hi, 20)), int(med), step=1)
            elif col == "satisfaction_score":
                row[col] = st.slider("Satisfaction score", 1, 5, int(round(med)))
            elif col == "office_share":
                row[col] = st.slider("Share of office units", 0.0, 1.0, 0.0, step=0.05)
            elif col == "age":
                row[col] = st.slider("Age", 18, 100, int(med))
            else:
                row[col] = float(st.number_input("Average unit price (USD)", float(lo * 0.5), float(hi * 2), float(med),
                                                 step=5000.0))
    for i, col in enumerate(cfg["binary"] + cfg["nominal"]):
        with cols[i % 3]:
            row[col] = st.selectbox(col.replace("_", " ").title(), meta["categories"][col])
    if st.button("Assign segment", type="primary"):
        X = transform_new_clients(pre, cfg, pd.DataFrame([row]))
        cid = int(kmeans.predict(X)[0])
        dist = np.linalg.norm(kmeans.cluster_centers_ - X, axis=1)
        name = meta["cluster_names"][str(cid)]
        st.success(f"Assigned to **C{cid + 1} · {name}**")
        d = pd.DataFrame({"segment": [f"C{i + 1} · {meta['cluster_names'][str(i)]}" for i in range(len(dist))],
                          "distance_to_centre": dist})
        show_plot(px.bar(d, x="segment", y="distance_to_centre", title="Distance to each cluster centre (smaller = closer)"), "pr_dist")
        ins = next(i for i in ctx["insights"] if i["cluster"] == cid)
        st.markdown("**Suggested approach:**")
        for r in ins["marketing_recommendations"][:4]:
            st.markdown(f"- {r}")


PAGES = {
    "📊 Executive Overview": page_overview,
    "🧩 Buyer Segmentation": page_segmentation,
    "💼 Investor Behaviour": page_investor,
    "🌍 Geographic Analysis": page_geography,
    "🔍 Segment Insights": page_insights,
    "🗂 Data Explorer": page_explorer,
    "🧪 Model Evaluation": page_evaluation,
    "🎯 Segment Predictor": page_predictor,
}


def main() -> None:
    st.markdown('<div class="hero"><h1>🏢 Buyer Segmentation & Investment Profiling</h1>'
                '<p>Machine-learning market intelligence for real estate · K-Means + Hierarchical clustering</p></div>',
                unsafe_allow_html=True)
    try:
        ctx = load_data()
    except FileNotFoundError as err:
        st.error(f"Missing generated files ({err}). Run `python train_pipeline.py` first, then reload this page.")
        st.stop()
    ctx["master"] = ctx["master"]
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")
    fm, ftx = sidebar_filters(ctx["master"], ctx["tx"])
    if fm.empty:
        st.warning("No buyers match the current filters. Remove a filter to continue.")
        st.stop()
    PAGES[page](fm, ftx, ctx)


main()
