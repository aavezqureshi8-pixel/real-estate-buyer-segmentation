"""
profiling.py
------------
Turns raw cluster numbers into business knowledge (PRD Step 6, project Parts 9-11):

* profile_clusters()       numbers per cluster (size, age, loan, investment, geography, property metrics)
* name_clusters()          DATA-DRIVEN names: built from the traits that most separate a cluster from everyone else
* test_prd_segments()      checks whether the PRD's example segments (Global Investors, First-Time Buyers,
                           Corporate Buyers, Luxury Investors) really exist in this data
* segment_tests()          chi-square / Kruskal-Wallis tests: which variables truly differ between clusters?
* investment_profile()     investor-focused view per cluster
* build_cluster_insights() customer / behaviour / investment / financing / geography profiles + recommendations,
                           with DATA-SUPPORTED findings kept separate from ASSUMPTIONS.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Traits used for naming. column -> (label when HIGH, label when LOW)
# ---------------------------------------------------------------------------
TRAITS: Dict[str, Tuple[str, str]] = {
    "n_units": ("High-Volume", "Low-Volume"),
    "avg_price": ("Premium-Ticket", "Value-Ticket"),
    "office_share": ("Office-Focused", "Residential-Only"),
    "satisfaction_score": ("Highly Satisfied", "Dissatisfied"),
    "age": ("Senior", "Younger"),
    "is_investment": ("Investment-Leaning", "Home-Leaning"),
    "loan_flag": ("Loan-Reliant", "Cash-Leaning"),
    "is_company": ("Corporate", "Individual"),
    "is_international": ("International", "Domestic"),
}
# 'Low' labels that are not interesting on their own (e.g. most clients are Individuals / Domestic)
IGNORE_LOW = {"is_company", "is_international", "office_share", "n_units"}
MIN_EFFECT = 0.30            # minimum |effect size| for a trait to be used in a name


def _top_share(series: pd.Series) -> Tuple[str, float]:
    vc = series.value_counts()
    return str(vc.index[0]), float(vc.iloc[0] / len(series))


def _summarise(df: pd.DataFrame, total_clients: int, total_spend: float) -> Dict:
    """All headline numbers for ONE group of clients (a cluster or the whole base)."""
    top_country, top_country_pct = _top_share(df["country"])
    top_region, top_region_pct = _top_share(df["region"])
    top_channel, top_channel_pct = _top_share(df["referral_channel"])
    return {
        "size": int(len(df)),
        "pct_of_clients": float(len(df) / total_clients),
        "avg_age": float(df["age"].mean()),
        "median_age": float(df["age"].median()),
        "pct_female": float((df["gender"] == "Female").mean()),
        "pct_company": float(df["is_company"].mean()),
        "investment_rate": float(df["is_investment"].mean()),
        "home_rate": float(1 - df["is_investment"].mean()),
        "loan_rate": float(df["loan_flag"].mean()),
        "avg_satisfaction": float(df["satisfaction_score"].mean()),
        "pct_international": float(df["is_international"].mean()),
        "top_country": top_country, "top_country_pct": top_country_pct,
        "top_region": top_region, "top_region_pct": top_region_pct,
        "top_channel": top_channel, "top_channel_pct": top_channel_pct,
        "avg_units": float(df["n_units"].mean()),
        "avg_price": float(df["avg_price"].mean()),
        "avg_total_spend": float(df["total_spend"].mean()),
        "avg_area": float(df["avg_area"].mean()),
        "office_share": float(df["office_share"].mean()),
        "avg_price_per_sqft": float(df["avg_price_per_sqft"].mean()),
        "total_spend": float(df["total_spend"].sum()),
        "revenue_share": float(df["total_spend"].sum() / total_spend),
    }


def profile_clusters(master: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """One row per cluster plus an 'All clients' row (index = cluster id, 'All' for the total)."""
    total_clients = len(master)
    total_spend = float(master["total_spend"].sum())
    rows = {}
    for cid, grp in master.groupby(cluster_col):
        rows[cid] = _summarise(grp, total_clients, total_spend)
    rows["All"] = _summarise(master, total_clients, total_spend)
    prof = pd.DataFrame(rows).T
    # value index: share of revenue / share of clients (>1 means the cluster brings more revenue than its size)
    prof["value_index"] = prof["revenue_share"] / prof["pct_of_clients"]
    return prof


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def trait_effects(master: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """
    Effect size per cluster and trait: (cluster mean - overall mean) / overall standard deviation.
    |0.3| = small-but-noticeable, |0.5| = medium, |0.8| = large (Cohen's d rule of thumb).
    """
    cols = [c for c in TRAITS if c in master.columns]
    overall_mean, overall_std = master[cols].mean(), master[cols].std().replace(0, np.nan)
    effects = (master.groupby(cluster_col)[cols].mean() - overall_mean) / overall_std
    return effects.fillna(0.0)


def name_clusters(master: pd.DataFrame, cluster_col: str = "cluster") -> Dict[int, str]:
    """
    Build a name for each cluster from its 2 strongest distinguishing traits (|effect| >= MIN_EFFECT).
    Names are therefore a description of the data, not a pre-set label.
    """
    eff = trait_effects(master, cluster_col)
    names: Dict[int, str] = {}
    used: Dict[str, int] = {}
    for cid in eff.index:
        row = eff.loc[cid]
        picks = []
        for col in row.abs().sort_values(ascending=False).index:
            d = row[col]
            if abs(d) < MIN_EFFECT:
                break
            label = TRAITS[col][0] if d > 0 else TRAITS[col][1]
            if d < 0 and col in IGNORE_LOW:
                continue
            picks.append(label)
            if len(picks) == 2:
                break
        name = (" ".join(picks) + " Buyers") if picks else "Mainstream Buyers"
        if name in used:                                   # keep names unique
            used[name] += 1
            name = f"{name} ({used[name]})"
        else:
            used[name] = 1
        names[int(cid)] = name
    return names


# ---------------------------------------------------------------------------
# Do the PRD's example segments exist in the data?
# ---------------------------------------------------------------------------
# condition = (metric column in profile, 'high' or 'low', required ratio vs overall)
PRD_SEGMENTS = {
    "Global Investors": {
        "prd_description": "High income, investment purchases (international)",
        "conditions": [("investment_rate", "high", 1.25), ("pct_international", "high", 1.25)],
        "not_testable": "income is not in the data",
    },
    "First-Time Buyers": {
        "prd_description": "Younger, loan dependent",
        "conditions": [("avg_age", "low", 0.85), ("loan_rate", "high", 1.25), ("investment_rate", "low", 0.90)],
        "not_testable": "'first-time' cannot be observed (every client owns 3+ units); personal-use + young + loans used as proxy",
    },
    "Corporate Buyers": {
        "prd_description": "Companies purchasing multiple units",
        "conditions": [("pct_company", "high", 1.25), ("avg_units", "high", 1.25)],
        "not_testable": "",
    },
    "Luxury Investors": {
        "prd_description": "High satisfaction, large investments",
        "conditions": [("avg_price", "high", 1.25), ("avg_satisfaction", "high", 1.10), ("investment_rate", "high", 1.25)],
        "not_testable": "",
    },
}


def test_prd_segments(profile: pd.DataFrame, names: Dict[int, str]) -> pd.DataFrame:
    """
    For each PRD segment find the closest cluster and check every condition.
    A condition is met when (cluster value / overall value) >= ratio for 'high' or <= ratio for 'low'.
    A concept is SUPPORTED only if ALL conditions are met by the same cluster.
    """
    overall = profile.loc["All"]
    clusters = [c for c in profile.index if c != "All"]
    rows = []
    for concept, spec in PRD_SEGMENTS.items():
        best = None
        for cid in clusters:
            score, met, detail = 0.0, 0, []
            for metric, direction, ratio in spec["conditions"]:
                r = float(profile.loc[cid, metric]) / float(overall[metric])
                ok = (r >= ratio) if direction == "high" else (r <= ratio)
                met += int(ok)
                # closeness: how far towards the threshold the cluster is (capped) - used only for picking the closest cluster
                score += (r if direction == "high" else 2 - r)
                detail.append(f"{metric}: {profile.loc[cid, metric]:.3g} vs {overall[metric]:.3g} overall "
                              f"(x{r:.2f}, need {'>=' if direction == 'high' else '<='} {ratio}) -> {'OK' if ok else 'no'}")
            cand = (met, score, cid, detail)
            if best is None or cand[:2] > best[:2]:
                best = cand
        met, _, cid, detail = best
        rows.append({
            "prd_segment": concept,
            "prd_description": spec["prd_description"],
            "closest_cluster": int(cid),
            "closest_cluster_name": names[int(cid)],
            "conditions_met": f"{met}/{len(spec['conditions'])}",
            "supported_by_data": bool(met == len(spec["conditions"])),
            "evidence": " | ".join(detail),
            "limitation": spec["not_testable"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests: which variables really differ between clusters?
# ---------------------------------------------------------------------------

def segment_tests(master: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """
    Categorical variables -> chi-square test of independence + Cramer's V (0 = unrelated, 1 = identical)
    Numeric variables     -> Kruskal-Wallis test + eta-squared (share of variance explained by clusters)
    NOTE: numeric variables that were used to build the clusters will always look 'significant' (circular).
    The informative results are for variables NOT used as inputs / that carry little weight (gender, region ...).
    """
    rows = []
    cats = ["acquisition_purpose", "loan_applied", "client_type", "referral_channel", "country", "gender"]
    for col in cats:
        if col not in master.columns:
            continue
        ct = pd.crosstab(master[cluster_col], master[col])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 else 0.0
        rows.append({"variable": col, "type": "categorical", "test": "chi-square", "statistic": float(chi2),
                     "p_value": float(p), "effect_size": v, "effect_size_name": "Cramer's V"})
    nums = ["age", "satisfaction_score", "n_units", "avg_price", "office_share"]
    for col in nums:
        if col not in master.columns:
            continue
        groups = [g[col].values for _, g in master.groupby(cluster_col)]
        h, p = stats.kruskal(*groups)
        grand = master[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((master[col] - grand) ** 2).sum()
        rows.append({"variable": col, "type": "numeric", "test": "Kruskal-Wallis", "statistic": float(h),
                     "p_value": float(p), "effect_size": float(ss_between / ss_total), "effect_size_name": "eta-squared"})
    out = pd.DataFrame(rows)
    out["significant_0.05"] = out["p_value"] < 0.05
    out["used_as_model_input"] = out["variable"].isin(
        ["acquisition_purpose", "loan_applied", "client_type", "referral_channel", "country",
         "age", "satisfaction_score", "n_units", "avg_price", "office_share"])
    return out


# ---------------------------------------------------------------------------
# Investment profile
# ---------------------------------------------------------------------------

def investment_profile(master: pd.DataFrame, profile: pd.DataFrame, names: Dict[int, str],
                       cluster_col: str = "cluster") -> pd.DataFrame:
    """Investor-oriented table: who invests, how much they spend, and which cluster is most valuable."""
    inv = master[master["is_investment"] == 1]
    total_inv_spend = float(inv["total_spend"].sum())
    overall_rate = float(profile.loc["All", "investment_rate"])
    rows = []
    for cid in sorted(master[cluster_col].unique()):
        sub = master[master[cluster_col] == cid]
        sub_inv = sub[sub["is_investment"] == 1]
        rows.append({
            "cluster": int(cid), "segment": names[int(cid)],
            "clients": int(len(sub)),
            "investors": int(len(sub_inv)),
            "investment_rate": float(sub["is_investment"].mean()),
            "investment_lift_vs_overall": float(sub["is_investment"].mean() / overall_rate),
            "investor_spend_usd": float(sub_inv["total_spend"].sum()),
            "share_of_all_investor_spend": float(sub_inv["total_spend"].sum() / total_inv_spend) if total_inv_spend else 0.0,
            "avg_units_per_investor": float(sub_inv["n_units"].mean()) if len(sub_inv) else np.nan,
            "avg_unit_price_investor": float(sub_inv["avg_price"].mean()) if len(sub_inv) else np.nan,
            "investor_loan_rate": float(sub_inv["loan_flag"].mean()) if len(sub_inv) else np.nan,
            "investor_avg_satisfaction": float(sub_inv["satisfaction_score"].mean()) if len(sub_inv) else np.nan,
            "investor_top_country": _top_share(sub_inv["country"])[0] if len(sub_inv) else "n/a",
        })
    out = pd.DataFrame(rows)
    out["investor_value_rank"] = out["investor_spend_usd"].rank(ascending=False).astype(int)
    return out


# ---------------------------------------------------------------------------
# Business insights (rule-based text built from the real numbers)
# ---------------------------------------------------------------------------

def _rel(value: float, overall: float, tol: float = 0.10) -> str:
    """'above' / 'below' / 'in line with' (within +-tol relative difference)."""
    if overall == 0:
        return "in line with"
    r = value / overall
    return "above" if r > 1 + tol else ("below" if r < 1 - tol else "in line with")


def _money(x: float) -> str:
    return f"${x:,.0f}"


def build_cluster_insights(master: pd.DataFrame, profile: pd.DataFrame, names: Dict[int, str],
                           effects: pd.DataFrame, inv_profile: pd.DataFrame,
                           cluster_col: str = "cluster") -> List[Dict]:
    """
    For every cluster return a dictionary with the 7 profiles requested in the brief.
    `data_supported` lists measured facts; `assumptions` lists interpretations that the data cannot prove.
    """
    overall = profile.loc["All"]
    overall_channel = master["referral_channel"].value_counts(normalize=True)
    overall_country = master["country"].value_counts(normalize=True)
    insights: List[Dict] = []
    for cid in sorted(master[cluster_col].unique()):
        p = profile.loc[cid]
        sub = master[master[cluster_col] == cid]
        e = effects.loc[cid]
        ip = inv_profile[inv_profile["cluster"] == cid].iloc[0]
        top_countries = (sub["country"].value_counts(normalize=True).head(3) * 100).round(1).to_dict()
        top_regions = (sub["region"].value_counts(normalize=True).head(3) * 100).round(1).to_dict()
        channels = (sub["referral_channel"].value_counts(normalize=True) * 100).round(1).to_dict()

        # ---------------- data-supported findings ----------------
        facts = [
            f"{int(p['size'])} clients ({p['pct_of_clients']*100:.1f}% of all buyers) generating "
            f"{p['revenue_share']*100:.1f}% of total sold value (value index {p['value_index']:.2f}).",
            f"Average age {p['avg_age']:.1f} years (overall {overall['avg_age']:.1f}); "
            f"{p['pct_company']*100:.1f}% are companies (overall {overall['pct_company']*100:.1f}%).",
            f"Investment purpose {p['investment_rate']*100:.1f}% (overall {overall['investment_rate']*100:.1f}%) - "
            f"{_rel(p['investment_rate'], overall['investment_rate'])} average.",
            f"Loan applied {p['loan_rate']*100:.1f}% (overall {overall['loan_rate']*100:.1f}%) - "
            f"{_rel(p['loan_rate'], overall['loan_rate'])} average.",
            f"Average satisfaction {p['avg_satisfaction']:.2f}/5 (overall {overall['avg_satisfaction']:.2f}).",
            f"Owns {p['avg_units']:.2f} units on average (overall {overall['avg_units']:.2f}); "
            f"average unit price {_money(p['avg_price'])} (overall {_money(overall['avg_price'])}); "
            f"{p['office_share']*100:.1f}% of their units are offices (overall {overall['office_share']*100:.1f}%).",
            f"Geography: {p['pct_international']*100:.1f}% live outside the USA (overall {overall['pct_international']*100:.1f}%); "
            f"top countries {top_countries}; top regions {top_regions}.",
            f"Referral channels: {channels}.",
        ]

        # ---------------- profiles ----------------
        customer = (f"{names[cid]}: average age {p['avg_age']:.0f}, {p['pct_female']*100:.0f}% female, "
                    f"{p['pct_company']*100:.1f}% companies, {p['pct_international']*100:.0f}% international.")
        behavioural = (f"Average satisfaction {p['avg_satisfaction']:.2f}/5 "
                       f"({_rel(p['avg_satisfaction'], overall['avg_satisfaction'], 0.05)} the overall {overall['avg_satisfaction']:.2f}); "
                       f"owns {p['avg_units']:.1f} units on average; most common channel is {p['top_channel']} "
                       f"({p['top_channel_pct']*100:.0f}%).")
        investment = (f"{p['investment_rate']*100:.1f}% buy for investment "
                      f"(lift {ip['investment_lift_vs_overall']:.2f}x vs overall); this cluster holds "
                      f"{ip['share_of_all_investor_spend']*100:.1f}% of all investor spend "
                      f"({int(ip['investors'])} investors, rank {int(ip['investor_value_rank'])} of {len(inv_profile)}). "
                      f"Average unit price {_money(p['avg_price'])}.")
        financing = (f"{p['loan_rate']*100:.1f}% applied for a loan "
                     f"({_rel(p['loan_rate'], overall['loan_rate'])} the overall {overall['loan_rate']*100:.1f}%).")
        geography = (f"{p['top_country']} is the main country ({p['top_country_pct']*100:.0f}% of the cluster); "
                     f"main region {p['top_region']} ({p['top_region_pct']*100:.0f}%).")

        # ---------------- recommendations (each tied to a measured trait) ----------------
        marketing: List[str] = []
        prop: List[str] = []
        assumptions: List[str] = []

        if e.get("satisfaction_score", 0) <= -0.5:
            marketing.append("Service-recovery and retention programme (follow-up calls, post-sale support): "
                             f"satisfaction is only {p['avg_satisfaction']:.2f}/5.")
            assumptions.append("The reason for low satisfaction is not in the data (could be price, service or delivery).")
        if e.get("satisfaction_score", 0) >= 0.5:
            marketing.append("Referral / ambassador programme and testimonials: "
                             f"these are the happiest buyers ({p['avg_satisfaction']:.2f}/5).")
        if e.get("n_units", 0) >= 0.5:
            marketing.append("Dedicated relationship manager and multi-unit / bulk-purchase packages: "
                             f"clients here buy {p['avg_units']:.1f} units on average.")
            prop.append("Offer several units in the same tower or adjacent towers as bundles.")
            assumptions.append("Large portfolios suggest portfolio-building behaviour, but the data does not show a higher "
                               f"'Investment' share ({p['investment_rate']*100:.0f}% vs {overall['investment_rate']*100:.0f}% overall).")
        if e.get("avg_price", 0) >= 0.8:
            marketing.append("Premium positioning: private viewings and a dedicated advisor for high-ticket buyers "
                             f"(they pay {_money(p['avg_price'])} per unit on average).")
        if e.get("age", 0) <= -0.4:
            marketing.append("Digital-first campaigns (social media, online calculators, virtual tours) for a younger audience.")
        if e.get("age", 0) >= 0.4:
            marketing.append("Advisor-led, relationship-based outreach (events, phone/in-person) for an older audience.")
        if e.get("loan_flag", 0) >= 0.15:
            marketing.append("Co-market with lenders: mortgage pre-approval offers and EMI/affordability calculators.")
        elif e.get("loan_flag", 0) <= -0.15:
            marketing.append("Highlight fast closing and cash-buyer incentives.")
        if e.get("is_investment", 0) >= 0.15:
            marketing.append("Yield- and capital-growth-focused messaging (rental return, appreciation).")
        elif e.get("is_investment", 0) <= -0.15:
            marketing.append("Lifestyle / owner-occupier messaging (neighbourhood, space, family fit).")
        if e.get("is_company", 0) >= 0.3:
            marketing.append("B2B outreach to companies (corporate accounts, bulk contracts).")
        if p["pct_international"] >= 1.2 * overall["pct_international"]:
            marketing.append("Cross-border purchase support: multilingual content, currency and legal guidance.")
        # channel advice only when the cluster's channel mix really differs from the overall mix (>= 5 points)
        ch_share = sub["referral_channel"].value_counts(normalize=True)
        ch_diff = (ch_share - overall_channel.reindex(ch_share.index).fillna(0))
        best_ch = ch_diff.idxmax()
        if ch_diff[best_ch] >= 0.05:
            marketing.append(f"Prioritise the {best_ch} channel: {ch_share[best_ch]*100:.0f}% of this cluster comes from it "
                             f"(overall {overall_channel[best_ch]*100:.0f}%).")
        if not marketing:
            marketing.append("Standard multi-channel campaign; no strong behavioural differentiator was found for this cluster.")

        if e.get("avg_price", 0) >= 0.4:
            prop.append(f"Premium units: this cluster pays {_money(p['avg_price'])} per unit on average "
                        f"(overall {_money(overall['avg_price'])}) - show the upper price band first.")
        elif e.get("avg_price", 0) <= -0.4:
            prop.append(f"Value units: average price paid is {_money(p['avg_price'])} "
                        f"(overall {_money(overall['avg_price'])}) - lead with entry-price apartments.")
        if e.get("office_share", 0) >= 0.4:
            prop.append(f"Office / commercial units: {p['office_share']*100:.0f}% of their purchases are offices "
                        f"(overall {overall['office_share']*100:.0f}%).")
        elif p["office_share"] <= 0.02:
            prop.append("Residential apartments: this cluster essentially buys no offices.")
        cty_diff = (sub["country"].value_counts(normalize=True) - overall_country.reindex(sub["country"].unique()).fillna(0))
        if cty_diff.max() >= 0.05:
            prop.append(f"Over-represented market: {cty_diff.idxmax()} ({sub['country'].value_counts(normalize=True)[cty_diff.idxmax()]*100:.0f}% "
                        f"vs {overall_country[cty_diff.idxmax()]*100:.0f}% overall).")
        else:
            prop.append("Geography does not set this segment apart (country mix is close to the overall mix), "
                        "so use the standard regional inventory.")
        if p["pct_international"] >= 1.2 * overall["pct_international"]:
            prop.append("Include listings suited to overseas buyers (remote purchase, managed-rental options).")

        assumptions.append("Marketing recommendations are rule-based interpretations of the measured traits; "
                           "their effect on conversion has not been tested (no campaign data available).")

        insights.append({
            "cluster": int(cid), "cluster_id": f"C{int(cid)+1}", "name": names[cid],
            "size": int(p["size"]), "pct": float(p["pct_of_clients"] * 100),
            "headline": f"{names[cid]} - {int(p['size'])} clients ({p['pct_of_clients']*100:.1f}%)",
            "customer_profile": customer, "behavioral_profile": behavioural,
            "investment_profile": investment, "financing_profile": financing,
            "geographic_profile": geography,
            "marketing_recommendations": marketing, "property_recommendations": prop,
            "data_supported_findings": facts, "assumptions": assumptions,
            "distinguishing_traits": {k: float(v) for k, v in e.sort_values(key=lambda s: -s.abs()).head(4).items()},
        })
    return insights
