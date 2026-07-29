"""
Oxano Capital — Portfolio Risk & Early-Warning Monitoring
Live demo dashboard. Run with:  streamlit run app.py

Data: expects the CSVs from Oxano_Streamlit_Ready_CSVs.zip in the same
folder as this script (or edit DATA_DIR below).
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Oxano Portfolio Risk Monitor", layout="wide", page_icon="🛡️")

# ---------------------------------------------------------------------------
# DATA LOADING (cached so toggling environment doesn't reload from disk)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    d = {}
    d["master"] = pd.read_csv(DATA_DIR / "realistic_company_master.csv", parse_dates=["investment_date"])
    d["scored"] = pd.read_csv(DATA_DIR / "realistic_scored_company_months.csv", parse_dates=["month"])
    d["perf"] = pd.read_csv(DATA_DIR / "realistic_model_performance_by_environment.csv")
    d["leadtime"] = pd.read_csv(DATA_DIR / "realistic_detection_lead_time.csv",
                                 parse_dates=["event_month", "first_high_risk_flag_month"])
    d["importance"] = pd.read_csv(DATA_DIR / "realistic_feature_importance.csv")
    d["montecarlo"] = pd.read_csv(DATA_DIR / "realistic_monte_carlo_cash_runway.csv")
    d["events"] = pd.read_csv(DATA_DIR / "realistic_distress_event_log.csv", parse_dates=["event_month"])
    return d

data = load_data()

FRIENDLY_NAMES = {
    "revenue_3m_pct_change": "Revenue trend (3-month)",
    "ebitda_margin": "EBITDA margin",
    "cash_runway_months": "Cash runway (months)",
    "accounts_receivable_days": "Receivable days",
    "accounts_payable_days": "Payable days",
    "inventory_turnover_days": "Inventory turnover days",
    "top5_customer_concentration_pct": "Top-5 customer concentration",
    "anomaly_flag": "Transaction anomaly flag",
    "monthly_report_submitted_on_time": "Reporting on time",
    "days_late_reporting": "Days late (reporting)",
    "leadership_turnover_event": "Leadership turnover event",
    "auditor_changed_event": "Auditor changed event",
    "days_late": "Days late (repayment)",
    "amount_paid_ratio": "Amount paid ratio",
    "receivable_payable_gap": "Receivable-payable gap",
}
TIER_COLOR = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}

# ---------------------------------------------------------------------------
# SIDEBAR — the live toggle
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Oxano Portfolio Monitor")
st.sidebar.caption("Live demo — synthetic data")

environments = list(data["master"]["environment"].unique())
env = st.sidebar.selectbox("Market environment", environments, index=1)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**What this toggle does:** switches the entire dashboard to a dataset "
    "generated under different macro-economic stress conditions — same "
    "portfolio, same model, different market noise. Nothing is pre-rendered "
    "per environment; every chart below recomputes from the selected slice."
)

# Filter everything to the selected environment
master_e = data["master"][data["master"].environment == env]
scored_e = data["scored"][data["scored"].environment == env]
perf_e = data["perf"][data["perf"].environment == env].iloc[0]
leadtime_e = data["leadtime"][data["leadtime"].environment == env]
importance_e = data["importance"][data["importance"].environment == env].copy()
importance_e["feature"] = importance_e["feature"].map(FRIENDLY_NAMES).fillna(importance_e["feature"])
mc_e = data["montecarlo"][data["montecarlo"].environment == env]

latest_month = scored_e["month"].max()
latest = scored_e[scored_e.month == latest_month]

# ---------------------------------------------------------------------------
# HEADER + KPI ROW
# ---------------------------------------------------------------------------
st.title("Portfolio Risk & Early-Warning Monitoring")
st.caption(f"Environment: **{env}**  ·  Snapshot month: **{latest_month.strftime('%B %Y')}**  ·  Data is synthetic, structured to mirror real portfolio monitoring")

k1, k2, k3, k4, k5 = st.columns(5)
total_capital = master_e["investment_amount_usd"].sum()
high_risk_ids = latest.loc[latest.risk_tier == "High", "company_id"]
at_risk_capital = master_e.loc[master_e.company_id.isin(
    latest.loc[latest.risk_tier.isin(["High", "Medium"]), "company_id"]), "investment_amount_usd"].sum()

k1.metric("Companies monitored", master_e["company_id"].nunique())
k2.metric("Capital deployed", f"${total_capital/1e6:.2f}M")
k3.metric("High risk right now", int((latest.risk_tier == "High").sum()))
k4.metric("Capital at High/Medium risk", f"${at_risk_capital/1e6:.2f}M",
          f"{at_risk_capital/total_capital*100:.0f}% of portfolio")
k5.metric("Model AUC-ROC (this environment)", f"{perf_e['model_auc_roc']:.2f}" if pd.notna(perf_e['model_auc_roc']) else "n/a")

st.markdown("---")

# ---------------------------------------------------------------------------
# PORTFOLIO RISK TABLE
# ---------------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Portfolio risk — current snapshot")
    view = latest.merge(master_e[["company_id", "company_name", "sector", "investment_type", "investment_amount_usd"]],
                         on="company_id").sort_values("distress_risk_score", ascending=False)
    view = view[["company_name", "sector", "investment_type", "investment_amount_usd",
                 "distress_risk_score", "risk_tier"]]
    view.columns = ["Company", "Sector", "Investment type", "Capital (USD)", "Risk score (0-100)", "Tier"]

    def tier_style(row):
        color = TIER_COLOR.get(row["Tier"], "#000000")
        return [f"color: {color}; font-weight: 600" if col == "Tier" else "" for col in row.index]

    st.dataframe(
        view.style.apply(tier_style, axis=1).format({"Capital (USD)": "${:,.0f}", "Risk score (0-100)": "{:.1f}"}),
        use_container_width=True, height=430,
    )

with right:
    st.subheader("What's driving the score")
    fig = px.bar(importance_e.sort_values("importance").tail(8), x="importance", y="feature",
                 orientation="h", color_discrete_sequence=["#1F3864"])
    fig.update_layout(showlegend=False, xaxis_title="Relative importance", yaxis_title="",
                       margin=dict(l=0, r=10, t=10, b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# MONTE CARLO CASH RUNWAY — the fan chart
# ---------------------------------------------------------------------------
st.subheader("Cash runway simulation — 6-month forward projection")
sel_company = st.selectbox(
    "Select a portfolio company",
    master_e.sort_values("company_name")["company_name"],
    key="mc_company",
)
sel_id = master_e.loc[master_e.company_name == sel_company, "company_id"].values[0]
mc_c = mc_e[mc_e.company_id == sel_id].sort_values("months_forward")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=mc_c.months_forward, y=mc_c.simulated_cash_p90, line=dict(width=0),
                           showlegend=False, hoverinfo="skip"))
fig2.add_trace(go.Scatter(x=mc_c.months_forward, y=mc_c.simulated_cash_p10, fill="tonexty",
                           fillcolor="rgba(31,56,100,0.15)", line=dict(width=0),
                           name="P10–P90 range", hoverinfo="skip"))
fig2.add_trace(go.Scatter(x=mc_c.months_forward, y=mc_c.simulated_cash_p50, line=dict(color="#1F3864", width=3),
                           name="Median (P50)"))
fig2.add_hline(y=0, line_dash="dash", line_color="#C62828", annotation_text="Cash-out line")
fig2.update_layout(xaxis_title="Months forward", yaxis_title="Simulated cash balance (USD)",
                    height=420, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig2, use_container_width=True)

last_row = mc_c[mc_c.months_forward == mc_c.months_forward.max()].iloc[0]
st.caption(
    f"Based on 500 simulated paths under **{env}** conditions: by month {int(last_row.months_forward)}, "
    f"there is a **{last_row.probability_cash_negative*100:.0f}% probability** {sel_company} runs out of cash "
    f"if current trends continue unchanged."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# DETECTION LEAD TIME — the proof-of-concept table
# ---------------------------------------------------------------------------
st.subheader("Early-warning track record (backtest on simulated history)")
if len(leadtime_e) == 0:
    st.info("No distress events occurred in this environment's simulated history.")
else:
    lt = leadtime_e[["company_name", "event_month", "first_high_risk_flag_month",
                      "detection_lead_time_months", "was_detected_before_event"]].copy()
    lt.columns = ["Company", "Event month", "First flagged High-risk", "Lead time (months)", "Caught before event?"]
    lt["Caught before event?"] = lt["Caught before event?"].map({True: "✅ Yes", False: "❌ Missed"})
    st.dataframe(lt, use_container_width=True, hide_index=True)
    caught = leadtime_e["was_detected_before_event"].mean()
    avg_lead = leadtime_e.loc[leadtime_e.was_detected_before_event, "detection_lead_time_months"].mean()
    st.caption(
        f"In this environment, **{caught*100:.0f}%** of distress events were flagged in advance, "
        f"with an average lead time of **{avg_lead:.1f} months** when caught. "
        f"False-positive rate at the High-risk threshold: **{perf_e['false_positive_rate_high_tier']*100:.1f}%**."
    )

st.markdown("---")
st.caption(
    "All data on this dashboard is synthetic, built to demonstrate data structure, modeling approach, "
    "and dashboard mechanics. Model performance on Oxano's real portfolio data will differ and the model "
    "would need to be retrained on actual reporting history."
)
