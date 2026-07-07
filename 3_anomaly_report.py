import plotly.graph_objects as go
import streamlit as st

from utils import apply_page_style, detect_anomalies, kpi_card, load_data, sidebar_branding

st.set_page_config(page_title="Anomaly Report", page_icon="🚨", layout="wide")
apply_page_style()
sidebar_branding("Anomaly Report")

df = load_data()

# --------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------
st.sidebar.subheader("🔎 Detection Settings")
z_window = st.sidebar.slider("Z-Score rolling window (weeks)", min_value=4, max_value=16, value=8, key="an_win")
z_threshold = st.sidebar.slider("Z-Score threshold (std devs)", min_value=1.0, max_value=3.5, value=2.0, step=0.5, key="an_thresh")
method_filter = st.sidebar.multiselect(
    "Show anomalies detected by",
    ["Isolation Forest", "Z-Score", "Both Methods"],
    default=["Isolation Forest", "Z-Score"],
    key="an_method",
)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown('<p class="page-title">🚨 Anomaly Detection Report</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Unusually high or low weekly sales, flagged using Isolation Forest and Z-Score methods</p>',
    unsafe_allow_html=True,
)

result = detect_anomalies(df, zscore_window=z_window, zscore_threshold=z_threshold)

iso_count = int(result["IsolationForest_Anomaly"].sum())
z_count = int(result["ZScore_Anomaly"].sum())
both_count = int(result["Both_Methods_Agree"].sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Total Weeks", f"{len(result)}")
with c2:
    kpi_card("Isolation Forest Flags", f"{iso_count}")
with c3:
    kpi_card("Z-Score Flags", f"{z_count}")
with c4:
    kpi_card("Flagged by Both", f"{both_count}")

st.write("")

# --------------------------------------------------------------------------
# ANOMALY CHART
# --------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📉 Weekly Sales with Anomalies Highlighted")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=result.index, y=result["Sales"], mode="lines", name="Weekly Sales",
    line=dict(color="#0E4C92", width=1.5),
))

if "Isolation Forest" in method_filter:
    iso_pts = result[result["IsolationForest_Anomaly"]]
    fig.add_trace(go.Scatter(
        x=iso_pts.index, y=iso_pts["Sales"], mode="markers", name="Isolation Forest Anomaly",
        marker=dict(color="#E74C3C", size=9, symbol="circle"),
    ))

if "Z-Score" in method_filter:
    z_pts = result[result["ZScore_Anomaly"]]
    fig.add_trace(go.Scatter(
        x=z_pts.index, y=z_pts["Sales"], mode="markers", name="Z-Score Anomaly",
        marker=dict(color="#F39C12", size=11, symbol="x"),
    ))

if "Both Methods" in method_filter:
    both_pts = result[result["Both_Methods_Agree"]]
    fig.add_trace(go.Scatter(
        x=both_pts.index, y=both_pts["Sales"], mode="markers", name="Flagged by Both",
        marker=dict(color="black", size=13, symbol="star"),
    ))

fig.update_layout(
    xaxis_title="Week", yaxis_title="Sales ($)", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# ANOMALY TABLE
# --------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📋 Detected Anomalies")

anomaly_rows = result[result["IsolationForest_Anomaly"] | result["ZScore_Anomaly"]].copy()
anomaly_rows.index.name = "Week"
anomaly_rows = anomaly_rows.reset_index()
anomaly_rows["Week"] = anomaly_rows["Week"].dt.strftime("%Y-%m-%d")
anomaly_rows["Sales"] = anomaly_rows["Sales"].round(2)
anomaly_rows["ZScore"] = anomaly_rows["ZScore"].round(2)
anomaly_rows = anomaly_rows[[
    "Week", "Sales", "ZScore", "IsolationForest_Anomaly", "ZScore_Anomaly", "Both_Methods_Agree"
]].sort_values("Week")

st.dataframe(anomaly_rows, width="stretch", hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("🧠 How to read this report"):
    st.write(
        """
        - **Isolation Forest** learns the overall pattern of weekly sales and flags the
          weeks that are structurally the most "different" from the rest, without
          assuming any particular distribution.
        - **Z-Score** flags a week if its sales deviate more than the chosen threshold
          (in standard deviations) from a rolling mean of recent weeks — a more local,
          recency-sensitive definition of "unusual."
        - Weeks flagged by **both methods** (⭐ on the chart) are the strongest
          candidates for genuine, business-relevant anomalies (e.g. holiday sales
          spikes, stock-outs, or data issues) and are worth investigating first.
        """
    )

st.markdown(
    '<div class="footer">Page 3 of 4 — Anomaly Report</div>', unsafe_allow_html=True
)