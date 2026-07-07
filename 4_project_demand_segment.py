import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import apply_page_style, kpi_card, load_data, segment_products, sidebar_branding

st.set_page_config(page_title="Product Demand Segments", page_icon="📦", layout="wide")
apply_page_style()
sidebar_branding("Product Demand Segments")

df = load_data()

# --------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------
st.sidebar.subheader("🔎 Clustering Settings")
n_clusters = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=6, value=4, key="seg_k")

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown('<p class="page-title">📦 Product Demand Segmentation</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Sub-categories grouped by demand behaviour using K-Means clustering on '
    'total sales, growth rate, volatility and average order value</p>',
    unsafe_allow_html=True,
)

features, k_range, inertias = segment_products(df, n_clusters=n_clusters)

# --------------------------------------------------------------------------
# KPI ROW
# --------------------------------------------------------------------------
n_segments = features["Segment_Label"].nunique()
largest_segment = features["Segment_Label"].value_counts().idxmax()
top_growth_sub = features.loc[features["Growth_Rate"].idxmax(), "Sub-Category"]
top_volatile_sub = features.loc[features["Volatility"].idxmax(), "Sub-Category"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Sub-Categories", f"{len(features)}")
with c2:
    kpi_card("Demand Segments", f"{n_segments}")
with c3:
    kpi_card("Fastest Growing", top_growth_sub)
with c4:
    kpi_card("Most Volatile", top_volatile_sub)

st.write("")

# --------------------------------------------------------------------------
# ELBOW METHOD + CLUSTER SCATTER
# --------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📐 Elbow Method")
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(k_range), y=inertias, mode="lines+markers", line=dict(color="#0E4C92"),
    ))
    fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#E67E22")
    fig_elbow.update_layout(xaxis_title="K (clusters)", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧩 Product Clusters (PCA 2D Projection)")
    fig_scatter = px.scatter(
        features, x="PCA1", y="PCA2", color="Segment_Label",
        size="Total_Sales", hover_name="Sub-Category",
        hover_data={"Total_Sales": ":.0f", "Growth_Rate": ":.1f", "Volatility": ":.0f", "PCA1": False, "PCA2": False},
    )
    fig_scatter.update_layout(legend_title_text="Segment")
    st.plotly_chart(fig_scatter, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SEGMENT TABLE
# --------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📋 Sub-Categories by Demand Segment")

display_table = features[[
    "Sub-Category", "Segment_Label", "Total_Sales", "Growth_Rate", "Volatility", "Avg_Order_Value"
]].sort_values(["Segment_Label", "Total_Sales"], ascending=[True, False]).copy()

display_table["Total_Sales"] = display_table["Total_Sales"].round(0)
display_table["Growth_Rate"] = display_table["Growth_Rate"].round(1)
display_table["Volatility"] = display_table["Volatility"].round(0)
display_table["Avg_Order_Value"] = display_table["Avg_Order_Value"].round(2)
display_table = display_table.rename(columns={
    "Segment_Label": "Segment",
    "Total_Sales": "Total Sales ($)",
    "Growth_Rate": "Growth Rate (%)",
    "Volatility": "Volatility ($)",
    "Avg_Order_Value": "Avg Order Value ($)",
})

st.dataframe(display_table, width="stretch", hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# STOCKING STRATEGY PER SEGMENT
# --------------------------------------------------------------------------
STRATEGY = {
    "High Volume, Stable Demand": "Maintain steady stock levels with standard reorder points — low risk, prioritize efficient replenishment over safety stock.",
    "Low Volume, High Volatility": "Keep lean stock and rely on faster reorder cycles or on-demand ordering to avoid overstock during quiet periods.",
    "Growing Demand": "Increase safety stock and reorder frequency ahead of demand — treat as a priority for future capacity planning.",
    "Declining Demand": "Reduce stock commitments, run promotions to clear inventory, and reassess supplier contracts.",
}

with st.expander("💡 Recommended Stocking Strategy per Segment"):
    for segment in sorted(features["Segment_Label"].unique()):
        subs = features.loc[features["Segment_Label"] == segment, "Sub-Category"].tolist()
        st.markdown(f"**{segment}** ({', '.join(subs)})")
        st.write(STRATEGY.get(segment, "Monitor closely and adjust stocking based on observed trends."))
        st.write("")

st.markdown(
    '<div class="footer">Page 4 of 4 — Product Demand Segments</div>', unsafe_allow_html=True
)