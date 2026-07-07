import plotly.express as px
import streamlit as st

from utils import apply_page_style, filter_data, get_filter_options, kpi_card, load_data, sidebar_branding

st.set_page_config(page_title="Sales Overview", page_icon="📊", layout="wide")
apply_page_style()
sidebar_branding("Sales Overview")

df = load_data()
categories, regions, sub_categories, years = get_filter_options(df)

# --------------------------------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------------------------------
st.sidebar.subheader("🔎 Filters")
sel_category = st.sidebar.selectbox("Category", categories, key="ov_cat")
sel_region = st.sidebar.selectbox("Region", regions, key="ov_reg")
sel_years = st.sidebar.multiselect("Year(s)", years, default=years, key="ov_year")

if st.sidebar.button("Reset Filters", width="stretch"):
    st.session_state["ov_cat"] = "All"
    st.session_state["ov_reg"] = "All"
    st.session_state["ov_year"] = years
    st.rerun()

filtered = filter_data(df, category=sel_category, region=sel_region, years=sel_years)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown('<p class="page-title">📊 Sales Overview Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Yearly, monthly, regional and category-wise sales performance</p>',
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No records match the selected filters. Please broaden your selection.")
    st.stop()

# --------------------------------------------------------------------------
# KPI ROW
# --------------------------------------------------------------------------
total_sales = filtered["Sales"].sum()
total_orders = filtered["Order ID"].nunique()
n_categories = filtered["Category"].nunique()
n_subcategories = filtered["Sub-Category"].nunique()
avg_order_value = total_sales / total_orders if total_orders else 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Total Sales", f"${total_sales:,.0f}")
with c2:
    kpi_card("Total Orders", f"{total_orders:,}")
with c3:
    kpi_card("Categories", f"{n_categories}")
with c4:
    kpi_card("Sub-Categories", f"{n_subcategories}")
with c5:
    kpi_card("Avg Order Value", f"${avg_order_value:,.0f}")

st.write("")

# --------------------------------------------------------------------------
# TOTAL SALES BY YEAR
# --------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📅 Total Sales by Year")
yearly = filtered.groupby("Year", as_index=False)["Sales"].sum()
fig_year = px.bar(
    yearly, x="Year", y="Sales", text_auto=".2s",
    color="Sales", color_continuous_scale="Blues",
)
fig_year.update_layout(showlegend=False, coloraxis_showscale=False, xaxis=dict(dtick=1))
st.plotly_chart(fig_year, width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# MONTHLY SALES TREND
# --------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📈 Monthly Sales Trend")
monthly = filtered.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
fig_month = px.line(monthly, x="Order Date", y="Sales", markers=True)
fig_month.update_traces(line_color="#0E4C92")
fig_month.update_layout(xaxis_title="Month", yaxis_title="Sales ($)")
st.plotly_chart(fig_month, width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# REGION & CATEGORY BREAKDOWN
# --------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🌎 Sales by Region")
    region_sales = filtered.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    fig_region = px.bar(
        region_sales, x="Region", y="Sales", color="Region", text_auto=".2s",
    )
    fig_region.update_layout(showlegend=False)
    st.plotly_chart(fig_region, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📦 Sales by Category")
    cat_sales = filtered.groupby("Category", as_index=False)["Sales"].sum()
    fig_cat = px.pie(cat_sales, names="Category", values="Sales", hole=0.45)
    st.plotly_chart(fig_cat, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# REGION x CATEGORY HEATMAP + SUB-CATEGORY TABLE
# --------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🧭 Region vs Category Breakdown")
pivot = filtered.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="sum", fill_value=0)
fig_heat = px.imshow(
    pivot, text_auto=".2s", color_continuous_scale="Blues", aspect="auto",
    labels=dict(color="Sales"),
)
st.plotly_chart(fig_heat, width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("📋 Sales by Sub-Category (table)"):
    sub_table = (
        filtered.groupby(["Category", "Sub-Category"], as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )
    sub_table["Sales"] = sub_table["Sales"].round(2)
    st.dataframe(sub_table, width="stretch", hide_index=True)

st.markdown(
    '<div class="footer">Page 1 of 4 — Sales Overview</div>', unsafe_allow_html=True
)