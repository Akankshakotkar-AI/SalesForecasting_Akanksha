import streamlit as st

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#0E4C92;
}

.subtitle{
    font-size:18px;
    color:gray;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.08);
    text-align:center;
}

.feature-card{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.08);
    min-height:170px;
}

.footer{
    text-align:center;
    color:gray;
    padding-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart--v1.png",
    width=80
)

st.sidebar.title("Sales Forecasting")

st.sidebar.markdown("---")

st.sidebar.success("Navigation")

st.sidebar.info("""
Use the pages menu above to open:

• Sales Overview

• Forecast Explorer

• Anomaly Report

• Product Demand
""")

st.sidebar.markdown("---")

st.sidebar.write("Project")
st.sidebar.caption("University Sales Forecasting Dashboard")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<p class="title">📈 Sales Forecasting Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Professional Business Analytics Dashboard built using Streamlit</p>',
    unsafe_allow_html=True
)

st.write("")

# -----------------------------
# KPI PLACEHOLDERS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
    <h4>Total Sales</h4>
    <h2>$0</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
    <h4>Total Orders</h4>
    <h2>0</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
    <h4>Categories</h4>
    <h2>0</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
    <h4>Sub-Categories</h4>
    <h2>0</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# -----------------------------
# DASHBOARD FEATURES
# -----------------------------
st.subheader("Dashboard Modules")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class="feature-card">

    <h3>📊 Sales Overview</h3>

    <p>
    Analyze yearly, monthly, regional and category-wise sales using interactive charts.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="feature-card">

    <h3>🚨 Anomaly Detection</h3>

    <p>
    Detect unusual sales transactions using Isolation Forest and visualize anomalies.
    </p>

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="feature-card">

    <h3>📈 Forecast Explorer</h3>

    <p>
    Explore future sales forecasts using the trained XGBoost forecasting model.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="feature-card">

    <h3>📦 Product Demand Segmentation</h3>

    <p>
    View customer demand clusters using PCA and K-Means segmentation.
    </p>

    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# -----------------------------
# PROJECT INFORMATION
# -----------------------------
with st.expander("📚 Project Information"):

    st.write("""
This dashboard contains four interactive modules:

- Sales Overview Dashboard
- Forecast Explorer
- Anomaly Detection Report
- Product Demand Segmentation

The forecasting model has already been trained.

Best Forecasting Model:

**XGBoost**

Use the sidebar to navigate between pages.
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    '<div class="footer">Developed for University Sales Forecasting Project using Streamlit</div>',
    unsafe_allow_html=True
)