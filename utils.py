"""
utils.py
--------
Shared data-loading, feature-engineering, forecasting, anomaly-detection and
clustering utilities used by every page of the Sales Forecasting Dashboard.

All heavy computations are wrapped with @st.cache_data so that switching
between pages / changing filters does not repeatedly reprocess the raw data.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA_PATH = "train.csv"

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}

PRIMARY = "#0E4C92"
BG = "#f5f7fb"


# --------------------------------------------------------------------------
# SHARED STYLING
# --------------------------------------------------------------------------
def apply_page_style():
    st.markdown(
        f"""
        <style>
        .main {{ background-color: {BG}; }}
        .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
        .page-title {{ font-size: 34px; font-weight: bold; color: {PRIMARY}; }}
        .page-subtitle {{ font-size: 16px; color: gray; margin-bottom: 10px; }}
        .metric-card {{
            background: white; padding: 18px; border-radius: 12px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.08); text-align: center;
        }}
        .metric-card h4 {{ margin: 0; color: gray; font-weight: 500; font-size: 14px; }}
        .metric-card h2 {{ margin: 4px 0 0 0; color: {PRIMARY}; }}
        .section-card {{
            background: white; padding: 18px 22px; border-radius: 14px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.06); margin-bottom: 18px;
        }}
        .footer {{ text-align: center; color: gray; padding-top: 30px; font-size: 13px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str):
    st.markdown(
        f"""<div class="metric-card"><h4>{label}</h4><h2>{value}</h2></div>""",
        unsafe_allow_html=True,
    )


def sidebar_branding(active_page: str):
    st.sidebar.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=64)
    st.sidebar.title("Sales Forecasting")
    st.sidebar.caption(f"📍 {active_page}")
    st.sidebar.markdown("---")


# --------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING (Task 1)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["Order Date"]).copy()

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["MonthName"] = df["Order Date"].dt.strftime("%b")
    df["Week"] = df["Order Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Order Date"].dt.day_name()
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Season"] = df["Month"].map(SEASON_MAP)
    df["ShipDelayDays"] = (df["Ship Date"] - df["Order Date"]).dt.days

    return df


@st.cache_data(show_spinner=False)
def get_filter_options(df: pd.DataFrame):
    categories = ["All"] + sorted(df["Category"].unique().tolist())
    regions = ["All"] + sorted(df["Region"].unique().tolist())
    sub_categories = ["All"] + sorted(df["Sub-Category"].unique().tolist())
    years = sorted(df["Year"].unique().tolist())
    return categories, regions, sub_categories, years


@st.cache_data(show_spinner=False)
def filter_data(df: pd.DataFrame, category="All", region="All", sub_category="All", years=None):
    data = df.copy()
    if category and category != "All":
        data = data[data["Category"] == category]
    if region and region != "All":
        data = data[data["Region"] == region]
    if sub_category and sub_category != "All":
        data = data[data["Sub-Category"] == sub_category]
    if years:
        data = data[data["Year"].isin(years)]
    return data


@st.cache_data(show_spinner=False)
def get_monthly_series(df: pd.DataFrame, category="All", region="All") -> pd.Series:
    data = filter_data(df, category=category, region=region)
    monthly = data.set_index("Order Date").resample("MS")["Sales"].sum()
    monthly = monthly.asfreq("MS").fillna(0)
    return monthly


# --------------------------------------------------------------------------
# FORECASTING — XGBoost (Task 3 winning model, used in production per Task 7)
# --------------------------------------------------------------------------
def _build_supervised_features(monthly: pd.Series, n_lags: int = 3) -> pd.DataFrame:
    d = pd.DataFrame({"y": monthly.values}, index=monthly.index)
    for lag in range(1, n_lags + 1):
        d[f"lag_{lag}"] = d["y"].shift(lag)
    d["rolling_mean_3"] = d["y"].shift(1).rolling(3).mean()
    d["Month"] = d.index.month
    d["Quarter"] = d.index.quarter
    d["Season"] = d["Month"].map(SEASON_MAP)
    d = pd.get_dummies(d, columns=["Season"], drop_first=True)
    d = d.dropna()
    return d


@st.cache_data(show_spinner=False)
def train_and_forecast(monthly: pd.Series, horizon: int = 3, test_size: int = 6):
    """
    Trains an XGBoost regressor on lag/rolling/calendar features, evaluates it
    on a held-out tail of the series, then refits on all data and produces a
    recursive multi-step forecast `horizon` months ahead.
    """
    feat = _build_supervised_features(monthly)
    feature_cols = [c for c in feat.columns if c != "y"]

    if len(feat) <= test_size + 5:
        test_size = max(1, len(feat) // 4)

    train, test = feat.iloc[:-test_size], feat.iloc[-test_size:]

    eval_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    )
    eval_model.fit(train[feature_cols], train["y"])
    test_pred = eval_model.predict(test[feature_cols])

    mae = mean_absolute_error(test["y"], test_pred)
    rmse = float(np.sqrt(mean_squared_error(test["y"], test_pred)))
    denom = np.where(test["y"].values == 0, np.nan, test["y"].values)
    mape = float(np.nanmean(np.abs((test["y"].values - test_pred) / denom)) * 100)

    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    )
    final_model.fit(feat[feature_cols], feat["y"])

    extended = monthly.copy()
    future_dates = pd.date_range(
        extended.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    forecasts = []
    for date in future_dates:
        lag1, lag2, lag3 = extended.iloc[-1], extended.iloc[-2], extended.iloc[-3]
        rolling_mean_3 = extended.iloc[-3:].mean()
        row = {
            "lag_1": lag1, "lag_2": lag2, "lag_3": lag3,
            "rolling_mean_3": rolling_mean_3,
            "Month": date.month, "Quarter": date.quarter,
        }
        season = SEASON_MAP[date.month]
        for col in feature_cols:
            if col.startswith("Season_"):
                row[col] = 1 if col == f"Season_{season}" else 0
        row_df = pd.DataFrame([row])[feature_cols]
        pred = float(final_model.predict(row_df)[0])
        forecasts.append(pred)
        extended.loc[date] = pred

    forecast_series = pd.Series(forecasts, index=future_dates)

    return {
        "forecast": forecast_series,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "test_actual": test["y"],
        "test_pred": pd.Series(test_pred, index=test.index),
    }


# --------------------------------------------------------------------------
# ANOMALY DETECTION (Task 5)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def detect_anomalies(df: pd.DataFrame, zscore_window: int = 8, zscore_threshold: float = 2.0):
    weekly = df.set_index("Order Date").resample("W")["Sales"].sum()
    result = pd.DataFrame({"Sales": weekly})

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_flags = iso.fit_predict(result[["Sales"]])
    result["IsolationForest_Anomaly"] = iso_flags == -1

    roll_mean = result["Sales"].rolling(zscore_window, min_periods=3).mean()
    roll_std = result["Sales"].rolling(zscore_window, min_periods=3).std()
    result["ZScore"] = (result["Sales"] - roll_mean) / roll_std
    result["ZScore_Anomaly"] = result["ZScore"].abs() > zscore_threshold

    result["Both_Methods_Agree"] = result["IsolationForest_Anomaly"] & result["ZScore_Anomaly"]
    return result


# --------------------------------------------------------------------------
# PRODUCT DEMAND SEGMENTATION (Task 6)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def segment_products(df: pd.DataFrame, n_clusters: int = 4):
    data = df.copy()
    data["YearMonth"] = data["Order Date"].dt.to_period("M")

    monthly_sub = data.groupby(["Sub-Category", "YearMonth"])["Sales"].sum().reset_index()

    agg = data.groupby("Sub-Category").agg(
        Total_Sales=("Sales", "sum"),
        Avg_Order_Value=("Sales", "mean"),
    ).reset_index()

    vol = (
        monthly_sub.groupby("Sub-Category")["Sales"]
        .std().fillna(0).reset_index()
        .rename(columns={"Sales": "Volatility"})
    )

    yearly = data.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()
    growth_rows = []
    for sub in yearly["Sub-Category"].unique():
        sub_y = yearly[yearly["Sub-Category"] == sub].sort_values("Year")
        if len(sub_y) >= 2 and sub_y["Sales"].iloc[0] > 0:
            growth = (sub_y["Sales"].iloc[-1] - sub_y["Sales"].iloc[0]) / sub_y["Sales"].iloc[0] * 100
        else:
            growth = 0.0
        growth_rows.append({"Sub-Category": sub, "Growth_Rate": growth})
    growth_df = pd.DataFrame(growth_rows)

    features = agg.merge(vol, on="Sub-Category").merge(growth_df, on="Sub-Category")

    feature_cols = ["Total_Sales", "Growth_Rate", "Volatility", "Avg_Order_Value"]
    X_scaled = StandardScaler().fit_transform(features[feature_cols].values)

    inertias = []
    k_range = list(range(1, min(9, len(features))))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    n_clusters = min(n_clusters, len(features))
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features["Cluster"] = km_final.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    features["PCA1"], features["PCA2"] = coords[:, 0], coords[:, 1]

    med_sales = features["Total_Sales"].median()
    med_growth = features["Growth_Rate"].median()
    med_vol = features["Volatility"].median()

    labels = {}
    for c in sorted(features["Cluster"].unique()):
        sub = features[features["Cluster"] == c]
        high_sales = sub["Total_Sales"].mean() >= med_sales
        high_growth = sub["Growth_Rate"].mean() >= med_growth
        high_vol = sub["Volatility"].mean() >= med_vol

        if high_sales and not high_vol:
            labels[c] = "High Volume, Stable Demand"
        elif not high_sales and high_vol:
            labels[c] = "Low Volume, High Volatility"
        elif high_growth:
            labels[c] = "Growing Demand"
        else:
            labels[c] = "Declining Demand"

    features["Segment_Label"] = features["Cluster"].map(labels)

    return features, k_range, inertias