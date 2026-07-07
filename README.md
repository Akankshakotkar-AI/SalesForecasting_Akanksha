# 📊 Sales Forecasting Dashboard

A Sales Forecasting and Business Analytics project developed using Python, Machine Learning, and Streamlit. This project analyzes historical retail sales data, forecasts future sales, detects anomalies, and segments products based on demand to support better business decision-making.

## 📌 Project Overview

The objective of this project is to help retail businesses improve inventory management and sales planning through data analysis and predictive modeling. An interactive Streamlit dashboard provides business users with easy access to key insights and forecasts.

## 🚀 Features

- Interactive Sales Overview Dashboard
- Sales Trend Analysis
- Monthly and Yearly Sales Visualization
- Sales Forecasting using XGBoost
- Forecast Explorer Dashboard
- Anomaly Detection using Isolation Forest and Z-Score
- Product Demand Segmentation using K-Means Clustering
- Interactive Filters (Region, Category, Year)
- KPI Cards and Business Insights
- Professional Multi-Page Streamlit Dashboard

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Streamlit
- Scikit-learn
- XGBoost
- Statsmodels
- Prophet

## 📂 Project Structure

```
SalesForecasting_Akanksha/
│
├── app.py
├── utils.py
├── analysis.ipynb
├── train.csv
├── requirements.txt
│
├── pages/
│   ├── 1_Sales_Overview.py
│   ├── 2_forecast_explorer.py
│   ├── 3_anomaly_report.py
│   └── 4_project_demand_segment.py
```

## 📈 Dashboard Modules

### 📊 Sales Overview
- Sales KPIs
- Monthly Sales Trend
- Sales by Region
- Sales by Category
- Top Sub-Categories

### 📈 Forecast Explorer
- Future Sales Prediction
- Interactive Forecast Visualization
- Model Performance Metrics

### 🚨 Anomaly Report
- Detection of Unusual Sales
- Interactive Charts
- Business Explanation

### 📦 Product Demand Segmentation
- Customer/Product Clustering
- Demand Segments
- Inventory Planning Insights

## 🎯 Machine Learning Models

- SARIMA
- Prophet
- XGBoost (Selected Best Model)
- Isolation Forest
- K-Means Clustering

## 📊 Dataset

The project uses the Superstore Sales Dataset (`train.csv`) containing:

- Order Date
- Ship Date
- Sales
- Region
- Category
- Sub-Category
- Order ID

## 💡 Business Benefits

- Improve demand forecasting
- Better inventory management
- Identify abnormal sales patterns
- Support data-driven decision making
- Reduce stock shortages and overstocking

## ▶️ How to Run

1. Clone this repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## 👩‍🎓 Developed By

**Akanksha Kotkar**

B.Tech (Artificial Intelligence & Data Science)

NMIMS Indore

---

⭐ If you found this project useful, consider giving it a star on GitHub!
