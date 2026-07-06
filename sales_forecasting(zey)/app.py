"""
Sales Forecast Dashboard
=========================
A simple Streamlit interface wrapping the forecasting logic from main.py.

Place this file INSIDE your existing project folder (same level as main.py),
since it imports src/data_generator.py, src/models.py and src/visualize.py as-is.

Install:
    pip install -r requirements.txt

Run:
    streamlit run app.py
"""
import sys
import os
# Alt klasordeki app.py dosyasinin ust klasorleri gorebilmesi icin pusula ekliyoruz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
import os
import tempfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.data_generator import generate_synthetic_data  # noqa: F401 (optional use)
from src.models import (
    PROPHET_AVAILABLE,
    LinearRegressionWrapper,
    MLPWrapper,
    ProphetWrapper,
    SARIMAXWrapper,
)
from src.visualize import plot_forecast_scenarios, plot_validation_comparison

# ---------------------------------------------------------------------------
# Global City Presets (Expanded for Global Climate Coverage)
# ---------------------------------------------------------------------------
CITY_PRESETS = {
    "Istanbul, Turkey": (41.0082, 28.9784),
    "Ankara, Turkey": (39.9334, 32.8597),
    "Izmir, Turkey": (38.4237, 27.1428),
    "Paris, France": (48.8566, 2.3522),
    "Lyon, France": (45.7640, 4.8357),
    "Berlin, Germany": (52.5200, 13.4050),
    "Frankfurt, Germany": (50.1109, 8.6821),
    "London, UK": (51.5074, -0.1278),
    "Madrid, Spain": (40.4168, -3.7038),
    "Rome, Italy": (41.9028, 12.4964),
    "Milan, Italy": (45.4642, 9.1900),
    "Amsterdam, Netherlands": (52.3676, 4.9041),
    "Brussels, Belgium": (50.8503, 4.3517),
    "Vienna, Austria": (48.2082, 16.3738),
    "Zurich, Switzerland": (47.3769, 8.5417),
    "Athens, Greece": (37.9838, 23.7275),
    "Lisbon, Portugal": (38.7223, -9.1393),
    "Warsaw, Poland": (52.2297, 21.0122),
    "Stockholm, Sweden": (59.3293, 18.0686),
    "Oslo, Norway": (59.9139, 10.7522),
    "Helsinki, Finland": (60.1699, 24.9384),
    "Copenhagen, Denmark": (55.6761, 12.5683),
    "New York, USA": (40.7128, -74.0060),
    "Los Angeles, USA": (34.0522, -118.2437),
    "Chicago, USA": (41.8781, -87.6298),
    "Miami, USA": (25.7617, -80.1918),
    "Toronto, Canada": (43.6532, -79.3832),
    "Vancouver, Canada": (49.2827, -123.1207),
    "Mexico City, Mexico": (19.4326, -99.1332),
    "Sao Paulo, Brazil": (-23.5505, -46.6333),
    "Buenos Aires, Argentina": (-34.6037, -58.3816),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Seoul, South Korea": (37.5665, 126.9780),
    "Shanghai, China": (31.2304, 121.4737),
    "Beijing, China": (39.9042, 116.4074),
    "Singapore, Singapore": (1.3521, 103.8198),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Melbourne, Australia": (-37.8136, 144.9631),
    "Mumbai, India": (19.0760, 72.8777),
    "New Delhi, India": (28.6139, 77.2090),
    "Dubai, UAE": (25.2048, 55.2708),
    "Riyadh, Saudi Arabia": (24.7136, 46.6753),
    "Doha, Qatar": (25.2854, 51.5310),
    "Cairo, Egypt": (30.0444, 31.2357),
    "Johannesburg, South Africa": (-26.2041, 28.0473),
    "Nairobi, Kenya": (-1.2921, 36.8219),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def sanity_check_predictions(preds, hist_max, model_name):
    preds = np.nan_to_num(preds, nan=0.0, posinf=hist_max * 3, neginf=0.0)
    if preds.max() > hist_max * 5:
        st.sidebar.warning(f"{model_name} predictions came out too high, they were capped.")
    if preds.max() < hist_max * 0.01:
        st.sidebar.warning(f"{model_name} predictions came out too low, they were capped.")
    # Ensure rounded integers for physical units like air conditioners
    return np.round(np.clip(preds, 0, hist_max * 3)).astype(int)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "WAPE (%)": round(wape, 2),
        "sMAPE (%)": round(smape, 2),
        "R-squared": round(r2, 4),
    }


def get_city_temperatures(dates, lat, lon):
    global_warming_rate = 0.04
    try:
        min_date = dates.min().strftime("%Y-%m-%d")
        max_date = dates.max().strftime("%Y-%m-%d")
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={min_date}&end_date={max_date}"
            f"&daily=temperature_2m_mean&timezone=UTC"
        )
        response = requests.get(url, verify=False, timeout=20)
        data = response.json()
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(data["daily"]["time"]),
                "Temperature": data["daily"]["temperature_2m_mean"],
            }
        )
        df["Temperature"] = df["Temperature"].ffill().bfill().fillna(15.0)
        df_monthly = df.groupby(pd.Grouper(key="Date", freq="MS")).mean().reset_index()

        if len(df_monthly) > 24:
            min_year = df_monthly["Date"].dt.year.min()
            max_year = df_monthly["Date"].dt.year.max()
            years_diff = max_year - min_year

            if years_diff >= 2:
                first_years = df_monthly[df_monthly["Date"].dt.year <= min_year + 1]["Temperature"].mean()
                last_years = df_monthly[df_monthly["Date"].dt.year >= max_year - 1]["Temperature"].mean()

                if pd.notna(first_years) and pd.notna(last_years):
                    calculated_rate = (last_years - first_years) / years_diff
                    global_warming_rate = np.clip(calculated_rate, 0.01, 0.15)
                    if np.isnan(global_warming_rate):
                        global_warming_rate = 0.04

        st.session_state["calculated_warming_rate"] = global_warming_rate
        return df_monthly
    except Exception as e:
        st.session_state["calculated_warming_rate"] = global_warming_rate
        records = []
        base, amp = 13, 8.5
        for dt in dates:
            angle = 2 * np.pi * (dt.month - 1) / 12
            temp = base - amp * np.cos(angle) + global_warming_rate * (dt.year - 2018)
            records.append({"Date": dt, "Temperature": temp})
        return pd.DataFrame(records)


def generate_future_scenarios(start, end, df_weather_hist):
    dates = pd.date_range(start, end, freq="MS")
    warming_rate = st.session_state.get("calculated_warming_rate", 0.04)

    monthly_clima = df_weather_hist.copy()
    monthly_clima["month"] = monthly_clima["Date"].dt.month
    monthly_avg = monthly_clima.groupby("month")["Temperature"].mean()
    ref_year = df_weather_hist["Date"].dt.year.min()

    records = []
    for dt in dates:
        m = dt.month
        base_temp = monthly_avg.get(m, monthly_avg.mean())
        year_diff = dt.year - ref_year
        records.append(
            {
                "Date": dt,
                "Temp_Normal": base_temp + warming_rate * year_diff,
                "Temp_Hot": base_temp + 2 + (warming_rate + 0.02) * year_diff,
                "Temp_Cold": base_temp - 1.2,
            }
        )
    return pd.DataFrame(records)


def load_and_preprocess_sales(df_raw, split_year=2025):
    df = df_raw.copy()
    df.columns = df.columns.str.lower()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    elif "year" in df.columns and "month" in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str) + "-01", errors="coerce"
        )
    else:
        raise ValueError("The dataset must contain a 'date' column, or both 'year' and 'month' columns.")

    n_bad_dates = int(df["date"].isna().sum())
    if n_bad_dates > 0:
        df = df.dropna(subset=["date"])

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

    train_mask = df["year"] < split_year
    lower_fixed = 500.0
    train_sales = df.loc[train_mask, "sales"]
    if train_sales.notna().sum() >= 4:
        q1, q3 = train_sales.quantile(0.25), train_sales.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
    else:
        upper_bound = np.inf

    anomalies = (df["sales"] < lower_fixed) | (df["sales"] > upper_bound)
    df.loc[anomalies, "sales"] = np.nan

    monthly_means = df[train_mask].groupby("month")["sales"].mean()
    global_monthly_means = df.groupby("month")["sales"].mean().fillna(500)

    def fill_val(row):
        if pd.isna(row["sales"]):
            m = row["month"]
            if m in monthly_means and not pd.isna(monthly_means[m]):
                return monthly_means[m]
            return global_monthly_means.get(m, 500)
        return row["sales"]

    df["sales"] = df.apply(fill_val, axis=1)

    # Force whole integer values across historic performance logs
    df["sales"] = np.round(df["sales"]).astype(int)
    return df.sort_values("date")[["date", "sales"]], int(anomalies.sum()), n_bad_dates


# ---------------------------------------------------------------------------
# Streamlit UI & Clean Corporate Theme Customization
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sales Forecast Hub", layout="wide", initial_sidebar_state="expanded")

BRAND_RED = "#8B0000"
BRAND_RED_HOVER = "#6E0000"

st.html(f"""
    <style>
    .corporate-line {{
        height: 4px;
        background-color: {BRAND_RED};
        border-radius: 2px;
        margin-top: 10px;
        margin-bottom: 2rem;
    }}
    div.stButton > button:first-child {{
        background-color: {BRAND_RED} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }}
    div.stButton > button:first-child:hover {{
        background-color: {BRAND_RED_HOVER} !important;
    }}
    </style>
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### Configuration")

    with st.expander("Location Settings", expanded=True):
        city_choice = st.selectbox("Select Target Location", sorted(list(CITY_PRESETS.keys())) + ["Enter manually"])
        if city_choice == "Enter manually":
            lat = st.number_input("Latitude", value=48.8566, format="%.4f")
            lon = st.number_input("Longitude", value=2.3522, format="%.4f")
        else:
            lat, lon = CITY_PRESETS[city_choice]
            st.caption(f"Coordinates: {lat}, {lon}")

    with st.expander("Model Timelines", expanded=True):
        start_year = st.number_input("Training Start Year", value=2022, step=1)
        split_year = st.number_input("Validation Split Year", value=2025, step=1)
        horizon_years = st.slider("Forecast Horizon (Years)", 1, 15, 10)

    st.markdown("##")
    run_button = st.button("Run Forecast Pipeline", type="primary", use_container_width=True)

# --- MAIN PAGE DASHBOARD HEADER ---
st.markdown(f"<h1 style='color:{BRAND_RED}; font-weight:800; margin-bottom:0;'>Sales Forecast Dashboard</h1>",
            unsafe_allow_html=True)
st.markdown("<div class='corporate-line'></div>", unsafe_allow_html=True)

# --- INITIAL FILE UPLOAD HANDLING & TEMPLATE CONSOLE (MAIN AREA) ---
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

# Analytical SVG Graphic to nicely occupy whitespace
st.markdown(
    "<div style='text-align: center; margin: 20px 0 10px 0;'>"
    "<svg width='80' height='80' viewBox='0 0 24 24' fill='none' stroke='#8B0000' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'>"
    "<line x1='18' y1='20' x2='18' y2='10'></line>"
    "<line x1='12' y1='20' x2='12' y2='4'></line>"
    "<line x1='6' y1='20' x2='6' y2='14'></line>"
    "<path d='M3 20h18'></path>"
    "<path d='M21 4l-7 7-4-4-7 7'></path>"
    "</svg>"
    "</div>",
    unsafe_allow_html=True
)

# --- REQUIRED DATA FORMAT EXPANDER (INTEGER METRICS) ---
with st.expander("📊 Required Excel Data Format & Template", expanded=True):
    st.markdown(
        "To ensure proper pipeline operation, data must be structured **monthly (not daily)**. "
        "Sales metrics must represent absolute unit volumes (integers) without decimal points."
    )

    # Pure discrete unit figures for real air conditioning unit simulations
    mock_data = pd.DataFrame({
        "Year": [2024, 2024, 2024, 2025],
        "Month": [10, 11, 12, 1],
        "Sales": [14200, 15800, 19100, 13400]
    })

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(mock_data, hide_index=True, use_container_width=True)
    with col2:
        st.markdown(
            "- **Year:** Full numeric year framework (e.g., `2024`)\n"
            "- **Month:** Numerical sequence representation from `1` to `12`\n"
            "- **Sales:** Total unit items processed (Whole numbers only)"
        )

        template_buffer = io.BytesIO()
        with pd.ExcelWriter(template_buffer, engine="openpyxl") as template_writer:
            mock_data.to_excel(template_writer, sheet_name="Template_Format", index=False)
        template_buffer.seek(0)

        st.download_button(
            label="📥 Download Reference Template Excel File",
            data=template_buffer,
            file_name="sales_data_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("### Upload Dataset")
uploaded_file = st.file_uploader("Upload historic sales data records (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"],
                                 label_visibility="collapsed")

if uploaded_file is None:
    st.info(
        "💡 **Getting Started:** Please check the required format details in the block above, or download the reference template sheet to structure your metrics.")
    st.stop()

if not run_button:
    st.warning(
        "⚡ **Configuration Ready:** Please adjust your model timelines on the sidebar and click **'Run Forecast Pipeline'** to compute results.")
    st.stop()

# ---------------------------------------------------------------------------
# Core Forecasting Engine Pipeline
# ---------------------------------------------------------------------------
with st.status("Initializing Pipeline Components...", expanded=True) as status:
    st.write("Extracting raw files and preprocessing features...")
    raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df_sales, n_anomalies, n_bad_dates = load_and_preprocess_sales(raw_df, split_year=split_year)

    if n_bad_dates > 0:
        st.sidebar.warning(f"Skipped {n_bad_dates} rows containing unreadable date formats.")
    if n_anomalies > 0:
        st.sidebar.info(f"Imputed {n_anomalies} extreme market anomalies using historic seasonal baselines.")

    if len(df_sales) == 0:
        st.error("Pipeline Aborted: The dataset does not match target features. Verify columns exist.")
        st.stop()

    data_min_year = int(df_sales["date"].dt.year.min())
    data_max_year = int(df_sales["date"].dt.year.max())
    if split_year > data_max_year:
        split_year = data_max_year

    st.write("Connecting to Open-Meteo API for historical climate vectors...")
    df_weather = get_city_temperatures(df_sales["date"], lat, lon)

    df_sales.columns = [c.lower() for c in df_sales.columns]
    df_weather.columns = [c.lower() for c in df_weather.columns]
    df_hist = df_sales.merge(df_weather, on="date", how="left")
    df_hist = df_hist.rename(columns={"sales": "Sales", "date": "Date", "temperature": "Temperature"})
    df_hist["trend"] = (df_hist["Date"].dt.year - 2018) * 12 + (df_hist["Date"].dt.month - 1)
    hist_max = df_hist["Sales"].max()

    split = pd.Timestamp(f"{split_year}-01-01")
    df_train = df_hist[(df_hist["Date"] < split) & (df_hist["Date"].dt.year >= start_year)].copy()
    df_test = df_hist[df_hist["Date"] >= split].copy()

    if len(df_test) == 0 or len(df_train) == 0:
        st.error("Validation Error: Out of bounds temporal range. Change target 'Start' or 'Split' year variables.")
        st.stop()

    models = {"Linear Regression": LinearRegressionWrapper(), "SARIMAX": SARIMAXWrapper(), "MLP": MLPWrapper()}
    if PROPHET_AVAILABLE:
        models["Prophet"] = ProphetWrapper()

    st.write("Evaluating machine learning models on validation splits...")
    preds_df = pd.DataFrame({"Date": df_test["Date"]})
    metrics = {}
    for name, model in models.items():
        model.fit(df_train)
        raw = model.predict(df_test)
        safe = sanity_check_predictions(raw, hist_max, name)
        preds_df[name] = safe
        metrics[name] = calculate_metrics(df_test["Sales"], safe)

    if "MLP" in preds_df.columns and "Linear Regression" in preds_df.columns:
        hybrid_raw = (0.6 * preds_df["MLP"]) + (0.4 * preds_df["Linear Regression"])
        hybrid_safe = sanity_check_predictions(hybrid_raw, hist_max, "Hybrid")
        preds_df["Hybrid"] = hybrid_safe
        metrics["Hybrid"] = calculate_metrics(df_test["Sales"], hybrid_safe)

    leaderboard = pd.DataFrame(metrics).T.sort_values("WAPE (%)")
    leaderboard["Success Rate (%)"] = np.maximum(100 - leaderboard["WAPE (%)"], 0).round(2)
    leaderboard = leaderboard[["Success Rate (%)", "WAPE (%)", "sMAPE (%)", "MAPE (%)", "MAE", "RMSE", "R-squared"]]
    best = leaderboard.index[0]

    st.write(f"Retraining optimal architecture ({best}) across complete time horizons...")
    if_hybrid = (best == "Hybrid")
    if if_hybrid:
        models["MLP"].fit(df_hist)
        models["Linear Regression"].fit(df_hist)
    else:
        best_model = models[best]
        best_model.fit(df_hist)

    future_start = df_hist["Date"].max() + pd.DateOffset(months=1)
    future_end = future_start + pd.DateOffset(years=horizon_years)
    df_future = generate_future_scenarios(future_start, future_end,
                                          df_weather.rename(columns={"temperature": "Temperature", "date": "Date"}))

    forecasts = {}
    for scenario in ["Normal", "Hot", "Cold"]:
        df_in = pd.DataFrame({
            "Date": df_future["Date"],
            "Temperature": df_future[f"Temp_{scenario}"],
            "trend": (df_future["Date"].dt.year - 2018) * 12 + (df_future["Date"].dt.month - 1),
        })
        raw = (0.6 * models["MLP"].predict(df_in)) + (
                    0.4 * models["Linear Regression"].predict(df_in)) if if_hybrid else best_model.predict(df_in)
        safe_preds = sanity_check_predictions(raw, hist_max, best)

        np.random.seed(42 if scenario == "Normal" else (43 if scenario == "Hot" else 44))
        hist_std = df_hist["Sales"].std()
        random_walk = np.cumsum(np.random.normal(0, hist_std * 0.015, size=len(safe_preds)))
        noise, current_noise = np.zeros(len(safe_preds)), 0
        for t in range(len(safe_preds)):
            current_noise = 0.7 * current_noise + np.random.normal(0, hist_std * 0.03)
            noise[t] = current_noise

        # Double clamp ensuring forecasts remain absolute positive physical integer items
        safe_preds = np.round(np.clip(safe_preds + random_walk + noise, 500, hist_max * 3)).astype(int)
        forecasts[scenario] = pd.DataFrame({"Date": df_future["Date"], "Forecast": safe_preds})

    status.update(label="✅ Computation Complete. Dashboard view generated.", state="complete")

# ---------------------------------------------------------------------------
# Executive UI Report Generation
# ---------------------------------------------------------------------------
st.markdown("##")

# --- HIGH-LEVEL EXEC CARDS (KPIs) ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Top Performing Model", value=best)
with kpi2:
    st.metric(label="Pipeline Accuracy", value=f"{leaderboard.loc[best, 'Success Rate (%)']}%")
with kpi3:
    st.metric(label="WAPE Error Rate", value=f"{leaderboard.loc[best, 'WAPE (%)']}%")
with kpi4:
    st.metric(label="Historic Horizon Range", value=f"{data_min_year} - {data_max_year}")

st.markdown("---")

# --- MULTI-TAB DISPLAY OUTLINE ---
tab1, tab2, tab3 = st.tabs(["Performance Leaderboard", "Model Validation Fit", "Scenario Forecasting Horizons"])

with tab1:
    st.markdown("### Model Selection Analytics")
    st.caption("Architectures are evaluated and sorted dynamically using Weighted Absolute Percentage Error (WAPE).")
    st.dataframe(leaderboard.style.highlight_max(axis=0, subset=['Success Rate (%)'], color='#f5e6e6'),
                 use_container_width=True)

# Generate exact whole unit clean integer values for the validation matrix download
preds_df_download = preds_df.copy()
for col in preds_df_download.columns:
    if col != "Date":
        preds_df_download[col] = np.round(preds_df_download[col]).astype(int)

with pd.ExcelWriter(buffer := io.BytesIO(), engine="openpyxl") as writer:
    leaderboard.to_excel(writer, sheet_name="Leaderboard")
    preds_df_download.to_excel(writer, sheet_name="Validation", index=False)
buffer.seek(0)

with tab2:
    st.markdown("### Historical Validation Comparison")
    st.caption(
        f"Cross-examination of actual historical performance trends against validation forecasts generated using model: {best}")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "validation_comparison.png")
        plot_validation_comparison(df_train, df_test, preds_df, best_model_name=best, save_path=path)
        st.image(path, use_container_width=True)

with tab3:
    st.markdown(f"### Climate-Simulated Forward Horizon ({horizon_years} Years)")
    st.caption(
        f"Future predictions computed under Normal, Hot, and Cold climate deviations leveraging {best} pipelines.")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "forecast_scenarios.png")
        plot_forecast_scenarios(df_hist, forecasts, best, save_path=path)
        st.image(path, use_container_width=True)

# --- GLOBAL DATA EXPORT CONSOLE ---
st.markdown("---")
st.markdown("### Export Executive Report Data")
st.download_button(
    label="Download Comprehensive Excel Results (.xlsx)",
    data=buffer,
    file_name=f"sales_forecast_report_{best}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)