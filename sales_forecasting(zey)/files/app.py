"""
Satış Tahmin Dashboard'u
=========================
main.py'deki tahminleme mantığını kullanan basit bir Streamlit arayüzü.

Bu dosyayı mevcut proje klasörünüzün İÇİNE koyun (main.py ile aynı seviyeye),
çünkü src/data_generator.py, src/models.py ve src/visualize.py modüllerini
olduğu gibi kullanıyor.

Kurulum:
    pip install -r requirements.txt

Çalıştırma:
    streamlit run app.py
"""

import io
import os
import tempfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.data_generator import generate_synthetic_data  # noqa: F401 (opsiyonel kullanım)
from src.models import (
    PROPHET_AVAILABLE,
    LinearRegressionWrapper,
    MLPWrapper,
    ProphetWrapper,
    SARIMAXWrapper,
)
from src.visualize import plot_forecast_scenarios, plot_validation_comparison

# ---------------------------------------------------------------------------
# Şehir ön ayarları (istediğiniz kadar ekleyebilirsiniz)
# ---------------------------------------------------------------------------
CITY_PRESETS = {
    "Paris, Fransa": (48.8566, 2.3522),
    "Lyon, Fransa": (45.7640, 4.8357),
    "Marsilya, Fransa": (43.2965, 5.3698),
    "İstanbul, Türkiye": (41.0082, 28.9784),
    "Ankara, Türkiye": (39.9334, 32.8597),
    "İzmir, Türkiye": (38.4237, 27.1428),
    "Berlin, Almanya": (52.5200, 13.4050),
    "Madrid, İspanya": (40.4168, -3.7038),
    "Londra, İngiltere": (51.5074, -0.1278),
    "Roma, İtalya": (41.9028, 12.4964),
}


# ---------------------------------------------------------------------------
# main.py'den taşınan / genelleştirilen yardımcı fonksiyonlar
# ---------------------------------------------------------------------------
def sanity_check_predictions(preds, hist_max, model_name):
    preds = np.nan_to_num(preds, nan=0.0, posinf=hist_max * 3, neginf=0.0)
    if preds.max() > hist_max * 5:
        st.warning(f"{model_name} tahmini çok yüksek çıktı, sınırlandırıldı.")
    if preds.max() < hist_max * 0.01:
        st.warning(f"{model_name} tahmini çok düşük çıktı, sınırlandırıldı.")
    return np.clip(preds, 0, hist_max * 3)


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


def get_city_temperatures(dates, lat, lon, warming_rate=0.04):
    """Paris'e kilitli get_paris_temperatures yerine, herhangi bir şehir için
    enlem/boylam parametresiyle geçmiş hava durumu çeker."""
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
        return df.groupby(pd.Grouper(key="Date", freq="MS")).mean().reset_index()
    except Exception as e:
        st.info(f"Hava durumu API'sine ulaşılamadı, genel iklim modeline geçildi: {e}")
        records = []
        base, amp = 13, 8.5
        for dt in dates:
            angle = 2 * np.pi * (dt.month - 1) / 12
            temp = base - amp * np.cos(angle) + warming_rate * (dt.year - 2018)
            records.append({"Date": dt, "Temperature": temp})
        return pd.DataFrame(records)


def generate_future_scenarios(start, end, df_weather_hist, warming_rate=0.04):
    """Sabit Paris sinüs eğrisi yerine, gerçek geçmiş hava verisinden aylık
    iklim ortalamalarını (climatology) çıkarıp geleceğe taşır. Böylece
    hangi şehir seçilirse seçilsin, o şehrin gerçek mevsimsel profiline
    göre senaryo üretilir."""
    dates = pd.date_range(start, end, freq="MS")

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
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    elif "year" in df.columns and "month" in df.columns:
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    else:
        raise ValueError("Veride 'date' veya hem 'year' hem de 'month' kolonları bulunmalıdır!")

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
    return df.sort_values("date")[["date", "sales"]], int(anomalies.sum())


# ---------------------------------------------------------------------------
# Streamlit Arayüzü
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Satış Tahmin Dashboard'u", layout="wide")
st.title("📈 Satış Tahmin Dashboard'u")
st.caption("Şehir/ülke bağımsız çalışır — hava durumu seçtiğiniz konuma göre otomatik çekilir.")

with st.sidebar:
    st.header("1. Veri")
    uploaded_file = st.file_uploader("Satış verisi (CSV veya Excel)", type=["csv", "xlsx", "xls"])

    st.header("2. Konum (Hava Durumu)")
    city_choice = st.selectbox("Şehir seçin", list(CITY_PRESETS.keys()) + ["Manuel gir"])
    if city_choice == "Manuel gir":
        lat = st.number_input("Enlem (latitude)", value=48.8566, format="%.4f")
        lon = st.number_input("Boylam (longitude)", value=2.3522, format="%.4f")
    else:
        lat, lon = CITY_PRESETS[city_choice]
        st.caption(f"Enlem: {lat}, Boylam: {lon}")

    st.header("3. Parametreler")
    split_year = st.number_input("Test/doğrulama başlangıç yılı", value=2025, step=1)
    start_year = st.number_input("Eğitim verisi başlangıç yılı", value=2022, step=1)
    warming_rate = st.slider("Isınma oranı (yıllık °C artışı)", 0.0, 0.2, 0.04, 0.01)
    horizon_years = st.slider("Gelecek tahmin ufku (yıl)", 1, 15, 10)

    run_button = st.button("🚀 Tahmini Çalıştır", type="primary", use_container_width=True)

if uploaded_file is None:
    st.info("Başlamak için soldan bir satış verisi dosyası yükleyin.")
    st.stop()

if not run_button:
    st.info("Parametreleri ayarlayıp **'Tahmini Çalıştır'** butonuna basın.")
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
with st.spinner("Veri işleniyor..."):
    raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df_sales, n_anomalies = load_and_preprocess_sales(raw_df, split_year=split_year)
    if n_anomalies > 0:
        st.warning(f"{n_anomalies} anomali tespit edildi ve mevsimsel ortalamayla dolduruldu.")

    if split_year > df_sales["date"].dt.year.max():
        split_year = df_sales["date"].dt.year.max()

with st.spinner(f"{city_choice if city_choice != 'Manuel gir' else 'seçilen konum'} için hava durumu çekiliyor..."):
    df_weather = get_city_temperatures(df_sales["date"], lat, lon, warming_rate)

df_sales.columns = [c.lower() for c in df_sales.columns]
df_weather.columns = [c.lower() for c in df_weather.columns]
df_hist = df_sales.merge(df_weather, on="date", how="left")
df_hist = df_hist.rename(columns={"sales": "Sales", "date": "Date", "temperature": "Temperature"})
df_hist["trend"] = (df_hist["Date"].dt.year - 2018) * 12 + (df_hist["Date"].dt.month - 1)
hist_max = df_hist["Sales"].max()

split = pd.Timestamp(f"{split_year}-01-01")
df_train = df_hist[(df_hist["Date"] < split) & (df_hist["Date"].dt.year >= start_year)]
df_test = df_hist[df_hist["Date"] >= split]

if len(df_test) == 0:
    st.error("Seçilen doğrulama yılı için veri bulunamadı. Lütfen 'Test başlangıç yılı' değerini kontrol edin.")
    st.stop()

models = {
    "Linear Regression": LinearRegressionWrapper(),
    "SARIMAX": SARIMAXWrapper(),
    "MLP": MLPWrapper(),
}
if PROPHET_AVAILABLE:
    models["Prophet"] = ProphetWrapper()

with st.spinner("Modeller eğitiliyor ve doğrulanıyor..."):
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

with st.spinner(f"En iyi model ({best}) tüm veriyle yeniden eğitiliyor ve {horizon_years} yıllık tahmin üretiliyor..."):
    if best == "Hybrid":
        models["MLP"].fit(df_hist)
        models["Linear Regression"].fit(df_hist)
    else:
        best_model = models[best]
        best_model.fit(df_hist)

    future_start = df_hist["Date"].max() + pd.DateOffset(months=1)
    future_end = future_start + pd.DateOffset(years=horizon_years)
    df_future = generate_future_scenarios(future_start, future_end, df_weather.rename(columns={"temperature": "Temperature", "date": "Date"}), warming_rate)

    forecasts = {}
    for scenario in ["Normal", "Hot", "Cold"]:
        df_in = pd.DataFrame(
            {
                "Date": df_future["Date"],
                "Temperature": df_future[f"Temp_{scenario}"],
                "trend": (df_future["Date"].dt.year - 2018) * 12 + (df_future["Date"].dt.month - 1),
            }
        )
        if best == "Hybrid":
            raw = (0.6 * models["MLP"].predict(df_in)) + (0.4 * models["Linear Regression"].predict(df_in))
        else:
            raw = best_model.predict(df_in)

        safe_preds = sanity_check_predictions(raw, hist_max, best)
        np.random.seed(42 if scenario == "Normal" else (43 if scenario == "Hot" else 44))
        hist_std = df_hist["Sales"].std()
        random_walk = np.cumsum(np.random.normal(0, hist_std * 0.015, size=len(safe_preds)))
        noise, current_noise = np.zeros(len(safe_preds)), 0
        for t in range(len(safe_preds)):
            current_noise = 0.7 * current_noise + np.random.normal(0, hist_std * 0.03)
            noise[t] = current_noise
        safe_preds = np.clip(safe_preds + random_walk + noise, 500, hist_max * 3)
        forecasts[scenario] = pd.DataFrame({"Date": df_future["Date"], "Forecast": safe_preds})

# ---------------------------------------------------------------------------
# Sonuçlar
# ---------------------------------------------------------------------------
st.success(f"Tamamlandı. En iyi model: **{best}** (Başarı Oranı: %{leaderboard.loc[best, 'Success Rate (%)']:.1f})")

tab1, tab2, tab3 = st.tabs(["🏆 Model Karşılaştırma", "✅ Doğrulama Grafiği", "🔮 Gelecek Tahmini"])

with tab1:
    st.dataframe(leaderboard, use_container_width=True)

with tab2:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "validation_comparison.png")
        plot_validation_comparison(df_train, df_test, preds_df, best_model_name=best, save_path=path)
        st.image(path, use_container_width=True)

with tab3:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "forecast_scenarios.png")
        plot_forecast_scenarios(df_hist, forecasts, best, save_path=path)
        st.image(path, use_container_width=True)

# Excel indirme
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    leaderboard.to_excel(writer, sheet_name="Leaderboard")
    preds_df.to_excel(writer, sheet_name="Validation")
buffer.seek(0)

st.download_button(
    "📥 Excel Sonuçlarını İndir",
    data=buffer,
    file_name="forecast_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
