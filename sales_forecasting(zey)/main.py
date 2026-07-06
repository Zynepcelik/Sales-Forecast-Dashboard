from sklearn.linear_model import Ridge

import argparse

import os

import numpy as np

import pandas as pd

import requests

import urllib3



# SSL sertifika doğrulama uyarılarını terminalde kalabalık yapmasın diye gizliyoruz

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



from src.data_generator import generate_synthetic_data

from src.models import (

    PROPHET_AVAILABLE,

    LinearRegressionWrapper,

    MLPWrapper,

    ProphetWrapper,

    SARIMAXWrapper,

)

from src.visualize import plot_forecast_scenarios, plot_validation_comparison





def is_stable_model(preds, hist_max):

    return not (

            preds.max() > hist_max * 5 or preds.max() < hist_max * 0.01

    )





def sanity_check_predictions(preds, hist_max, model_name):

    preds = np.nan_to_num(preds, nan=0.0, posinf=hist_max * 3, neginf=0.0)



    if preds.max() > hist_max * 5:

        print(f"\n[UYARI] {model_name} tahmini cok yuksek, patlamis olabilir!")



    if preds.max() < hist_max * 0.01:

        print(f"\n[UYARI] {model_name} tahmini cok dusuk, cokmus olabilir!")



    return np.clip(preds, 0, hist_max * 3)





def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sales-data", type=str, default="data/historical_sales.csv")

    parser.add_argument("--generate-synthetic", action="store_true")

    parser.add_argument("--output-excel", type=str, default="data/forecast_results.xlsx")

    parser.add_argument("--start-year", type=int, default=2022, help="Egitim verisi baslangic yili (en iyi sonuc icin son yillari baz alir)")

    parser.add_argument("--split-year", type=int, default=2025)

    parser.add_argument("--warming-rate", type=float, default=0.04)

    return parser.parse_args()







def calculate_metrics(y_true, y_pred):

    """DÜZELTME: Klasik MAPE, gerçek değer sıfıra yakın olduğunda (ör. 2025

    Ağustos-Eylül aylarındaki neredeyse-sıfır satışlar) tek bir noktadan

    binlerce puanlık hata üretip tüm ortalamayı bozabiliyordu. Bunun yerine

    toplam-bazlı WAPE (Weighted Absolute Percentage Error) ve simetrik

    sMAPE eklendi; ikisi de tekil sıfıra-yakın noktalara karşı çok daha

    dayanıklı ve "Success Rate" hesaplaması için WAPE kullanılacak.

    Eski MAPE değeri de referans olması açısından hâlâ hesaplanıp tabloya

    ekleniyor, sadece leaderboard sıralamasında/kazanan seçiminde artık

    kullanılmıyor.

    """

    y_true = np.asarray(y_true, dtype=float)

    y_pred = np.asarray(y_pred, dtype=float)



    mae = np.mean(np.abs(y_true - y_pred))

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))



    eps = 1e-8



    # Eski MAPE - artik sadece bilgi amacli, siralamada kullanilmiyor

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100



    # WAPE: toplam mutlak hata / toplam gercek deger. Tekil sifira-yakin

    # noktalardan (payda kucukken) etkilenmez, cunku hem pay hem payda

    # butun seri uzerinden toplanir.

    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps) * 100



    # sMAPE: hem gercek hem tahmin paydada oldugu icin asiri buyumeyi

    # engeller, 0-200 araliginda kalir.

    smape = np.mean(

        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)

    ) * 100



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





def get_paris_temperatures(dates, warming_rate=0.04):

    try:

        min_date = dates.min().strftime("%Y-%m-%d")

        max_date = dates.max().strftime("%Y-%m-%d")



        url = (

            "https://archive-api.open-meteo.com/v1/archive"

            f"?latitude=48.8566&longitude=2.3522"

            f"&start_date={min_date}&end_date={max_date}"

            f"&daily=temperature_2m_mean&timezone=UTC"

        )



        response = requests.get(url, verify=False)

        data = response.json()



        df = pd.DataFrame({

            "Date": pd.to_datetime(data["daily"]["time"]),

            "Temperature": data["daily"]["temperature_2m_mean"],

        })



        df = df.groupby(pd.Grouper(key="Date", freq="MS")).mean().reset_index()

        return df



    except Exception as e:

        print("Fallback climate model:", e)

        records = []

        base = 13

        amp = 8.5



        for dt in dates:

            angle = 2 * np.pi * (dt.month - 1) / 12

            temp = base - amp * np.cos(angle) + warming_rate * (dt.year - 2018)

            records.append({"Date": dt, "Temperature": temp})



        return pd.DataFrame(records)





def generate_future_scenarios(start, end, warming_rate=0.04):

    dates = pd.date_range(start, end, freq="MS")

    records = []

    base, amp = 13, 8.5



    for dt in dates:

        angle = 2 * np.pi * (dt.month - 1) / 12

        year_diff = dt.year - 2018



        records.append({

            "Date": dt,

            "Temp_Normal": base - amp * np.cos(angle) + warming_rate * year_diff,

            "Temp_Hot": base - amp * np.cos(angle) + 2 + (warming_rate + 0.02) * year_diff,

            "Temp_Cold": base - 1.2 - amp * np.cos(angle),

        })



    return pd.DataFrame(records)





def load_and_preprocess_sales(file_path, split_year=2025):

    if not os.path.exists(file_path):

        raise FileNotFoundError(file_path)



    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    df.columns = df.columns.str.lower()



    # Tarih kolonlarını otomatik algılama ve oluşturma

    if "date" in df.columns:

        df["date"] = pd.to_datetime(df["date"])

        df["year"] = df["date"].dt.year

        df["month"] = df["date"].dt.month

    elif "year" in df.columns and "month" in df.columns:

        df["date"] = pd.to_datetime(

            df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"

        )

    else:

        raise ValueError("Veride 'date' veya hem 'year' hem de 'month' kolonları bulunmalıdır!")



    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)



    # Eğitim verisi (split_year öncesi) üzerinden alt VE üst sınır anomali tespiti.

    # DÜZELTME: Önceden sadece "sales < 500" kontrol ediliyordu; tek seferlik

    # aşırı yüksek satış sıçramaları (ör. 2020/2022'deki zirveler) hiç

    # filtrelenmiyordu. Bu tür tek seferlik olaylar aylık mevsimsel

    # ortalamaları çarpıtıp modelin gelecekte de benzer patlamalar

    # bekleyeceği (ya da tam tersi, gerçek mevsimselliği öğrenemeyeceği)

    # bir duruma yol açabiliyordu. IQR tabanlı bir üst sınır eklendi.

    train_mask = df["year"] < split_year



    lower_fixed = 500.0

    train_sales = df.loc[train_mask, "sales"]

    if train_sales.notna().sum() >= 4:

        q1 = train_sales.quantile(0.25)

        q3 = train_sales.quantile(0.75)

        iqr = q3 - q1

        upper_bound = q3 + 3 * iqr  # agresif olmayan, sadece asiri uc degerleri yakalayan bir esik

    else:

        upper_bound = np.inf



    anomalies = (df["sales"] < lower_fixed) | (df["sales"] > upper_bound)

    n_anomalies = int(anomalies.sum())

    if n_anomalies > 0:

        print(f"[BILGI] {n_anomalies} anomali tespit edildi (alt sinir: {lower_fixed}, ust sinir: {upper_bound:.1f}) ve mevsimsel ortalamayla dolduruldu.")



    df.loc[anomalies, "sales"] = np.nan



    # Eğitim verileri üzerinden (split_year öncesi tüm geçmiş) her ayın ortalamasını hesaplıyoruz

    monthly_means = df[train_mask].groupby("month")["sales"].mean()



    # Eğitim verisinde o aya ait veri yoksa genel ortalamayı (veya 500) kullanıyoruz

    global_monthly_means = df.groupby("month")["sales"].mean().fillna(500)



    # Eksik değerleri mevsimsel ortalamayla dolduruyoruz

    def fill_val(row):

        if pd.isna(row["sales"]):

            m = row["month"]

            if m in monthly_means and not pd.isna(monthly_means[m]):

                return monthly_means[m]

            return global_monthly_means.get(m, 500)

        return row["sales"]



    df["sales"] = df.apply(fill_val, axis=1)



    return df.sort_values("date")[["date", "sales"]]


def main():
    args = parse_args()

    if args.generate_synthetic or not os.path.exists(args.sales_data):
        generate_synthetic_data("data", 2018, args.split_year)
        args.sales_data = "data/historical_sales.csv"

    df_sales = load_and_preprocess_sales(args.sales_data, split_year=args.split_year)

    if args.split_year > df_sales["date"].dt.year.max():
        args.split_year = df_sales["date"].dt.year.max()

    df_weather = get_paris_temperatures(df_sales["date"], args.warming_rate)

    df_sales.columns = [c.lower() for c in df_sales.columns]
    df_weather.columns = [c.lower() for c in df_weather.columns]

    df_hist = df_sales.merge(df_weather, on="date", how="left")
    df_hist = df_hist.rename(columns={"sales": "Sales", "date": "Date", "temperature": "Temperature"})
    df_hist["trend"] = (df_hist["Date"].dt.year - 2018) * 12 + (df_hist["Date"].dt.month - 1)

    hist_max = df_hist["Sales"].max()

    split = pd.Timestamp(f"{args.split_year}-01-01")
    df_train = df_hist[(df_hist["Date"] < split) & (df_hist["Date"].dt.year >= args.start_year)]
    df_test = df_hist[df_hist["Date"] >= split]

    models = {
        "Linear Regression": LinearRegressionWrapper(),
        "SARIMAX": SARIMAXWrapper(),
        "MLP": MLPWrapper(),
    }

    if PROPHET_AVAILABLE:
        models["Prophet"] = ProphetWrapper()

    preds_df = pd.DataFrame({"Date": df_test["Date"]})
    metrics = {}

    # Mevcut modellerin eğitilmesi ve tahmin üretmesi
    for name, model in models.items():
        model.fit(df_train)
        raw = model.predict(df_test)
        safe = sanity_check_predictions(raw, hist_max, name)

        preds_df[name] = safe
        metrics[name] = calculate_metrics(df_test["Sales"], safe)

    # ==================================================================
    # YENİ EKLENEN KISIM: HİBRİT MODEL (MLP + LINEAR REGRESSION ENSEMBLE)
    # ==================================================================
    if "MLP" in preds_df.columns and "Linear Regression" in preds_df.columns:
        print("\n[BİLGİ] MLP ve Linear Regression birleştirilerek Hibrit Model oluşturuluyor...")
        # İki modelin tahminlerinin ağırlıklı ortalamasını alıyoruz (%60 MLP, %40 Lineer Regresyon)
        hybrid_raw = (0.6 * preds_df["MLP"]) + (0.4 * preds_df["Linear Regression"])
        hybrid_safe = sanity_check_predictions(hybrid_raw, hist_max, "Hybrid")

        preds_df["Hybrid"] = hybrid_safe
        metrics["Hybrid"] = calculate_metrics(df_test["Sales"], hybrid_safe)
    # ==================================================================

    leaderboard = pd.DataFrame(metrics).T
    leaderboard = leaderboard.sort_values("WAPE (%)")
    leaderboard["Success Rate (%)"] = np.maximum(100 - leaderboard["WAPE (%)"], 0).round(2)

    cols = ["Success Rate (%)", "WAPE (%)", "sMAPE (%)", "MAPE (%)", "MAE", "RMSE", "R-squared"]
    leaderboard = leaderboard[cols]

    print("\n" + "=" * 80)
    print("                     MODEL PERFORMANS TABLOSU (LEADERBOARD)                    ")
    print("=" * 80)
    print(leaderboard.to_string())
    print("=" * 80 + "\n")

    best = leaderboard.index[0]

    print(f"[EN IYI MODEL] En iyi model secildi: {best}")

    # ==================================================================
    # YENİ EKLENEN KISIM: GELECEK TAHMİNİ İÇİN HİBRİT MODEL KONTROLÜ
    # ==================================================================
    if best == "Hybrid":
        print("Gelecek tahmini icin hem MLP hem Linear Regression modelleri tum veriyle yeniden egitiliyor...")
        models["MLP"].fit(df_hist)
        models["Linear Regression"].fit(df_hist)
    else:
        print(f"Gelecek tahmini icin {best} modeli tum veriyle yeniden egitiliyor...")
        best_model = models[best]
        best_model.fit(df_hist)
    # ==================================================================

    future_start = df_hist["Date"].max() + pd.DateOffset(months=1)
    future_end = future_start + pd.DateOffset(years=10)

    df_future = generate_future_scenarios(future_start, future_end)

    forecasts = {}
    for scenario in ["Normal", "Hot", "Cold"]:
        df_in = pd.DataFrame({
            "Date": df_future["Date"],
            "Temperature": df_future[f"Temp_{scenario}"],
            "trend": (df_future["Date"].dt.year - 2018) * 12 + (df_future["Date"].dt.month - 1),
        })

        # Eğer en iyi modelimiz Hibrit ise, iki modeli de çalıştırıp sonuçları birleştiriyoruz
        if best == "Hybrid":
            raw_mlp = models["MLP"].predict(df_in)
            raw_lr = models["Linear Regression"].predict(df_in)
            raw = (0.6 * raw_mlp) + (0.4 * raw_lr)
        else:
            raw = best_model.predict(df_in)

        safe_preds = sanity_check_predictions(raw, hist_max, best)

        np.random.seed(42 if scenario == "Normal" else (43 if scenario == "Hot" else 44))
        hist_std = df_hist["Sales"].std()

        random_walk = np.cumsum(np.random.normal(0, hist_std * 0.015, size=len(safe_preds)))
        noise = np.zeros(len(safe_preds))
        current_noise = 0
        for t in range(len(safe_preds)):
            current_noise = 0.7 * current_noise + np.random.normal(0, hist_std * 0.03)
            noise[t] = current_noise

        safe_preds = np.clip(safe_preds + random_walk + noise, 500, hist_max * 3)

        forecasts[scenario] = pd.DataFrame({
            "Date": df_future["Date"],
            "Forecast": safe_preds
        })

    os.makedirs("data", exist_ok=True)

    print("Grafikler cizdiriliyor ve Excel kaydediliyor...")
    plot_validation_comparison(df_train, df_test, preds_df, best_model_name=best,
                               save_path="data/validation_comparison.png")
    plot_forecast_scenarios(df_hist, forecasts, best, save_path="data/forecast_scenarios.png")

    with pd.ExcelWriter(args.output_excel) as writer:
        leaderboard.to_excel(writer, sheet_name="Leaderboard")
        preds_df.to_excel(writer, sheet_name="Validation")

    if os.name == "nt":
        print("\nSonuclar data/ klasorune kaydedildi. Klasor aciliyor...")
        os.system(f'explorer "{os.path.abspath("data")}"')





if __name__ == "__main__":

    main()