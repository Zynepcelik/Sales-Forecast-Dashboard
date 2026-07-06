import os
import numpy as np
import pandas as pd


def generate_synthetic_data(output_dir="data", start_year=2018, end_year=2025):
    """Sentetik satış verisi üretir ve negatif/düşük değerleri engeller."""
    os.makedirs(output_dir, exist_ok=True)

    # Tarih aralığını oluşturma
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq="MS")
    n_periods = len(dates)

    # Trend, mevsimsellik ve gürültü (noise) hesaplama
    time = np.arange(n_periods)
    trend = 200 + 1.5 * time
    seasonal = 50 * np.sin(2 * np.pi * time / 12)
    noise = np.random.normal(0, 80, size=n_periods)

    sales = trend + seasonal + noise

    # --- YENİ EKLENEN ADIM (GÜRÜLTÜDEN KAYNAKLI NEGATİF/DÜŞÜK DEĞERLERİ ENGELLER) ---
    sales = np.clip(sales, a_min=100, a_max=None)  # Minimum 100 birim yapar

    # Veri çerçevesini oluşturup kaydetme
    df = pd.DataFrame({
        "Date": dates,
        "sales": sales.round(2)
    })

    output_path = os.path.join(output_dir, "historical_sales.csv")
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated successfully at: {output_path}")