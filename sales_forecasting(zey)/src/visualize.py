import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns

# Görselleri daha profesyonel ve modern hale getirmek için Seaborn temasını uyguluyoruz
sns.set_theme(style="whitegrid", palette="muted")

# Model -> (renk, marker, çizgi stili) eşlemesi
_MODEL_STYLES = {
    "Linear Regression": dict(color="#f59e0b", marker="s", linestyle="--"),
    "SARIMAX": dict(color="#3b82f6", marker="D", linestyle="-"),
    "MLP": dict(color="#10b981", marker="^", linestyle=":"),
    "Prophet": dict(color="#8b5cf6", marker="v", linestyle="-."),
}


def plot_validation_comparison(df_train_recent, df_test_actual, df_preds, best_model_name=None,
                                save_path="validation_comparison_2.jpg"):
    """
    Model Validation Comparison grafiğini çizer.
    """
    plt.figure(figsize=(15, 6))

    # DÜZELTME: Sabit '2023-01-01' tarihi yerine, eğitim verisinin KENDİ
    # aralığına göre dinamik bir kesim noktası kullanıyoruz (son 24 ay).
    # Böylece split_year/start_year hangi yıla ayarlanırsa ayarlansın,
    # ya da farklı bir şirket/şehir verisi yüklensin, filtre her zaman
    # geçerli veri döndürür ve boş kalıp NaN'a düşmez.
    if len(df_train_recent) > 0:
        cutoff_date = df_train_recent['Date'].max() - pd.DateOffset(months=24)
        df_train_filtered = df_train_recent[df_train_recent['Date'] >= cutoff_date].copy()
        if len(df_train_filtered) == 0:
            # Beklenmedik bir durumda yine de boş kalırsa, filtrelemeden
            # tüm eğitim verisine geri dön (grafik daha kalabalık olur
            # ama en azından çöker).
            df_train_filtered = df_train_recent.copy()
    else:
        df_train_filtered = df_train_recent.copy()

    if len(df_test_actual) > 0:
        first_test_row = df_test_actual.iloc[0]
        conn_row = pd.DataFrame({
            'Date': [first_test_row['Date']],
            'Sales': [first_test_row['Sales']]
        })
        df_train_plot = pd.concat([df_train_filtered, conn_row]).sort_values('Date')
    else:
        df_train_plot = df_train_filtered

    # 1. Eğitilen Tarihsel Veri (Düz Slate Gri Çizgi)
    plt.plot(df_train_plot['Date'], df_train_plot['Sales'], color='#94a3b8', linewidth=2,
             label='Historical Sales (Train)')

    # 2. Gerçekleşen Satışlar (Düz Koyu Lacivert Çizgi + Yuvarlak Markerlar)
    plt.plot(df_test_actual['Date'], df_test_actual['Sales'], color='#1e293b', linestyle='-', marker='o',
             markersize=6, markeredgecolor='white', markeredgewidth=1, linewidth=2.5,
             label='Actual Sales (Test)')

    # 3. Model Tahminleri
    for name, style in _MODEL_STYLES.items():
        if name not in df_preds.columns:
            continue

        is_best = (name == best_model_name)
        label = f"{name} (En İyi Model)" if is_best else name

        plt.plot(
            df_preds['Date'], df_preds[name],
            color=style["color"], marker=style["marker"],
            markersize=6 if is_best else 5,
            markeredgecolor='white', markeredgewidth=0.7,
            linestyle=style["linestyle"], linewidth=2.6 if is_best else 1.8,
            zorder=5 if is_best else 3,
            label=label,
        )

    # Grafik Tasarım Ayarları
    plt.title('Model Validation Comparison', fontsize=14, fontweight='bold', color='#1e293b', pad=15)
    plt.xlabel('Date', fontsize=11, color='#475569', labelpad=10)
    plt.ylabel('Sales Quantity (Units)', fontsize=11, color='#475569', labelpad=10)

    ax = plt.gca()
    ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')

    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.9, fontsize=10)

    # Tarih Formatlama
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    # DÜZELTME: min_x/max_x hesaplanırken kullanılan kaynaklar boşsa
    # (df_train_filtered ya da df_preds), NaT/NaN yerine güvenli bir
    # varsayılana düşüyoruz.
    candidate_min_dates = []
    if len(df_train_filtered) > 0:
        candidate_min_dates.append(df_train_filtered['Date'].min())
    if len(df_test_actual) > 0:
        candidate_min_dates.append(df_test_actual['Date'].min())

    candidate_max_dates = []
    if len(df_preds) > 0:
        candidate_max_dates.append(df_preds['Date'].max())
    if len(df_test_actual) > 0:
        candidate_max_dates.append(df_test_actual['Date'].max())

    if candidate_min_dates:
        min_x = min(candidate_min_dates) - pd.DateOffset(months=1)
    else:
        min_x = pd.Timestamp.now() - pd.DateOffset(months=25)

    if candidate_max_dates:
        max_x = max(candidate_max_dates) + pd.DateOffset(months=1)
    else:
        max_x = pd.Timestamp.now() + pd.DateOffset(months=1)

    plt.xlim(min_x, max_x)

    all_sales_vals = list(df_train_filtered['Sales']) + list(df_test_actual['Sales'])
    for col in df_preds.columns:
        if col != 'Date':
            all_sales_vals += list(df_preds[col])
    max_y = max(all_sales_vals) if len(all_sales_vals) > 0 else 65000
    plt.ylim(0, max_y * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_forecast_scenarios(df_historical, df_scenarios, best_model_name, save_path="forecast_scenarios_2.jpg"):
    """
    N-Year Sales Forecast grafiğini çizer.
    """
    plt.figure(figsize=(15, 6))

    # 1. Tarihsel Satışlar
    plt.plot(df_historical['Date'], df_historical['Sales'], color='#334155', linewidth=2,
             label=f'Historical Sales ({df_historical["Date"].dt.year.min()}-{df_historical["Date"].dt.year.max()})')

    df_normal = df_scenarios['Normal'].copy()
    df_hot = df_scenarios['Hot'].copy()
    df_cold = df_scenarios['Cold'].copy()

    last_hist_date = df_historical['Date'].iloc[-1]
    last_hist_sales = df_historical['Sales'].iloc[-1]

    conn_scen = pd.DataFrame({
        'Date': [last_hist_date],
        'Forecast': [last_hist_sales]
    })

    df_normal_plot = pd.concat([conn_scen, df_normal]).sort_values('Date')
    df_hot_plot = pd.concat([conn_scen, df_hot]).sort_values('Date')
    df_cold_plot = pd.concat([conn_scen, df_cold]).sort_values('Date')

    # 2. Senaryo Çizgileri
    plt.plot(df_normal_plot['Date'], df_normal_plot['Forecast'], color='#0d9488', linewidth=2.5, linestyle='-', label='Normal Weather Scenario')
    plt.plot(df_hot_plot['Date'], df_hot_plot['Forecast'], color='#ef4444', linewidth=2, linestyle='--', label='Hot Weather Scenario')
    plt.plot(df_cold_plot['Date'], df_cold_plot['Forecast'], color='#3b82f6', linewidth=2, linestyle='--', label='Cold Weather Scenario')

    # 3. Güven Aralığı Görünümü
    plt.fill_between(df_normal_plot['Date'], df_cold_plot['Forecast'], df_hot_plot['Forecast'], color='#64748b', alpha=0.1)

    # Geçmiş ve Gelecek Ayrım Çizgisi
    plt.axvline(x=last_hist_date, color='#94a3b8', linestyle=':', linewidth=1.5)

    # Grafik Tasarım Ayarları
    plt.title(f'{last_hist_date.year + 1}-{df_normal["Date"].dt.year.max()} Sales Forecast via {best_model_name}',
              fontsize=14, fontweight='bold', color='#1e293b', pad=15)
    plt.xlabel('Date', fontsize=11, color='#475569', labelpad=10)
    plt.ylabel('Forecasted Sales Quantity (Units)', fontsize=11, color='#475569', labelpad=10)

    ax = plt.gca()
    ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')

    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.9, fontsize=10)

    # Eksen Ayarları
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    # DÜZELTME: Sabit (2018-01-01, 2038-06-01) tarihleri yerine, verinin
    # KENDİ aralığına göre dinamik hesaplıyoruz. Böylece hangi yıl
    # aralığında veri yüklenirse yüklensin (veya kaç yıllık tahmin
    # ufku seçilirse seçilsin) grafik doğru pencereyi gösterir.
    all_dates = pd.concat([
        df_historical['Date'],
        df_normal_plot['Date'],
        df_hot_plot['Date'],
        df_cold_plot['Date'],
    ])
    if len(all_dates) > 0:
        min_x = all_dates.min() - pd.DateOffset(months=1)
        max_x = all_dates.max() + pd.DateOffset(months=6)
    else:
        min_x = pd.Timestamp.now() - pd.DateOffset(years=1)
        max_x = pd.Timestamp.now() + pd.DateOffset(years=1)
    plt.xlim(min_x, max_x)

    max_forecast_y = max(
        df_historical['Sales'].max(),
        df_normal_plot['Forecast'].max(),
        df_hot_plot['Forecast'].max(),
        df_cold_plot['Forecast'].max()
    )
    plt.ylim(0, max_forecast_y * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()