import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def _add_seasonal_features(df):
    """Ham 'trend' sayacı yerine sınırlı (bounded) mevsimsel özellikler üretir.

    Ham trend (yıl*12+ay gibi) eğitim aralığının dışına çıkıldığında modelleri
    (özellikle MLP ve doğrusal modelleri) kontrolsüz ekstrapolasyona zorluyordu.
    sin/cos dönüşümü her zaman [-1, 1] aralığında kaldığı için, model gelecekte
    kaç yıl uzakta olursa olsun girdi aralığı sabit kalır; sadece mevsimsellik
    ve sıcaklık ilişkisinden öğrenir.
    """
    month = df["Date"].dt.month
    out = pd.DataFrame({
        "Temperature": df["Temperature"].values,
        "month_sin": np.sin(2 * np.pi * month / 12).values,
        "month_cos": np.cos(2 * np.pi * month / 12).values,
    })
    return out


def _raw_trend(df):
    return ((df["Date"].dt.year - 2018) * 12 + (df["Date"].dt.month - 1)).values.astype(float)


def _bounded_trend_ratio(raw_trend, trend_scale):
    """DÜZELTME (KRİTİK BUG FIX): Önceki versiyon trend'i trend_scale'e bölüp
    normalize ediyordu ama bu oranı SINIRLAMIYORDU. Eğitim penceresi kısaysa
    (ör. varsayılan --start-year 2022 -> sadece 3 yıl/36 ay), tahmin edilen
    tarih eğitim aralığının dışına (ör. 2025) çıktığında oran 1.0'ı aşıyor ve
    -eğer trend katsayısı negatifse (bu veri setinde 2022-2024 arası düşüş
    eğiliminde)- tahmin negatife düşüp np.maximum(...,0) ile SIFIRA çakılıyordu.
    Gerçek veriyle test edildiğinde MLP modeli tam olarak bu yüzden Ağustos
    2025'ten itibaren 0 tahmin ediyordu.

    Çözüm: oranı [0, 1] aralığında SIKI şekilde sınırlıyoruz (clip). Yani
    model, eğitim penceresinin en son ayındaki trend seviyesinin ötesine asla
    ekstrapolasyon yapmıyor; oradan sonrası tamamen mevsimsel (month_sin/cos)
    ve sıcaklık bileşenlerinden geliyor. Bu, kısa eğitim pencerelerinde bile
    tahminlerin sıfıra çökmesini engelliyor.
    """
    return np.clip(raw_trend / trend_scale, 0.0, 1.0)


class LinearRegressionWrapper:
    """'trend' artık ham (sınırsız büyüyen) bir sayaç olarak değil, eğitim
    penceresine göre [0, 1] aralığında SIKI şekilde sınırlanmış bir oran
    olarak veriliyor. Bkz. _bounded_trend_ratio docstring'i.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.trend_scale = 1.0

    def fit(self, df):
        X = _add_seasonal_features(df)
        raw_trend = _raw_trend(df)
        self.trend_scale = max(float(raw_trend.max()), 1.0)
        X["trend"] = _bounded_trend_ratio(raw_trend, self.trend_scale)

        y = df["Sales"]
        self.model.fit(X, y)

    def predict(self, df):
        X = _add_seasonal_features(df)
        raw_trend = _raw_trend(df)
        X["trend"] = _bounded_trend_ratio(raw_trend, self.trend_scale)

        preds = self.model.predict(X)
        return np.maximum(preds, 0)


class SARIMAXWrapper:
    """SARIMAX: trendi exog yerine modelin kendi d/D fark alma mekanizmasına
    bırakıyoruz. Sadece sıcaklık ve mevsimsel sin/cos exog olarak veriliyor.
    """

    def __init__(self):
        # DÜZELTME: En iyi tahmin başarısını veren SARIMAX order (1, 1, 1) x (1, 0, 0, 12) olarak güncellendi.
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 0, 0, 12)

        # Ölçekleyici ve model nesnelerini tanımlıyoruz
        self.scaler = StandardScaler()
        self.model_res = None
        self.hist_max = None

    def fit(self, df):
        df_reset = df.reset_index(drop=True)
        exog = self.scaler.fit_transform(_add_seasonal_features(df_reset))
        self.hist_max = df_reset["Sales"].max()

        model = SARIMAX(
            df_reset["Sales"].values,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_res = model.fit(disp=False, maxiter=300, method="lbfgs")
        return self

    def predict(self, df, return_conf_int=False, alpha=0.20):
        if self.model_res is None:
            raise ValueError(
                "Model henüz eğitilmedi. Önce .fit() metodunu çağırın."
            )

        df_reset = df.reset_index(drop=True)
        exog = self.scaler.transform(_add_seasonal_features(df_reset))
        result = self.model_res.get_forecast(steps=len(df_reset), exog=exog)
        preds = np.asarray(result.predicted_mean)
        preds = np.clip(preds, 0, self.hist_max * 3)

        if return_conf_int:
            # DÜZELTME: main.py'deki yapay random-walk + AR(1) gürültüsü yerine
            # SARIMAX'ın istatistiksel olarak anlamlı güven aralığını kullanmak
            # için eklendi. alpha=0.20 -> %80 güven aralığı.
            conf = result.conf_int(alpha=alpha)
            lower = np.clip(np.asarray(conf[:, 0]), 0, self.hist_max * 3)
            upper = np.clip(np.asarray(conf[:, 1]), 0, self.hist_max * 3)
            return preds, lower, upper

        return preds


class MLPWrapper:
    def __init__(self):
        self.trend_model = LinearRegression()
        self.model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            max_iter=3000,
            early_stopping=False,
            learning_rate="adaptive",
            learning_rate_init=0.01,
            alpha=0.01,
            random_state=42,
        )
        self.scaler_X = StandardScaler()
        self.trend_scale = 1.0

    def fit(self, df):
        # 1. Detrending
        # DÜZELTME (KRİTİK BUG FIX): trend burada da artik [0,1] araliginda
        # sinirlaniyor (LinearRegressionWrapper'daki ayni mantik). Eskiden
        # ham trend degeri kullaniliyordu; kisa egitim penceresinde (ör. 3
        # yil) ve dusus egiliminde bir trend katsayisinda, gelecege dogru
        # ekstrapolasyon trend_pred'i çok negatif yapip toplam tahmini
        # sifira cakiyordu (gercek veriyle test edildi: Agustos 2025'ten
        # itibaren tam olarak 0 tahmin ediyordu).
        raw_trend = _raw_trend(df)
        self.trend_scale = max(float(raw_trend.max()), 1.0)
        trend = _bounded_trend_ratio(raw_trend, self.trend_scale).reshape(-1, 1)
        self.trend_model.fit(trend, df["Sales"].values)
        trend_pred = self.trend_model.predict(trend)

        y_detrended = df["Sales"].values - trend_pred

        # 2. Features
        X = self.scaler_X.fit_transform(_add_seasonal_features(df))

        # 3. Fit MLP
        self.model.fit(X, y_detrended)

        print(
            f" [MLP Debug] n_iter: {self.model.n_iter_}, final_loss: {self.model.loss_:.4f}"
        )
        return self

    def predict(self, df):
        # 1. Trend Prediction (sinirlanmis trend orani ile - bkz. fit())
        raw_trend = _raw_trend(df)
        trend = _bounded_trend_ratio(raw_trend, self.trend_scale).reshape(-1, 1)
        trend_pred = self.trend_model.predict(trend)

        # 2. MLP Residual Prediction
        X = self.scaler_X.transform(_add_seasonal_features(df))
        residual_pred = self.model.predict(X)

        # 3. Combine
        preds = trend_pred + residual_pred
        return np.maximum(preds, 0)


class ProphetWrapper:
    """Prophet, kendi içinde otomatik changepoint (trend kırılma noktası)
    tespiti yapan bir model olduğu için LinearRegression/MLP'deki gibi elle
    bounded-trend hack'ine ihtiyaç duymaz; 2018-2022 yükseliş, 2022-2025
    düşüş gibi TEK bir doğrusal eğimle açıklanamayan (non-monotonic) trendi
    doğal olarak yakalayabilir. Bu yüzden Prophet, bu projedeki en gerçekçi
    trend modelleme seçeneğidir.
    """

    def __init__(self):
        self.model = None

    def fit(self, df):
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet kütüphanesi yüklü değil. Kurmak için: pip install prophet"
            )

        ph_df = pd.DataFrame({
            "ds": df["Date"],
            "y": df["Sales"],
            "Temperature": df["Temperature"]
        })

        # DÜZELTME: Aylık veride varsayılan 25 changepoint, kısa eğitim
        # pencerelerinde (ör. 36 aylık) gürültüye aşırı uyum (overfitting)
        # riski taşıyor. Veri uzunluğuna göre makul bir üst sınır konuldu
        # (yaklaşık her 4 ayda bir changepoint, en fazla 25).
        n_changepoints = max(3, min(25, len(ph_df) // 4))

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # DÜZELTME: 0.008'den 0.05'e çıkararak trende daha duyarlı yaptık
            seasonality_prior_scale=10.0,  # DÜZELTME: Mevsimsel dalgalanmaları daha cesur tahmin etmesini sağlar
            n_changepoints=n_changepoints,
            changepoint_range=0.9,
        )
        self.model.add_regressor("Temperature", prior_scale=0.5)
        self.model.fit(ph_df)

    def predict(self, df):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet kütüphanesi yüklü değil.")

        ph_df = pd.DataFrame({
            "ds": df["Date"],
            "Temperature": df["Temperature"]
        })

        forecast = self.model.predict(ph_df)
        preds = forecast["yhat"].values
        return np.maximum(preds, 0)