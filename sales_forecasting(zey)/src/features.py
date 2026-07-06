import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_cyclical_features(df, date_col="Date"):
    """
    Adds sine and cosine features based on the month of the date.
    This creates smooth periodic features representing seasonality.
    """
    df_out = df.copy()
    months = df_out[date_col].dt.month
    
    df_out["sin_month"] = np.sin(2 * np.pi * months / 12.0)
    df_out["cos_month"] = np.cos(2 * np.pi * months / 12.0)
    
    return df_out

def add_trend_feature(df, start_date, date_col="Date"):
    """
    Adds a linear trend variable representing the number of months since start_date.
    """
    df_out = df.copy()
    # Calculate months since start date
    df_out["Trend"] = ((df_out[date_col].dt.year - start_date.year) * 12 + 
                       (df_out[date_col].dt.month - start_date.month))
    return df_out

def prepare_feature_matrix(
    df,
    start_date=None,
    temp_col="Temperature",
    date_col="Date",
    target_col="Sales",
    fit_scaler=False,
    scaler=None
):
    """
    Prepares the full feature matrix for modeling.
    Features:
    - Trend (Months since start)
    - Temperature (Linear)
    - Temperature_Sq (Quadratic)
    - sin_month (Cyclical Month)
    - cos_month (Cyclical Month)
    
    Returns:
    - X: pandas DataFrame of features
    - y: pandas Series of target (if target_col is in df, else None)
    - scaler: fitted StandardScaler (if fit_scaler=True or scaler is provided, else None)
    - start_date: the Timestamp used as the trend start date
    """
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    
    # 1. Establish start_date for trend calculation
    if start_date is None:
        start_date = df_temp[date_col].min()
        
    # 2. Add features
    df_temp = add_cyclical_features(df_temp, date_col=date_col)
    df_temp = add_trend_feature(df_temp, start_date=start_date, date_col=date_col)
    # 3. Separate features and target
    feature_cols = ["Trend", "sin_month", "cos_month"]
    if temp_col in df_temp.columns:
        feature_cols.extend([temp_col])
        
    X = df_temp[feature_cols].copy()
    
    # 4. Extract target if exists
    y = None
    if target_col in df_temp.columns:
        y = df_temp[target_col].copy()
        
    # 5. Scaling logic (useful for Neural Networks / MLP)
    scaled_X = X.copy()
    if fit_scaler:
        scaler = StandardScaler()
        scaled_X[X.columns] = scaler.fit_transform(X)
    elif scaler is not None:
        scaled_X[X.columns] = scaler.transform(X)
        
    return X, y, scaled_X, scaler, start_date
