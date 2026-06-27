import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Scans a DataFrame to detect columns containing dates or timestamps.
    """
    date_cols = []
    n_rows = len(df)
    if n_rows == 0:
        return []

    # Check already inferred datetimes
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue
            
        # For object or category columns, check if a high proportion can be parsed as dates
        if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            # Sample non-null values
            sample = df[col].dropna().head(100).astype(str)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    valid_pct = parsed.notnull().sum() / len(sample)
                    if valid_pct > 0.8: # More than 80% parse successfully
                        date_cols.append(col)
                except Exception:
                    pass
                    
    return date_cols

def resample_time_series(
    df: pd.DataFrame, 
    date_col: str, 
    target_col: str, 
    freq: str = "D", 
    agg_func: str = "mean"
) -> Tuple[pd.Series, List[str]]:
    """
    Resamples a time series to a regular frequency (Daily, Weekly, Monthly)
    and aggregates target values, filling missing gaps using linear interpolation.
    """
    logs = []
    
    # 1. Prepare datetime index
    temp_df = df[[date_col, target_col]].copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
    temp_df = temp_df.dropna(subset=[date_col])
    
    # Ensure target is numeric
    temp_df[target_col] = pd.to_numeric(temp_df[target_col], errors='coerce')
    temp_df = temp_df.dropna(subset=[target_col])
    
    if len(temp_df) < 5:
        raise ValueError("Insufficient chronological data (less than 5 valid numeric points).")
        
    temp_df = temp_df.sort_values(by=date_col)
    temp_df = temp_df.set_index(date_col)
    
    # 2. Resample
    series = temp_df[target_col]
    if agg_func == "sum":
        resampled = series.resample(freq).sum()
    else:
        resampled = series.resample(freq).mean()
        
    logs.append(f"Resampled series to frequency '{freq}' using {agg_func}. Rows count: {len(resampled)}")
    
    # 3. Fill missing gaps
    null_count = resampled.isnull().sum()
    if null_count > 0:
        resampled = resampled.interpolate(method='linear')
        # If there are still NaNs at edges, forward/backward fill
        resampled = resampled.ffill().bfill()
        logs.append(f"Imputed {null_count} missing time steps in resampled index using linear interpolation.")
        
    return resampled, logs

def forecast_future(
    series: pd.Series, 
    horizon: int = 12, 
    model_type: str = "rf"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Fits an autoregressive model (with 3 lags) on the resampled series,
    and runs a recursive multi-step forecasting loop.
    Returns:
    - History DataFrame (actuals, fitted)
    - Forecast DataFrame (forecast, lower_bound, upper_bound)
    - Metrics Dictionary
    - Processing logs
    """
    logs = []
    n_samples = len(series)
    
    if n_samples < 6:
        raise ValueError("Forecasting requires at least 6 resampled data points to construct lag features.")
        
    # 1. Feature Engineering (Lags: t-1, t-2, t-3)
    df_lags = pd.DataFrame(index=series.index)
    df_lags["y"] = series
    df_lags["lag_1"] = series.shift(1)
    df_lags["lag_2"] = series.shift(2)
    df_lags["lag_3"] = series.shift(3)
    
    # Rolling mean of t-1, t-2, t-3
    df_lags["rolling_mean_3"] = df_lags[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
    
    # Drop rows with NaNs (first 3 rows)
    train_df = df_lags.dropna()
    
    X = train_df[["lag_1", "lag_2", "lag_3", "rolling_mean_3"]].values
    y = train_df["y"].values
    
    # 2. Fit Model
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_name = "Autoregressive Random Forest"
    else:
        model = LinearRegression()
        model_name = "Autoregressive Linear Regression"
        
    model.fit(X, y)
    fitted_values = model.predict(X)
    
    # Evaluate model on training set
    r2 = r2_score(y, fitted_values)
    mae = mean_absolute_error(y, fitted_values)
    rmse = np.sqrt(mean_squared_error(y, fitted_values))
    
    logs.append(f"Fitted {model_name} model. R2: {r2:.4f}, MAE: {mae:.4f}")
    
    # Create History DF
    history_df = pd.DataFrame(index=series.index)
    history_df["actual"] = series
    # Pad fitted values for the first 3 dropped rows with NaNs
    padded_fitted = [np.nan, np.nan, np.nan] + list(fitted_values)
    history_df["fitted"] = padded_fitted
    
    # Compute residuals standard deviation
    residuals = y - fitted_values
    residual_std = np.std(residuals)
    
    # 3. Recursive Forecasting Loop
    # Initialize lag buffer with last 3 actual values
    buffer = list(series.values[-3:]) # [y_n-2, y_n-1, y_n]
    
    forecast_values = []
    lower_bounds = []
    upper_bounds = []
    
    for h in range(1, horizon + 1):
        # buffer structure: [t-2, t-1, t]
        lag_1 = buffer[-1]
        lag_2 = buffer[-2]
        lag_3 = buffer[-3]
        roll_mean = np.mean([lag_1, lag_2, lag_3])
        
        feat = np.array([[lag_1, lag_2, lag_3, roll_mean]])
        pred = float(model.predict(feat)[0])
        forecast_values.append(pred)
        
        # Growing uncertainty interval: error propagates at sqrt(h)
        margin = 1.96 * residual_std * np.sqrt(h)
        lower_bounds.append(pred - margin)
        upper_bounds.append(pred + margin)
        
        # Update lag buffer: drop first element, append predicted value
        buffer.pop(0)
        buffer.append(pred)
        
    # Generate future timestamps
    freq = series.index.freq
    if freq is None:
        # Inferred freq
        freq = pd.infer_freq(series.index)
        if freq is None:
            # Fallback to date diff
            freq = series.index[1] - series.index[0]
            
    future_dates = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=freq)[1:]
    
    # Create Forecast DF
    forecast_df = pd.DataFrame(index=future_dates)
    forecast_df["forecast"] = forecast_values
    forecast_df["lower_bound"] = lower_bounds
    forecast_df["upper_bound"] = upper_bounds
    
    # 4. Calculate trend summary stats
    first_half_avg = series.iloc[:n_samples//2].mean()
    second_half_avg = series.iloc[n_samples//2:].mean()
    growth_rate = ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg != 0 else 0
    
    # Forecast direction
    forecast_avg = np.mean(forecast_values)
    history_last_val = series.iloc[-1]
    forecast_pct_change = ((forecast_avg - history_last_val) / history_last_val) * 100 if history_last_val != 0 else 0
    
    if forecast_pct_change > 2.0:
        direction = "Upward / Growth Trend"
    elif forecast_pct_change < -2.0:
        direction = "Downward / Decline Trend"
    else:
        direction = "Stable / Sideways Trend"
        
    metrics = {
        "model_r2": float(r2),
        "model_mae": float(mae),
        "model_rmse": float(rmse),
        "historical_growth_pct": float(growth_rate),
        "forecast_average": float(forecast_avg),
        "forecast_direction": direction,
        "forecast_pct_change": float(forecast_pct_change)
    }
    
    return history_df, forecast_df, metrics, logs
