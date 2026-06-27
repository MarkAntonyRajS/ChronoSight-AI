import os
import io
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from sklearn.ensemble import IsolationForest

def optimize_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Optimizes memory usage of a pandas DataFrame by downcasting numeric types
    and converting low-cardinality object columns to categories.
    """
    logs = []
    start_mem = df.memory_usage().sum() / (1024 ** 2)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # If object, check cardinality
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_total > 0 and (num_unique / num_total) < 0.5 and num_unique < 1000:
                df[col] = df[col].astype('category')
                logs.append(f"Converted column '{col}' to category (unique values: {num_unique})")

    end_mem = df.memory_usage().sum() / (1024 ** 2)
    saved = ((start_mem - end_mem) / start_mem) * 100 if start_mem > 0 else 0
    logs.append(f"Memory usage optimized from {start_mem:.2f}MB to {end_mem:.2f}MB (Reduced by {saved:.1f}%)")
    
    return df, logs

def load_data(file_source: Union[str, io.BytesIO, Any], file_name: str = "") -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads data from CSV, Excel, or Parquet. Applies memory optimization.
    """
    logs = []
    df = pd.DataFrame()
    
    # Extract extension
    ext = ""
    if isinstance(file_source, str):
        ext = os.path.splitext(file_source)[1].lower()
        file_size = os.path.getsize(file_source) / (1024 * 1024)
        logs.append(f"Loading file from path: {file_source} ({file_size:.2f} MB)")
    else:
        ext = os.path.splitext(file_name)[1].lower() if file_name else ".csv"
        # BytesIO support
        if hasattr(file_source, "size"):
            file_size = file_source.size / (1024 * 1024)
            logs.append(f"Loading uploaded file: {file_name} ({file_size:.2f} MB)")
        else:
            logs.append(f"Loading uploaded file buffer: {file_name}")

    try:
        if ext == '.csv':
            # Check if large, use chunksize if file is massive (mock chunk logic or read direct if memory allows)
            df = pd.read_csv(file_source)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_source)
        elif ext in ['.parquet', '.pq']:
            df = pd.read_parquet(file_source)
        else:
            # Fallback to csv if unknown
            df = pd.read_csv(file_source)
            
        logs.append(f"Successfully loaded dataset with shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Optimize memory
        df, opt_logs = optimize_memory(df)
        logs.extend(opt_logs)
        
    except Exception as e:
        logs.append(f"Error during file ingestion: {str(e)}")
        raise e
        
    return df, logs

def audit_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Scans the DataFrame for missing values, structural duplicates, and data type anomalies.
    """
    audit = {}
    
    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100 if len(df) > 0 else 0
    missing_info = {}
    for col in df.columns:
        if missing_counts[col] > 0:
            missing_info[col] = {
                "count": int(missing_counts[col]),
                "percentage": float(round(missing_pct[col], 2))
            }
    
    audit["missing_values"] = {
        "total_cells": int(df.isnull().sum().sum()),
        "total_percentage": float(round((df.isnull().sum().sum() / df.size) * 100, 2)) if df.size > 0 else 0.0,
        "by_column": missing_info
    }
    
    # Structural duplicates
    duplicate_count = int(df.duplicated().sum())
    audit["duplicates"] = {
        "count": duplicate_count,
        "percentage": float(round((duplicate_count / len(df)) * 100, 2)) if len(df) > 0 else 0.0
    }
    
    # Data type anomalies (e.g. object columns where many strings can be converted to numbers)
    anomalous_cols = {}
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            # Sample non-null values
            sample = df[col].dropna().head(100).astype(str)
            if len(sample) > 0:
                # Remove common currency symbols, commas, percent signs, and check if it's numeric
                cleaned_sample = sample.str.replace(r'[$,%\s]', '', regex=True).str.replace(r',', '', regex=True)
                is_num = pd.to_numeric(cleaned_sample, errors='coerce')
                num_valid = is_num.notnull().sum()
                if num_valid / len(sample) > 0.8: # More than 80% can be converted to numbers
                    anomalous_cols[col] = {
                        "inferred_type": "numeric",
                        "reason": "Contains numeric strings with formatting (symbols, spaces, or commas)"
                    }
                    
    audit["type_anomalies"] = anomalous_cols
    
    # Column types summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    audit["columns_summary"] = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "total_numeric": len(numeric_cols),
        "total_categorical": len(categorical_cols)
    }
    
    return audit

def clean_data(
    df: pd.DataFrame, 
    impute_num: str = "skew_based", 
    impute_cat: str = "mode", 
    convert_anomalies: bool = True,
    remove_duplicates: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cleans the DataFrame: converts type anomalies, removes duplicates, and imputes missing values.
    """
    logs = []
    cleaned_df = df.copy()
    
    # 1. Deduplication
    if remove_duplicates:
        dup_count = cleaned_df.duplicated().sum()
        if dup_count > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
            logs.append(f"Removed {dup_count} duplicate rows.")
            
    # 2. Type Anomalies Correction
    audit = audit_data(cleaned_df)
    anomalies = audit["type_anomalies"]
    
    if convert_anomalies and anomalies:
        for col, info in anomalies.items():
            if info["inferred_type"] == "numeric":
                try:
                    # Clean currency symbols, commas, spaces, percentages
                    series = cleaned_df[col].astype(str)
                    cleaned_series = series.str.replace(r'[$,%\s]', '', regex=True).str.replace(r',', '', regex=True)
                    # Convert to numeric
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    cleaned_df[col] = numeric_series
                    logs.append(f"Corrected anomaly: Converted column '{col}' from text to numeric.")
                except Exception as e:
                    logs.append(f"Failed to convert column '{col}' to numeric: {str(e)}")

    # Re-evaluate types after conversion
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 3. Missing Value Imputation
    # Numeric Imputation
    for col in numeric_cols:
        null_count = cleaned_df[col].isnull().sum()
        if null_count > 0:
            skew = cleaned_df[col].skew()
            # Decide strategy
            strategy = "mean"
            if impute_num == "skew_based":
                # If skewed, use median. Otherwise use mean.
                strategy = "median" if abs(skew) > 1.0 else "mean"
            elif impute_num in ["mean", "median"]:
                strategy = impute_num
                
            if strategy == "median":
                val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(val)
                logs.append(f"Imputed {null_count} missing values in numeric '{col}' using Median ({val:.4f}) due to skewness ({skew:.2f})")
            else:
                val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(val)
                logs.append(f"Imputed {null_count} missing values in numeric '{col}' using Mean ({val:.4f}) due to low skewness ({skew:.2f})")

    # Categorical Imputation
    for col in categorical_cols:
        null_count = cleaned_df[col].isnull().sum()
        if null_count > 0:
            if impute_cat == "mode":
                mode_series = cleaned_df[col].mode()
                if not mode_series.empty:
                    val = mode_series[0]
                    # If categorical column is category dtype, add category if not present
                    if isinstance(cleaned_df[col].dtype, pd.CategoricalDtype) and val not in cleaned_df[col].cat.categories:
                        cleaned_df[col] = cleaned_df[col].cat.add_categories([val])
                    cleaned_df[col] = cleaned_df[col].fillna(val)
                    logs.append(f"Imputed {null_count} missing values in categorical '{col}' using Mode ('{val}')")
                else:
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                    logs.append(f"Imputed {null_count} missing values in categorical '{col}' with 'Unknown' (No mode available)")
            else:
                # Custom value or simple fill
                cleaned_df[col] = cleaned_df[col].fillna("Missing")
                logs.append(f"Imputed {null_count} missing values in categorical '{col}' with 'Missing'")

    return cleaned_df, logs

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Detects outliers in a series using the Interquartile Range (IQR) method.
    Returns a boolean mask where True indicates an outlier.
    """
    if pd.api.types.is_numeric_dtype(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (series < lower_bound) | (series > upper_bound)
    return pd.Series(False, index=series.index)

def treat_outliers(
    df: pd.DataFrame, 
    columns: List[str], 
    method: str = "iqr", 
    action: str = "cap", 
    contamination: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Detects and treats outliers in specified numeric columns.
    Actions: 'cap' (winsorize / clamp to bounds), 'remove' (drop rows with outliers).
    """
    logs = []
    treated_df = df.copy()
    outlier_info = {}
    
    if not columns:
        return treated_df, {}, ["No columns selected for outlier treatment."]

    if method == "iqr":
        rows_to_drop = set()
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            mask = detect_outliers_iqr(df[col])
            outlier_count = mask.sum()
            outlier_info[col] = int(outlier_count)
            
            if outlier_count > 0:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                if action == "cap":
                    treated_df[col] = treated_df[col].clip(lower=lower, upper=upper)
                    logs.append(f"Capped {outlier_count} outliers in '{col}' to range [{lower:.2f}, {upper:.2f}].")
                elif action == "remove":
                    outlier_indices = df.index[mask].tolist()
                    rows_to_drop.update(outlier_indices)
                    
        if action == "remove" and rows_to_drop:
            treated_df = treated_df.drop(index=list(rows_to_drop)).reset_index(drop=True)
            logs.append(f"Removed {len(rows_to_drop)} rows containing outliers across columns: {columns}.")
            
    elif method == "isolation_forest":
        # Fit on numerical subset of selected columns
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            # Run Isolation Forest
            # Impute temporarily if there are NaNs (though cleaning should be done first)
            temp_df = df[numeric_cols].fillna(df[numeric_cols].median())
            
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(temp_df)
            mask = preds == -1
            outlier_count = mask.sum()
            
            outlier_info["multivariate_isolation_forest"] = int(outlier_count)
            
            if outlier_count > 0:
                if action == "remove":
                    treated_df = treated_df.loc[~mask].reset_index(drop=True)
                    logs.append(f"Removed {outlier_count} multivariate outliers using Isolation Forest (contamination={contamination}).")
                elif action == "cap":
                    # For isolation forest, capping is not directly applicable in a simple bound way, 
                    # so we fall back to capping columns individually using IQR or log warning
                    logs.append("Capping is not directly supported for Isolation Forest. Capping individual columns via IQR boundaries instead.")
                    for col in numeric_cols:
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        treated_df[col] = treated_df[col].clip(lower=lower, upper=upper)
                    logs.append(f"Capped individual columns {numeric_cols} to their IQR bounds.")
                    
    return treated_df, outlier_info, logs
