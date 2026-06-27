import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Union

def get_descriptive_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Computes detailed descriptive statistics for both numeric and categorical columns.
    """
    stats_dict = {}
    
    # Numeric descriptive stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc_num = df[numeric_cols].describe().T
        # Add skewness and kurtosis
        desc_num['skewness'] = df[numeric_cols].skew()
        desc_num['kurtosis'] = df[numeric_cols].kurtosis()
        stats_dict['numeric'] = desc_num
    else:
        stats_dict['numeric'] = pd.DataFrame()
        
    # Categorical descriptive stats
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        desc_cat_list = []
        for col in cat_cols:
            non_null = df[col].dropna()
            total = len(df[col])
            missing = df[col].isnull().sum()
            missing_pct = (missing / total) * 100 if total > 0 else 0
            
            unique_count = df[col].nunique()
            mode_series = df[col].mode()
            mode_val = mode_series[0] if not mode_series.empty else "N/A"
            
            if mode_val != "N/A":
                mode_freq = (df[col] == mode_val).sum()
                mode_pct = (mode_freq / total) * 100 if total > 0 else 0
            else:
                mode_freq = 0
                mode_pct = 0
                
            desc_cat_list.append({
                "column": col,
                "data_type": str(df[col].dtype),
                "total_rows": total,
                "missing": missing,
                "missing_pct": round(missing_pct, 2),
                "unique_values": unique_count,
                "mode": str(mode_val),
                "mode_frequency": mode_freq,
                "mode_pct": round(mode_pct, 2)
            })
        stats_dict['categorical'] = pd.DataFrame(desc_cat_list).set_index("column")
    else:
        stats_dict['categorical'] = pd.DataFrame()
        
    return stats_dict

def get_correlations(df: pd.DataFrame, method: str = "pearson") -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Computes the correlation matrix for numeric columns and extracts the top 5 strongest relationships.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(), []
        
    # Calculate matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Extract top 5 relationships
    # We want to get the upper triangle of the matrix to avoid duplicate pairs
    pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            val = corr_matrix.loc[col1, col2]
            if not np.isnan(val):
                pairs.append({
                    "var1": col1,
                    "var2": col2,
                    "correlation": float(val),
                    "abs_correlation": float(abs(val))
                })
                
    # Sort by absolute correlation descending
    pairs_sorted = sorted(pairs, key=lambda x: x["abs_correlation"], reverse=True)
    top_5 = pairs_sorted[:5]
    
    return corr_matrix, top_5

def check_normality(series: pd.Series) -> bool:
    """
    Checks if a series is normally distributed using Shapiro-Wilk (N <= 5000) or Kolmogorov-Smirnov (N > 5000).
    Returns True if normally distributed (p-value > 0.05).
    """
    clean_series = series.dropna()
    n = len(clean_series)
    if n < 3:
        return False  # Not enough data
        
    if n <= 5000:
        stat, p = stats.shapiro(clean_series)
    else:
        # Use D'Agostino's K-squared test
        stat, p = stats.normaltest(clean_series)
        
    return p > 0.05

def run_hypothesis_test(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
    """
    Automatically selects and runs the appropriate hypothesis test between two variables.
    """
    if var1 not in df.columns or var2 not in df.columns:
        raise ValueError(f"Variables '{var1}' and/or '{var2}' do not exist in the DataFrame.")
        
    v1_numeric = pd.api.types.is_numeric_dtype(df[var1])
    v2_numeric = pd.api.types.is_numeric_dtype(df[var2])
    
    result = {
        "var1": var1,
        "var2": var2,
        "test_name": "",
        "statistic": 0.0,
        "p_value": 1.0,
        "interpretation": "",
        "null_rejected": False,
        "details": {}
    }
    
    try:
        # Scenario 1: Numeric vs. Numeric (Correlation significance test)
        if v1_numeric and v2_numeric:
            cleaned = df[[var1, var2]].dropna()
            if len(cleaned) < 5:
                result["interpretation"] = "Insufficient data points (less than 5 non-null rows) to run a correlation test."
                return result
                
            n1_normal = check_normality(cleaned[var1])
            n2_normal = check_normality(cleaned[var2])
            
            if n1_normal and n2_normal:
                test_name = "Pearson Correlation Test"
                stat, p = stats.pearsonr(cleaned[var1], cleaned[var2])
                dist_info = "Both variables are normally distributed."
            else:
                test_name = "Spearman Rank Correlation Test"
                stat, p = stats.spearmanr(cleaned[var1], cleaned[var2])
                dist_info = "One or both variables are not normally distributed."
                
            null_rejected = p < 0.05
            interpretation = (
                f"{test_name} indicates a correlation coefficient of {stat:.4f} with a p-value of {p:.4e}. "
                f"({dist_info}) "
            )
            if null_rejected:
                interpretation += f"The correlation is statistically significant (reject Null Hypothesis). There is a significant linear/monotonic relationship between '{var1}' and '{var2}'."
            else:
                interpretation += f"The correlation is not statistically significant (fail to reject Null Hypothesis). There is no significant relationship between '{var1}' and '{var2}'."
                
            result.update({
                "test_name": test_name,
                "statistic": float(stat),
                "p_value": float(p),
                "interpretation": interpretation,
                "null_rejected": bool(null_rejected),
                "details": {
                    "correlation_coefficient": float(stat),
                    "n_samples": len(cleaned)
                }
            })
            
        # Scenario 2: Categorical vs. Categorical (Chi-Square Independence Test)
        elif not v1_numeric and not v2_numeric:
            cleaned = df[[var1, var2]].dropna()
            if len(cleaned) < 10:
                result["interpretation"] = "Insufficient data points (less than 10 non-null rows) to run Chi-Square test."
                return result
                
            contingency_table = pd.crosstab(cleaned[var1], cleaned[var2])
            
            # If table is 1D or empty
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                result["interpretation"] = f"Chi-Square requires both variables to have at least 2 categories. '{var1}' has {contingency_table.shape[0]}, '{var2}' has {contingency_table.shape[1]}."
                return result
                
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            null_rejected = p < 0.05
            
            interpretation = (
                f"Chi-Square Test of Independence yields a statistic of {chi2:.4f} and a p-value of {p:.4e} with {dof} degrees of freedom. "
            )
            if null_rejected:
                interpretation += f"The association is statistically significant (reject Null Hypothesis). The categorical distributions of '{var1}' and '{var2}' are dependent on each other."
            else:
                interpretation += f"The association is not statistically significant (fail to reject Null Hypothesis). The categorical distributions of '{var1}' and '{var2}' are independent."
                
            result.update({
                "test_name": "Chi-Square Test of Independence",
                "statistic": float(chi2),
                "p_value": float(p),
                "interpretation": interpretation,
                "null_rejected": bool(null_rejected),
                "details": {
                    "degrees_of_freedom": int(dof),
                    "contingency_table": contingency_table.to_dict()
                }
            })
            
        # Scenario 3: Categorical vs. Numeric (Group Comparisons: T-test, ANOVA, Kruskal-Wallis, MWU)
        else:
            # Let var_cat be the categorical one and var_num be the numeric one
            if not v1_numeric:
                var_cat = var1
                var_num = var2
            else:
                var_cat = var2
                var_num = var1
                
            # Filter non-null and get groups
            cleaned = df[[var_cat, var_num]].dropna()
            categories = cleaned[var_cat].unique()
            
            if len(categories) < 2:
                result["interpretation"] = f"Categorical column '{var_cat}' has only {len(categories)} unique category. Group comparison requires at least 2 groups."
                return result
                
            groups = [cleaned[cleaned[var_cat] == cat][var_num].values for cat in categories]
            
            # Check normality of each group
            all_normal = True
            group_means = {}
            for cat, gp in zip(categories, groups):
                group_means[str(cat)] = float(np.mean(gp)) if len(gp) > 0 else 0.0
                if len(gp) >= 3:
                    if not check_normality(pd.Series(gp)):
                        all_normal = False
                else:
                    all_normal = False
                    
            # 2 Groups
            if len(categories) == 2:
                if all_normal:
                    # Run Welch's T-test (does not assume equal variance)
                    test_name = "Welch's Two-Sample T-Test"
                    stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    dist_info = "Both groups are normally distributed."
                else:
                    # Run Mann-Whitney U test
                    test_name = "Mann-Whitney U Test"
                    stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    dist_info = "One or both groups are not normally distributed."
                    
                null_rejected = p < 0.05
                interpretation = (
                    f"{test_name} yields a statistic of {stat:.4f} and a p-value of {p:.4e}. "
                    f"({dist_info}) "
                )
                if null_rejected:
                    interpretation += f"The difference between the two groups is statistically significant (reject Null Hypothesis). The mean/median of '{var_num}' differs across the levels of '{var_cat}'."
                else:
                    interpretation += f"The difference between the two groups is not statistically significant (fail to reject Null Hypothesis). There is no significant difference in '{var_num}' between the levels of '{var_cat}'."
                    
                result.update({
                    "test_name": test_name,
                    "statistic": float(stat),
                    "p_value": float(p),
                    "interpretation": interpretation,
                    "null_rejected": bool(null_rejected),
                    "details": {
                        "group_means": group_means,
                        "group_counts": {str(cat): len(gp) for cat, gp in zip(categories, groups)}
                    }
                })
                
            # > 2 Groups
            else:
                if all_normal:
                    test_name = "One-Way ANOVA Test"
                    stat, p = stats.f_oneway(*groups)
                    dist_info = "All groups are normally distributed."
                else:
                    test_name = "Kruskal-Wallis H Test (Non-parametric ANOVA)"
                    stat, p = stats.kruskal(*groups)
                    dist_info = "One or more groups are not normally distributed."
                    
                null_rejected = p < 0.05
                interpretation = (
                    f"{test_name} yields a statistic of {stat:.4f} and a p-value of {p:.4e}. "
                    f"({dist_info}) "
                )
                if null_rejected:
                    interpretation += f"The difference across multiple groups is statistically significant (reject Null Hypothesis). At least one category level in '{var_cat}' has a significantly different distribution of '{var_num}'."
                else:
                    interpretation += f"The difference across multiple groups is not statistically significant (fail to reject Null Hypothesis). There is no significant variance in '{var_num}' across categories of '{var_cat}'."
                    
                result.update({
                    "test_name": test_name,
                    "statistic": float(stat),
                    "p_value": float(p),
                    "interpretation": interpretation,
                    "null_rejected": bool(null_rejected),
                    "details": {
                        "group_means": group_means,
                        "group_counts": {str(cat): len(gp) for cat, gp in zip(categories, groups)}
                    }
                })
                
    except Exception as e:
        result["interpretation"] = f"An error occurred while executing the statistical test: {str(e)}"
        
    return result
