import os
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Component & Util Imports
from components.cleaning import load_data, audit_data, clean_data, treat_outliers
from components.analytics import get_descriptive_stats, get_correlations, run_hypothesis_test
from components.models import run_automl, AutoMLPipeline
from components.forecasting import detect_date_columns, resample_time_series, forecast_future
from utils.llm_helper import generate_ai_report
from utils.report_generator import generate_markdown_report, compile_pdf_report

# Page Setup
st.set_page_config(
    page_title="ChronoSight AI - Smart Analyzer & Report Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style Application
st.markdown("""
<style>
    .main-header {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(135deg, #1E293B, #0D9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 18px;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0D9488 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate test dataset
def generate_sample_data() -> pd.DataFrame:
    """Generates a synthetic dataset containing type anomalies, missing values, duplicates, and outliers."""
    np.random.seed(42)
    n = 250
    
    # Customer Churn Synthetic Dataset
    # Base Registration Date timeline spanning 3 years
    base_date = pd.to_datetime("2023-01-01")
    reg_dates = [base_date + pd.DateOffset(days=int(d)) for d in np.random.randint(0, 1000, n)]
    
    data = {
        "Customer_ID": [f"C-{1000+i}" for i in range(n)],
        "Registration_Date": reg_dates,
        "Age": np.random.randint(18, 70, n),
        "Monthly_Charges": np.round(np.random.uniform(20.0, 150.0, n), 2),
        "Total_Spent": np.zeros(n),
        "Tenure_Months": np.random.randint(1, 72, n),
        "Contract_Type": np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.5, 0.3, 0.2]),
        "Support_Tickets": np.random.poisson(lam=1.5, size=n),
        "Churn_Flag": np.random.choice(["Yes", "No"], n, p=[0.25, 0.75])
    }
    
    # Add relationship: Total_Spent = Tenure * Monthly_Charges + noise
    data["Total_Spent"] = np.round(data["Tenure_Months"] * data["Monthly_Charges"] + np.random.normal(0, 15, n), 2)
    
    df = pd.DataFrame(data)
    
    # Inject missing values
    df.loc[df.sample(frac=0.04, random_state=1).index, 'Age'] = np.nan
    df.loc[df.sample(frac=0.06, random_state=2).index, 'Monthly_Charges'] = np.nan
    df.loc[df.sample(frac=0.05, random_state=3).index, 'Contract_Type'] = np.nan
    
    # Inject outlier: some customers having monthly charge of $1200
    outliers_idx = df.sample(frac=0.02, random_state=4).index
    df.loc[outliers_idx, 'Monthly_Charges'] = 1200.0
    
    # Inject duplicates
    duplicates = df.sample(n=5, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # format one column with currency signs to test type anomaly converter
    df["Monthly_Charges"] = df["Monthly_Charges"].apply(lambda x: f"${x:,.2f}" if not np.isnan(x) else np.nan)
    
    return df

# Initialize Session States
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
if "active_df" not in st.session_state:
    st.session_state["active_df"] = None
if "cleaning_logs" not in st.session_state:
    st.session_state["cleaning_logs"] = []
if "audit_results" not in st.session_state:
    st.session_state["audit_results"] = {}
if "ml_pipeline" not in st.session_state:
    st.session_state["ml_pipeline"] = None
if "ml_logs" not in st.session_state:
    st.session_state["ml_logs"] = []
if "ai_report" not in st.session_state:
    st.session_state["ai_report"] = ""
if "forecast_history" not in st.session_state:
    st.session_state["forecast_history"] = None
if "forecast_future" not in st.session_state:
    st.session_state["forecast_future"] = None
if "forecast_metrics" not in st.session_state:
    st.session_state["forecast_metrics"] = {}
if "forecast_logs" not in st.session_state:
    st.session_state["forecast_logs"] = []

# Sidebar Ingest & Config
st.sidebar.markdown("### 📊 Ingestion & Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Tabular File (CSV, Excel, Parquet)",
    type=["csv", "xlsx", "xls", "parquet", "pq"]
)

# Button to load sample data if user wants to play instantly
if uploaded_file is None:
    if st.sidebar.button("💡 Load Sample Churn Dataset"):
        df_sample = generate_sample_data()
        st.session_state["raw_df"] = df_sample
        # Save initial file loaded log
        st.session_state["cleaning_logs"] = ["Loaded synthetic churn dataset containing 255 rows."]
        st.toast("Loaded sample dataset!", icon="💡")
else:
    # Trigger load from uploaded file
    try:
        df_loaded, load_logs = load_data(uploaded_file, uploaded_file.name)
        st.session_state["raw_df"] = df_loaded
        st.session_state["cleaning_logs"] = load_logs
        st.toast("Loaded uploaded dataset successfully!", icon="✅")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

# Ensure subsequent configs only display if data is loaded
if st.session_state["raw_df"] is not None:
    raw_df = st.session_state["raw_df"]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧹 Clean & Impute Options")
    
    dedup = st.sidebar.checkbox("Remove Duplicate Rows", value=True)
    conv_anom = st.sidebar.checkbox("Convert Numeric-Stored Strings", value=True)
    
    impute_num_strat = st.sidebar.selectbox(
        "Numerical Imputation",
        options=["skew_based", "mean", "median"],
        index=0,
        format_func=lambda x: "Skew-Based (Mean/Median)" if x == "skew_based" else x.capitalize()
    )
    
    impute_cat_strat = st.sidebar.selectbox(
        "Categorical Imputation",
        options=["mode", "fill_missing"],
        index=0,
        format_func=lambda x: "Mode (Most Frequent)" if x == "mode" else "Fill 'Missing'"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ Outlier Mitigation")
    
    outlier_action = st.sidebar.selectbox(
        "Treatment Strategy",
        options=["cap", "remove", "none"],
        index=0,
        format_func=lambda x: "Cap to IQR Bounds" if x == "cap" else "Remove Row Outliers" if x == "remove" else "Do Not Treat"
    )
    
    # Process cleaning reactively on configuration changes
    cleaned_df, cleaning_logs = clean_data(
        raw_df,
        impute_num=impute_num_strat,
        impute_cat=impute_cat_strat,
        convert_anomalies=conv_anom,
        remove_duplicates=dedup
    )
    
    audit_res = audit_data(cleaned_df)
    
    # Handle outliers if checked
    numeric_cols = audit_res["columns_summary"]["numeric"]
    selected_outlier_cols = []
    
    if outlier_action != "none" and len(numeric_cols) > 0:
        selected_outlier_cols = st.sidebar.multiselect(
            "Columns to Treat Outliers",
            options=numeric_cols,
            default=numeric_cols
        )
        
        active_df, outlier_info, outlier_logs = treat_outliers(
            cleaned_df,
            columns=selected_outlier_cols,
            method="iqr",
            action=outlier_action
        )
        # Combine cleaning and outlier logs
        all_logs = st.session_state["cleaning_logs"] + cleaning_logs + outlier_logs
        # Deduplicate logs while preserving order
        seen = set()
        st.session_state["cleaning_logs"] = [x for x in all_logs if not (x in seen or seen.add(x))]
        st.session_state["active_df"] = active_df
    else:
        st.session_state["active_df"] = cleaned_df
        st.session_state["cleaning_logs"] = list(dict.fromkeys(st.session_state["cleaning_logs"] + cleaning_logs))
        
    st.session_state["cleaned_df"] = cleaned_df
    st.session_state["audit_results"] = audit_res
    
    # Target selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Machine Learning Target")
    all_cols = st.session_state["active_df"].columns.tolist()
    target_col = st.sidebar.selectbox(
        "Select Target Variable",
        options=["None (Clustering Mode)"] + all_cols,
        index=0
    )
    selected_target = None if target_col == "None (Clustering Mode)" else target_col
    
    # LLM Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Executive Report Config")
    llm_provider = st.sidebar.selectbox("LLM Provider", options=["Gemini", "OpenAI", "Ollama"], index=0)
    
    if llm_provider == "Gemini":
        llm_key = st.sidebar.text_input("Gemini API Key", type="password", help="Input your Google Gemini API Key")
        llm_model = st.sidebar.selectbox("Model Name", options=["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    elif llm_provider == "OpenAI":
        llm_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Input your OpenAI API Key")
        llm_model = st.sidebar.selectbox("Model Name", options=["gpt-4o-mini", "gpt-4o"], index=0)
    else: # Ollama
        llm_key = st.sidebar.text_input("Ollama Host", value="http://localhost:11434", help="Default local address of Ollama")
        llm_model = st.sidebar.text_input("Model Name", value="llama3", help="Specify local model name (e.g. llama3, mistral)")
        
    business_context = st.sidebar.text_area(
        "Commercial Focus/Context",
        value="",
        placeholder="e.g. Focus on predicting churn risk to reduce recurring revenue loss. Keep report summary brief."
    )

# --- MAIN APP VIEW ---
st.markdown("<div class='main-header'>📊 ChronoSight AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Autonomous Data Audit, AutoML Engine & Time-Series Forecasting.</div>", unsafe_allow_html=True)

if st.session_state["raw_df"] is None:
    # Welcome Layout
    st.info("👋 Welcome! Please upload a data file (CSV, Excel, Parquet) in the sidebar or click **Load Sample Churn Dataset** to test-drive the analyzer pipeline.")
    
    # Feature showcase grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🧹 1. Ingest & Auto-Clean
        - File chunking & downcasting
        - Imputes nulls based on skewness
        - Fixes formatting type anomalies
        - Capping/removing IQR outliers
        """)
    with col2:
        st.markdown("""
        ### 📐 2. Statistical Engine
        - Continuous & discrete descriptives
        - Pearson/Spearman matrix checks
        - Dynamic hypothesis testing
        - Interactive distributions
        """)
    with col3:
        st.markdown("""
        ### 🔮 3. AutoML & AI Reports
        - Autodetect target task logic
        - Preprocess & fit baseline pipelines
        - Tuning & feature importances
        - LLM business report compiler
        """)
else:
    # We have data! Let's display the tabs
    tab_overview, tab_stats, tab_trends, tab_ml, tab_ai = st.tabs([
        "📁 Ingestion & Cleaning",
        "📐 Statistical Analysis",
        "📈 Trends & Forecasting",
        "🔮 AutoML Playground",
        "🤖 AI Executive Report"
    ])
    
    active_df = st.session_state["active_df"]
    raw_df = st.session_state["raw_df"]
    audit_results = st.session_state["audit_results"]
    
    # ----------------------------------------------------
    # TAB 1: INGESTION & DATA CLEANING LOGS
    # ----------------------------------------------------
    with tab_overview:
        st.markdown("## 📁 Ingestion & Cleaning Audit")
        
        # High level metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Rows", active_df.shape[0], delta=active_df.shape[0] - raw_df.shape[0])
        with m2:
            st.metric("Total Columns", active_df.shape[1])
        with m3:
            st.metric("Duplicates Removed", raw_df.duplicated().sum() if dedup else 0)
        with m4:
            st.metric("Total Missing Cells", audit_results["missing_values"]["total_cells"])
            
        col_tables1, col_tables2 = st.columns(2)
        with col_tables1:
            st.subheader("Raw Uploaded Dataset (Sample)")
            st.dataframe(raw_df.head(20), use_container_width=True)
            
        with col_tables2:
            st.subheader("Processed & Cleaned Dataset (Sample)")
            st.dataframe(active_df.head(20), use_container_width=True)
            # Add download for clean dataset
            csv = active_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Cleaned CSV Dataset",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.markdown("---")
        # Logs and audits section
        with st.expander("🧹 Technical Processing & Cleaning Logs", expanded=True):
            if st.session_state["cleaning_logs"]:
                for log in st.session_state["cleaning_logs"]:
                    st.write(f"- {log}")
            else:
                st.write("No modifications were triggered on the raw data.")
                
        with st.expander("🔍 Column Audit Report Details", expanded=False):
            st.json(st.session_state["audit_results"])
            
    # ----------------------------------------------------
    # TAB 2: STATISTICAL ANALYSIS & HYPOTHESIS TESTING
    # ----------------------------------------------------
    with tab_stats:
        st.markdown("## 📐 Statistical Engine")
        
        # Get descriptive statistics
        desc_stats = get_descriptive_stats(active_df)
        
        # Display numeric descriptives
        if not desc_stats["numeric"].empty:
            st.subheader("Descriptive Statistics: Numerical Variables")
            st.dataframe(desc_stats["numeric"], use_container_width=True)
        
        # Display categorical descriptives
        if not desc_stats["categorical"].empty:
            st.subheader("Descriptive Statistics: Categorical Variables")
            st.dataframe(desc_stats["categorical"], use_container_width=True)
            
        st.markdown("---")
        
        # Distributions Plotting
        st.subheader("📊 Interactive Distribution Inspector")
        dist_col1, dist_col2 = st.columns([1, 3])
        with dist_col1:
            inspect_col = st.selectbox("Select Column to Analyze", options=active_df.columns.tolist())
            
        with dist_col2:
            if inspect_col:
                if pd.api.types.is_numeric_dtype(active_df[inspect_col]):
                    fig = px.histogram(
                        active_df, x=inspect_col, marginal="box", 
                        title=f"Distribution & Spread of {inspect_col}",
                        color_discrete_sequence=["#0D9488"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    counts = active_df[inspect_col].value_counts().reset_index()
                    counts.columns = [inspect_col, "count"]
                    fig = px.bar(
                        counts, x=inspect_col, y="count", 
                        title=f"Category Distribution of {inspect_col}",
                        color_discrete_sequence=["#0D9488"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
        st.markdown("---")
        
        # Correlation Matrices
        st.subheader("🔗 Linear & Non-Linear Variable Correlations")
        corr_method = st.selectbox("Correlation Method", options=["pearson", "spearman"], format_func=lambda x: "Pearson (Linear)" if x == "pearson" else "Spearman (Rank / Non-linear)")
        
        corr_matrix, top_5_corrs = get_correlations(active_df, method=corr_method)
        
        if not corr_matrix.empty:
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = px.imshow(
                    corr_matrix, 
                    color_continuous_scale="RdBu", 
                    zmin=-1.0, zmax=1.0,
                    title=f"{corr_method.capitalize()} Correlation Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("#### Strongest Relationships")
                if top_5_corrs:
                    for idx, item in enumerate(top_5_corrs):
                        st.markdown(
                            f"**{idx+1}. {item['var1']}** ↔ **{item['var2']}**  \n"
                            f"Coefficient: `{item['correlation']:.4f}` (Abs: `{item['abs_correlation']:.4f}`)"
                        )
                else:
                    st.write("No numeric column pairs found to compute correlation.")
        else:
            st.info("Not enough numeric columns to generate a correlation heatmap.")
            
        st.markdown("---")
        
        # Hypothesis Testing Playground
        st.subheader("🔬 Dynamic Hypothesis Testing Playground")
        st.write("Select two variables. The engine will inspect distributions and run the correct statistical test (T-Test, ANOVA, Chi-Square, MWU, Pearson correlation).")
        
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            hyp_var1 = st.selectbox("Select Variable 1", options=all_cols, key="hyp_v1")
        with h_col2:
            # Filter out variable 1
            other_cols = [c for c in all_cols if c != hyp_var1]
            hyp_var2 = st.selectbox("Select Variable 2 (Comparison)", options=other_cols, key="hyp_v2")
            
        with h_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_test_btn = st.button("🧪 Execute Statistical Test", use_container_width=True)
            
        if run_test_btn:
            try:
                test_output = run_hypothesis_test(active_df, hyp_var1, hyp_var2)
                st.success(f"Hypothesis Test Executed: **{test_output['test_name']}**")
                
                # Metrics
                st.metric("Test Statistic", f"{test_output['statistic']:.4f}")
                st.metric("p-value", f"{test_output['p_value']:.4e}")
                
                # Highlight outcome
                if test_output["null_rejected"]:
                    st.warning("⚠️ **Null Hypothesis Rejected (p < 0.05)**. There is a statistically significant relationship or difference.")
                else:
                    st.info("ℹ️ **Fail to Reject Null Hypothesis (p >= 0.05)**. No statistically significant relationship or difference detected.")
                    
                st.markdown("##### Narrative Interpretation")
                st.info(test_output["interpretation"])
                
                # Save test to session state to pass to LLM report later
                st.session_state["last_hypothesis_test"] = test_output
            except Exception as e:
                st.error(f"Could not execute hypothesis test: {str(e)}")

    # ----------------------------------------------------
    # TAB 3: TRENDS & FORECASTING
    # ----------------------------------------------------
    with tab_trends:
        st.markdown("## 📈 Historical Trends & Time-Series Forecasting")
        st.write("Scan date dimensions, aggregate target metrics, and run recursive machine learning models to project future values.")
        
        # Detect candidate date columns
        candidate_date_cols = detect_date_columns(active_df)
        
        if not candidate_date_cols:
            st.info("ℹ️ **No Date/Time variables detected in this dataset.** To unlock historical trends and machine learning forecasting, upload a dataset containing timestamps or date values (e.g. YYYY-MM-DD).")
        else:
            col_tf1, col_tf2 = st.columns([1, 3])
            
            with col_tf1:
                st.markdown("#### ⚙️ Forecasting Config")
                selected_date_col = st.selectbox("Date / Timestamp Field", options=candidate_date_cols, key="ts_date_col")
                
                # Numeric variables for forecasting target
                numeric_targets = active_df.select_dtypes(include=[np.number]).columns.tolist()
                selected_target_metric = st.selectbox("Target Metric to Forecast", options=numeric_targets, key="ts_target_col")
                
                ts_freq = st.selectbox(
                    "Resampling Frequency", 
                    options=["D", "W", "M"], 
                    format_func=lambda x: "Daily" if x=="D" else "Weekly" if x=="W" else "Monthly",
                    key="ts_frequency"
                )
                
                ts_agg = st.selectbox(
                    "Aggregation Method", 
                    options=["mean", "sum"], 
                    format_func=lambda x: "Average / Mean" if x=="mean" else "Cumulative Sum",
                    key="ts_aggregation"
                )
                
                ts_horizon = st.slider(
                    "Forecast Horizon (Future Periods)", 
                    min_value=5, 
                    max_value=36, 
                    value=12,
                    key="ts_horizon_slider"
                )
                
                ts_model = st.selectbox(
                    "Forecasting Model Base", 
                    options=["rf", "lr"], 
                    format_func=lambda x: "Random Forest Regressor" if x=="rf" else "Linear Regression",
                    key="ts_model_base"
                )
                
                run_forecast_btn = st.button("📈 Compute Future Forecasts", use_container_width=True)
                
            with col_tf2:
                if run_forecast_btn:
                    with st.spinner("Aggregating historical timelines and projecting forecast curve..."):
                        try:
                            # 1. Resample
                            resampled_ts, resample_logs = resample_time_series(
                                active_df, 
                                selected_date_col, 
                                selected_target_metric, 
                                freq=ts_freq, 
                                agg_func=ts_agg
                            )
                            # 2. Forecast
                            hist_df, fore_df, ts_mets, fore_logs = forecast_future(
                                resampled_ts, 
                                horizon=ts_horizon, 
                                model_type=ts_model
                            )
                            
                            st.session_state["forecast_history"] = hist_df
                            st.session_state["forecast_future"] = fore_df
                            # Add contextual details to metrics
                            ts_mets.update({
                                "date_col": selected_date_col,
                                "target_col": selected_target_metric,
                                "freq": ts_freq,
                                "horizon": ts_horizon
                            })
                            st.session_state["forecast_metrics"] = ts_mets
                            st.session_state["forecast_logs"] = resample_logs + fore_logs
                            st.toast("Forecasting compiled successfully!", icon="📈")
                        except Exception as e:
                            st.error(f"Forecasting Engine Error: {str(e)}")
                            st.session_state["forecast_history"] = None
                            
                # Show results if available
                if st.session_state["forecast_history"] is not None:
                    h_df = st.session_state["forecast_history"]
                    f_df = st.session_state["forecast_future"]
                    f_mets = st.session_state["forecast_metrics"]
                    
                    st.markdown(f"### 📊 Time-Series Projections: {f_mets['target_col']}")
                    
                    # Custom Plotly ribbon forecast chart
                    fig = go.Figure()
                    
                    # Actuals history
                    fig.add_trace(go.Scatter(
                        x=h_df.index, y=h_df['actual'], 
                        name='Actual History', 
                        line=dict(color='#0D9488', width=2)
                    ))
                    # Model fit history
                    fig.add_trace(go.Scatter(
                        x=h_df.index, y=h_df['fitted'], 
                        name='Autoregressive Fit', 
                        line=dict(color='#64748B', width=1.5, dash='dash')
                    ))
                    # Forecast projection
                    fig.add_trace(go.Scatter(
                        x=f_df.index, y=f_df['forecast'], 
                        name='Forecast Curve', 
                        line=dict(color='#F59E0B', width=3)
                    ))
                    # Shaded uncertainty envelope
                    fig.add_trace(go.Scatter(
                        x=list(f_df.index) + list(f_df.index)[::-1],
                        y=list(f_df['upper_bound']) + list(f_df['lower_bound'])[::-1],
                        fill='toself',
                        fillcolor='rgba(245, 158, 11, 0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis_title="Date Timeline",
                        yaxis_title=f_mets['target_col'],
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics Grid
                    st.markdown("#### Key Forecast Findings")
                    f_c1, f_c2, f_c3 = st.columns(3)
                    with f_c1:
                        st.metric("Expected Future Trajectory", f_mets['forecast_direction'])
                    with f_c2:
                        st.metric("Historical Change Rate", f"{f_mets['historical_growth_pct']:.2f}%")
                    with f_c3:
                        st.metric("Expected Value Change", f"{f_mets['forecast_pct_change']:.2f}%")
                        
                    # Logs
                    with st.expander("🖥️ Forecasting Log Details"):
                        for log in st.session_state.get("forecast_logs", []):
                            st.write(f"- {log}")
                else:
                    st.info("Adjust the configuration settings on the left panel and click **Compute Future Forecasts** to build predictions.")

    # ----------------------------------------------------
    # TAB 4: AUTOML PLAYGROUND
    # ----------------------------------------------------
    with tab_ml:
        st.markdown("## 🔮 Automated Machine Learning (AutoML) Pipeline")
        
        # AutoML Trigger
        if selected_target:
            st.write(f"Predictive Target Selected: **{selected_target}**")
        else:
            st.write("No Target Selected. The AutoML pipeline will partition the dataset using **Unsupervised K-Means Clustering**.")
            
        ml_btn = st.button("🚀 Train AutoML Model Pipeline", use_container_width=True)
        
        if ml_btn:
            with st.spinner("Automating preprocessing, training models, and tuning parameters..."):
                try:
                    pipeline, ml_logs = run_automl(active_df, selected_target)
                    st.session_state["ml_pipeline"] = pipeline
                    st.session_state["ml_logs"] = ml_logs
                    st.toast("Model training completed!", icon="🎉")
                except Exception as e:
                    st.error(f"AutoML Training Error: {str(e)}")
                    st.session_state["ml_pipeline"] = None
                    
        # If pipeline has been trained, display results
        if st.session_state["ml_pipeline"] is not None:
            pipeline: AutoMLPipeline = st.session_state["ml_pipeline"]
            
            st.markdown("### 🏆 Best Model Fit Summary")
            
            # Cards
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Model Selected", pipeline.model_name)
            with col_m2:
                st.metric("Task Type", pipeline.task_type.upper())
                
            # Model metrics
            st.subheader("Performance Assessment (Test Split / Validation)")
            met_cols = st.columns(len(pipeline.metrics))
            for col_widget, (k, v) in zip(met_cols, pipeline.metrics.items()):
                with col_widget:
                    st.metric(k, f"{v:.4f}")
                    
            # Feature Importance or Clusters plotting
            if pipeline.task_type == "clustering":
                st.subheader("Clustering Profile")
                labels = pipeline.best_model.labels_
                # Add cluster labels to a temporary plotting dataframe
                plot_df = active_df.copy()
                plot_df["Cluster"] = [f"Cluster {l}" for l in labels]
                
                # Pick top 2 numeric columns for visual scatter
                num_vars = plot_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_vars) >= 2:
                    fig = px.scatter(
                        plot_df, x=num_vars[0], y=num_vars[1], color="Cluster",
                        title=f"K-Means Clustering Visualized (Features: {num_vars[0]} vs {num_vars[1]})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Cluster Distribution count:")
                    st.dataframe(plot_df["Cluster"].value_counts())
            else:
                # Supervised
                if not pipeline.feature_importance_df.empty:
                    st.subheader("🌟 Predictive Feature Importance")
                    fig = px.bar(
                        pipeline.feature_importance_df.head(15), 
                        x="importance", y="feature", 
                        orientation="h",
                        title="Top Predictive Driver Importance Coefficients",
                        color_discrete_sequence=["#0D9488"]
                    )
                    # Sort bars ascending
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
            # Predict new data
            if pipeline.task_type != "clustering":
                st.markdown("---")
                st.subheader("🔮 Predict New Observation Inputs")
                st.write("Enter values below to run the observations through the preprocessing and AutoML pipeline.")
                
                with st.form("prediction_form"):
                    input_inputs = {}
                    
                    # Split features for rendering grid columns
                    cols_to_render = pipeline.feature_cols
                    grid_columns = st.columns(3)
                    
                    for idx, col in enumerate(cols_to_render):
                        col_widget = grid_columns[idx % 3]
                        with col_widget:
                            if col in pipeline.categorical_options:
                                input_inputs[col] = st.selectbox(
                                    col, 
                                    options=pipeline.categorical_options[col],
                                    key=f"predict_{col}"
                                )
                            elif col in pipeline.numeric_bounds:
                                bounds = pipeline.numeric_bounds[col]
                                # Create reasonable inputs
                                val_min = bounds["min"]
                                val_max = bounds["max"]
                                val_med = bounds["median"]
                                # Check bounds validity
                                if val_min == val_max:
                                    val_max = val_min + 1.0
                                    
                                input_inputs[col] = st.number_input(
                                    col,
                                    min_value=float(val_min),
                                    max_value=float(val_max),
                                    value=float(val_med),
                                    key=f"predict_{col}"
                                )
                                
                    predict_submit = st.form_submit_button("🔮 Predict Target Value", use_container_width=True)
                    
                if predict_submit:
                    pred_val, probs = pipeline.predict(input_inputs)
                    
                    st.markdown("#### ⚡ Pipeline Prediction Output")
                    
                    c_p1, c_p2 = st.columns(2)
                    with c_p1:
                        st.metric("Predicted Class/Value", str(pred_val))
                    with c_p2:
                        if probs is not None:
                            st.write("Class Probabilities:")
                            classes = pipeline.best_model.classes_
                            for cls_val, prob in zip(classes, probs):
                                st.write(f"- Class **{cls_val}**: {prob:.2%}")
                                st.progress(float(prob))
                                
            # Display logs
            with st.expander("🖥️ AutoML Training Logs"):
                if st.session_state["ml_logs"]:
                    for log in st.session_state["ml_logs"]:
                        st.write(f"- {log}")
                        
        else:
            st.info("Train the AutoML model pipeline above to unlock evaluations and make interactive predictions.")

    # ----------------------------------------------------
    # TAB 4: AI SUMMARY & EXPORT DOWNLOAD
    # ----------------------------------------------------
    with tab_ai:
        st.markdown("## 🤖 AI Analytical Synthesis & Executive Report")
        st.write("Combine quantitative insights, data audits, correlations, and ML model outputs into a rich business brief.")
        
        ai_btn = st.button("🤖 Generate AI Executive Report", use_container_width=True)
        
        # Build analysis summaries to send
        summary_payload = {
            "rows": active_df.shape[0],
            "columns": active_df.shape[1],
            "audit": audit_results,
            "top_correlations": top_5_corrs if 'top_5_corrs' in locals() else [],
            "hypothesis_results": st.session_state.get("last_hypothesis_test", {}),
            "ml_results": {},
            "forecasting_results": st.session_state.get("forecast_metrics", {})
        }
        
        # Populate ML metrics in payload if trained
        if st.session_state["ml_pipeline"] is not None:
            pipe: AutoMLPipeline = st.session_state["ml_pipeline"]
            feat_imp_list = []
            if not pipe.feature_importance_df.empty:
                feat_imp_list = pipe.feature_importance_df.to_dict('records')
                
            summary_payload["ml_results"] = {
                "task_type": pipe.task_type,
                "model_name": pipe.model_name,
                "metrics": pipe.metrics,
                "feature_importance": feat_imp_list
            }
            
        if ai_btn:
            with st.spinner("Synthesizing metrics and drafting executive summary..."):
                report = generate_ai_report(
                    provider=llm_provider,
                    api_key=llm_key,
                    model_name=llm_model,
                    data_summary=summary_payload,
                    user_context=business_context
                )
                st.session_state["ai_report"] = report
                st.toast("Report compiled!", icon="🤖")
                
        # Display report if available
        if st.session_state["ai_report"]:
            st.markdown("---")
            st.markdown(st.session_state["ai_report"])
            
            # Export Buttons
            st.markdown("---")
            st.subheader("📥 Export & Download Executive Report")
            
            # Prep Markdown text
            md_content = generate_markdown_report(summary_payload, st.session_state["ai_report"])
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.download_button(
                    "📝 Download Markdown Brief",
                    data=md_content,
                    file_name="executive_data_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                
            with c_d2:
                # PDF Compilation with temporary file buffer
                # Wrap inside a button context or prepare on-the-fly
                with st.spinner("Compiling PDF document and rendering layouts..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp_path = tmp.name
                        try:
                            compile_pdf_report(summary_payload, st.session_state["ai_report"], tmp_path)
                            with open(tmp_path, "rb") as f:
                                pdf_bytes = f.read()
                                
                            st.download_button(
                                "📄 Download Premium PDF Report",
                                data=pdf_bytes,
                                file_name="executive_data_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    except Exception as e:
                        st.error(f"Could not generate PDF download binary: {str(e)}")
        else:
            st.info("Click **Generate AI Executive Report** to draft the executive summary and unlock PDF/Markdown downloads.")
