# ChronoSight AI 📊

**ChronoSight AI** is a complete, production-ready, and fully automated data analysis, machine learning modeling, and report generation platform built from scratch in Python. It ingests raw tabular datasets, runs autonomous data cleaning, executes statistical analyses, trains machine learning pipelines (AutoML), forecasts future trends, and synthesizes executive business summaries using an LLM.

---

## ✨ Key Features

### 🧹 1. Autonomous Data Ingestion & Cleaning
- **Multi-Format Support**: Reads CSV, Excel (`.xlsx`), and Parquet formats.
- **Memory Optimization**: Downcasts numeric data types and converts low-cardinality strings to categories to reduce memory footprint.
- **Data Auditing**: Automatically scans for missing cells, structural duplicates, and type anomalies (e.g., currency symbols, commas, or percent signs in string-inferred columns).
- **Autonomous Treatment**: Imputes numerical missing values based on column skewness (chooses Median for skewed distributions, Mean otherwise) and categorical values using Mode.
- **Outlier Mitigation**: Isolates and caps or removes outliers using the Interquartile Range (IQR) method and Isolation Forest.

### 📐 2. Statistical Engine & Adaptive Hypothesis Testing
- **Descriptive Metrics**: Computes means, medians, standard deviation, skewness, and kurtosis.
- **Correlation Analysis**: Calculates Pearson & Spearman correlation matrices and isolates the top 5 strongest relationships.
- **Adaptive Hypothesis Testing**: Scans two user-selected columns, analyzes their data types/distributions, and automatically routes them to the correct statistical test:
  - *Numeric vs. Numeric*: Pearson or Spearman Rank Correlation significance tests.
  - *Categorical (2 groups) vs. Numeric*: Welch's T-Test or Mann-Whitney U test.
  - *Categorical (>2 groups) vs. Numeric*: One-Way ANOVA or Kruskal-Wallis.
  - *Categorical vs. Categorical*: Chi-Square test of independence.
  - Returns exact test statistics, p-values, and a plain-English narrative interpretation.

### 📈 3. Time-Series Trends & Forecasting
- **Index Auto-Detection**: Scans and flags candidates for date/timestamp dimensions.
- **Temporal Resampling**: Aggregates target metrics daily, weekly, or monthly using Mean or Sum, linearly interpolating chronological gaps.
- **Recursive Lag-Modeling**: Fits an autoregressive Random Forest or Linear Regression model using past lags and rolling averages.
- **Uncertainty Envelope**: Projects future intervals along with a shaded confidence boundary ($\pm 1.96 \times$ residual error propagation).

### 🔮 4. AutoML Playground
- **Auto-Task Selection**: Detects target data types to toggle between **Regression** (continuous target), **Classification** (categorical target), or **Unsupervised Clustering (K-Means)**.
- **Feature Selection**: Filters out high-cardinality text (like IDs), zero-variance columns, and columns containing mostly missing values.
- **Tuned Model Fitting**: Fits and evaluates baseline pipelines, performs randomized hyperparameter tuning on the best performer, and displays validation metrics ($R^2$, MAE, RMSE for regression; Accuracy, F1-Score, ROC-AUC for classification; Silhouette score for clustering).
- **Interactive Predictions**: Generates dynamic input controls (sliders and dropdowns) allowing users to enter custom observation values and run them live through the preprocessors and best-fit model pipeline.

### 🤖 5. AI Business Brief Synthesis & Exports
- **API Adapters**: Native integration with Google Gemini, OpenAI, and local Ollama.
- **Executive Summaries**: Combines metrics, audits, hypothesis tests, AutoML evaluations, and forecasts into a structured prompt for the LLM to draft actionable, data-backed business briefs.
- **Export Formats**: Provides one-click downloads for clean CSV datasets, formatted Markdown reports, and styled PDF executive briefs.

---

## 📁 Project Directory Layout

```text
data_analyzer/
├── requirements.txt            # Python environment dependencies
├── README.md                   # Project landing page & documentation
├── app.py                      # Main Streamlit dashboard interface
├── utils/
│   ├── __init__.py
│   ├── llm_helper.py           # API adapters (Gemini, OpenAI, Ollama)
│   └── report_generator.py     # PDF & Markdown compilers
└── components/
    ├── __init__.py
    ├── cleaning.py             # Memory optimization, audits, and imputation
    ├── analytics.py            # Descriptives, correlations, hypothesis testing
    ├── forecasting.py          # Temporal index resampling, lag models, projections
    └── models.py               # Preprocessing pipelines, AutoML, and prediction forms
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have **Python 3.11+** installed on your system.

### 1. Clone the Codebase
Move the project files into your local directory.

### 2. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
This will launch the application server. Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 💡 Quick Start Guide
1. Launch the app and click the **"💡 Load Sample Churn Dataset"** button in the sidebar to populate the application instantly with a pre-configured synthetic dataset (includes dates, missing values, formatting anomalies, and outliers).
2. Browse through the tabs:
   - **📁 Ingestion & Cleaning**: Inspect clean tables and processing logs.
   - **📐 Statistical Analysis**: Inspect correlation heatmaps and execute dynamic hypothesis tests.
   - **📈 Trends & Forecasting**: Set `Registration_Date` as the timeline index and forecast numeric fields.
   - **🔮 AutoML Playground**: Select a target column (e.g. `Churn_Flag`), fit models, and test live predictions.
   - **🤖 AI Executive Report**: Enter your API Key or Ollama host, generate a comprehensive text brief, and download the report as a PDF!
