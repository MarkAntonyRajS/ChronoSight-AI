import requests
import json
from typing import Dict, Any, Optional

def generate_ai_report(
    provider: str,
    api_key: str,
    model_name: str,
    data_summary: Dict[str, Any],
    user_context: str = ""
) -> str:
    """
    Sends structured analysis data to the selected LLM provider and retrieves
    a comprehensive business analysis report.
    """
    prompt = build_analysis_prompt(data_summary, user_context)
    
    provider = provider.lower()
    if provider == "gemini":
        return call_gemini(api_key, model_name, prompt)
    elif provider == "openai":
        return call_openai(api_key, model_name, prompt)
    elif provider == "ollama":
        # Usually Ollama doesn't require an API Key, API Key will contain host address if customized, or default to localhost
        host = api_key if (api_key and api_key.startswith("http")) else "http://localhost:11434"
        return call_ollama(host, model_name, prompt)
    else:
        return f"Error: Unsupported LLM provider '{provider}' selected."

def build_analysis_prompt(summary: Dict[str, Any], user_context: str) -> str:
    """
    Constructs a highly detailed markdown prompt containing tabular details,
    statistical findings, and model metrics for the LLM to synthesize.
    """
    # Parse summary fields
    audit = summary.get("audit", {})
    cols_summary = audit.get("columns_summary", {})
    num_cols = cols_summary.get("numeric", [])
    cat_cols = cols_summary.get("categorical", [])
    missing = audit.get("missing_values", {})
    duplicates = audit.get("duplicates", {})
    
    top_corrs = summary.get("top_correlations", [])
    hyp_results = summary.get("hypothesis_results", {})
    ml_results = summary.get("ml_results", {})
    
    # Format correlations list
    corr_text = ""
    if top_corrs:
        for idx, item in enumerate(top_corrs):
            corr_text += f"{idx+1}. '{item['var1']}' and '{item['var2']}' (r = {item['correlation']:.3f})\n"
    else:
        corr_text = "No significant numeric correlations found.\n"
        
    # Format hypothesis testing
    hyp_text = "No specific hypothesis tests were run."
    if hyp_results:
        hyp_text = (
            f"Test Conducted: {hyp_results.get('test_name')}\n"
            f"Variables: '{hyp_results.get('var1')}' vs. '{hyp_results.get('var2')}'\n"
            f"Statistic: {hyp_results.get('statistic'):.4f}, p-value: {hyp_results.get('p_value'):.4e}\n"
            f"Interpretation: {hyp_results.get('interpretation')}"
        )
        
    # Format ML findings
    ml_text = "No ML modeling was performed."
    if ml_results:
        task_type = ml_results.get("task_type", "").upper()
        best_model = ml_results.get("model_name", "N/A")
        metrics = ml_results.get("metrics", {})
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        ml_text = (
            f"Task Type: {task_type}\n"
            f"Best Model: {best_model}\n"
            f"Performance Metrics: {metrics_str}\n"
        )
        if ml_results.get("feature_importance"):
            ml_text += "Top Features by Importance:\n"
            for idx, item in enumerate(ml_results["feature_importance"][:5]):
                ml_text += f" - {item['feature']}: {item['importance']:.2%}\n"

    # Format Time-series Forecasting findings
    ts_results = summary.get("forecasting_results", {})
    ts_text = "No time-series forecasting analysis was performed."
    if ts_results:
        ts_text = (
            f"Date Index Column: '{ts_results.get('date_col')}'\n"
            f"Forecasted Variable: '{ts_results.get('target_col')}' (Frequency: '{ts_results.get('freq')}')\n"
            f"Forecast Horizon: {ts_results.get('horizon')} periods\n"
            f"Forecasting Model R2: {ts_results.get('model_r2'):.4f}, MAE: {ts_results.get('model_mae'):.4f}\n"
            f"Historical Growth Rate: {ts_results.get('historical_growth_pct'):.2f}%\n"
            f"Forecast Trend Direction: {ts_results.get('forecast_direction')}\n"
            f"Expected Average Future Value: {ts_results.get('forecast_average'):.4f} (Change: {ts_results.get('forecast_pct_change'):.2f}%)\n"
        )

    # Assemble prompt
    prompt = f"""You are a world-class AI Solution Architect, Senior Python Data Scientist, and Business Intelligence consultant. 
Your goal is to write a highly professional, comprehensive executive business report based on the following automated data analysis.

--- DATASET OVERVIEW ---
- Rows: {summary.get('rows')}
- Columns: {summary.get('columns')}
- Numeric Columns: {cols_summary.get('total_numeric')} ({', '.join(num_cols[:15])}...)
- Categorical Columns: {cols_summary.get('total_categorical')} ({', '.join(cat_cols[:15])}...)
- Missing Value Cells: {missing.get('total_cells')} ({missing.get('total_percentage')}% of dataset)
- Duplicate Rows: {duplicates.get('count')} ({duplicates.get('percentage')}%)

--- KEY STATISTICAL FINDINGS ---
Strongest Variable Correlations (Pearson/Spearman):
{corr_text}

Hypothesis Testing Results:
{hyp_text}

--- HISTORICAL TRENDS & FORECASTING INSIGHTS ---
{ts_text}

--- MACHINE LEARNING & PREDICTIVE MODEL METRICS ---
{ml_text}

--- USER-PROVIDED BUSINESS CONTEXT / OBJECTIVE ---
{user_context if user_context else "Analyze this dataset for general commercial insight, identify anomalies, and give growth recommendations."}

--- REPORT REQUIREMENTS ---
Please structure your report into the following three distinct sections using markdown headers:

# Executive Summary
Provide a clean, high-level summary of the dataset. What are the key takeaways? Speak directly to executives (C-level) and state what the data reveals in 2-3 concise paragraphs.

# Deep Dive Insights & Data Relationships
Explain what the correlations and hypothesis tests actually mean for the business. Translate mathematical findings (coefficients, p-values, feature importances) into real-world business dynamics. What causes what? What is the predictive power of the models?

# Actionable Business Recommendations
List 4-5 concrete, data-backed recommendations. Each recommendation must:
1. Reference a specific stat or ML insight from the dataset.
2. Provide a practical action step.
3. Define the expected business outcome or risk mitigation.

Be direct, precise, and professional. Avoid generic AI fluff. Focus heavily on context-driven data patterns.
"""
    return prompt

def call_gemini(api_key: str, model_name: str, prompt: str) -> str:
    """Calls Google Gemini API."""
    if not api_key:
        return "Error: Gemini API Key is missing. Please set it in the configuration."
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Use default model name if not provided
        model_name = model_name or "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Failed to generate report via Gemini API: {str(e)}"

def call_openai(api_key: str, model_name: str, prompt: str) -> str:
    """Calls OpenAI API."""
    if not api_key:
        return "Error: OpenAI API Key is missing. Please set it in the configuration."
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        model_name = model_name or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate report via OpenAI API: {str(e)}"

def call_ollama(host: str, model_name: str, prompt: str) -> str:
    """Calls a local Ollama instance."""
    url = f"{host}/api/generate"
    model_name = model_name or "llama3"
    
    try:
        response = requests.post(
            url,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=180 # Generous timeout for local models
        )
        if response.status_code == 200:
            return response.json().get("response", "No response content received.")
        else:
            return f"Ollama request failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"Failed to generate report via Ollama: {str(e)}. Make sure Ollama is running and accessible at {host}."
