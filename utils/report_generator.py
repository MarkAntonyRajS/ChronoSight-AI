import re
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from fpdf import FPDF

def clean_pdf_text(text: str) -> str:
    """
    Cleans text of non-Latin-1 characters, smart quotes, em-dashes, and other
    symbols that crash standard FPDF fonts.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Replace common unicode quotes and symbols
    replacements = {
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u2014': '-',  # Em dash
        '\u2013': '-',  # En dash
        '\u2022': '*',  # Bullet point
        '\u2265': '>=', # Greater than or equal
        '\u2264': '<=', # Less than or equal
        '\u00b1': '+/-',# Plus-minus
        '\u2212': '-',  # Minus sign
    }
    
    for key, val in replacements.items():
        text = text.replace(key, val)
        
    # Remove any remaining non-latin1 characters
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return text

class ExecutivePDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(15, 20, 15)
        self.alias_nb_pages()
        
    def header(self):
        # Header banner (only on page 2 onwards)
        if self.page_no() > 1:
            self.set_fill_color(30, 41, 59) # Slate 800
            self.rect(0, 0, 210, 12, 'F')
            self.set_y(2)
            self.set_text_color(255, 255, 255)
            self.set_font('helvetica', 'B', 8)
            self.cell(0, 8, 'CHRONOSIGHT AI - EXECUTIVE ANALYTICS REPORT', 0, 0, 'L')
            self.set_font('helvetica', '', 8)
            self.cell(0, 8, datetime.now().strftime('%Y-%m-%d'), 0, 0, 'R')
            self.set_y(15)
            self.ln(5)

    def footer(self):
        # Footer (only on page 2 onwards)
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.set_text_color(100, 116, 139) # Slate 500
            # Line separator
            self.set_draw_color(226, 232, 240) # Slate 200
            self.line(15, self.get_y(), 195, self.get_y())
            self.ln(2)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def add_section_header(self, title: str):
        self.set_text_color(30, 41, 59) # Slate 800
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, clean_pdf_text(title), 0, 1, 'L')
        # Draw underline accent
        self.set_draw_color(13, 148, 136) # Teal 600
        self.set_thickness(1.5)
        self.line(self.get_x(), self.get_y(), self.get_x() + 40, self.get_y())
        self.set_thickness(0.2) # reset
        self.ln(8)

    def add_subsection_header(self, title: str):
        self.set_text_color(51, 65, 85) # Slate 700
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 8, clean_pdf_text(title), 0, 1, 'L')
        self.ln(2)

    def add_paragraph(self, text: str):
        self.set_text_color(71, 85, 105) # Slate 600
        self.set_font('helvetica', '', 10)
        # Multi_cell handles word wrap
        self.multi_cell(0, 6, clean_pdf_text(text))
        self.ln(4)

    def add_bullet_point(self, text: str):
        self.set_text_color(71, 85, 105) # Slate 600
        self.set_font('helvetica', '', 10)
        # Bullet offset
        self.cell(8, 6, '-', 0, 0, 'R')
        self.multi_cell(0, 6, clean_pdf_text(text))
        self.ln(2)

def generate_markdown_report(summary: Dict[str, Any], ai_findings: str) -> str:
    """
    Compiles a comprehensive markdown report.
    """
    audit = summary.get("audit", {})
    cols_summary = audit.get("columns_summary", {})
    missing = audit.get("missing_values", {})
    duplicates = audit.get("duplicates", {})
    
    top_corrs = summary.get("top_correlations", [])
    hyp_results = summary.get("hypothesis_results", {})
    ml_results = summary.get("ml_results", {})
    
    md = f"""# ChronoSight AI - Executive Analytics & Forecast Report
**Date of Analysis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset Shape:** {summary.get('rows')} rows, {summary.get('columns')} columns

---

## 1. Data Ingestion & Audit Summary
- **Total Missing Cells:** {missing.get('total_cells')} ({missing.get('total_percentage')}%)
- **Duplicate Rows:** {duplicates.get('count')} ({duplicates.get('percentage')}%)
- **Numeric Fields:** {cols_summary.get('total_numeric')}
- **Categorical Fields:** {cols_summary.get('total_categorical')}

### Column Integrity Checklist
| Column Name | Missing Count | Missing % |
|---|---|---|
"""
    # Add columns details
    for col, info in missing.get("by_column", {}).items():
        md += f"| {col} | {info['count']} | {info['percentage']}% |\n"
    if not missing.get("by_column"):
        md += "| (None) | 0 | 0.0% |\n"
        
    md += "\n---\n\n## 2. Statistical Analysis & Insights\n"
    
    # Correlations
    md += "### Top 5 Strongest Correlations\n"
    if top_corrs:
        md += "| Variable 1 | Variable 2 | Correlation Coeff (r) |\n|---|---|---|\n"
        for item in top_corrs:
            md += f"| {item['var1']} | {item['var2']} | {item['correlation']:.4f} |\n"
    else:
        md += "*No numerical variables found for correlation analysis.*\n"
        
    # Hypothesis test
    md += "\n### Hypothesis Testing Logs\n"
    if hyp_results:
        md += f"""- **Test Performed:** {hyp_results.get('test_name')}
- **Variables Tested:** `{hyp_results.get('var1')}` vs. `{hyp_results.get('var2')}`
- **Test Statistic:** {hyp_results.get('statistic'):.4f}
- **p-value:** {hyp_results.get('p_value'):.4e}
- **Interpretation:** {hyp_results.get('interpretation')}
"""
    else:
        md += "*No hypothesis tests were selected for this run.*\n"

    md += "\n---\n\n## 3. Automated Machine Learning (AutoML) Insights\n"
    
    if ml_results:
        task_type = ml_results.get("task_type", "").upper()
        best_model = ml_results.get("model_name", "N/A")
        metrics = ml_results.get("metrics", {})
        metrics_str = ", ".join([f"**{k}**: {v:.4f}" for k, v in metrics.items()])
        
        md += f"""- **Task Mode:** {task_type}
- **Selected Best Model:** {best_model}
- **Validation Score:** {metrics_str}

### Feature Importance Summary
| Feature | Normalized Importance |
|---|---|
"""
        for item in ml_results.get("feature_importance", [])[:10]:
            md += f"| {item['feature']} | {item['importance']:.2%} |\n"
        if not ml_results.get("feature_importance"):
            md += "| (No importance values generated) | - |\n"
    else:
        md += "*AutoML model training was skipped or target variable was not configured.*\n"
        
    # Time-series forecasting
    md += "\n---\n\n## 4. Time-Series Trends & Forecasting Insights\n"
    ts_results = summary.get("forecasting_results", {})
    if ts_results:
        md += f"""- **Date Variable:** `{ts_results.get('date_col')}`
- **Forecast Metric:** `{ts_results.get('target_col')}`
- **Aggregated Frequency:** {ts_results.get('freq')}
- **Historical Growth Rate:** {ts_results.get('historical_growth_pct'):.2f}%
- **Forecast Model Fit (R2):** {ts_results.get('model_r2'):.4f}
- **Forecast Horizon:** {ts_results.get('horizon')} steps
- **Projected Future Trajectory:** **{ts_results.get('forecast_direction')}** (Average predicted: {ts_results.get('forecast_average'):.4f})
"""
    else:
        md += "*No time-series trend forecasting was performed.*\n"
        
    md += "\n---\n\n## 5. AI-Generated Executive Synthesis\n"
    md += ai_findings
    
    return md

def compile_pdf_report(summary: Dict[str, Any], ai_findings: str, output_path: str):
    """
    Compiles an Executive PDF report using fpdf2 and saves it.
    """
    pdf = ExecutivePDFReport()
    pdf.add_page()
    
    # --- PAGE 1: TITLE PAGE & DATA AUDIT ---
    # Title Block
    pdf.set_fill_color(30, 41, 59) # Slate 800
    pdf.rect(0, 0, 210, 80, 'F')
    
    pdf.set_y(25)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('helvetica', 'B', 24)
    pdf.cell(0, 12, 'CHRONOSIGHT AI REPORT', 0, 1, 'C')
    
    pdf.set_font('helvetica', 'I', 12)
    pdf.set_text_color(13, 148, 136) # Teal 600
    pdf.cell(0, 8, 'Autonomous Data Audit, Statistical Engine & Predictive Insights', 0, 1, 'C')
    
    pdf.set_y(85)
    pdf.set_text_color(100, 116, 139) # Slate 500
    pdf.set_font('helvetica', '', 9)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(5)
    
    # Ingestion details
    pdf.add_section_header("1. Data Ingestion & Audit")
    
    audit = summary.get("audit", {})
    cols_summary = audit.get("columns_summary", {})
    missing = audit.get("missing_values", {})
    duplicates = audit.get("duplicates", {})
    
    # Audit summary grid
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_text_color(51, 65, 85)
    
    # Draw simple key value table
    data_metrics = [
        ("Total Rows", str(summary.get("rows"))),
        ("Total Columns", str(summary.get("columns"))),
        ("Numeric Columns", str(cols_summary.get("total_numeric"))),
        ("Categorical Columns", str(cols_summary.get("total_categorical"))),
        ("Missing Data Cells", f"{missing.get('total_cells')} ({missing.get('total_percentage')}%)"),
        ("Duplicate Rows", f"{duplicates.get('count')} ({duplicates.get('percentage')}%)")
    ]
    
    # Table styling parameters
    col_w_key = 60
    col_w_val = 110
    row_h = 7
    
    pdf.set_fill_color(241, 245, 249) # Light gray table header
    
    # Set headers
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(col_w_key, row_h, "Audit Metric", 1, 0, 'L', True)
    pdf.cell(col_w_val, row_h, "Observed Value", 1, 1, 'L', True)
    
    # Fill values
    pdf.set_font('helvetica', '', 10)
    pdf.set_text_color(71, 85, 105)
    for key, val in data_metrics:
        pdf.cell(col_w_key, row_h, clean_pdf_text(key), 1, 0, 'L')
        pdf.cell(col_w_val, row_h, clean_pdf_text(val), 1, 1, 'L')
        
    pdf.ln(10)
    
    # --- PAGE 2: STATISTICAL ANALYSIS ---
    pdf.add_page()
    pdf.add_section_header("2. Quantitative & Hypothesis Testing")
    
    # Correlations
    pdf.add_subsection_header("Strongest Variable Interactions")
    top_corrs = summary.get("top_correlations", [])
    if top_corrs:
        pdf.set_fill_color(241, 245, 249)
        pdf.set_font('helvetica', 'B', 9)
        pdf.cell(60, row_h, "Variable 1", 1, 0, 'L', True)
        pdf.cell(60, row_h, "Variable 2", 1, 0, 'L', True)
        pdf.cell(50, row_h, "Correlation Coeff (r)", 1, 1, 'C', True)
        
        pdf.set_font('helvetica', '', 9)
        for item in top_corrs:
            pdf.cell(60, row_h, clean_pdf_text(item['var1']), 1, 0, 'L')
            pdf.cell(60, row_h, clean_pdf_text(item['var2']), 1, 0, 'L')
            pdf.cell(50, row_h, f"{item['correlation']:.4f}", 1, 1, 'C')
    else:
        pdf.add_paragraph("No numerical relationships identified.")
        
    pdf.ln(8)
    
    # Hypothesis test
    pdf.add_subsection_header("Hypothesis Testing Summary")
    hyp_results = summary.get("hypothesis_results", {})
    if hyp_results:
        test_info = (
            f"Test Conducted: {hyp_results.get('test_name')}\n"
            f"Variables Evaluated: '{hyp_results.get('var1')}' vs. '{hyp_results.get('var2')}'\n"
            f"Observed Test Statistic: {hyp_results.get('statistic'):.4f}\n"
            f"Exact p-value: {hyp_results.get('p_value'):.4e}\n\n"
            f"Scientific Interpretation: {hyp_results.get('interpretation')}"
        )
        # Wrap inside multi_cell
        pdf.multi_cell(0, 6, clean_pdf_text(test_info), 1, 'L')
    else:
        pdf.add_paragraph("No hypothesis testing was executed for this run.")
        
    pdf.ln(10)
    
    # --- PAGE 3: AUTOML PIPELINE ---
    pdf.add_page()
    pdf.add_section_header("3. Machine Learning Insights")
    
    ml_results = summary.get("ml_results", {})
    if ml_results:
        task_type = ml_results.get("task_type", "").upper()
        best_model = ml_results.get("model_name", "N/A")
        metrics = ml_results.get("metrics", {})
        
        pdf.add_subsection_header(f"Automated Model Selection ({task_type})")
        pdf.add_paragraph(f"The AutoML engine evaluated multiple algorithms. The best model selected is {best_model}.")
        
        # Metrics Table
        pdf.set_fill_color(241, 245, 249)
        pdf.set_font('helvetica', 'B', 9)
        pdf.cell(85, row_h, "Evaluation Metric", 1, 0, 'L', True)
        pdf.cell(85, row_h, "Test Set Score", 1, 1, 'C', True)
        
        pdf.set_font('helvetica', '', 9)
        for k, v in metrics.items():
            pdf.cell(85, row_h, clean_pdf_text(k), 1, 0, 'L')
            pdf.cell(85, row_h, f"{v:.4f}", 1, 1, 'C')
            
        pdf.ln(8)
        
        # Feature importance
        pdf.add_subsection_header("Top Predictive Drivers")
        feat_imp = ml_results.get("feature_importance", [])
        if feat_imp:
            pdf.set_fill_color(241, 245, 249)
            pdf.set_font('helvetica', 'B', 9)
            pdf.cell(100, row_h, "Predictive Feature", 1, 0, 'L', True)
            pdf.cell(70, row_h, "Relative Contribution", 1, 1, 'C', True)
            
            pdf.set_font('helvetica', '', 9)
            for item in feat_imp[:8]: # top 8
                pdf.cell(100, row_h, clean_pdf_text(item['feature']), 1, 0, 'L')
                pdf.cell(70, row_h, f"{item['importance']:.2%}", 1, 1, 'C')
        else:
            pdf.add_paragraph("Feature importances are unavailable for the selected model.")
    else:
        pdf.add_paragraph("Machine learning modeling was not performed.")
        
    # --- PAGE 4: TIME-SERIES FORECASTING ---
    pdf.add_page()
    pdf.add_section_header("4. Time-Series Trends & Forecasting")
    
    ts_results = summary.get("forecasting_results", {})
    if ts_results:
        pdf.add_subsection_header(f"Autoregressive Forecast ({ts_results.get('target_col')} over time)")
        pdf.add_paragraph(f"The engine resampled the data based on the '{ts_results.get('date_col')}' temporal column at a frequency of '{ts_results.get('freq')}'.")
        
        # Grid of Forecasting Stats
        ts_metrics = [
            ("Date Column", str(ts_results.get("date_col"))),
            ("Target Variable", str(ts_results.get("target_col"))),
            ("Resampling Frequency", str(ts_results.get("freq"))),
            ("Historical Growth Rate", f"{ts_results.get('historical_growth_pct'):.2f}%"),
            ("Model Fit R-Squared", f"{ts_results.get('model_r2'):.4f}"),
            ("Model MAE", f"{ts_results.get('model_mae'):.4f}"),
            ("Forecast Horizon", f"{ts_results.get('horizon')} periods"),
            ("Expected Future Trend", str(ts_results.get("forecast_direction"))),
            ("Predicted Average Value", f"{ts_results.get('forecast_average'):.4f} ({ts_results.get('forecast_pct_change'):.2f}%)")
        ]
        
        pdf.set_fill_color(241, 245, 249)
        pdf.set_font('helvetica', 'B', 9)
        pdf.cell(85, row_h, "Forecasting Metric", 1, 0, 'L', True)
        pdf.cell(85, row_h, "Observed/Predicted Value", 1, 1, 'L', True)
        
        pdf.set_font('helvetica', '', 9)
        for k, v in ts_metrics:
            pdf.cell(85, row_h, clean_pdf_text(k), 1, 0, 'L')
            pdf.cell(85, row_h, clean_pdf_text(v), 1, 1, 'L')
    else:
        pdf.add_paragraph("Time-series trend forecasting analysis was not performed for this run.")
        
    pdf.ln(10)
    
    # --- PAGE 5+: AI EXECUTIVE SUMMARY & SYNTHESIS ---
    pdf.add_page()
    pdf.add_section_header("5. AI Analytical Synthesis")
    
    # Process AI findings (convert markdown text to FPDF layout)
    # Simple parser: Split text into lines, search for headers, bullet points and paragraphs
    lines = ai_findings.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Headers: e.g. # Header or ## Header
        if line.startswith("#"):
            # Count hashes
            header_level = len(re.match(r"^#+", line).group(0))
            title_text = line.replace("#", "").strip()
            
            if header_level == 1:
                pdf.ln(5)
                pdf.set_text_color(30, 41, 59)
                pdf.set_font('helvetica', 'B', 14)
                pdf.cell(0, 10, clean_pdf_text(title_text), 0, 1, 'L')
                pdf.ln(2)
            else:
                pdf.ln(3)
                pdf.set_text_color(51, 65, 85)
                pdf.set_font('helvetica', 'B', 11)
                pdf.cell(0, 8, clean_pdf_text(title_text), 0, 1, 'L')
                pdf.ln(1)
                
        # Bullet points: e.g. - item or * item
        elif line.startswith("-") or line.startswith("*"):
            bullet_text = re.sub(r"^[-*]\s*", "", line).strip()
            pdf.add_bullet_point(bullet_text)
            
        # Standard paragraphs
        else:
            # Check if bold formatting inside paragraph, we strip it out for PDF simplicity
            cleaned_line = line.replace("**", "").replace("*", "")
            pdf.add_paragraph(cleaned_line)
            
    # Save file
    pdf.output(output_path)
