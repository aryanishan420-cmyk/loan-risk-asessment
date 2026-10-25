# Intelligent Loan Risk Assessment using Machine Learning (AegisCredit.AI)

An end-to-end Machine Learning-powered Decision Support System for retail banking loan default prediction, risk tiering, explainable factor attribution, and relational SQL portfolio analytics.

---

## 🌟 Key Features

1. **End-to-End Machine Learning Pipeline:**
   - 12,000 synthetic banking records with realistic distributions (LendingClub / Fannie Mae distribution curves).
   - Domain-specific financial feature engineering: Loan-to-Income (LTI), Payment-to-Income (PTI), Debt Utilization Stress Index, and Delinquency Tenure ratio.
   - Candidate models trained & benchmarked: **Logistic Regression (Champion)**, **Random Forest**, and **LightGBM**.
   - Asymmetric cost-minimizing threshold optimization ($C_{\text{FN}} = 4.0 \times C_{\text{FP}}$) capturing 83.4% of defaults.

2. **Enterprise Relational Database (SQLite):**
   - Normalized 3NF schema with `applicants`, `loans`, `credit_profiles`, and `risk_evaluations` audit tables.
   - Pre-compiled analytical views: `vw_portfolio_risk_summary`, `vw_high_risk_pipeline`.
   - Real-time audit logging for every decision made.

3. **Interactive Decision Support Web Application:**
   - **Portfolio Analytics:** Real-time KPI cards, grade default curves, and capital exposure doughnut charts (Chart.js).
   - **Risk Underwriter Console:** 1-Click persona presets, interactive sliders, animated risk gauge, tiered decision badges (`AUTO-APPROVE`, `MANUAL REVIEW`, `ADVERSE ACTION`), recommended interest rates, and SHAP-style explainable risk factor drivers.
   - **SQL Query Studio:** Interactive in-browser SQL terminal with pre-built queries, execution time measurement in ms, dynamic results table, and one-click **CSV export**.
   - **Model Performance Hub:** Multi-model comparative metrics, interactive ROC curves, and Top 10 feature importance rankings.
   - **Macroeconomic Stress Testing (Basel III):** Real-time scenario shock simulator (interest rate hike, wage cut, debt surge).

---

## 📁 Project Directory Structure

```
loan-risk-assessment/
│
├── data/
│   ├── generate_dataset.py       # Realistic 12,000 records data generator
│   ├── loan_data_raw.csv         # Raw synthetic banking dataset
│   └── loan_data_processed.csv   # Preprocessed & engineered dataset
│
├── database/
│   ├── schema.sql                # SQLite DDL with 3NF tables, indexes & views
│   ├── init_db.py                # Database population and query utilities
│   └── loan_risk.db              # Relational SQLite database
│
├── src/
│   ├── preprocessing.py          # Data cleaning & financial feature engineering
│   ├── train_models.py           # Training Logistic Regression, Random Forest, LightGBM
│   ├── predict.py                # Inference engine & explainability attribution
│   └── evaluate.py               # Evaluation curves & threshold optimization
│
├── models/
│   ├── best_model.joblib         # Exported champion ML model
│   ├── scaler.joblib             # Standard scaler object
│   ├── feature_cols.joblib       # 31 engineered feature definitions
│   ├── model_metadata.json       # Top features and thresholds
│   └── evaluation_results.json   # Benchmark metrics and ROC points
│
├── app/
│   ├── app.py                    # Multi-threaded Flask REST API backend
│   ├── templates/
│   │   └── index.html            # Luxury dark-mode Single Page Application
│   └── static/
│       ├── css/style.css         # Modern fintech CSS design system
│       └── js/main.js            # Client controller & Chart.js visualizers
│
├── docs/
│   └── PROJECT_REPORT.md         # Comprehensive academic project report
│
├── run_pipeline.py               # Master 1-click end-to-end pipeline script
├── test_app_endpoints.py         # Automated HTTP test suite
└── requirements.txt              # Project dependencies
```

---

## 🚀 Quick Start Guide

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Data Gen + SQL + ML Training)
```bash
python run_pipeline.py
```

### 3. Launch the Web Application
```bash
python app/app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5050`**

### 4. Run the Automated API Test Suite
```bash
python test_app_endpoints.py
```

---

## 📡 REST API Endpoints

- `GET /`: Single-Page Application Web Dashboard
- `GET /api/portfolio_stats`: Portfolio exposure, default rate, grade breakdown, and recent audit trail
- `POST /api/predict`: Real-time loan scoring and explainable driver attribution
- `POST /api/sql_query`: Safe analytic SQL query execution on `loan_risk.db`
- `GET /api/model_metrics`: Comparative evaluation metrics and ROC curve points
- `POST /api/stress_test`: Basel III macro scenario stress testing simulation
- `GET /api/sample_applicants`: Curated borrower personas for 1-click testing
