"""
Intelligent Loan Risk Assessment - Decision Support Web Server
Powered by Flask, SQLite, and Scikit-Learn.
"""

import os
import sys
import json
import sqlite3
import time
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
sys.path.append(os.path.join(BASE_DIR, 'database'))

from predict import LoanRiskPredictor
from init_db import get_db_connection, log_prediction_audit

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static')
)

predictor = LoanRiskPredictor()

# In-memory cached portfolio sample for instant stress simulation
RAW_CSV_PATH = os.path.join(BASE_DIR, 'data', 'loan_data_raw.csv')
if os.path.exists(RAW_CSV_PATH):
    PORTFOLIO_CACHE_DF = pd.read_csv(RAW_CSV_PATH).sample(n=1200, random_state=42).reset_index(drop=True)
else:
    PORTFOLIO_CACHE_DF = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        result = predictor.predict(data)
        
        # Log decision into SQLite audit table
        applicant_id = data.get('applicant_id', f"APP-LIVE-{int(time.time() * 1000) % 1000000}")
        try:
            log_prediction_audit(
                applicant_id=applicant_id,
                model_version=result['model_name'],
                default_prob=result['default_probability'],
                risk_tier=result['risk_tier'],
                decision=result['decision']
            )
        except Exception as audit_err:
            print("Audit log warning:", audit_err)
            
        result['applicant_id'] = applicant_id
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/portfolio_stats', methods=['GET'])
def api_portfolio_stats():
    try:
        conn = get_db_connection()
        # Summary view
        df_summary = pd.read_sql_query("SELECT * FROM vw_portfolio_risk_summary;", conn)
        
        # General aggregates
        total_loans = int(df_summary['total_loans'].sum())
        total_exposure = float(df_summary['total_exposure_usd'].sum())
        total_defaults = int(df_summary['default_count'].sum())
        overall_default_rate = round((total_defaults / total_loans) * 100.0, 2)
        avg_interest_rate = round(float(df_summary['avg_interest_rate'].mean()), 2)
        avg_credit_score = round(float(df_summary['avg_credit_score'].mean()), 1)
        
        # Grade breakdown
        grades_data = {
            'labels': df_summary['loan_grade'].tolist(),
            'counts': df_summary['total_loans'].tolist(),
            'default_rates': df_summary['default_rate_pct'].tolist(),
            'exposure': df_summary['total_exposure_usd'].tolist()
        }
        
        # Recent audit evaluations
        df_recent = pd.read_sql_query(
            "SELECT * FROM risk_evaluations ORDER BY evaluated_at DESC LIMIT 8;", conn
        )
        recent_evals = df_recent.to_dict(orient='records')
        
        conn.close()
        
        return jsonify({
            'success': True,
            'metrics': {
                'total_loans': total_loans,
                'total_exposure_usd': total_exposure,
                'total_defaults': total_defaults,
                'overall_default_rate': overall_default_rate,
                'avg_interest_rate': avg_interest_rate,
                'avg_credit_score': avg_credit_score
            },
            'grades_data': grades_data,
            'recent_evaluations': recent_evals
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model_metrics', methods=['GET'])
def api_model_metrics():
    try:
        metrics_file = os.path.join(BASE_DIR, 'models', 'evaluation_results.json')
        meta_file = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
        
        with open(metrics_file, 'r') as f:
            eval_data = json.load(f)
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
            
        return jsonify({
            'success': True,
            'models': eval_data,
            'metadata': metadata
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sql_query', methods=['POST'])
def api_sql_query():
    try:
        data = request.get_json(force=True)
        query = data.get('query', '').strip()
        
        # Security: only permit SELECT queries
        if not query.lower().startswith('select'):
            return jsonify({'success': False, 'error': 'Only SELECT analytic queries are permitted.'}), 400
            
        start_time = time.time()
        conn = get_db_connection()
        df = pd.read_sql_query(query, conn)
        conn.close()
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        
        # Format results
        columns = df.columns.tolist()
        rows = df.head(100).to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'columns': columns,
            'rows': rows,
            'row_count': len(df),
            'execution_time_ms': elapsed_ms
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stress_test', methods=['POST'])
def api_stress_test():
    """Simulates economic stress scenarios on loan portfolio with vectorized batch scoring."""
    try:
        data = request.get_json(force=True)
        interest_rate_delta = float(data.get('interest_rate_delta', 2.0))
        income_shock_pct = float(data.get('income_shock_pct', -10.0))
        dti_shock_pct = float(data.get('dti_shock_pct', 15.0))
        
        df = PORTFOLIO_CACHE_DF.copy()
        
        # Base metrics
        base_defaults = int(df['loan_status'].sum())
        base_default_rate = round((base_defaults / len(df)) * 100.0, 2)
        base_loss_usd = float(df[df['loan_status'] == 1]['loan_amount'].sum())
        
        # Apply shocks
        df['annual_income'] = df['annual_income'] * (1.0 + (income_shock_pct / 100.0))
        df['interest_rate'] = df['interest_rate'] + interest_rate_delta
        df['dti'] = df['dti'] * (1.0 + (dti_shock_pct / 100.0))
        
        # Vectorized feature engineering
        df['loan_to_income'] = df['loan_amount'] / np.maximum(df['annual_income'], 1000.0)
        
        monthly_r = (df['interest_rate'] / 100.0) / 12.0
        n_term = df['loan_term_months']
        installments = df['loan_amount'] * monthly_r * ((1 + monthly_r) ** n_term) / (((1 + monthly_r) ** n_term) - 1)
        df['monthly_installment'] = np.round(installments, 2)
        df['payment_to_income'] = df['monthly_installment'] / (np.maximum(df['annual_income'], 1000.0) / 12.0)
        df['debt_util_stress'] = (df['dti'] * 0.5) + (df['revolving_utilization'] * 0.5)
        df['delinq_frequency'] = df['delinquencies_2yrs'] / np.maximum(df['credit_history_length'], 1.0)
        df['credit_tenure_ratio'] = df['credit_history_length'] / np.maximum(df['age'], 18.0)
        df['recent_inquiry_flag'] = (df['inquiries_last_6m'] >= 2).astype(int)
        
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
        df['grade_num'] = df['loan_grade'].map(grade_map).fillna(3)
        df['fico_grade_ratio'] = df['credit_score'] / (df['grade_num'] * 100.0)
        
        # Dummies
        df['home_ownership_OTHER'] = (df['home_ownership'] == 'OTHER').astype(int)
        df['home_ownership_OWN'] = (df['home_ownership'] == 'OWN').astype(int)
        df['home_ownership_RENT'] = (df['home_ownership'] == 'RENT').astype(int)
        
        df['loan_intent_EDUCATION'] = (df['loan_intent'] == 'EDUCATION').astype(int)
        df['loan_intent_HOME_IMPROVEMENT'] = (df['loan_intent'] == 'HOME_IMPROVEMENT').astype(int)
        df['loan_intent_MEDICAL'] = (df['loan_intent'] == 'MEDICAL').astype(int)
        df['loan_intent_SMALL_BUSINESS'] = (df['loan_intent'] == 'SMALL_BUSINESS').astype(int)
        df['loan_intent_DEBT_CONSOLIDATION'] = (df['loan_intent'] == 'DEBT_CONSOLIDATION').astype(int)
        
        for col in predictor.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        aligned_df = df[predictor.feature_cols]
        scaled_mat = predictor.scaler.transform(aligned_df)
        scaled_df = pd.DataFrame(scaled_mat, columns=predictor.feature_cols)
        
        stressed_probs = predictor.model.predict_proba(scaled_df)[:, 1]
        
        opt_thresh = predictor.metadata.get('optimal_threshold', 0.395)
        stressed_flags = stressed_probs >= opt_thresh
        stressed_defaults = int(np.sum(stressed_flags))
        stressed_default_rate = round((stressed_defaults / len(df)) * 100.0, 2)
        estimated_stress_loss_usd = float(df.loc[stressed_flags, 'loan_amount'].sum())
        
        return jsonify({
            'success': True,
            'portfolio_size': len(df),
            'base_default_rate': base_default_rate,
            'stressed_default_rate': stressed_default_rate,
            'delta_rate': round(stressed_default_rate - base_default_rate, 2),
            'base_loss_usd': round(base_loss_usd, 2),
            'estimated_stress_loss_usd': round(estimated_stress_loss_usd, 2),
            'loss_increase_usd': round(estimated_stress_loss_usd - base_loss_usd, 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/sample_applicants', methods=['GET'])
def api_sample_applicants():
    """Returns curated personas for 1-click risk simulations."""
    samples = [
        {
            'label': 'Prime Corporate Professional (Low Risk)',
            'applicant_id': 'LN-PRIME-001',
            'age': 44,
            'annual_income': 135000,
            'employment_length': 12,
            'home_ownership': 'MORTGAGE',
            'loan_intent': 'HOME_IMPROVEMENT',
            'loan_grade': 'A',
            'loan_amount': 25000,
            'loan_term_months': 36,
            'interest_rate': 7.5,
            'credit_score': 785,
            'dti': 12.5,
            'credit_history_length': 18,
            'revolving_utilization': 15.2,
            'delinquencies_2yrs': 0,
            'public_records': 0,
            'inquiries_last_6m': 0,
            'open_accounts': 14
        },
        {
            'label': 'Average Debt Consolidation Borrower (Moderate)',
            'applicant_id': 'LN-MED-002',
            'age': 34,
            'annual_income': 62000,
            'employment_length': 4,
            'home_ownership': 'RENT',
            'loan_intent': 'DEBT_CONSOLIDATION',
            'loan_grade': 'B',
            'loan_amount': 15000,
            'loan_term_months': 36,
            'interest_rate': 11.2,
            'credit_score': 695,
            'dti': 21.0,
            'credit_history_length': 9,
            'revolving_utilization': 45.0,
            'delinquencies_2yrs': 0,
            'public_records': 0,
            'inquiries_last_6m': 1,
            'open_accounts': 8
        },
        {
            'label': 'Stressed Borrower with Delinquencies (High Risk)',
            'applicant_id': 'LN-HIGH-003',
            'age': 28,
            'annual_income': 38000,
            'employment_length': 1,
            'home_ownership': 'RENT',
            'loan_intent': 'CREDIT_CARD',
            'loan_grade': 'D',
            'loan_amount': 18000,
            'loan_term_months': 60,
            'interest_rate': 18.5,
            'credit_score': 590,
            'dti': 38.5,
            'credit_history_length': 4,
            'revolving_utilization': 84.0,
            'delinquencies_2yrs': 2,
            'public_records': 1,
            'inquiries_last_6m': 4,
            'open_accounts': 6
        },
        {
            'label': 'Subprime Small Business Applicant (Severe)',
            'applicant_id': 'LN-CRIT-004',
            'age': 51,
            'annual_income': 42000,
            'employment_length': 2,
            'home_ownership': 'RENT',
            'loan_intent': 'SMALL_BUSINESS',
            'loan_grade': 'F',
            'loan_amount': 22000,
            'loan_term_months': 60,
            'interest_rate': 28.5,
            'credit_score': 540,
            'dti': 46.0,
            'credit_history_length': 15,
            'revolving_utilization': 92.5,
            'delinquencies_2yrs': 3,
            'public_records': 1,
            'inquiries_last_6m': 5,
            'open_accounts': 11
        }
    ]
    return jsonify({'success': True, 'samples': samples})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting Intelligent Loan Risk Decision Support Server on port {port}...")
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
