"""
Database Initialization and Management Script
Loads raw loan dataset into normalized SQLite tables and validates analytical queries.
"""

import sqlite3
import pandas as pd
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'loan_risk.db')
SCHEMA_PATH = os.path.join(BASE_DIR, 'database', 'schema.sql')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'loan_data_raw.csv')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database(populate: bool = True):
    print(f"Initializing database at: {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(schema_sql)
    conn.commit()
    print("Database schema created successfully.")
    
    if populate and os.path.exists(RAW_DATA_PATH):
        print(f"Populating database from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH)
        
        # 1. Applicants DataFrame
        applicants_df = df[['applicant_id', 'age', 'annual_income', 'employment_length', 'home_ownership']].copy()
        applicants_df.drop_duplicates(subset=['applicant_id'], inplace=True)
        applicants_df.to_sql('applicants', conn, if_exists='replace', index=False)
        
        # 2. Loans DataFrame
        loans_df = df[['applicant_id', 'loan_amount', 'loan_term_months', 'loan_intent', 'loan_grade', 'interest_rate', 'loan_status']].copy()
        loans_df['loan_id'] = [f"LN-FAC-{i+1:06d}" for i in range(len(loans_df))]
        cols = ['loan_id', 'applicant_id', 'loan_amount', 'loan_term_months', 'loan_intent', 'loan_grade', 'interest_rate', 'loan_status']
        loans_df = loans_df[cols]
        loans_df.to_sql('loans', conn, if_exists='replace', index=False)
        
        # 3. Credit Profiles DataFrame
        credit_df = df[['applicant_id', 'credit_score', 'dti', 'credit_history_length', 
                        'revolving_utilization', 'delinquencies_2yrs', 'public_records', 
                        'inquiries_last_6m', 'open_accounts']].copy()
        credit_df.to_sql('credit_profiles', conn, if_exists='replace', index=False)
        
        conn.commit()
        print(f"Successfully inserted {len(applicants_df)} applicants, {len(loans_df)} loans, and {len(credit_df)} credit profiles.")
        
    # Re-apply schema views and indexes if tables were replaced
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()

def run_analytical_query(query: str):
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def log_prediction_audit(applicant_id: str, model_version: str, default_prob: float, risk_tier: str, decision: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO risk_evaluations (applicant_id, model_version, default_probability, risk_tier, recommended_decision)
            VALUES (?, ?, ?, ?, ?)
        """, (applicant_id, model_version, round(float(default_prob), 4), risk_tier, decision))
        conn.commit()
    finally:
        conn.close()

if __name__ == '__main__':
    init_database()
    print("\n--- Testing Portfolio Risk Summary View ---")
    summary = run_analytical_query("SELECT * FROM vw_portfolio_risk_summary;")
    print(summary.to_string(index=False))
