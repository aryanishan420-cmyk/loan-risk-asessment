"""
Master Pipeline Orchestrator
Executes the entire end-to-end Machine Learning and Database Pipeline.
"""

import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))
sys.path.append(os.path.join(BASE_DIR, 'database'))
sys.path.append(os.path.join(BASE_DIR, 'data'))

from generate_dataset import generate_loan_dataset
from init_db import init_database
from train_models import train_and_evaluate_all
from predict import LoanRiskPredictor

def run_master_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("  INTELLIGENT LOAN RISK ASSESSMENT - END-TO-END PIPELINE")
    print("=" * 70)
    
    # 1. Data Generation
    print("\n[Step 1/4] Generating Synthetic Financial Loan Dataset...")
    raw_csv = os.path.join(BASE_DIR, 'data', 'loan_data_raw.csv')
    df = generate_loan_dataset(n_samples=12000, random_seed=42)
    df.to_csv(raw_csv, index=False)
    print(f" -> Generated 12,000 credit records at: {raw_csv}")
    
    # 2. Database Initialization
    print("\n[Step 2/4] Initializing Relational SQLite Database & Analytical Views...")
    init_database(populate=True)
    print(" -> SQLite schema, normalized tables & analytical views ready.")
    
    # 3. Preprocessing, Feature Engineering & ML Model Training
    print("\n[Step 3/4] Training ML Models (Logistic Regression, Random Forest, LightGBM)...")
    results = train_and_evaluate_all()
    print(" -> Model artifacts, scalers, and evaluation metrics persisted.")
    
    # 4. Pipeline Verification
    print("\n[Step 4/4] Validating Inference Decision Engine...")
    predictor = LoanRiskPredictor()
    sample_lead = {
        'age': 45,
        'annual_income': 95000,
        'employment_length': 8,
        'home_ownership': 'OWN',
        'loan_intent': 'HOME_IMPROVEMENT',
        'loan_grade': 'A',
        'loan_amount': 25000,
        'loan_term_months': 36,
        'interest_rate': 7.5,
        'credit_score': 770,
        'dti': 14.2,
        'credit_history_length': 18,
        'revolving_utilization': 18.5,
        'delinquencies_2yrs': 0,
        'public_records': 0,
        'inquiries_last_6m': 0,
        'open_accounts': 12
    }
    decision = predictor.predict(sample_lead)
    print(f" -> Sample Lead Evaluation: {decision['risk_tier']} | Decision: {decision['decision']}")
    print(f" -> Predicted Default Probability: {decision['default_probability_pct']}%")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  END-TO-END PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
    print("=" * 70)

if __name__ == '__main__':
    run_master_pipeline()
