"""
Loan Risk Inference Engine & Decision Support System
Computes default probability, classifies risk tier, recommends loan terms,
and generates explainable risk factor attribution.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class LoanRiskPredictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        self.feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
            self.metadata = json.load(f)

    def _engineer_row(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        
        # Financial ratios
        loan_amt = float(data.get('loan_amount', 10000))
        income = max(float(data.get('annual_income', 50000)), 1000.0)
        rate = float(data.get('interest_rate', 12.0))
        term = int(data.get('loan_term_months', 36))
        
        df['loan_to_income'] = loan_amt / income
        
        monthly_rate = (rate / 100.0) / 12.0
        if monthly_rate == 0:
            installment = loan_amt / term
        else:
            installment = loan_amt * monthly_rate * ((1 + monthly_rate) ** term) / (((1 + monthly_rate) ** term) - 1)
            
        df['monthly_installment'] = round(installment, 2)
        df['payment_to_income'] = installment / (income / 12.0)
        df['debt_util_stress'] = (float(data.get('dti', 15.0)) * 0.5) + (float(data.get('revolving_utilization', 30.0)) * 0.5)
        df['delinq_frequency'] = float(data.get('delinquencies_2yrs', 0)) / max(float(data.get('credit_history_length', 5)), 1.0)
        df['credit_tenure_ratio'] = float(data.get('credit_history_length', 5)) / max(float(data.get('age', 35)), 18.0)
        df['recent_inquiry_flag'] = int(float(data.get('inquiries_last_6m', 0)) >= 2)
        
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
        grade_num = grade_map.get(str(data.get('loan_grade', 'B')).upper(), 2)
        df['grade_num'] = grade_num
        df['fico_grade_ratio'] = float(data.get('credit_score', 700)) / (grade_num * 100.0)
        
        # Categorical dummies
        home_own = str(data.get('home_ownership', 'MORTGAGE')).upper()
        df['home_ownership_OTHER'] = 1 if home_own == 'OTHER' else 0
        df['home_ownership_OWN'] = 1 if home_own == 'OWN' else 0
        df['home_ownership_RENT'] = 1 if home_own == 'RENT' else 0
        
        intent = str(data.get('loan_intent', 'DEBT_CONSOLIDATION')).upper()
        df['loan_intent_EDUCATION'] = 1 if intent == 'EDUCATION' else 0
        df['loan_intent_HOME_IMPROVEMENT'] = 1 if intent == 'HOME_IMPROVEMENT' else 0
        df['loan_intent_MEDICAL'] = 1 if intent == 'MEDICAL' else 0
        df['loan_intent_SMALL_BUSINESS'] = 1 if intent == 'SMALL_BUSINESS' else 0
        df['loan_intent_DEBT_CONSOLIDATION'] = 1 if intent == 'DEBT_CONSOLIDATION' else 0
        
        # Fill missing features with 0
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        return df[self.feature_cols]

    def predict(self, applicant: dict) -> dict:
        df_row = self._engineer_row(applicant)
        scaled_array = self.scaler.transform(df_row)
        scaled_df = pd.DataFrame(scaled_array, columns=self.feature_cols)
        
        # Probability of default
        prob_default = float(self.model.predict_proba(scaled_df)[0][1])
        prob_repay = 1.0 - prob_default
        
        # Determine risk tier & underwriting decision
        if prob_default < 0.12:
            risk_tier = "Tier 1: Prime Low Risk"
            decision = "AUTO-APPROVE"
            decision_code = "APPROVE"
            rate_adjustment = -0.75
            max_recommended_amount = float(applicant.get('loan_amount', 10000))
        elif prob_default < 0.25:
            risk_tier = "Tier 2: Near Prime Moderate"
            decision = "APPROVE WITH STANDARD TERMS"
            decision_code = "APPROVE"
            rate_adjustment = 0.0
            max_recommended_amount = float(applicant.get('loan_amount', 10000))
        elif prob_default < 0.42:
            risk_tier = "Tier 3: Elevated Risk"
            decision = "REFER TO MANUAL UNDERWRITING"
            decision_code = "REVIEW"
            rate_adjustment = +2.50
            max_recommended_amount = round(float(applicant.get('loan_amount', 10000)) * 0.75, -2)
        else:
            risk_tier = "Tier 4: Severe Default Risk"
            decision = "ADVERSE ACTION / REJECT"
            decision_code = "REJECT"
            rate_adjustment = +5.00
            max_recommended_amount = 0.0
            
        base_rate = float(applicant.get('interest_rate', 12.0))
        offered_rate = max(round(base_rate + rate_adjustment, 2), 4.5)
        
        # Explainable factor analysis (attribution)
        factors = []
        dti = float(applicant.get('dti', 15))
        fico = float(applicant.get('credit_score', 700))
        util = float(applicant.get('revolving_utilization', 30))
        delinq = int(applicant.get('delinquencies_2yrs', 0))
        pub_rec = int(applicant.get('public_records', 0))
        inq = int(applicant.get('inquiries_last_6m', 0))
        income = float(applicant.get('annual_income', 50000))
        emp_len = int(applicant.get('employment_length', 5))
        
        # Credit Score impact
        if fico >= 740:
            factors.append({'factor': 'Excellent Credit Score (FICO > 740)', 'impact': 'positive', 'weight': -0.15})
        elif fico < 620:
            factors.append({'factor': 'Subprime Credit Score (FICO < 620)', 'impact': 'negative', 'weight': +0.22})
            
        # DTI impact
        if dti > 35.0:
            factors.append({'factor': f'High Debt-to-Income Ratio ({dti}%)', 'impact': 'negative', 'weight': +0.18})
        elif dti < 15.0:
            factors.append({'factor': f'Healthy Debt-to-Income Ratio ({dti}%)', 'impact': 'positive', 'weight': -0.10})
            
        # Utilization
        if util > 65.0:
            factors.append({'factor': f'High Revolving Line Utilization ({util}%)', 'impact': 'negative', 'weight': +0.14})
        elif util < 25.0:
            factors.append({'factor': f'Conservative Revolving Utilization ({util}%)', 'impact': 'positive', 'weight': -0.08})
            
        # Delinquencies
        if delinq > 0:
            factors.append({'factor': f'Recent Delinquencies ({delinq} past due)', 'impact': 'negative', 'weight': +0.16 * delinq})
            
        # Public records
        if pub_rec > 0:
            factors.append({'factor': f'Derogatory Public Record/Bankruptcy ({pub_rec})', 'impact': 'negative', 'weight': +0.25 * pub_rec})
            
        # Employment tenure
        if emp_len >= 8:
            factors.append({'factor': f'Stable Employment Tenure ({emp_len} years)', 'impact': 'positive', 'weight': -0.09})
        elif emp_len <= 1:
            factors.append({'factor': f'Limited Employment History ({emp_len} years)', 'impact': 'negative', 'weight': +0.07})
            
        # Recent hard inquiries
        if inq >= 3:
            factors.append({'factor': f'Multiple Recent Hard Inquiries ({inq} in 6m)', 'impact': 'negative', 'weight': +0.11})
            
        if not factors:
            factors.append({'factor': 'Balanced Financial and Credit Metrics', 'impact': 'neutral', 'weight': 0.0})
            
        return {
            'default_probability': round(prob_default, 4),
            'default_probability_pct': round(prob_default * 100.0, 2),
            'repayment_probability_pct': round(prob_repay * 100.0, 2),
            'risk_tier': risk_tier,
            'decision': decision,
            'decision_code': decision_code,
            'recommended_rate': offered_rate,
            'max_approved_amount': max_recommended_amount,
            'risk_factors': factors,
            'model_name': self.metadata.get('champion_model', 'LightGBM')
        }

if __name__ == '__main__':
    predictor = LoanRiskPredictor()
    test_applicant = {
        'age': 38,
        'annual_income': 85000,
        'employment_length': 6,
        'home_ownership': 'MORTGAGE',
        'loan_intent': 'DEBT_CONSOLIDATION',
        'loan_grade': 'B',
        'loan_amount': 18000,
        'loan_term_months': 36,
        'interest_rate': 10.5,
        'credit_score': 725,
        'dti': 18.5,
        'credit_history_length': 12,
        'revolving_utilization': 32.0,
        'delinquencies_2yrs': 0,
        'public_records': 0,
        'inquiries_last_6m': 1,
        'open_accounts': 9
    }
    res = predictor.predict(test_applicant)
    print("Inference test result:")
    print(json.dumps(res, indent=2))
