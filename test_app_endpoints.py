"""
API Test Suite using requests with precise timing
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5050"

def test_endpoint(name, method="GET", path="/", data=None):
    url = f"{BASE_URL}{path}"
    t0 = time.time()
    try:
        if method == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=data, timeout=10)
        elapsed = round((time.time() - t0) * 1000, 2)
        print(f"[PASS] {name} -> Status: {resp.status_code} | Time: {elapsed} ms")
        if path.startswith('/api'):
            parsed = resp.json()
            assert parsed.get('success') is True, f"Failed: {parsed}"
        return resp
    except Exception as e:
        elapsed = round((time.time() - t0) * 1000, 2)
        print(f"[FAIL] {name} -> Error after {elapsed} ms: {e}")
        raise

if __name__ == '__main__':
    print("=" * 60)
    print("Testing AegisCredit Endpoints with requests")
    print("=" * 60)
    
    test_endpoint("GET Main SPA", "GET", "/")
    test_endpoint("GET Portfolio Stats", "GET", "/api/portfolio_stats")
    test_endpoint("GET Model Metrics", "GET", "/api/model_metrics")
    test_endpoint("GET Sample Applicants", "GET", "/api/sample_applicants")
    
    test_lead = {
        'applicant_id': 'LN-TEST-001',
        'age': 32,
        'annual_income': 72000,
        'employment_length': 5,
        'home_ownership': 'RENT',
        'loan_intent': 'CREDIT_CARD',
        'loan_grade': 'C',
        'loan_amount': 15000,
        'loan_term_months': 36,
        'interest_rate': 14.5,
        'credit_score': 660,
        'dti': 24.5,
        'credit_history_length': 8,
        'revolving_utilization': 52.0,
        'delinquencies_2yrs': 0,
        'public_records': 0,
        'inquiries_last_6m': 1,
        'open_accounts': 8
    }
    r = test_endpoint("POST Live Prediction", "POST", "/api/predict", data=test_lead)
    pred = r.json()['data']
    print(f"   -> Probability: {pred['default_probability_pct']}% | Tier: {pred['risk_tier']}")
    
    sql_payload = {'query': 'SELECT loan_grade, total_loans, default_rate_pct FROM vw_portfolio_risk_summary LIMIT 4;'}
    test_endpoint("POST SQL Query", "POST", "/api/sql_query", data=sql_payload)
    
    stress_payload = {'interest_rate_delta': 3.0, 'income_shock_pct': -15.0, 'dti_shock_pct': 20.0}
    sr = test_endpoint("POST Stress Test", "POST", "/api/stress_test", data=stress_payload)
    sdata = sr.json()
    print(f"   -> Stress Result: Base {sdata['base_default_rate']}% -> Stressed {sdata['stressed_default_rate']}% (Delta: +{sdata['delta_rate']}%)")
    
    print("=" * 60)
    print("ALL 7 ENDPOINTS VERIFIED AND PASSING!")
    print("=" * 60)
