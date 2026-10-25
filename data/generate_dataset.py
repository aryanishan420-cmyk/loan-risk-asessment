"""
Synthetic Loan Application Dataset Generator
Generates realistic financial and credit history data mimicking real-world retail banking datasets (such as LendingClub/Fannie Mae).
"""

import numpy as np
import pandas as pd
import os

def generate_loan_dataset(n_samples: int = 12000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    applicant_ids = [f"LN-2026-{10000 + i}" for i in range(n_samples)]
    
    # Demographics and Employment
    age = np.random.normal(loc=41, scale=11.5, size=n_samples).clip(21, 72).astype(int)
    
    employment_lengths = np.random.choice(
        [0, 1, 2, 3, 5, 7, 10, 15, 20],
        size=n_samples,
        p=[0.08, 0.10, 0.12, 0.15, 0.18, 0.15, 0.12, 0.06, 0.04]
    )
    
    home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
        size=n_samples,
        p=[0.42, 0.46, 0.11, 0.01]
    )
    
    annual_income = np.random.lognormal(mean=11.0, sigma=0.55, size=n_samples)
    annual_income = np.round(annual_income.clip(18000, 350000), 2)
    
    loan_intent = np.random.choice(
        ['DEBT_CONSOLIDATION', 'CREDIT_CARD', 'HOME_IMPROVEMENT', 'SMALL_BUSINESS', 'MEDICAL', 'EDUCATION'],
        size=n_samples,
        p=[0.45, 0.22, 0.13, 0.08, 0.07, 0.05]
    )
    
    loan_grade = np.random.choice(
        ['A', 'B', 'C', 'D', 'E', 'F'],
        size=n_samples,
        p=[0.20, 0.30, 0.25, 0.15, 0.07, 0.03]
    )
    
    # Grade-dependent interest rates and base risk
    grade_rate_map = {
        'A': (6.5, 9.5),
        'B': (9.5, 12.8),
        'C': (12.8, 16.5),
        'D': (16.5, 21.0),
        'E': (21.0, 26.0),
        'F': (26.0, 32.0)
    }
    
    int_rates = []
    for g in loan_grade:
        low, high = grade_rate_map[g]
        int_rates.append(np.round(np.random.uniform(low, high), 2))
    int_rates = np.array(int_rates)
    
    # Loan Amount proportional to income but capped
    loan_amounts = []
    for inc in annual_income:
        max_possible = min(inc * 0.55, 45000)
        min_possible = 2000
        val = np.random.triangular(min_possible, min_possible + 0.35 * (max_possible - min_possible), max_possible)
        loan_amounts.append(np.round(val, -2))
    loan_amounts = np.array(loan_amounts)
    
    loan_term = np.random.choice([36, 60], size=n_samples, p=[0.72, 0.28])
    
    # Credit History and Bureau Metrics
    # FICO Credit Score: 300 to 850
    # Positively correlates with income and age, inversely with loan grade
    grade_fico_means = {'A': 750, 'B': 710, 'C': 670, 'D': 635, 'E': 600, 'F': 565}
    credit_scores = []
    for g in loan_grade:
        mean_s = grade_fico_means[g]
        score = int(np.clip(np.random.normal(loc=mean_s, scale=28), 500, 840))
        credit_scores.append(score)
    credit_scores = np.array(credit_scores)
    
    # Debt-To-Income (DTI) ratio (percentage)
    dti = np.random.gamma(shape=3.5, scale=4.8, size=n_samples).clip(3.0, 58.0)
    dti = np.round(dti, 2)
    
    # Credit history length in years
    cb_person_cred_hist_length = (age - 18 - np.random.uniform(0, 5, size=n_samples)).clip(1, 45).astype(int)
    
    # Revolving balance utilization (0 to 100%)
    revolving_util = np.random.beta(a=2.5, b=3.5, size=n_samples) * 100
    revolving_util = np.round(revolving_util.clip(1.0, 99.5), 2)
    
    # Past Delinquencies in last 2 years
    delinq_2yrs = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.76, 0.15, 0.05, 0.03, 0.01])
    
    # Derogatory Public Records (bankruptcies, liens)
    public_records = np.random.choice([0, 1, 2], size=n_samples, p=[0.88, 0.10, 0.02])
    
    # Inquiries in last 6 months
    inquiries_last_6m = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.48, 0.26, 0.13, 0.07, 0.04, 0.02])
    
    # Total Open Accounts
    open_acc = np.random.poisson(lam=11, size=n_samples).clip(2, 38)
    
    # Calculate Ground-Truth Default Probability using a calibrated Latent Financial Stress Index
    # Financial theory: Log-odds of default increases with DTI, Interest Rate, Revolving Util, Delinquencies,
    # and decreases with Credit Score, Annual Income, Employment Length, and Age.
    z = (
        - 4.2
        + 0.045 * dti
        + 0.10 * int_rates
        + 0.022 * revolving_util
        + 0.42 * delinq_2yrs
        + 0.55 * public_records
        + 0.18 * inquiries_last_6m
        - 0.0075 * (credit_scores - 600)
        - 0.35 * np.log(annual_income / 15000)
        - 0.035 * employment_lengths
        + 0.000025 * loan_amounts
        + (loan_term == 60) * 0.35
    )
    
    default_prob = 1.0 / (1.0 + np.exp(-z))
    # Default event (1 = Defaulted/Charged-Off, 0 = Fully Paid/Current)
    loan_status = (np.random.uniform(0, 1, size=n_samples) < default_prob).astype(int)
    
    df = pd.DataFrame({
        'applicant_id': applicant_ids,
        'age': age,
        'annual_income': annual_income,
        'employment_length': employment_lengths,
        'home_ownership': home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amount': loan_amounts,
        'loan_term_months': loan_term,
        'interest_rate': int_rates,
        'credit_score': credit_scores,
        'dti': dti,
        'credit_history_length': cb_person_cred_hist_length,
        'revolving_utilization': revolving_util,
        'delinquencies_2yrs': delinq_2yrs,
        'public_records': public_records,
        'inquiries_last_6m': inquiries_last_6m,
        'open_accounts': open_acc,
        'default_probability_ground_truth': np.round(default_prob, 4),
        'loan_status': loan_status  # Target variable
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'loan_data_raw.csv')
    print("Generating synthetic loan risk dataset (12,000 records)...")
    df = generate_loan_dataset(n_samples=12000, random_seed=42)
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully created at: {out_path}")
    print(f"Total Rows: {len(df)}, Features: {df.shape[1]}")
    print(f"Overall Default Rate: {df['loan_status'].mean():.2%}")
    print(df[['loan_grade', 'credit_score', 'dti', 'annual_income', 'loan_status']].head())
