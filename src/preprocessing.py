"""
Data Preprocessing & Feature Engineering Pipeline
Handles financial ratios, categorical encoding, scaling, and train-test splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'loan_data_raw.csv')
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'loan_data_processed.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Grade ordinal mapping: A=1, B=2, C=3, D=4, E=5, F=6
GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

def calculate_monthly_installment(loan_amount, annual_interest_rate, term_months):
    """Computes fixed monthly amortized installment amount."""
    monthly_rate = (annual_interest_rate / 100.0) / 12.0
    if monthly_rate == 0:
        return loan_amount / term_months
    numerator = loan_amount * monthly_rate * ((1 + monthly_rate) ** term_months)
    denominator = ((1 + monthly_rate) ** term_months) - 1
    return numerator / denominator

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes domain-specific financial risk ratios."""
    df_feat = df.copy()
    
    # 1. Loan to Annual Income Ratio (leverage)
    df_feat['loan_to_income'] = df_feat['loan_amount'] / np.maximum(df_feat['annual_income'], 1000.0)
    
    # 2. Estimated Monthly Installment & Payment-to-Monthly-Income (PTI)
    installments = [
        calculate_monthly_installment(amt, rate, term)
        for amt, rate, term in zip(df_feat['loan_amount'], df_feat['interest_rate'], df_feat['loan_term_months'])
    ]
    df_feat['monthly_installment'] = np.round(installments, 2)
    df_feat['payment_to_income'] = df_feat['monthly_installment'] / (np.maximum(df_feat['annual_income'], 1000.0) / 12.0)
    
    # 3. Credit Utilization and DTI combined stress factor
    df_feat['debt_util_stress'] = (df_feat['dti'] * 0.5) + (df_feat['revolving_utilization'] * 0.5)
    
    # 4. Delinquency frequency relative to credit history tenure
    df_feat['delinq_frequency'] = df_feat['delinquencies_2yrs'] / np.maximum(df_feat['credit_history_length'], 1.0)
    
    # 5. Credit age relative to total applicant age
    df_feat['credit_tenure_ratio'] = df_feat['credit_history_length'] / np.maximum(df_feat['age'], 18.0)
    
    # 6. High risk inquiry flag
    df_feat['recent_inquiry_flag'] = (df_feat['inquiries_last_6m'] >= 2).astype(int)
    
    # 7. Credit Score to Grade divergence
    df_feat['grade_num'] = df_feat['loan_grade'].map(GRADE_MAP).fillna(3)
    df_feat['fico_grade_ratio'] = df_feat['credit_score'] / (df_feat['grade_num'] * 100.0)

    return df_feat

def build_preprocessing_pipeline(save_artifacts: bool = True):
    print("Loading raw data from:", DATA_RAW_PATH)
    df = pd.read_csv(DATA_RAW_PATH)
    
    # Feature Engineering
    df_eng = engineer_features(df)
    
    # Categorical columns to dummy encode
    cat_cols = ['home_ownership', 'loan_intent']
    df_encoded = pd.get_dummies(df_eng, columns=cat_cols, drop_first=True, dtype=int)
    
    # Define feature lists
    id_cols = ['applicant_id', 'default_probability_ground_truth', 'loan_status']
    feature_cols = [c for c in df_encoded.columns if c not in id_cols and c != 'loan_grade']
    
    X = df_encoded[feature_cols].copy()
    y = df_encoded['loan_status'].copy()
    
    print(f"Total Features engineered: {len(feature_cols)}")
    print("Features list:", feature_cols)
    
    # Train-test split (80-20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    if save_artifacts:
        os.makedirs(MODELS_DIR, exist_ok=True)
        # Save Scaler and Feature Names
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        # Save engineered dataset
        df_encoded.to_csv(DATA_PROCESSED_PATH, index=False)
        print(f"Processed dataset saved to {DATA_PROCESSED_PATH}")
        print("Preprocessing artifacts saved successfully.")
        
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled_df,
        'X_test_scaled': X_test_scaled_df,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

def transform_single_applicant(applicant_dict: dict, scaler=None, feature_cols=None) -> np.ndarray:
    """Prepares a single applicant record from dictionary into scaled feature vector."""
    if scaler is None:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    if feature_cols is None:
        feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
    df_raw = pd.DataFrame([applicant_dict])
    df_eng = engineer_features(df_raw)
    
    cat_cols = ['home_ownership', 'loan_intent']
    df_encoded = pd.get_dummies(df_eng, columns=cat_cols, drop_first=False, dtype=int)
    
    # Align columns to feature_cols
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_aligned = df_encoded[feature_cols]
    scaled_vector = scaler.transform(df_aligned)
    return scaled_vector, df_aligned

if __name__ == '__main__':
    data_dict = build_preprocessing_pipeline(save_artifacts=True)
    print("Preprocessing pipeline verification completed.")
