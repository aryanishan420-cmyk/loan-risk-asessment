-- Intelligent Loan Risk Assessment Relational Database Schema
-- SQLite3 compliant DDL

-- 1. Applicants Table: Demographic and Employment profile
CREATE TABLE IF NOT EXISTS applicants (
    applicant_id VARCHAR(32) PRIMARY KEY,
    age INTEGER NOT NULL CHECK (age >= 18),
    annual_income REAL NOT NULL CHECK (annual_income > 0),
    employment_length INTEGER NOT NULL CHECK (employment_length >= 0),
    home_ownership VARCHAR(20) NOT NULL CHECK (home_ownership IN ('RENT', 'MORTGAGE', 'OWN', 'OTHER')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Loans Table: Specific loan facilities requested
CREATE TABLE IF NOT EXISTS loans (
    loan_id VARCHAR(32) PRIMARY KEY,
    applicant_id VARCHAR(32) NOT NULL,
    loan_amount REAL NOT NULL CHECK (loan_amount > 0),
    loan_term_months INTEGER NOT NULL CHECK (loan_term_months IN (36, 60)),
    loan_intent VARCHAR(40) NOT NULL,
    loan_grade VARCHAR(2) NOT NULL CHECK (loan_grade IN ('A', 'B', 'C', 'D', 'E', 'F')),
    interest_rate REAL NOT NULL,
    loan_status INTEGER NOT NULL CHECK (loan_status IN (0, 1)), -- 0: Current/Paid, 1: Default
    FOREIGN KEY (applicant_id) REFERENCES applicants(applicant_id) ON DELETE CASCADE
);

-- 3. Credit Profiles: Bureau metrics & risk factors
CREATE TABLE IF NOT EXISTS credit_profiles (
    applicant_id VARCHAR(32) PRIMARY KEY,
    credit_score INTEGER NOT NULL CHECK (credit_score BETWEEN 300 AND 850),
    dti REAL NOT NULL,
    credit_history_length INTEGER NOT NULL,
    revolving_utilization REAL NOT NULL,
    delinquencies_2yrs INTEGER NOT NULL,
    public_records INTEGER NOT NULL,
    inquiries_last_6m INTEGER NOT NULL,
    open_accounts INTEGER NOT NULL,
    FOREIGN KEY (applicant_id) REFERENCES applicants(applicant_id) ON DELETE CASCADE
);

-- 4. Risk Model Evaluations: Automated inference audit trail
CREATE TABLE IF NOT EXISTS risk_evaluations (
    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id VARCHAR(32) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    default_probability REAL NOT NULL,
    risk_tier VARCHAR(20) NOT NULL,
    recommended_decision VARCHAR(30) NOT NULL,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (applicant_id) REFERENCES applicants(applicant_id)
);

-- Indexes for high-throughput queries
CREATE INDEX IF NOT EXISTS idx_loans_grade ON loans(loan_grade);
CREATE INDEX IF NOT EXISTS idx_loans_status ON loans(loan_status);
CREATE INDEX IF NOT EXISTS idx_credit_score ON credit_profiles(credit_score);
CREATE INDEX IF NOT EXISTS idx_credit_dti ON credit_profiles(dti);
CREATE INDEX IF NOT EXISTS idx_risk_eval_tier ON risk_evaluations(risk_tier);

-- Analytical View 1: Loan Grade Portfolio Risk Analysis
CREATE VIEW IF NOT EXISTS vw_portfolio_risk_summary AS
SELECT 
    l.loan_grade,
    COUNT(l.loan_id) AS total_loans,
    ROUND(SUM(l.loan_amount), 2) AS total_exposure_usd,
    ROUND(AVG(l.loan_amount), 2) AS avg_loan_size,
    ROUND(AVG(l.interest_rate), 2) AS avg_interest_rate,
    ROUND(AVG(c.credit_score), 1) AS avg_credit_score,
    ROUND(AVG(c.dti), 2) AS avg_dti,
    SUM(l.loan_status) AS default_count,
    ROUND(AVG(l.loan_status) * 100.0, 2) AS default_rate_pct
FROM loans l
JOIN credit_profiles c ON l.applicant_id = c.applicant_id
GROUP BY l.loan_grade
ORDER BY l.loan_grade ASC;

-- Analytical View 2: High Risk Underwriting Watchlist
CREATE VIEW IF NOT EXISTS vw_high_risk_pipeline AS
SELECT 
    a.applicant_id,
    a.annual_income,
    l.loan_amount,
    l.loan_grade,
    c.credit_score,
    c.dti,
    c.revolving_utilization,
    c.delinquencies_2yrs,
    CASE 
        WHEN c.credit_score < 600 OR c.dti > 40.0 OR c.delinquencies_2yrs > 1 THEN 'HIGH RISK CRITERIA'
        ELSE 'STANDARD'
    END AS risk_flag
FROM applicants a
JOIN loans l ON a.applicant_id = l.applicant_id
JOIN credit_profiles c ON a.applicant_id = c.applicant_id
WHERE c.credit_score < 620 OR c.dti > 35.0;
