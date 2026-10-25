"""
Machine Learning Model Training & Comparative Evaluation
Trains Logistic Regression, Random Forest, and LightGBM / Gradient Boosting models.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
import lightgbm as lgb

from preprocessing import build_preprocessing_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def compute_optimal_threshold(y_true, y_probs, cost_fn=5.0, cost_fp=1.0):
    """
    Finds threshold that minimizes expected financial loss in retail lending:
    Cost of False Negative (Defaulting loan approved) >> Cost of False Positive (Good customer denied).
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_cost = float('inf')
    best_th = 0.50
    
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        cm = confusion_matrix(y_true, preds)
        # cm: [[TN, FP], [FN, TP]]
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            total_loss = (fn * cost_fn) + (fp * cost_fp)
            if total_loss < best_cost:
                best_cost = total_loss
                best_th = th
                
    return round(float(best_th), 3)

def train_and_evaluate_all():
    print("=" * 60)
    print("Executing Machine Learning Training Pipeline")
    print("=" * 60)
    
    data = build_preprocessing_pipeline(save_artifacts=True)
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_cols = data['feature_cols']
    
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=8,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.04, max_depth=6,
            num_leaves=31, random_state=42, class_weight='balanced', verbose=-1
        )
    }
    
    results = {}
    best_auc = -1.0
    best_model_name = ""
    best_model_obj = None
    
    for name, model in models.items():
        print(f"\nTraining [{name}]...")
        model.fit(X_train_scaled, y_train)
        
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Standard threshold metrics
        y_pred_std = (y_probs >= 0.50).astype(int)
        
        roc_auc = float(roc_auc_score(y_test, y_probs))
        pr_auc = float(average_precision_score(y_test, y_probs))
        brier = float(brier_score_loss(y_test, y_probs))
        
        # Cost-optimized banking threshold
        opt_th = compute_optimal_threshold(y_test, y_probs, cost_fn=4.0, cost_fp=1.0)
        y_pred_opt = (y_probs >= opt_th).astype(int)
        
        acc = float(accuracy_score(y_test, y_pred_opt))
        prec = float(precision_score(y_test, y_pred_opt))
        rec = float(recall_score(y_test, y_pred_opt))
        f1 = float(f1_score(y_test, y_pred_opt))
        cm = confusion_matrix(y_test, y_pred_opt).tolist()
        
        # Calculate ROC curve samples for charting
        fpr, tpr, roc_thresh = roc_curve(y_test, y_probs)
        # Subsample points for lightweight web transmission
        sample_indices = np.linspace(0, len(fpr) - 1, 35, dtype=int)
        roc_points = [
            {'fpr': round(float(fpr[i]), 4), 'tpr': round(float(tpr[i]), 4)}
            for i in sample_indices
        ]
        
        # Feature importances / coefficients
        feat_importances = {}
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            # Normalize to percentage
            norm_imps = (imps / np.sum(imps)) * 100.0
            for col, imp in sorted(zip(feature_cols, norm_imps), key=lambda x: x[1], reverse=True):
                feat_importances[col] = round(float(imp), 2)
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_[0])
            norm_coefs = (coefs / np.sum(coefs)) * 100.0
            for col, imp in sorted(zip(feature_cols, norm_coefs), key=lambda x: x[1], reverse=True):
                feat_importances[col] = round(float(imp), 2)
                
        results[name] = {
            'roc_auc': round(roc_auc, 4),
            'pr_auc': round(pr_auc, 4),
            'brier_score': round(brier, 4),
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'optimal_threshold': opt_th,
            'confusion_matrix': cm,
            'roc_points': roc_points,
            'top_features': dict(list(feat_importances.items())[:10])
        }
        
        print(f"[{name}] Results -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Optimal Threshold: {opt_th} | Recall: {rec:.4f} | Precision: {prec:.4f}")
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = name
            best_model_obj = model
            
    print("\n" + "=" * 60)
    print(f"Champion Model Selected: [{best_model_name}] with ROC-AUC: {best_auc:.4f}")
    print("=" * 60)
    
    # Save Champion Model
    champion_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(best_model_obj, champion_path)
    print(f"Champion model saved to: {champion_path}")
    
    # Save Metadata and Evaluation JSON
    metadata = {
        'champion_model': best_model_name,
        'roc_auc': results[best_model_name]['roc_auc'],
        'optimal_threshold': results[best_model_name]['optimal_threshold'],
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'trained_samples': len(X_train_scaled),
        'test_samples': len(X_test_scaled),
        'top_features': results[best_model_name]['top_features']
    }
    
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    with open(os.path.join(MODELS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("All training metrics and artifacts persisted.")
    return results

if __name__ == '__main__':
    train_and_evaluate_all()
