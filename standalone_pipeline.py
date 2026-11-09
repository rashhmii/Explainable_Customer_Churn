"""
Standalone End-to-End Pipeline
Project: Explainable Customer Churn Prediction Using Data Analytics
Person 1: Data Science & Machine Learning Deliverables
Center for Data Analytics & Digital Intelligence
Center Lead: Dr Chatrapathy K | Faculty Guide: Dr Ragaventhiran J
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import shap

# ==========================================
# 1. DATA INGESTION & PREPROCESSING
# ==========================================
def load_and_preprocess(filepath: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(filepath)

    # Coerce TotalCharges blanks to NaN and float
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True),
            errors='coerce'
        )

    customer_ids = df['customerID'] if 'customerID' in df.columns else pd.Series(df.index, name='customerID')

    target_col = 'Churn'
    if target_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            y = df[target_col].astype(int)
        else:
            y = df[target_col].astype(str).str.strip().map({
                'Yes': 1, 'No': 0, '1': 1, '0': 0, 'True': 1, 'False': 0
            }).fillna(0).astype(int)
    else:
        raise ValueError(f"Target column '{target_col}' not found.")

    drop_cols = [target_col]
    if 'customerID' in df.columns:
        drop_cols.append('customerID')
    X = df.drop(columns=drop_cols)

    if 'SeniorCitizen' in X.columns:
        X['SeniorCitizen'] = X['SeniorCitizen'].astype(str)

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()

    # Stratified Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids, test_size=test_size, random_state=random_state, stratify=y
    )

    # Leak-free ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    raw_names = preprocessor.get_feature_names_out()
    feature_names = [n.replace('num__', '').replace('cat__', '') for n in raw_names]

    return (
        X_train_raw, X_test_raw, y_train, y_test,
        ids_train, ids_test, X_train_trans, X_test_trans,
        preprocessor, feature_names
    )


# ==========================================
# 2. MODEL EXPLORATION & BENCHMARKING
# ==========================================
def train_and_benchmark(X_train_trans, y_train, X_test_trans, y_test):
    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)
    pos_weight = float((len(y_train_arr) - np.sum(y_train_arr)) / np.sum(y_train_arr))

    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=10, min_samples_leaf=4,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
    }

    # 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}

    for name, model in models.items():
        roc_list, pr_list, f1_list, brier_list = [], [], [], []
        for tr_idx, val_idx in cv.split(X_train_trans, y_train_arr):
            X_tr, X_val = X_train_trans[tr_idx], X_train_trans[val_idx]
            y_tr, y_val = y_train_arr[tr_idx], y_train_arr[val_idx]

            if name == 'Logistic Regression':
                m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
            elif name == 'Random Forest':
                m = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
            else:
                m = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, scale_pos_weight=(len(y_tr)-sum(y_tr))/sum(y_tr),
                                  eval_metric='logloss', random_state=42, n_jobs=-1)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_val)[:, 1]
            roc_list.append(roc_auc_score(y_val, p))
            pr_list.append(average_precision_score(y_val, p))
            f1_list.append(f1_score(y_val, (p >= 0.5).astype(int), zero_division=0))
            brier_list.append(brier_score_loss(y_val, p))

        cv_summary[name] = {
            'CV_ROC_AUC': float(np.mean(roc_list)),
            'CV_PR_AUC': float(np.mean(pr_list)),
            'CV_F1': float(np.mean(f1_list)),
            'CV_Brier': float(np.mean(brier_list))
        }

    # Full Train and Held-Out Test Evaluation
    test_summary = {}
    best_score = -1.0
    best_name = None

    for name, model in models.items():
        model.fit(X_train_trans, y_train_arr)
        probs = model.predict_proba(X_test_trans)[:, 1]
        preds = (probs >= 0.5).astype(int)

        roc = roc_auc_score(y_test_arr, probs)
        pr = average_precision_score(y_test_arr, probs)
        f1 = f1_score(y_test_arr, preds, zero_division=0)
        rec = recall_score(y_test_arr, preds, zero_division=0)
        prec = precision_score(y_test_arr, preds, zero_division=0)
        acc = accuracy_score(y_test_arr, preds)
        brier = brier_score_loss(y_test_arr, probs)

        test_summary[name] = {
            'ROC_AUC': float(roc), 'PR_AUC': float(pr), 'F1': float(f1),
            'Recall': float(rec), 'Precision': float(prec), 'Accuracy': float(acc),
            'Brier_Score': float(brier)
        }

        composite = 0.6 * roc + 0.4 * pr
        if composite > best_score:
            best_score = composite
            best_name = name

    return models[best_name], best_name, test_summary, cv_summary, models


# ==========================================
# 3. FEATURE 5: RELIABILITY & CALIBRATION
# ==========================================
def calibrate_model(best_model, X_train_trans, y_train, X_test_trans, y_test):
    calibrator = CalibratedClassifierCV(estimator=best_model, method='sigmoid', cv=5)
    calibrator.fit(X_train_trans, np.array(y_train))

    uncal_probs = best_model.predict_proba(X_test_trans)[:, 1]
    cal_probs = calibrator.predict_proba(X_test_trans)[:, 1]

    uncal_brier = brier_score_loss(y_test, uncal_probs)
    cal_brier = brier_score_loss(y_test, cal_probs)

    return calibrator, uncal_probs, cal_probs, uncal_brier, cal_brier


# ==========================================
# 4. FEATURE 2: EARLY-WARNING SYSTEM
# ==========================================
def evaluate_early_warning(customer_series, churn_prob: float):
    if churn_prob >= 0.65:
        risk_tier = "Critical"
        urgency = "Immediate Intervention (< 24h)"
        action_sla = 24
    elif churn_prob >= 0.50:
        risk_tier = "High"
        urgency = "Urgent Retention Playbook (< 48h)"
        action_sla = 48
    elif churn_prob >= 0.35:
        risk_tier = "Medium"
        urgency = "Automated Retention Workflow (Within 7 Days)"
        action_sla = 168
    else:
        risk_tier = "Low"
        urgency = "Standard Loyalty Routine"
        action_sla = 720

    triggers = []
    tactics = []

    tenure = float(customer_series.get('tenure', 24))
    contract = str(customer_series.get('Contract', ''))
    monthly_charges = float(customer_series.get('MonthlyCharges', 0.0))
    payment_method = str(customer_series.get('PaymentMethod', ''))
    internet_service = str(customer_series.get('InternetService', ''))
    tech_support = str(customer_series.get('TechSupport', ''))
    online_security = str(customer_series.get('OnlineSecurity', ''))

    if tenure <= 6:
        triggers.append(f"Onboarding Hazard: Tenure is {int(tenure)} months (under 6m threshold)")
        tactics.append("Deploy VIP Onboarding Concierge Call to resolve early product blockers.")
    elif tenure <= 12:
        triggers.append(f"First-Year Fragility: Tenure is {int(tenure)} months")
        tactics.append("Offer 12-month loyalty milestone reward.")

    if 'Month-to-month' in contract:
        triggers.append("Contractual Friction: Month-to-month contract (frictionless churn barrier)")
        tactics.append("Offer 15% rate lock discount upon transitioning to 1-Year or 2-Year contract.")

    if monthly_charges >= 75.0:
        triggers.append(f"Financial Pressure: Elevated monthly charges (${monthly_charges:.2f}/mo)")
        tactics.append("Perform account plan audit to bundle services and apply $10 monthly credit.")

    if internet_service == 'Fiber optic' and (tech_support in ['No', 'No internet service'] or online_security in ['No', 'No internet service']):
        triggers.append("Service Gap: Fiber Optic service lacking TechSupport / OnlineSecurity")
        tactics.append("Provision complimentary 6-month TechSupport and Security upgrade.")

    if 'Electronic check' in payment_method:
        triggers.append("Payment Instability: Manual Electronic Check payment channel")
        tactics.append("Grant $10 bill credit upon enrolling into Bank/Card Auto-Pay.")

    if not triggers:
        triggers.append("Stable Engagement Profile: High multi-service loyalty")
        tactics.append("Maintain quarterly value newsletter and milestone rewards.")

    return {
        'churn_probability': round(float(churn_prob), 4),
        'risk_score_index': round(float(churn_prob) * 100, 1),
        'risk_tier': risk_tier,
        'action_sla_hours': action_sla,
        'intervention_urgency': urgency,
        'warning_triggers': triggers,
        'recommended_retention_playbook': tactics
    }


# ==========================================
# 5. FEATURE 1: SHAP EXPLANATION MODULE
# ==========================================
def explain_with_shap(model, X_train_trans, X_test_trans, feature_names):
    explainer = shap.TreeExplainer(model, data=X_train_trans[:200], feature_names=feature_names)
    raw_shap = explainer(X_test_trans)
    if len(raw_shap.shape) == 3 and raw_shap.shape[-1] == 2:
        shap_explanation = raw_shap[:, :, 1]
    else:
        shap_explanation = raw_shap
    shap_explanation.feature_names = feature_names
    return explainer, shap_explanation


# ==========================================
# 6. PIPELINE ORCHESTRATOR & SERIALIZATION
# ==========================================
def main():
    dataset_file = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    artifacts_dir = "artifacts"
    plots_dir = os.path.join(artifacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("Step 1: Loading & Preprocessing...")
    (
        X_train_raw, X_test_raw, y_train, y_test,
        ids_train, ids_test, X_train_trans, X_test_trans,
        preprocessor, feature_names
    ) = load_and_preprocess(dataset_file)

    print("Step 2: Model Benchmarking...")
    best_model, best_name, test_results, cv_results, all_models = train_and_benchmark(
        X_train_trans, y_train, X_test_trans, y_test
    )
    print(f"Champion Model: {best_name}")

    print("Step 3: Reliability & Probability Calibration (Feature 5)...")
    calibrator, uncal_probs, cal_probs, uncal_brier, cal_brier = calibrate_model(
        best_model, X_train_trans, y_train, X_test_trans, y_test
    )
    print(f"Uncalibrated Brier: {uncal_brier:.4f} | Calibrated Brier: {cal_brier:.4f}")

    # Plot Calibration
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, uncal_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10)

    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], 'k--', label="Ideal Calibration")
    plt.plot(prob_pred_uncal, prob_true_uncal, 's-', color='#e74c3c', label=f"Uncalibrated (Brier: {uncal_brier:.4f})")
    plt.plot(prob_pred_cal, prob_true_cal, 'o-', color='#2ecc71', label=f"Calibrated (Brier: {cal_brier:.4f})")
    plt.title("Probability Calibration Curves (Feature 5)", fontsize=13, fontweight='bold')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "calibration_reliability_diagram.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Step 4: Behavioral Early-Warning Diagnostics (Feature 2)...")
    early_warnings = []
    for i in range(len(X_test_raw)):
        diag = evaluate_early_warning(X_test_raw.iloc[i], cal_probs[i])
        early_warnings.append({
            'customerID': str(ids_test.iloc[i]),
            'churn_probability': diag['churn_probability'],
            'risk_score': diag['risk_score_index'],
            'risk_tier': diag['risk_tier'],
            'action_sla_hours': diag['action_sla_hours'],
            'intervention_urgency': diag['intervention_urgency'],
            'triggers': " | ".join(diag['warning_triggers']),
            'playbook': " | ".join(diag['recommended_retention_playbook'])
        })
    df_warnings = pd.DataFrame(early_warnings)
    df_warnings.to_csv(os.path.join(artifacts_dir, "early_warning_cohort_summary.csv"), index=False)

    print("Step 5: SHAP Explainability Engine (Feature 1)...")
    explainer, shap_explanation = explain_with_shap(best_model, X_train_trans, X_test_trans, feature_names)

    # Global plots
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_explanation, max_display=15, show=False)
    plt.title("Global Feature Importance (SHAP Bar Plot)", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(plots_dir, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.title("Global Feature Attribution (SHAP Beeswarm)", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(plots_dir, "shap_beeswarm.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Local Waterfall
    high_risk_idx = df_warnings[df_warnings['risk_tier'] == 'Critical'].index[0]
    high_risk_id = df_warnings.iloc[high_risk_idx]['customerID']
    high_risk_prob = df_warnings.iloc[high_risk_idx]['churn_probability']

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation[high_risk_idx], max_display=8, show=False)
    plt.title(f"Local SHAP Risk Attribution: Customer {high_risk_id} (P(Churn)={high_risk_prob:.2%})", fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(plots_dir, f"shap_waterfall_{high_risk_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Step 6: Serializing Pipeline Artifacts...")
    joblib.dump(calibrator, os.path.join(artifacts_dir, "best_calibrated_model.joblib"))
    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
    joblib.dump(explainer, os.path.join(artifacts_dir, "shap_explainer.joblib"))

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'champion_model': best_name,
        'cross_validation': cv_results,
        'test_evaluation': test_results,
        'calibration': {
            'uncalibrated_brier': uncal_brier,
            'calibrated_brier': cal_brier,
            'brier_improvement_pct': ((uncal_brier - cal_brier) / uncal_brier) * 100
        },
        'early_warning_tiers': df_warnings['risk_tier'].value_counts().to_dict()
    }
    with open(os.path.join(artifacts_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Execution complete! All artifacts and plots generated.")

if __name__ == "__main__":
    main()
