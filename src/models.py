"""
Module: Model Exploration, Benchmarking & Reliability Calibration
Project: Explainable Customer Churn Prediction Using Data Analytics
Center for Data Analytics & Digital Intelligence
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
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
    classification_report
)


def get_model_candidates(pos_weight: float) -> Dict[str, Any]:
    """
    Initializes candidate classifiers:
      1. Baseline: Logistic Regression (balanced class weights, L2)
      2. Ensemble Bagging: Random Forest Classifier (balanced class weights)
      3. Gradient Boosting: XGBoost Classifier (scale_pos_weight)
    """
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }


def cross_validate_candidates(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Performs 5-Fold Stratified Cross-Validation on training data across candidate models.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_summary = {}

    for name, model in models.items():
        roc_scores = []
        pr_scores = []
        f1_scores = []
        rec_scores = []
        prec_scores = []
        brier_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx], \
                                       y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

            # Clone/fit model
            fold_model = sklearn_clone_or_instantiate(model, name, (len(y_fold_train) - sum(y_fold_train)) / sum(y_fold_train))
            fold_model.fit(X_fold_train, y_fold_train)

            probs = fold_model.predict_proba(X_fold_val)[:, 1]
            preds = (probs >= 0.5).astype(int)

            roc_scores.append(roc_auc_score(y_fold_val, probs))
            pr_scores.append(average_precision_score(y_fold_val, probs))
            f1_scores.append(f1_score(y_fold_val, preds, zero_division=0))
            rec_scores.append(recall_score(y_fold_val, preds, zero_division=0))
            prec_scores.append(precision_score(y_fold_val, preds, zero_division=0))
            brier_scores.append(brier_score_loss(y_fold_val, probs))

        cv_summary[name] = {
            'CV_ROC_AUC_mean': float(np.mean(roc_scores)),
            'CV_ROC_AUC_std': float(np.std(roc_scores)),
            'CV_PR_AUC_mean': float(np.mean(pr_scores)),
            'CV_PR_AUC_std': float(np.std(pr_scores)),
            'CV_F1_mean': float(np.mean(f1_scores)),
            'CV_F1_std': float(np.std(f1_scores)),
            'CV_Recall_mean': float(np.mean(rec_scores)),
            'CV_Recall_std': float(np.std(rec_scores)),
            'CV_Precision_mean': float(np.mean(prec_scores)),
            'CV_Precision_std': float(np.std(prec_scores)),
            'CV_Brier_mean': float(np.mean(brier_scores)),
            'CV_Brier_std': float(np.std(brier_scores))
        }

    return cv_summary


def sklearn_clone_or_instantiate(base_model, name: str, pos_weight: float):
    """
    Helper to cleanly instantiate a fresh estimator for cross-validation folds.
    """
    if name == 'Logistic Regression':
        return LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
    elif name == 'Random Forest':
        return RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=10, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1)
    else:
        return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1)


def benchmark_and_select_model(
    X_train_trans: np.ndarray,
    y_train: pd.Series,
    X_test_trans: np.ndarray,
    y_test: pd.Series
) -> Tuple[Any, str, Dict[str, Any], Dict[str, Any]]:
    """
    Trains candidate models on the full training set, evaluates on the held-out test set,
    and selects the top model based on ROC-AUC and PR-AUC.
    """
    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)
    pos_weight = float((len(y_train_arr) - np.sum(y_train_arr)) / np.sum(y_train_arr))

    models = get_model_candidates(pos_weight)

    # 1. Cross-validation evaluation
    cv_results = cross_validate_candidates(models, X_train_trans, y_train, n_splits=5)

    # 2. Fit and held-out test evaluation
    test_results = {}
    best_score = -1.0
    best_model_name = None
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_trans, y_train_arr)
        fitted_models[name] = model

        probs = model.predict_proba(X_test_trans)[:, 1]
        preds = (probs >= 0.5).astype(int)

        roc = roc_auc_score(y_test_arr, probs)
        pr = average_precision_score(y_test_arr, probs)
        f1 = f1_score(y_test_arr, preds, zero_division=0)
        rec = recall_score(y_test_arr, preds, zero_division=0)
        prec = precision_score(y_test_arr, preds, zero_division=0)
        acc = accuracy_score(y_test_arr, preds)
        brier = brier_score_loss(y_test_arr, probs)
        cm = confusion_matrix(y_test_arr, preds).tolist()

        test_results[name] = {
            'ROC_AUC': float(roc),
            'PR_AUC': float(pr),
            'F1': float(f1),
            'Recall': float(rec),
            'Precision': float(prec),
            'Accuracy': float(acc),
            'Brier_Score': float(brier),
            'Confusion_Matrix': cm,
            'Classification_Report': classification_report(y_test_arr, preds, output_dict=True)
        }

        # Model selection metric: prioritize ROC-AUC followed by PR-AUC
        composite_score = 0.6 * roc + 0.4 * pr
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name

    best_estimator = fitted_models[best_model_name]
    return best_estimator, best_model_name, test_results, cv_results


def calibrate_estimator(
    base_estimator: Any,
    X_train_trans: np.ndarray,
    y_train: pd.Series,
    X_test_trans: np.ndarray,
    y_test: pd.Series,
    method: str = 'sigmoid'
) -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
    """
    Feature 5: Probability Calibration & Reliability Assessment.
    Calibrates probabilities using Platt scaling ('sigmoid') via CalibratedClassifierCV.
    Assesses Brier Score pre- and post-calibration on the test set.
    """
    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    # Use 5-fold CV calibration on the training data
    calibrator = CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=5
    )
    calibrator.fit(X_train_trans, y_train_arr)

    # Probabilities
    uncal_probs = base_estimator.predict_proba(X_test_trans)[:, 1]
    cal_probs = calibrator.predict_proba(X_test_trans)[:, 1]

    uncal_brier = brier_score_loss(y_test_arr, uncal_probs)
    cal_brier = brier_score_loss(y_test_arr, cal_probs)

    uncal_roc = roc_auc_score(y_test_arr, uncal_probs)
    cal_roc = roc_auc_score(y_test_arr, cal_probs)

    uncal_pr = average_precision_score(y_test_arr, uncal_probs)
    cal_pr = average_precision_score(y_test_arr, cal_probs)

    # Calibration curves (Reliability diagrams)
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test_arr, uncal_probs, n_bins=10, strategy='uniform')
    prob_true_cal, prob_pred_cal = calibration_curve(y_test_arr, cal_probs, n_bins=10, strategy='uniform')

    calibration_metrics = {
        'uncalibrated_brier_score': float(uncal_brier),
        'calibrated_brier_score': float(cal_brier),
        'brier_improvement_pct': float(((uncal_brier - cal_brier) / uncal_brier) * 100),
        'uncalibrated_roc_auc': float(uncal_roc),
        'calibrated_roc_auc': float(cal_roc),
        'uncalibrated_pr_auc': float(uncal_pr),
        'calibrated_pr_auc': float(cal_pr),
        'curve_data': {
            'prob_true_uncal': prob_true_uncal.tolist(),
            'prob_pred_uncal': prob_pred_uncal.tolist(),
            'prob_true_cal': prob_true_cal.tolist(),
            'prob_pred_cal': prob_pred_cal.tolist()
        }
    }

    return calibrator, calibration_metrics
