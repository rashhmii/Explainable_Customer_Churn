"""
Module: Evaluation Visualizations & Diagnostics
Project: Explainable Customer Churn Prediction Using Data Analytics
Center for Data Analytics & Digital Intelligence
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, Any


def plot_calibration_curves(
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    y_test: np.ndarray,
    uncal_brier: float,
    cal_brier: float,
    output_dir: str
) -> str:
    """
    Plots reliability diagram (calibration curves) before and after Platt scaling.
    """
    from sklearn.calibration import calibration_curve

    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, uncal_probs, n_bins=10, strategy='uniform')
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10, strategy='uniform')

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", linewidth=1.5)
    ax.plot(
        prob_pred_uncal, prob_true_uncal, "s-", color="#e74c3c", linewidth=2,
        label=f"Uncalibrated Model (Brier: {uncal_brier:.4f})"
    )
    ax.plot(
        prob_pred_cal, prob_true_cal, "o-", color="#2ecc71", linewidth=2.5,
        label=f"Calibrated Model (Platt/Sigmoid, Brier: {cal_brier:.4f})"
    )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight='bold')
    ax.set_ylabel("Empirical Fraction of Positives (Observed)", fontsize=12, fontweight='bold')
    ax.set_title("Probability Calibration Reliability Diagram (Feature 5)", fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "calibration_reliability_diagram.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def plot_model_comparisons(
    models: Dict[str, Any],
    X_test_trans: np.ndarray,
    y_test: np.ndarray,
    output_dir: str
) -> Dict[str, str]:
    """
    Generates ROC and Precision-Recall benchmarking curves for all models.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    palette = {'Logistic Regression': '#3498db', 'Random Forest': '#f39c12', 'XGBoost': '#9b59b6'}

    # ROC Curves
    for name, model in models.items():
        probs = model.predict_proba(X_test_trans)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        color = palette.get(name, '#34495e')
        ax1.plot(fpr, tpr, label=f"{name}", color=color, linewidth=2)

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Guess')
    ax1.set_title("ROC Curves Comparison", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Precision-Recall Curves
    for name, model in models.items():
        probs = model.predict_proba(X_test_trans)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, probs)
        color = palette.get(name, '#34495e')
        ax2.plot(recall, precision, label=f"{name}", color=color, linewidth=2)

    ax2.set_title("Precision-Recall Curves Comparison", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    benchmarking_path = os.path.join(output_dir, "model_benchmark_roc_pr_curves.png")
    plt.savefig(benchmarking_path, dpi=300)
    plt.close()

    return {'benchmark_curves': benchmarking_path}


def plot_confusion_matrix_heatmap(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str
) -> str:
    """
    Plots annotated confusion matrix heatmap.
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'],
                annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold')
    ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight='bold')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    return cm_path

