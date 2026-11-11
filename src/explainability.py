"""
Module: Explainable AI with SHAP (Feature 1)
Project: Explainable Customer Churn Prediction Using Data Analytics
Center for Data Analytics & Digital Intelligence
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from typing import Dict, Any, List, Tuple


class NativeXGBoostExplainer:
    """Compatibility-safe SHAP adapter backed by XGBoost's ``pred_contribs``.

    Recent XGBoost releases serialise ``base_score`` as a vector.  Some SHAP
    releases cannot parse that model representation in ``TreeExplainer``.  The
    booster itself can calculate exact tree contributions, so this adapter
    returns those values as a normal :class:`shap.Explanation` without relying
    on SHAP's private model parser.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def __call__(self, X: np.ndarray) -> shap.Explanation:
        values = np.asarray(X)
        contributions = self.model.get_booster().predict(
            xgb.DMatrix(values, feature_names=self.feature_names), pred_contribs=True
        )
        return shap.Explanation(
            values=contributions[:, :-1],
            base_values=contributions[:, -1],
            data=values,
            feature_names=self.feature_names,
        )


def build_shap_explainer(
    model: Any,
    model_name: str,
    X_train_trans: np.ndarray,
    feature_names: List[str]
) -> shap.Explainer:
    """
    Initializes the optimal SHAP explainer based on model architecture:
      - Tree-based (XGBoost, Random Forest): TreeExplainer
      - Linear (Logistic Regression): LinearExplainer
      - Fallback: General Explainer
    """
    if 'XGBoost' in model_name:
        # Uses XGBoost's exact contribution API and remains compatible with
        # current XGBoost model serialisation formats.
        explainer = NativeXGBoostExplainer(model, feature_names)
    elif 'Random Forest' in model_name:
        explainer = shap.TreeExplainer(model, data=X_train_trans[:200], feature_names=feature_names)
    elif 'Logistic' in model_name:
        explainer = shap.LinearExplainer(model, X_train_trans[:500], feature_names=feature_names)
    else:
        explainer = shap.Explainer(model, X_train_trans[:200], feature_names=feature_names)

    return explainer


def compute_shap_values(
    explainer: shap.Explainer,
    X_eval: np.ndarray,
    feature_names: List[str]
) -> shap.Explanation:
    """
    Computes SHAP values on the evaluation set and wraps them into a typed shap.Explanation object.
    """
    raw_shap = explainer(X_eval)

    # For multi-output or binary classifiers returning 2 classes, select the positive class (Churn = 1)
    if len(raw_shap.shape) == 3 and raw_shap.shape[-1] == 2:
        shap_explanation = raw_shap[:, :, 1]
    else:
        shap_explanation = raw_shap

    # Ensure feature names are aligned
    shap_explanation.feature_names = feature_names
    return shap_explanation


def generate_global_shap_visualizations(
    shap_explanation: shap.Explanation,
    output_dir: str
) -> Dict[str, str]:
    """
    Generates and persists:
      1. Global Summary Bar Plot (Mean Absolute SHAP Impact)
      2. Global Beeswarm Plot (Directional Feature Effects)
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # 1. Summary Bar Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_explanation, max_display=15, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_summary_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['summary_bar'] = bar_path

    # 2. Beeswarm Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.title("Global Feature Attribution (SHAP Beeswarm Summary)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['beeswarm'] = beeswarm_path

    return plot_paths


def generate_local_waterfall_explanation(
    shap_explanation: shap.Explanation,
    sample_idx: int,
    customer_id: str,
    raw_features: pd.Series,
    churn_probability: float,
    output_dir: str,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Feature 1: Local Prediction Explanations.
    Generates a personalized SHAP waterfall plot and structured attribution report
    detailing exact positive and negative drivers of churn risk for an individual customer.
    """
    os.makedirs(output_dir, exist_ok=True)
    customer_shap = shap_explanation[sample_idx]

    # Save Waterfall Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(customer_shap, max_display=top_k, show=False)
    plt.title(f"Local SHAP Risk Attribution: Customer {customer_id} (P(Churn)={churn_probability:.2%})",
              fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    waterfall_path = os.path.join(output_dir, f"shap_waterfall_{customer_id}.png")
    plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Extract structured drivers
    feature_names = customer_shap.feature_names
    shap_vals = customer_shap.values
    data_vals = customer_shap.data

    paired = []
    for f_name, s_val, d_val in zip(feature_names, shap_vals, data_vals):
        paired.append({
            'feature': f_name,
            'shap_value': float(s_val),
            'feature_value': float(d_val) if isinstance(d_val, (int, float, np.number)) else str(d_val)
        })

    # Sort positive (churn accelerators) and negative (retention anchors)
    churn_accelerators = sorted([p for p in paired if p['shap_value'] > 0], key=lambda x: x['shap_value'], reverse=True)[:5]
    retention_anchors = sorted([p for p in paired if p['shap_value'] < 0], key=lambda x: x['shap_value'])[:5]

    return {
        'customer_id': str(customer_id),
        'churn_probability': float(churn_probability),
        'base_value': float(customer_shap.base_values) if hasattr(customer_shap, 'base_values') else None,
        'waterfall_plot_path': waterfall_path,
        'top_churn_accelerators': churn_accelerators,
        'top_retention_anchors': retention_anchors,
        'raw_customer_context': raw_features.to_dict()
    }

