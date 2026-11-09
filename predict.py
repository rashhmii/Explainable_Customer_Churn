"""
Production Inference & Scoring Engine (CLI / Service)
Project: Explainable Customer Churn Prediction Using Data Analytics
Center for Data Analytics & Digital Intelligence
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from src.preprocessing import get_clean_feature_names
from src.early_warning import evaluate_customer_early_warning, batch_early_warning_detection


class ChurnInferenceEngine:
    """
    Production inference engine loading calibrated estimator, preprocessor,
    and SHAP explainer to provide real-time churn diagnostics.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "best_calibrated_model.joblib")
        self.preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
        self.explainer_path = os.path.join(artifacts_dir, "shap_explainer.joblib")

        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Missing serialized artifacts in '{artifacts_dir}'. Please run 'python main.py' first."
            )

        # Load serialized pipeline objects
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.explainer = joblib.load(self.explainer_path) if os.path.exists(self.explainer_path) else None

        # Extract feature names
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        cat_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        self.feature_names = get_clean_feature_names(self.preprocessor, num_cols, cat_cols)

    def _prepare_dataframe(self, record: dict) -> pd.DataFrame:
        """Sanitizes and aligns a single customer record."""
        df = pd.DataFrame([record])
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])

        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True),
                errors='coerce'
            )
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

        return df

    def predict_single(
        self,
        customer_data: dict,
        customer_id: str = "NEW_CUSTOMER",
        generate_waterfall: bool = True,
        output_dir: str = "artifacts/plots"
    ) -> dict:
        """
        Executes end-to-end scoring, early-warning classification, and local SHAP decomposition.
        """
        raw_df = self._prepare_dataframe(customer_data)
        X_trans = self.preprocessor.transform(raw_df)

        # Calibrated churn probability
        churn_prob = float(self.model.predict_proba(X_trans)[:, 1][0])

        # Feature 2: Early-Warning Diagnostic System
        raw_series = pd.Series(customer_data)
        early_warning = evaluate_customer_early_warning(raw_series, churn_prob)

        # Feature 1: Local SHAP Explanation
        shap_details = {}
        waterfall_path = None

        if self.explainer is not None:
            shap_vals = self.explainer(X_trans)
            if len(shap_vals.shape) == 3 and shap_vals.shape[-1] == 2:
                customer_shap = shap_vals[0, :, 1]
            else:
                customer_shap = shap_vals[0]

            customer_shap.feature_names = self.feature_names

            # Extract top positive (accelerators) and negative (retention anchors)
            paired = []
            for f_name, s_val, d_val in zip(self.feature_names, customer_shap.values, customer_shap.data):
                paired.append({
                    'feature': f_name,
                    'shap_impact': float(s_val),
                    'feature_value': float(d_val) if isinstance(d_val, (int, float, np.number)) else str(d_val)
                })

            accelerators = sorted([p for p in paired if p['shap_impact'] > 0], key=lambda x: x['shap_impact'], reverse=True)[:5]
            anchors = sorted([p for p in paired if p['shap_impact'] < 0], key=lambda x: x['shap_impact'])[:5]

            shap_details = {
                'base_value': float(customer_shap.base_values) if hasattr(customer_shap, 'base_values') else None,
                'top_churn_accelerators': accelerators,
                'top_retention_anchors': anchors
            }

            if generate_waterfall:
                os.makedirs(output_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(10, 7))
                shap.plots.waterfall(customer_shap, max_display=8, show=False)
                plt.title(f"Local SHAP Risk Attribution: {customer_id} (P(Churn)={churn_prob:.1%})",
                          fontsize=13, fontweight='bold', pad=12)
                plt.tight_layout()
                waterfall_path = os.path.join(output_dir, f"shap_waterfall_{customer_id}.png")
                plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                plt.close()
                shap_details['waterfall_plot_path'] = waterfall_path

        return {
            'customer_id': customer_id,
            'churn_probability': round(churn_prob, 4),
            'risk_score_index': round(churn_prob * 100, 1),
            'risk_tier': early_warning['risk_tier'],
            'action_sla_hours': early_warning['action_sla_hours'],
            'intervention_urgency': early_warning['intervention_urgency'],
            'active_triggers': early_warning['active_triggers'],
            'recommended_retention_playbook': early_warning['recommended_retention_playbook'],
            'shap_explanation': shap_details
        }

    def predict_batch(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        """
        Executes batch scoring and early warning risk calculation across a CSV file.
        """
        raw_df = pd.read_csv(input_csv)
        customer_ids = raw_df['customerID'] if 'customerID' in raw_df.columns else pd.Series(raw_df.index, name='customerID')

        sanitized_df = raw_df.copy()
        if 'customerID' in sanitized_df.columns:
            sanitized_df = sanitized_df.drop(columns=['customerID'])
        if 'Churn' in sanitized_df.columns:
            sanitized_df = sanitized_df.drop(columns=['Churn'])

        if 'TotalCharges' in sanitized_df.columns:
            sanitized_df['TotalCharges'] = pd.to_numeric(
                sanitized_df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True),
                errors='coerce'
            )
        if 'SeniorCitizen' in sanitized_df.columns:
            sanitized_df['SeniorCitizen'] = sanitized_df['SeniorCitizen'].astype(str)

        X_trans = self.preprocessor.transform(sanitized_df)
        probs = self.model.predict_proba(X_trans)[:, 1]

        scored_df = batch_early_warning_detection(sanitized_df, customer_ids, probs)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        scored_df.to_csv(output_csv, index=False)
        print(f"Successfully processed {len(scored_df)} accounts. Exported to: {output_csv}")
        return scored_df


def get_default_sample_customer() -> dict:
    """Returns an illustrative high-risk customer profile."""
    return {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 3,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.85,
        'TotalCharges': 269.55
    }


def main():
    parser = argparse.ArgumentParser(description="Explainable Churn Prediction Inference Engine")
    parser.add_argument("--sample", action="store_true", help="Score sample high-risk customer profile")
    parser.add_argument("--input-json", type=str, help="Path to customer JSON payload")
    parser.add_argument("--input-csv", type=str, help="Path to batch input CSV file")
    parser.add_argument("--output-csv", type=str, default="artifacts/scored_batch_output.csv", help="Path for batch output CSV")
    parser.add_argument("--customer-id", type=str, default="SAMPLE_001", help="Customer ID for single prediction")

    args = parser.parse_args()
    engine = ChurnInferenceEngine()

    if args.input_csv:
        print(f"Processing batch from: {args.input_csv}")
        engine.predict_batch(args.input_csv, args.output_csv)
    elif args.input_json:
        with open(args.input_json, 'r') as f:
            data = json.load(f)
        result = engine.predict_single(data, customer_id=args.customer_id)
        print(json.dumps(result, indent=2))
    elif args.sample or len(sys.argv) == 1:
        print("Scoring default illustrative customer profile...")
        data = get_default_sample_customer()
        result = engine.predict_single(data, customer_id="SAMPLE_VULNERABLE_01")
        print(json.dumps(result, indent=2))
        print("\nGenerated local explanation waterfall plot saved to:", result['shap_explanation'].get('waterfall_plot_path'))


if __name__ == "__main__":
    main()
