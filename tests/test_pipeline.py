"""
Unit & Integration Tests
Project: Explainable Customer Churn Prediction Using Data Analytics
Center for Data Analytics & Digital Intelligence
"""

import os
import unittest
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import clean_churn_dataset, build_preprocessor
from src.early_warning import evaluate_customer_early_warning


class TestCustomerChurnPipeline(unittest.TestCase):

    def setUp(self):
        self.artifacts_dir = "artifacts"
        self.dataset_path = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    def test_clean_churn_dataset(self):
        """Test TotalCharges coercion, target mapping, and customerID isolation."""
        sample_df = pd.DataFrame({
            'customerID': ['001-TEST', '002-TEST'],
            'gender': ['Female', 'Male'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'No'],
            'tenure': [1, 24],
            'PhoneService': ['No', 'Yes'],
            'MultipleLines': ['No phone service', 'No'],
            'InternetService': ['DSL', 'Fiber optic'],
            'OnlineSecurity': ['No', 'Yes'],
            'OnlineBackup': ['Yes', 'No'],
            'DeviceProtection': ['No', 'Yes'],
            'TechSupport': ['No', 'No'],
            'StreamingTV': ['No', 'Yes'],
            'StreamingMovies': ['No', 'No'],
            'Contract': ['Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check'],
            'MonthlyCharges': [29.85, 75.50],
            'TotalCharges': [' ', '1812.0'],  # Contains blank space
            'Churn': ['Yes', 'No']
        })

        X, y, customer_ids = clean_churn_dataset(sample_df)

        self.assertEqual(len(customer_ids), 2)
        self.assertEqual(customer_ids.iloc[0], '001-TEST')
        self.assertEqual(list(y), [1, 0])
        self.assertTrue(np.isnan(X['TotalCharges'].iloc[0]))
        self.assertEqual(X['TotalCharges'].iloc[1], 1812.0)
        self.assertNotIn('Churn', X.columns)
        self.assertNotIn('customerID', X.columns)

    def test_preprocessor_leak_free(self):
        """Test ColumnTransformer imputation and standard scaling."""
        df = pd.read_csv(self.dataset_path)
        X, y, _ = clean_churn_dataset(df)

        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()

        preprocessor = build_preprocessor(num_cols, cat_cols)
        X_trans = preprocessor.fit_transform(X)

        self.assertEqual(X_trans.shape[0], len(df))
        self.assertEqual(X_trans.shape[1], 30)  # 30 engineered features
        self.assertFalse(np.isnan(X_trans).any())  # Zero remaining NaNs

    def test_early_warning_critical_tier(self):
        """Test early warning logic for high-risk customer."""
        customer_series = pd.Series({
            'tenure': 2,
            'Contract': 'Month-to-month',
            'MonthlyCharges': 95.0,
            'PaymentMethod': 'Electronic check',
            'InternetService': 'Fiber optic',
            'TechSupport': 'No',
            'OnlineSecurity': 'No'
        })
        diag = evaluate_customer_early_warning(customer_series, churn_prob=0.82)

        self.assertEqual(diag['risk_tier'], 'Critical')
        self.assertEqual(diag['action_sla_hours'], 24)
        self.assertGreaterEqual(diag['trigger_count'], 3)
        trigger_codes = [t['code'] for t in diag['active_triggers']]
        self.assertIn('TRG_TENURE_CRITICAL', trigger_codes)
        self.assertIn('TRG_CONTRACT_MTM', trigger_codes)
        self.assertIn('TRG_HIGH_CHARGE', trigger_codes)

    def test_early_warning_low_tier(self):
        """Test early warning logic for loyal low-risk customer."""
        customer_series = pd.Series({
            'tenure': 65,
            'Contract': 'Two year',
            'MonthlyCharges': 45.0,
            'PaymentMethod': 'Bank transfer (automatic)',
            'InternetService': 'DSL',
            'TechSupport': 'Yes',
            'OnlineSecurity': 'Yes'
        })
        diag = evaluate_customer_early_warning(customer_series, churn_prob=0.08)

        self.assertEqual(diag['risk_tier'], 'Low')
        self.assertEqual(diag['action_sla_hours'], 720)
        self.assertEqual(diag['risk_tier'], 'Low')

    def test_serialized_artifacts_integrity(self):
        """Verify that saved serialized joblib artifacts load and generate valid predictions."""
        model_path = os.path.join(self.artifacts_dir, "best_calibrated_model.joblib")
        preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.joblib")
        explainer_path = os.path.join(self.artifacts_dir, "shap_explainer.joblib")

        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(preprocessor_path))
        self.assertTrue(os.path.exists(explainer_path))

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        explainer = joblib.load(explainer_path)

        # Load small batch for inference verification
        df = pd.read_csv(self.dataset_path).head(5)
        X, _, _ = clean_churn_dataset(df)

        X_trans = preprocessor.transform(X)
        probs = model.predict_proba(X_trans)[:, 1]

        self.assertEqual(len(probs), 5)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


if __name__ == '__main__':
    unittest.main()
