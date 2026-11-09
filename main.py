"""
Main Execution Script: Explainable Customer Churn Prediction Pipeline
Deliverables for Person 1: Data Science & Machine Learning
Center for Data Analytics & Digital Intelligence
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Local Modules
from src.preprocessing import prepare_data_pipeline
from src.models import (
    get_model_candidates,
    benchmark_and_select_model,
    calibrate_estimator
)
from src.early_warning import (
    evaluate_customer_early_warning,
    batch_early_warning_detection
)
from src.explainability import (
    build_shap_explainer,
    compute_shap_values,
    generate_global_shap_visualizations,
    generate_local_waterfall_explanation
)
from src.evaluation_plots import (
    plot_calibration_curves,
    plot_model_comparisons,
    plot_confusion_matrix_heatmap
)


def run_person1_pipeline(
    dataset_path: str = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    artifacts_dir: str = "artifacts"
):
    print("=" * 80)
    print(" EXPLAINABLE CUSTOMER CHURN PREDICTION USING DATA ANALYTICS")
    print(" Person 1: Data Science & Machine Learning Deliverables Execution")
    print(" Center for Data Analytics & Digital Intelligence")
    print("=" * 80)

    plots_dir = os.path.join(artifacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Data Ingestion & Preprocessing
    # ---------------------------------------------------------
    print("\n[Module 1/6] Ingesting & Preprocessing Telco Dataset...")
    data_bundle = prepare_data_pipeline(dataset_path, test_size=0.2, random_state=42)

    X_train_raw = data_bundle['X_train_raw']
    X_test_raw = data_bundle['X_test_raw']
    y_train = data_bundle['y_train']
    y_test = data_bundle['y_test']
    ids_train = data_bundle['ids_train']
    ids_test = data_bundle['ids_test']
    X_train_trans = data_bundle['X_train_trans']
    X_test_trans = data_bundle['X_test_trans']
    preprocessor = data_bundle['preprocessor']
    feature_names = data_bundle['feature_names']

    print(f"  Training cohort size : {X_train_trans.shape[0]} customers, {X_train_trans.shape[1]} engineered features")
    print(f"  Test cohort size     : {X_test_trans.shape[0]} customers")
    print(f"  Class balance (Train): Churned={sum(y_train)} ({sum(y_train)/len(y_train):.1%}), Retained={len(y_train)-sum(y_train)}")

    # ---------------------------------------------------------
    # 2. Model Exploration, Benchmarking & Selection
    # ---------------------------------------------------------
    print("\n[Module 2/6] Training & Benchmarking Candidate Estimators (5-Fold Stratified CV)...")
    best_estimator, best_model_name, test_results, cv_results = benchmark_and_select_model(
        X_train_trans, y_train, X_test_trans, y_test
    )

    print("\n  --- Cross-Validation Results (Mean ± Std) ---")
    for name, m in cv_results.items():
        print(f"  {name:20s} | ROC-AUC: {m['CV_ROC_AUC_mean']:.4f} ± {m['CV_ROC_AUC_std']:.4f} | PR-AUC: {m['CV_PR_AUC_mean']:.4f} | F1: {m['CV_F1_mean']:.4f} | Brier: {m['CV_Brier_mean']:.4f}")

    print("\n  --- Held-Out Test Set Results ---")
    for name, m in test_results.items():
        print(f"  {name:20s} | ROC-AUC: {m['ROC_AUC']:.4f} | PR-AUC: {m['PR_AUC']:.4f} | F1: {m['F1']:.4f} | Recall: {m['Recall']:.4f} | Brier: {m['Brier_Score']:.4f}")

    print(f"\n  >>> Champion Model Selected: {best_model_name} <<<")

    # Benchmarking plots
    pos_weight = float((len(y_train) - np.sum(y_train)) / np.sum(y_train))
    all_models = get_model_candidates(pos_weight)
    for name, m in all_models.items():
        m.fit(X_train_trans, y_train)
    plot_model_comparisons(all_models, X_test_trans, np.array(y_test), plots_dir)

    # ---------------------------------------------------------
    # 3. Feature 5: Reliability & Probability Calibration
    # ---------------------------------------------------------
    print("\n[Module 3/6] Applying Platt Scaling Probability Calibration (Feature 5)...")
    calibrated_model, calibration_metrics = calibrate_estimator(
        best_estimator, X_train_trans, y_train, X_test_trans, y_test, method='sigmoid'
    )

    uncal_probs = best_estimator.predict_proba(X_test_trans)[:, 1]
    cal_probs = calibrated_model.predict_proba(X_test_trans)[:, 1]
    cal_preds = (cal_probs >= 0.5).astype(int)

    print(f"  Pre-Calibration Brier Score  : {calibration_metrics['uncalibrated_brier_score']:.4f}")
    print(f"  Post-Calibration Brier Score : {calibration_metrics['calibrated_brier_score']:.4f}")
    print(f"  Brier Loss Optimization      : {calibration_metrics['brier_improvement_pct']:.2f}% improvement")

    # Generate Calibration Reliability Diagram
    rel_plot_path = plot_calibration_curves(
        uncal_probs, cal_probs, np.array(y_test),
        calibration_metrics['uncalibrated_brier_score'],
        calibration_metrics['calibrated_brier_score'],
        plots_dir
    )
    print(f"  Saved reliability diagram to : {rel_plot_path}")

    # Generate Confusion Matrix
    cm_plot_path = plot_confusion_matrix_heatmap(
        np.array(y_test), cal_preds, f"Calibrated {best_model_name}", plots_dir
    )
    print(f"  Saved confusion matrix to    : {cm_plot_path}")

    # ---------------------------------------------------------
    # 4. Feature 2: Behavioral Early-Warning Detection Module
    # ---------------------------------------------------------
    print("\n[Module 4/6] Executing Behavioral Early-Warning Detection (Feature 2)...")
    early_warning_df = batch_early_warning_detection(X_test_raw, ids_test, cal_probs)

    tier_counts = early_warning_df['risk_tier'].value_counts()
    print("  Early-Warning Cohort Distribution:")
    for tier, count in tier_counts.items():
        pct = count / len(early_warning_df) * 100
        print(f"    - {tier:10s}: {count:4d} customers ({pct:.1f}%)")

    # Identify top high-risk customer for local case study
    critical_cases = early_warning_df[early_warning_df['risk_tier'] == 'Critical']
    exemplar_pos = int(critical_cases.index[0]) if len(critical_cases) > 0 else 0
    exemplar_id = str(ids_test.iloc[exemplar_pos])
    exemplar_record = X_test_raw.iloc[exemplar_pos]
    exemplar_prob = float(cal_probs[exemplar_pos])

    exemplar_warning = evaluate_customer_early_warning(exemplar_record, exemplar_prob)

    cohort_summary_path = os.path.join(artifacts_dir, "early_warning_cohort_summary.csv")
    early_warning_df.to_csv(cohort_summary_path, index=False)
    print(f"  Exported early warning cohort: {cohort_summary_path}")

    # ---------------------------------------------------------
    # 5. Feature 1: Explainable AI with SHAP
    # ---------------------------------------------------------
    print("\n[Module 5/6] Generating SHAP Explainability Engine (Feature 1)...")
    explainer = build_shap_explainer(best_estimator, best_model_name, X_train_trans, feature_names)
    
    # Compute SHAP values for test cohort
    print("  Calculating test SHAP attributions...")
    shap_explanation = compute_shap_values(explainer, X_test_trans, feature_names)

    # Global Visualizations
    shap_plot_paths = generate_global_shap_visualizations(shap_explanation, plots_dir)
    print(f"  Saved SHAP summary bar plot  : {shap_plot_paths['summary_bar']}")
    print(f"  Saved SHAP beeswarm plot     : {shap_plot_paths['beeswarm']}")

    # Local Customer Explanation Waterfall
    local_explanation = generate_local_waterfall_explanation(
        shap_explanation,
        sample_idx=exemplar_pos,
        customer_id=exemplar_id,
        raw_features=exemplar_record,
        churn_probability=exemplar_prob,
        output_dir=plots_dir
    )
    print(f"  Saved local waterfall plot   : {local_explanation['waterfall_plot_path']}")

    # ---------------------------------------------------------
    # 6. Serialization & Artifact Export
    # ---------------------------------------------------------
    print("\n[Module 6/6] Serializing Production Pipelines & Artifacts...")

    # Joblib serializations
    model_artifact_path = os.path.join(artifacts_dir, "best_calibrated_model.joblib")
    preprocessor_artifact_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    explainer_artifact_path = os.path.join(artifacts_dir, "shap_explainer.joblib")

    joblib.dump(calibrated_model, model_artifact_path)
    joblib.dump(preprocessor, preprocessor_artifact_path)
    joblib.dump(explainer, explainer_artifact_path)

    # Metrics JSON
    metrics_payload = {
        'timestamp': datetime.now().isoformat(),
        'champion_model': best_model_name,
        'cross_validation_benchmark': cv_results,
        'held_out_test_benchmark': test_results,
        'feature_5_calibration_metrics': {
            'uncalibrated_brier_score': calibration_metrics['uncalibrated_brier_score'],
            'calibrated_brier_score': calibration_metrics['calibrated_brier_score'],
            'brier_improvement_pct': calibration_metrics['brier_improvement_pct'],
            'uncalibrated_roc_auc': calibration_metrics['uncalibrated_roc_auc'],
            'calibrated_roc_auc': calibration_metrics['calibrated_roc_auc']
        },
        'feature_2_early_warning_distribution': tier_counts.to_dict(),
        'feature_names': feature_names
    }

    metrics_json_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    # Exemplar diagnostic profile JSON
    exemplar_payload = {
        'customer_id': exemplar_id,
        'churn_probability': exemplar_prob,
        'early_warning_profile': exemplar_warning,
        'shap_attributions': {
            'top_churn_accelerators': local_explanation['top_churn_accelerators'],
            'top_retention_anchors': local_explanation['top_retention_anchors']
        }
    }
    exemplar_json_path = os.path.join(artifacts_dir, f"customer_profile_{exemplar_id}.json")
    with open(exemplar_json_path, 'w') as f:
        json.dump(exemplar_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f" Artifacts generated in: {os.path.abspath(artifacts_dir)}")
    print(f"  - Model binary        : {model_artifact_path}")
    print(f"  - Preprocessor        : {preprocessor_artifact_path}")
    print(f"  - SHAP Explainer      : {explainer_artifact_path}")
    print(f"  - Benchmark Metrics   : {metrics_json_path}")
    print(f"  - Cohort Risk Tiers   : {cohort_summary_path}")
    print(f"  - Exemplar Profile    : {exemplar_json_path}")
    print(" Visualizations in:", plots_dir)
    print("  - calibration_reliability_diagram.png")
    print("  - model_benchmark_roc_pr_curves.png")
    print("  - confusion_matrix.png")
    print("  - shap_summary_bar.png")
    print("  - shap_beeswarm.png")
    print(f"  - shap_waterfall_{exemplar_id}.png")
    print("=" * 80)


if __name__ == "__main__":
    run_person1_pipeline()
