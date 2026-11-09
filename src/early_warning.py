"""
Module: Behavioral Early-Warning Detection System (Feature 2)
Project: Explainable Customer Churn Prediction Using Data Analytics
Center for Data Analytics & Digital Intelligence
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def evaluate_customer_early_warning(customer_record: pd.Series, churn_prob: float) -> Dict[str, Any]:
    """
    Evaluates dynamic behavioral early-warning risk scores and actionable triggers
    for an individual customer based on contractual, tenure, financial, and product signals.
    """
    # 1. Dynamic Risk Tiering
    if churn_prob >= 0.65:
        risk_tier = "Critical"
        urgency = "Immediate Intervention (< 24h)"
        action_sla_hours = 24
    elif churn_prob >= 0.50:
        risk_tier = "High"
        urgency = "Urgent Retention Playbook (< 48h)"
        action_sla_hours = 48
    elif churn_prob >= 0.35:
        risk_tier = "Medium"
        urgency = "Automated Retention Workflow (Within 7 Days)"
        action_sla_hours = 168
    else:
        risk_tier = "Low"
        urgency = "Standard Loyalty Routine"
        action_sla_hours = 720

    # 2. Behavioral Trigger Analysis
    triggers = []
    retention_tactics = []

    tenure = float(customer_record.get('tenure', 24))
    contract = str(customer_record.get('Contract', ''))
    monthly_charges = float(customer_record.get('MonthlyCharges', 0.0))
    payment_method = str(customer_record.get('PaymentMethod', ''))
    internet_service = str(customer_record.get('InternetService', ''))
    tech_support = str(customer_record.get('TechSupport', ''))
    online_security = str(customer_record.get('OnlineSecurity', ''))
    paperless = str(customer_record.get('PaperlessBilling', ''))

    # Tenure Triggers
    if tenure <= 6:
        triggers.append({
            'code': 'TRG_TENURE_CRITICAL',
            'severity': 'Critical',
            'description': f"Onboarding Vulnerability: Extremely short tenure ({int(tenure)} months); customer has not crossed retention stickiness threshold."
        })
        retention_tactics.append("Trigger VIP Welcome Call & Onboarding Concierge to eliminate early product adoption friction.")
    elif tenure <= 12:
        triggers.append({
            'code': 'TRG_TENURE_YEAR1',
            'severity': 'High',
            'description': f"First-Year Fragility: Customer tenure ({int(tenure)} months) is within the highest hazard rate window."
        })
        retention_tactics.append("Offer 12-month loyalty milestone reward with personalized usage review.")

    # Contract Triggers
    if 'Month-to-month' in contract:
        triggers.append({
            'code': 'TRG_CONTRACT_MTM',
            'severity': 'High',
            'description': "Contractual Friction: Month-to-month agreement provides zero switching barrier or exit friction."
        })
        retention_tactics.append("Incentivize 1-Year or 2-Year Contract migration via 15% rate lock-in discount.")

    # Financial / Cost Pressure Triggers
    if monthly_charges >= 75.0:
        triggers.append({
            'code': 'TRG_HIGH_CHARGE',
            'severity': 'Medium',
            'description': f"Elevated Financial Exposure: High monthly rate (${monthly_charges:.2f}/mo) amplifies price elasticity and competitor shopping."
        })
        retention_tactics.append("Offer tier bundle optimization or temporary $10/mo promotional bill credit.")

    # Service & Tech Support Deficits
    if internet_service == 'Fiber optic':
        if tech_support in ['No', 'No internet service'] or online_security in ['No', 'No internet service']:
            triggers.append({
                'code': 'TRG_FIBER_UNPROTECTED',
                'severity': 'High',
                'description': "High-Bandwidth Support Deficit: Fiber optic customer lacking Online Security or Tech Support (susceptible to unresolved service outages)."
            })
            retention_tactics.append("Provision complimentary 6-month TechSupport and OnlineSecurity add-on package.")

    # Payment Channel Friction
    if 'Electronic check' in payment_method:
        triggers.append({
            'code': 'TRG_PAYMENT_ECHECK',
            'severity': 'Medium',
            'description': "Friction-Prone Billing: Electronic check payment method exhibits highest historical churn correlation vs Auto-Pay."
        })
        retention_tactics.append("Provide a one-time $10 account credit upon enrolling in Credit Card or Bank Auto-Pay.")

    # Baseline default if no triggers fired
    if not triggers:
        triggers.append({
            'code': 'TRG_STABLE_PROFILE',
            'severity': 'Low',
            'description': "Healthy Engagement Profile: Long tenure and multi-service adoption."
        })
        retention_tactics.append("Maintain brand affinity via quarterly value-add newsletters and referral incentives.")

    # Risk Score Synthesis (0 - 100 Scale)
    behavioral_risk_index = round(churn_prob * 100, 1)

    return {
        'churn_probability': round(float(churn_prob), 4),
        'behavioral_risk_index': behavioral_risk_index,
        'risk_tier': risk_tier,
        'action_sla_hours': action_sla_hours,
        'intervention_urgency': urgency,
        'trigger_count': len(triggers),
        'active_triggers': triggers,
        'recommended_retention_playbook': retention_tactics
    }


def batch_early_warning_detection(
    X_df: pd.DataFrame,
    customer_ids: pd.Series,
    churn_probabilities: np.ndarray
) -> pd.DataFrame:
    """
    Executes early-warning diagnostics across an entire cohort of customers.
    """
    records = []
    for i in range(len(X_df)):
        c_series = X_df.iloc[i]
        c_id = customer_ids.iloc[i] if hasattr(customer_ids, 'iloc') else str(i)
        prob = churn_probabilities[i]

        diag = evaluate_customer_early_warning(c_series, prob)
        trigger_descriptions = " | ".join([t['description'] for t in diag['active_triggers']])
        tactics_summary = " | ".join(diag['recommended_retention_playbook'])

        records.append({
            'customerID': c_id,
            'churn_probability': diag['churn_probability'],
            'risk_index': diag['behavioral_risk_index'],
            'risk_tier': diag['risk_tier'],
            'action_sla_hours': diag['action_sla_hours'],
            'intervention_urgency': diag['intervention_urgency'],
            'trigger_count': diag['trigger_count'],
            'warning_triggers': trigger_descriptions,
            'retention_playbook': tactics_summary
        })

    return pd.DataFrame(records)

