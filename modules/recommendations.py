"""Transparent, rule-based retention suggestions informed by SHAP factors."""

from typing import Any, Dict, Iterable, List


# Rules matched against the engineered feature name.  Names containing "_" are one-hot
# dummy columns and must match EXACTLY (substring matching made 'TechSupport_No' fire on
# 'TechSupport_No internet service').  Bare names ('tenure') match as substrings.
RULES = (
    ("Contract_Month-to-month", "Contract review", "Offer an appropriate 1- or 2-year contract with a rate-lock incentive."),
    ("MonthlyCharges", "Pricing review", "Review plan fit and offer a lower-cost bundle or temporary bill credit."),
    ("PaymentMethod_Electronic check", "Billing assistance", "Offer auto-pay enrolment support and a one-time billing incentive."),
    ("tenure", "Onboarding and loyalty", "Provide proactive onboarding or a loyalty milestone outreach."),
    ("InternetService_Fiber optic", "Service support", "Offer a service-quality check and complimentary support or security add-ons."),
)

# The training encoder uses OneHotEncoder(drop='first'), so the reference categories
# 'Contract_Month-to-month', 'TechSupport_No' and 'OnlineSecurity_No' are NOT columns the model
# or SHAP can ever report.  Those customers surface through the *other* dummies (e.g.
# 'Contract_Two year' with a positive SHAP value), so these rules match on the customer's raw value.
RAW_RULES = (
    ("Contract", "Month-to-month", "Contract review", "Offer an appropriate 1- or 2-year contract with a rate-lock incentive."),
    ("TechSupport", "No", "Service support", "Offer proactive technical-support outreach."),
    ("OnlineSecurity", "No", "Service support", "Offer an online-security add-on or trial."),
)


def _matches(token: str, feature: str) -> bool:
    return feature == token if "_" in token else token in feature


def build_recommendations(shap_factors: Iterable[Dict[str, Any]], risk_tier: str,
                          early_warning_actions: Iterable[str] = ()) -> Dict[str, Any]:
    """Return de-duplicated suggestions; these are not causal treatment claims."""
    priority = {"Critical": "Immediate", "High": "High", "Medium": "Planned", "Low": "Standard"}.get(risk_tier, "Standard")
    recommendations: List[Dict[str, str]] = []
    used_categories = set()

    for factor in shap_factors:
        feature = str(factor.get("feature", ""))
        for token, category, action in RULES:
            if _matches(token, feature) and category not in used_categories:
                recommendations.append({"priority": priority, "category": category, "reason": feature, "action": action})
                used_categories.add(category)
        source, raw = factor.get("source_feature"), factor.get("raw_value")
        for src, value, category, action in RAW_RULES:
            if source == src and raw == value and category not in used_categories:
                recommendations.append({"priority": priority, "category": category, "reason": f"{src} = {value}", "action": action})
                used_categories.add(category)

    for action in early_warning_actions:
        if action and action not in {item["action"] for item in recommendations}:
            recommendations.append({"priority": priority, "category": "Early-warning playbook", "reason": "Rule-based warning trigger", "action": action})

    if not recommendations:
        recommendations.append({"priority": priority, "category": "Relationship management", "reason": "No dominant modifiable risk factor", "action": "Maintain regular value-focused customer outreach."})

    return {
        "priority": priority,
        "disclaimer": "Suggestions are rule-based decision support, not evidence that an intervention will prevent churn.",
        "recommendations": recommendations,
    }
