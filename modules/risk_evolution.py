"""Predefined feasible scenario comparison for churn-risk evolution."""

from typing import Any, Dict, List

from modules.what_if import simulate_changes


def default_scenarios(customer: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create business-plausible scenarios without modifying protected fields."""
    scenarios: Dict[str, Dict[str, Any]] = {"Current profile": {}}
    if customer.get("Contract") == "Month-to-month":
        scenarios["Contract commitment"] = {"Contract": "One year"}
    if float(customer.get("MonthlyCharges", 0)) >= 75:
        scenarios["Plan-price review"] = {"MonthlyCharges": round(float(customer["MonthlyCharges"]) * 0.85, 2)}
    service_changes = {}
    if customer.get("TechSupport") == "No":
        service_changes["TechSupport"] = "Yes"
    if customer.get("OnlineSecurity") == "No":
        service_changes["OnlineSecurity"] = "Yes"
    if service_changes:
        scenarios["Support and security"] = service_changes
    if customer.get("PaymentMethod") == "Electronic check":
        scenarios["Auto-pay enrolment"] = {"PaymentMethod": "Bank transfer (automatic)"}
    return scenarios


def compare_risk_evolution(engine: Any, customer: Dict[str, Any], customer_id: str = "SIMULATION") -> List[Dict[str, Any]]:
    """Score the current profile and each independent hypothetical scenario."""
    rows = []
    for name, changes in default_scenarios(customer).items():
        result = simulate_changes(engine, customer, changes, customer_id=customer_id)
        rows.append({
            "scenario": name,
            "predicted_churn_probability": result["simulated_probability"],
            "risk_tier": result["simulated_risk_tier"],
            "risk_change_percentage_points": result["risk_change_percentage_points"],
            "changes": changes,
        })
    return rows
