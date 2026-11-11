"""Constrained, user-controlled churn what-if simulations."""

from copy import deepcopy
from typing import Any, Dict


# Identifiers and outcomes must never be changed in a simulated intervention.
IMMUTABLE_FIELDS = {"customerID", "Churn", "gender", "SeniorCitizen", "Dependents"}


def simulate_changes(engine: Any, customer: Dict[str, Any], changes: Dict[str, Any],
                     customer_id: str = "SIMULATION") -> Dict[str, Any]:
    """Score an original profile and one feasible, manually supplied scenario.

    ``engine`` is deliberately duck-typed so this module stays independent of
    the UI and can be unit-tested with a small scoring stub.
    """
    forbidden = set(changes).intersection(IMMUTABLE_FIELDS)
    if forbidden:
        raise ValueError("Immutable fields cannot be changed: " + ", ".join(sorted(forbidden)))

    unknown = set(changes).difference(customer)
    if unknown:
        raise ValueError("Unknown customer fields: " + ", ".join(sorted(unknown)))

    original = deepcopy(customer)
    scenario = deepcopy(customer)
    scenario.update(changes)

    baseline = engine.predict_single(original, customer_id=customer_id, generate_waterfall=False)
    simulated = engine.predict_single(scenario, customer_id=f"{customer_id}_WHAT_IF", generate_waterfall=False)
    original_probability = float(baseline["churn_probability"])
    simulated_probability = float(simulated["churn_probability"])

    return {
        "customer_id": customer_id,
        "original_probability": original_probability,
        "simulated_probability": simulated_probability,
        "risk_change": simulated_probability - original_probability,
        "risk_change_percentage_points": round((simulated_probability - original_probability) * 100, 2),
        "original_risk_tier": baseline["risk_tier"],
        "simulated_risk_tier": simulated["risk_tier"],
        "changed_fields": {key: {"from": original[key], "to": scenario[key]} for key in changes},
        "simulated_customer": scenario,
    }
