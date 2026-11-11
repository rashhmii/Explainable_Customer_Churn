from modules.recommendations import build_recommendations
from modules.risk_evolution import compare_risk_evolution
from modules.what_if import simulate_changes


class StubEngine:
    def predict_single(self, customer, customer_id, generate_waterfall=False):
        probability = 0.70 if customer["Contract"] == "Month-to-month" else 0.30
        return {"churn_probability": probability, "risk_tier": "Critical" if probability >= 0.65 else "Low"}


CUSTOMER = {"Contract": "Month-to-month", "MonthlyCharges": 90.0, "TechSupport": "No", "OnlineSecurity": "No", "PaymentMethod": "Electronic check"}


def test_what_if_compares_same_customer():
    result = simulate_changes(StubEngine(), CUSTOMER, {"Contract": "One year"})
    assert result["original_probability"] == 0.70
    assert result["simulated_probability"] == 0.30
    assert result["risk_change_percentage_points"] == -40.0


def test_what_if_rejects_immutable_changes():
    try:
        simulate_changes(StubEngine(), CUSTOMER, {"customerID": "changed"})
    except ValueError as exc:
        assert "Immutable" in str(exc)
    else:
        raise AssertionError("Expected immutable-field validation")


def test_recommendations_use_shap_and_warning_actions():
    result = build_recommendations([{"feature": "Contract_Month-to-month"}], "Critical", ["Call customer"])
    assert result["priority"] == "Immediate"
    assert any(item["category"] == "Contract review" for item in result["recommendations"])


def test_risk_evolution_includes_current_and_interventions():
    results = compare_risk_evolution(StubEngine(), CUSTOMER)
    assert results[0]["scenario"] == "Current profile"
    assert any(row["scenario"] == "Contract commitment" for row in results)
