from modules.recommendations import build_recommendations


def cats(factors):
    return [r["category"] for r in build_recommendations(factors, "High")["recommendations"]]


def test_month_to_month_detected_via_raw_value_with_one_hot_reference_dropped():
    f = [{"feature": "Contract_Two year", "source_feature": "Contract", "raw_value": "Month-to-month"}]
    assert "Contract review" in cats(f)


def test_tech_support_rule_ignores_no_internet_service():
    f = [{"feature": "TechSupport_No internet service", "source_feature": "TechSupport", "raw_value": "No internet service"}]
    assert "Service support" not in cats(f)


def test_tech_support_rule_fires_for_customer_without_support():
    f = [{"feature": "TechSupport_Yes", "source_feature": "TechSupport", "raw_value": "No"}]
    assert "Service support" in cats(f)
