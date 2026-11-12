"""API tests against the REAL saved artifacts (no fake engine, nothing fabricated)."""
import io
import math
import os
import tempfile

import pandas as pd
import pytest

from app import create_app
from predict import get_default_sample_customer

DATASET = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["MAX_BATCH_ROWS"] = 300
    return app.test_client()


@pytest.fixture
def customer():
    return get_default_sample_customer()


def test_health_and_model_info(client):
    assert client.get("/health").get_json()["model_loaded"] is True
    info = client.get("/model-info").get_json()
    assert info["feature_count"] == 19 and info["champion_model"] == "XGBoost"
    assert "Month-to-month" in info["allowed_values"]["Contract"]


def test_predict_real_engine(client, customer):
    d = client.post("/predict", json={**customer, "customerID": "C-1"}).get_json()["data"]
    assert 0 <= d["churn_probability"] <= 1 and d["risk_tier"] in {"Low", "Medium", "High", "Critical"}
    assert d["shap_explanation"]["explained_model_probability"] is not None
    assert d["shap_explanation"]["top_churn_accelerators"][0]["raw_value"] is not None
    assert any(r["category"] == "Contract review" for r in d["retention_recommendations"]["recommendations"])


@pytest.mark.parametrize("mutation", [
    {"Contract": "Weekly"}, {"SeniorCitizen": "maybe"}, {"tenure": -5}, {"tenure": "abc"}, {"tenure": "inf"},
    {"tenure": True}, {"MonthlyCharges": None}, {"gender": "female"}, {"Contract": None}, {"TotalCharges": "abc"},
])
def test_invalid_values_rejected_400(client, customer, mutation):
    r = client.post("/predict", json={**customer, **mutation})
    assert r.status_code == 400 and r.get_json()["status"] == "error"


def test_blank_total_charges_is_valid(client, customer):
    for blank in (" ", "", None):
        assert client.post("/predict", json={**customer, "TotalCharges": blank}).status_code == 200


def test_senior_citizen_forms_are_equivalent(client, customer):
    p = lambda v: client.post("/predict", json={**customer, "SeniorCitizen": v}).get_json()["data"]["churn_probability"]
    assert p("Yes") == p(1) == p(True) == p("1") != p(0)


def test_http_edge_cases(client, customer):
    assert client.post("/predict", data="x", content_type="text/plain").status_code == 415
    assert client.post("/predict", data="{bad", content_type="application/json").status_code == 400
    assert client.post("/predict", json={**customer, "customerID": "../../x"}).status_code == 400
    assert client.get("/nope").status_code == 404


def test_batch_full_dataset_slice_with_blank_total_charges(client):
    df = pd.read_csv(DATASET)
    blank = df[df.TotalCharges.str.strip() == ""].head(3)
    sub = pd.concat([df.head(40), blank])
    r = client.post("/predict-batch", data={"file": (io.BytesIO(sub.to_csv(index=False).encode()), "x.csv")})
    assert r.status_code == 200 and r.get_json()["count"] == 43


def test_batch_matches_cli_predict_batch(client):
    from predict import ChurnInferenceEngine
    df = pd.read_csv(DATASET).head(40)
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "i.csv"), os.path.join(d, "o.csv")
        df.to_csv(src, index=False)
        cli = ChurnInferenceEngine().predict_batch(src, dst).set_index("customerID")["churn_probability"]
    j = client.post("/predict-batch", data={"file": (io.BytesIO(df.to_csv(index=False).encode()), "x.csv")}).get_json()
    api = {x["customer_id"]: x["churn_probability"] for x in j["data"]}
    assert all(math.isclose(api[k], v, abs_tol=1e-9) for k, v in cli.items())


def test_batch_reports_every_bad_row_and_partial_mode(client, customer):
    recs = [{**customer, "customerID": "A"}, {**customer, "customerID": "B", "Contract": "Weekly"},
            {**customer, "customerID": "C", "tenure": "abc"}]
    r = client.post("/predict-batch", json=recs)
    assert r.status_code == 400 and [x["row"] for x in r.get_json()["rejected"]] == [2, 3]
    ok = client.post("/predict-batch?allow_partial=true", json=recs).get_json()
    assert ok["count"] == 1 and len(ok["rejected"]) == 2 and ok["summary"]["rows_rejected"] == 2


def test_batch_limits(client, customer):
    assert client.post("/predict-batch", json=[customer] * 301).status_code == 413
    assert client.post("/predict-batch", data={"file": (io.BytesIO(b""), "e.csv")}).status_code == 400
