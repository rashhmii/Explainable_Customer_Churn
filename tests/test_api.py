"""Black-box API contract tests using Flask's test client."""

import io

import pytest
from werkzeug.test import Client
from werkzeug.wrappers import Response

from app import create_app


class FakeEngine:
    required_features = ['tenure', 'Contract', 'MonthlyCharges']

    def predict_single(self, customer_data, customer_id, generate_waterfall=False):
        missing = set(self.required_features) - set(customer_data)
        if missing:
            raise ValueError('Missing required customer fields: ' + ', '.join(sorted(missing)))
        return {
            'customer_id': customer_id,
            'churn_probability': 0.7,
            'risk_tier': 'Critical',
            'retention_recommendations': {'priority': 'Immediate', 'recommendations': []},
        }


@pytest.fixture()
def client():
    # Use the WSGI client directly so the contract tests are not coupled to
    # Flask's optional test-client implementation details.
    return Client(create_app(FakeEngine()), Response)


def customer():
    return {'customerID': 'C-1', 'tenure': 2, 'Contract': 'Month-to-month', 'MonthlyCharges': 90.0}


def test_health_and_model_info(client):
    assert client.get('/health').get_json()['model_loaded'] is True
    info = client.get('/model-info').get_json()
    assert info['feature_count'] == 3
    assert 'Critical' in info['risk_thresholds']


def test_predict_returns_scored_customer(client):
    response = client.post('/predict', json=customer())
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['data']['customer_id'] == 'C-1'


def test_predict_rejects_invalid_json_and_missing_fields(client):
    assert client.post('/predict', json=['not', 'an', 'object']).status_code == 400
    response = client.post('/predict', json={'tenure': 2})
    assert response.status_code == 400
    assert 'Missing required' in response.get_json()['message']


def test_batch_supports_json_and_csv(client):
    response = client.post('/predict-batch', json={'customers': [customer(), {**customer(), 'customerID': 'C-2'}]})
    assert response.status_code == 200
    assert response.get_json()['count'] == 2

    csv = 'customerID,tenure,Contract,MonthlyCharges\nC-3,2,Month-to-month,90.0\n'
    response = client.post('/predict-batch', data={'file': (io.BytesIO(csv.encode()), 'customers.csv')})
    assert response.status_code == 200
    assert response.get_json()['data'][0]['customer_id'] == 'C-3'
