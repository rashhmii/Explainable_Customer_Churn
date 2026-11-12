"""Flask REST API for explainable customer-churn scoring."""

from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from predict import ChurnInferenceEngine


def create_app(engine: ChurnInferenceEngine | None = None) -> Flask:
    """Create an app; dependency injection keeps API tests fast and isolated."""
    app = Flask(__name__)
    app.json.sort_keys = False

    if engine is None:
        try:
            engine = ChurnInferenceEngine(os.getenv('ARTIFACTS_DIR', 'artifacts'))
        except Exception as exc:  # App stays observable even before training.
            app.logger.exception('Unable to load model artifacts')
            app.config['MODEL_ERROR'] = str(exc)
            engine = None
    app.config['ENGINE'] = engine

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        return jsonify(status='error', error=exc.name, message=exc.description), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        app.logger.exception('Unhandled API error')
        return jsonify(status='error', error='Internal Server Error', message=str(exc)), 500

    def service_unavailable():
        return jsonify(
            status='error', error='Service Unavailable',
            message=app.config.get('MODEL_ERROR', 'Model artifacts are not loaded.')
        ), 503

    @app.get('/health')
    def health():
        loaded = app.config['ENGINE'] is not None
        return jsonify(status='ok' if loaded else 'degraded', model_loaded=loaded, service='Customer Churn API'), 200 if loaded else 503

    @app.get('/model-info')
    def model_info():
        active_engine = app.config['ENGINE']
        if active_engine is None:
            return service_unavailable()
        return jsonify(
            model_loaded=True,
            features=active_engine.required_features,
            feature_count=len(active_engine.required_features),
            risk_thresholds={'Low': '< 0.35', 'Medium': '0.35 - < 0.50', 'High': '0.50 - < 0.65', 'Critical': '>= 0.65'}
        )

    @app.post('/predict')
    def predict():
        active_engine = app.config['ENGINE']
        if active_engine is None:
            return service_unavailable()
        payload: Any = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(status='error', message='Expected a JSON object.'), 400
        customer_id = str(payload.pop('customerID', 'NEW_CUSTOMER'))
        try:
            result = active_engine.predict_single(payload, customer_id=customer_id, generate_waterfall=False)
        except ValueError as exc:
            return jsonify(status='error', message=str(exc)), 400
        return jsonify(status='success', data=result)

    @app.post('/predict-batch')
    def predict_batch():
        active_engine = app.config['ENGINE']
        if active_engine is None:
            return service_unavailable()
        if 'file' in request.files:
            uploaded = request.files['file']
            if not uploaded.filename or not uploaded.filename.lower().endswith('.csv'):
                return jsonify(status='error', message='Upload a non-empty .csv file.'), 400
            try:
                records = pd.read_csv(uploaded.stream).to_dict(orient='records')
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                return jsonify(status='error', message=f'Invalid CSV: {exc}'), 400
        else:
            payload = request.get_json(silent=True)
            records = payload.get('customers') if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                return jsonify(status='error', message='Expected a JSON array or {"customers": [...]}.') , 400
        if not records:
            return jsonify(status='error', message='At least one customer record is required.'), 400

        results = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                return jsonify(status='error', message=f'Record {index} must be an object.'), 400
            customer_id = str(record.get('customerID', f'ROW_{index + 1}'))
            customer = {key: value for key, value in record.items() if key not in {'customerID', 'Churn'}}
            try:
                results.append(active_engine.predict_single(customer, customer_id=customer_id, generate_waterfall=False))
            except ValueError as exc:
                return jsonify(status='error', message=f'Record {index}: {exc}'), 400
        return jsonify(status='success', count=len(results), data=results)

    return app


app = create_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=False)
