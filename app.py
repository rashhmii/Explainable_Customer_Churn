"""Flask REST API for explainable customer-churn scoring."""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from predict import ChurnInferenceEngine

from flask_cors import CORS


ID_PATTERN = re.compile(r'^[A-Za-z0-9._\-]{1,64}$')


def create_app(engine: ChurnInferenceEngine | None = None) -> Flask:
    """Create an app; dependency injection keeps API tests fast and isolated."""
    app = Flask(__name__)

    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin.strip()
    ]
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

    app.json.sort_keys = False
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_BYTES', 5 * 1024 * 1024))
    app.config['MAX_BATCH_ROWS'] = int(os.getenv('MAX_BATCH_ROWS', 2000))

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
        return jsonify(status='error', error='Internal Server Error', message='An unexpected error occurred.'), 500

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
            optional_features=['TotalCharges'],
            allowed_values=getattr(active_engine, 'allowed_values', None),
            champion_model=getattr(active_engine, 'metrics', {}).get('champion_model'),
            calibration_metrics=getattr(active_engine, 'metrics', {}).get('feature_5_calibration_metrics'),
            explanation_note='SHAP values explain the uncalibrated model score (log-odds); churn_probability is calibrated.',
            max_batch_rows=app.config['MAX_BATCH_ROWS'],
            risk_thresholds={'Low': '< 0.35', 'Medium': '0.35 - < 0.50', 'High': '0.50 - < 0.65', 'Critical': '>= 0.65'}
        )

    @app.post('/predict')
    def predict():
        active_engine = app.config['ENGINE']
        if active_engine is None:
            return service_unavailable()
        if not request.is_json:
            return jsonify(status='error', message='Content-Type must be application/json.'), 415
        payload: Any = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(status='error', message='Expected a JSON object.'), 400
        customer_id = str(payload.pop('customerID', 'NEW_CUSTOMER')).strip()
        if not ID_PATTERN.match(customer_id):
            return jsonify(status='error', message="customerID must be 1-64 characters of letters, digits, '.', '_' or '-'."), 400
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
                # dtype=str keeps raw text so validation (not pandas) decides what is valid, e.g. blank TotalCharges
                records = pd.read_csv(uploaded.stream, dtype=str, keep_default_na=False).to_dict(orient='records')
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                return jsonify(status='error', message=f'Invalid CSV: {exc}'), 400
        else:
            payload = request.get_json(silent=True)
            records = payload.get('customers') if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                return jsonify(status='error', message='Expected a JSON array or {"customers": [...]}.'), 400
        if not records:
            return jsonify(status='error', message='At least one customer record is required.'), 400
        if len(records) > app.config['MAX_BATCH_ROWS']:
            return jsonify(status='error', message=f"Batch has {len(records)} records; the limit is {app.config['MAX_BATCH_ROWS']}."), 413

        allow_partial = request.args.get('allow_partial', 'false').lower() in ('1', 'true', 'yes')
        results, rejected = [], []
        for index, record in enumerate(records):
            row = index + 1
            if not isinstance(record, dict):
                rejected.append({'row': row, 'message': 'Record must be an object.'})
                continue
            customer_id = str(record.get('customerID', f'ROW_{row}')).strip()
            if not ID_PATTERN.match(customer_id):
                rejected.append({'row': row, 'message': "customerID must be 1-64 characters of letters, digits, '.', '_' or '-'."})
                continue
            customer = {key: value for key, value in record.items() if key not in {'customerID', 'Churn'}}
            try:
                results.append(active_engine.predict_single(customer, customer_id=customer_id, generate_waterfall=False))
            except ValueError as exc:
                rejected.append({'row': row, 'customer_id': customer_id, 'message': str(exc)})

        if rejected and (not allow_partial or not results):
            return jsonify(status='error', message=f'{len(rejected)} of {len(records)} record(s) are invalid; '
                           'nothing was scored. Use ?allow_partial=true to score the valid rows.',
                           rejected=rejected[:100]), 400
        body = dict(status='success', count=len(results), data=results,
                    summary={'rows_received': len(records), 'rows_scored': len(results), 'rows_rejected': len(rejected),
                             'risk_tier_counts': dict(Counter(r['risk_tier'] for r in results))})
        if rejected:
            body['rejected'] = rejected[:100]
        return jsonify(body)


    return app


app = create_app()


if __name__ == '__main__':
    app.run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '5000')), debug=False)
