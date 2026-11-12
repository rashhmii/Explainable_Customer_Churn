# Explainable Customer Churn Prediction

This project predicts Telco customer churn, explains model predictions with SHAP,
assigns early-warning risk tiers, and supplies decision-support simulations.

## Implemented capabilities

- Leak-aware preprocessing and comparison of Logistic Regression, Random Forest,
  and XGBoost models.
- Platt-scaled probability calibration and evaluation plots.
- Global and customer-level SHAP explanations.
- Rule-based early-warning triggers and retention playbooks.
- CLI single-customer and batch scoring via `predict.py`.
- Flask REST API with validated single and batch scoring endpoints.
- UI-independent what-if simulation, SHAP-aware retention recommendations, and
  multi-scenario risk-evolution comparison in `modules/`.

## Run

```powershell
python -m pip install -r requirements.txt
python main.py                 # train and generate artifacts
python predict.py --sample     # score an illustrative customer
python -m pytest -q            # run tests
python app.py                  # start the API on http://127.0.0.1:5000
```

## API

`GET /health` reports whether model artifacts loaded. `GET /model-info` returns
the required feature contract. `POST /predict` accepts one JSON customer record
(optionally with `customerID`); `POST /predict-batch` accepts a JSON array,
`{"customers": [...]}`, or a `file` CSV upload. Run the API tests with
`python -m pytest tests/test_api.py -q`.

The standard Telco dataset is a static snapshot. Early-warning and all simulated
interventions are decision-support demonstrations, not proof of behaviour over
time or causal retention effects.

## Next milestone

The remaining planned feature is the Streamlit dashboard, which should consume
the completed modules after the backend and artifacts have been verified.
