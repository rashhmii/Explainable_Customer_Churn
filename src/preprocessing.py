"""
Module: Data Ingestion & Preprocessing
Project: Explainable Customer Churn Prediction Using Data Analytics
Center for Data Analytics & Digital Intelligence
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw Telco customer churn dataset from CSV.
    """
    df = pd.read_csv(filepath)
    return df


def clean_churn_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans raw Telco churn dataframe:
      - Coerces TotalCharges blanks/spaces to numeric NaN.
      - Maps Churn string ('Yes', 'No') to binary (1, 0).
      - Isolates customer IDs for retention tracking.
    """
    df_clean = df.copy()

    # TotalCharges coercion
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'].replace(r'^\s*$', np.nan, regex=True),
            errors='coerce'
        )

    # Customer ID extraction
    customer_ids = df_clean['customerID'] if 'customerID' in df_clean.columns else pd.Series(df_clean.index, name='customerID')

    # Target variable mapping
    target_col = 'Churn'
    if target_col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[target_col]):
            y = df_clean[target_col].astype(int)
        else:
            y = df_clean[target_col].astype(str).str.strip().map({
                'Yes': 1, 'No': 0, '1': 1, '0': 0, 'True': 1, 'False': 0
            }).fillna(0).astype(int)
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Drop target and identifier from feature set
    cols_to_drop = [target_col]
    if 'customerID' in df_clean.columns:
        cols_to_drop.append('customerID')
    
    X = df_clean.drop(columns=cols_to_drop)

    # Ensure SeniorCitizen is treated as categorical
    if 'SeniorCitizen' in X.columns:
        X['SeniorCitizen'] = X['SeniorCitizen'].astype(str)

    return X, y, customer_ids


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Constructs an scikit-learn ColumnTransformer ensuring strict leak-free processing:
      - Numerical: Median imputation + StandardScaler
      - Categorical: Most frequent imputation + OneHotEncoder (drop='first')
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def get_clean_feature_names(preprocessor: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """
    Extracts human-readable feature names from the fitted ColumnTransformer.
    """
    try:
        raw_names = preprocessor.get_feature_names_out()
        cleaned_names = []
        for name in raw_names:
            # Strip transformer prefixes like 'num__' and 'cat__'
            if name.startswith('num__'):
                cleaned_names.append(name.replace('num__', ''))
            elif name.startswith('cat__'):
                cleaned_names.append(name.replace('cat__', ''))
            else:
                cleaned_names.append(name)
        return cleaned_names
    except Exception:
        # Fallback if get_feature_names_out encounters an issue
        ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_encoded_names = list(ohe.get_feature_names_out(cat_cols))
        return list(num_cols) + cat_encoded_names


def prepare_data_pipeline(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes end-to-end data ingestion, validation, stratified splitting, and transformation.
    """
    df = load_raw_data(filepath)
    X, y, customer_ids = clean_churn_dataset(df)

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()

    # Stratified Train-Test Split (80/20)
    (
        X_train_raw, X_test_raw,
        y_train, y_test,
        ids_train, ids_test
    ) = train_test_split(
        X, y, customer_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Build and fit preprocessor strictly on X_train_raw to prevent data leakage
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    feature_names = get_clean_feature_names(preprocessor, num_cols, cat_cols)

    return {
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'y_train': y_train,
        'y_test': y_test,
        'ids_train': ids_train,
        'ids_test': ids_test,
        'X_train_trans': X_train_trans,
        'X_test_trans': X_test_trans,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }
