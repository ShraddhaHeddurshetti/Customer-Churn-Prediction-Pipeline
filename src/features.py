import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to add new features before encoding."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # New feature 1: avg monthly spend
        X["avg_monthly_spend"] = np.where(
            X["tenure"] > 0,
            X["TotalCharges"] / X["tenure"],
            X["MonthlyCharges"]   # fallback for new customers
        )

        # New feature 2: is new customer (< 6 months)
        X["is_new_customer"] = (X["tenure"] < 6).astype(int)

        # New feature 3: has multiple services
        X["has_multiple_services"] = (
            (X["PhoneService"] == 1) &
            (X["InternetService"] != "No")
        ).astype(int)

        # New feature 4: high value customer
        X["is_high_value"] = (
            X["MonthlyCharges"] > X["MonthlyCharges"].median()
        ).astype(int)

        return X


def build_full_pipeline(model, num_cols, cat_cols):
    """
    Full sklearn pipeline:
    1. Feature engineering
    2. Preprocessing (scale numerics, encode categoricals)
    3. Model
    """
    # After feature engineering, new numeric columns are added
    extended_num = num_cols + [
        "avg_monthly_spend", "is_new_customer",
        "has_multiple_services", "is_high_value"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), extended_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    pipeline = Pipeline([
        ("engineer", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def apply_smote(X_train, y_train, random_state=42):
    """Oversample minority class to fix imbalance."""
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — Class 0: {(y_res == 0).sum()} | Class 1: {(y_res == 1).sum()}")
    return X_res, y_res
