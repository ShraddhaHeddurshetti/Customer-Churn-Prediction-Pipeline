import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Fix TotalCharges — convert spaces to NaN, then to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2. Drop the 11 rows with missing TotalCharges (new customers with tenure=0)
    df = df.dropna(subset=["TotalCharges"])

    # 3. Drop customerID — not a feature
    df = df.drop(columns=["customerID"])

    # 4. Encode target: Yes → 1, No → 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # 5. Convert binary Yes/No columns to 1/0
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)

    # 6. SeniorCitizen is already 0/1 — verify
    assert df["SeniorCitizen"].isin([0, 1]).all()

    print(f"Cleaned shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.1%}")
    return df


def split_data(df: pd.DataFrame, target: str = "Churn",
               test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,            # preserve class ratio in both splits
        random_state=random_state
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


def get_feature_types(X: pd.DataFrame):
    """Identify numeric vs categorical columns."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Numeric ({len(num_cols)}): {num_cols}")
    print(f"Categorical ({len(cat_cols)}): {cat_cols}")
    return num_cols, cat_cols
