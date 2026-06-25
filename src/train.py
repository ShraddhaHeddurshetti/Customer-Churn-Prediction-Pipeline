import mlflow
import mlflow.sklearn
import joblib
import os
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.preprocess import load_and_clean, split_data, get_feature_types
from src.features import build_full_pipeline

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("churn-prediction-v1")

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=3,   # handles imbalance: ~73/27 ratio
        random_state=42, eval_metric="logloss",
        use_label_encoder=False
    ),
}


def train_and_log(model_name: str, model, X_train, X_test, y_train, y_test,
                  num_cols, cat_cols) -> tuple:
    """Train one model, log everything to MLflow, return (run_id, metrics)."""
    pipeline = build_full_pipeline(model, num_cols, cat_cols)

    with mlflow.start_run(run_name=model_name) as run:
        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
        }

        # Log params
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, "model",
                                  registered_model_name=f"churn-{model_name}")

        # Save locally
        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        joblib.dump(pipeline, models_dir / f"{model_name}.pkl")

        print(f"\n{model_name}: ROC-AUC={metrics['roc_auc']} | F1={metrics['f1']}")
        return run.info.run_id, metrics


def run_all_training():
    # Load data
    data_path = str(ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    num_cols, cat_cols = get_feature_types(X_train)

    results = {}
    for name, model in MODELS.items():
        run_id, metrics = train_and_log(
            name, model, X_train, X_test, y_train, y_test, num_cols, cat_cols
        )
        results[name] = {"run_id": run_id, **metrics}

    # Print comparison table
    comparison = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON:")
    print(comparison.to_string())

    best_model = comparison.index[0]
    print(f"\nBest model: {best_model}")
    return best_model


if __name__ == "__main__":
    run_all_training()
