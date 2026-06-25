import matplotlib.pyplot as plt
import numpy as np
import mlflow
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay, classification_report
)


def plot_confusion_matrix(pipeline, X_test, y_test, run_id: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax, cmap="Purples", colorbar=False
    )
    ax.set_title("Confusion Matrix")
    path = "models/confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(path)


def plot_roc_curve(pipeline, X_test, y_test, run_id: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, color="#6366f1")
    ax.set_title("ROC Curve")
    path = "models/roc_curve.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(path)


def plot_precision_recall(pipeline, X_test, y_test, run_id: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax, color="#10b981"
    )
    ax.set_title("Precision-Recall Curve")
    path = "models/pr_curve.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(path)


def shap_analysis(pipeline, X_test):
    """Generate SHAP values for tree-based models."""
    # Extract preprocessed data (all steps except the final model)
    X_processed = pipeline[:-1].transform(X_test)

    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_processed, show=False)
    plt.title("SHAP Feature Importance")
    path = "models/shap_summary.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path


def print_classification_report(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
