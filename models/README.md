# Models

Trained `.pkl` files are saved here after running:

```bash
python -m src.train
```

Files generated:
- `logistic_regression.pkl`
- `random_forest.pkl`
- `xgboost.pkl`  ← used by Streamlit app
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `shap_summary.png`

These files are in `.gitignore` (large binaries). Regenerate locally by running training.
