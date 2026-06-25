# 🔮 Customer Churn Prediction Pipeline

> End-to-end ML pipeline to predict customer churn using the Telco Customer Churn dataset.  
> Stack: Python · scikit-learn · XGBoost · MLflow · SHAP · Streamlit

---

## 🏆 Results

| Model | ROC-AUC | F1 | Recall |
|---|---|---|---|
| XGBoost | ~0.85 | ~0.63 | ~0.79 |
| Random Forest | ~0.83 | ~0.61 | ~0.76 |
| Logistic Regression | ~0.83 | ~0.60 | ~0.75 |

> ⚠️ **Key finding:** Month-to-month contracts churn at **42%** vs only **3%** for 2-year contracts.

---

## 📁 Project Structure

```
customer-churn-pipeline/
├── data/
│   ├── raw/              # WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb      # exploratory analysis
├── src/
│   ├── preprocess.py     # data cleaning + train/test split
│   ├── features.py       # feature engineering + sklearn pipeline
│   ├── train.py          # MLflow training (3 models)
│   └── evaluate.py       # confusion matrix, ROC, SHAP
├── app/
│   └── app.py            # Streamlit scoring UI
├── models/               # saved .pkl files (after training)
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-pipeline.git
cd Customer Churn Prediction Pipeline

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Dataset

Go to [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Download and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`

Or via Kaggle API:
```bash
kaggle datasets download -d blastchar/telco-customer-churn --unzip -p data/raw/
```

### 3. Run EDA Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train All Models

```bash
python -m src.train
```

This trains Logistic Regression, Random Forest, and XGBoost — logging all metrics + artifacts to MLflow.

### 5. View MLflow Dashboard

```bash
mlflow ui
# Opens at http://127.0.0.1:5000
```

### 6. Run Streamlit App

```bash
streamlit run app/app.py
```

---

## 🧠 Feature Engineering

Custom features added in `src/features.py`:

| Feature | Formula | Why |
|---|---|---|
| `avg_monthly_spend` | TotalCharges / tenure | Detects high-spend short-tenure risk |
| `is_new_customer` | tenure < 6 | New customers churn at higher rates |
| `has_multiple_services` | PhoneService & Internet != No | Service bundling reduces churn |
| `is_high_value` | MonthlyCharges > median | High-value segment worth retaining |

---

## 📊 App Features

- **Single Customer Mode** — fill in customer details, get instant churn probability + gauge chart
- **Batch Mode** — upload a CSV, score all customers, download results
- **Action Recommendations** — High / Medium / Low risk with business actions suggested

---

## 🔗 Dataset

[IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
7,043 customers · 21 features · 26.5% churn rate
