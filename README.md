<p align="center">
  <h1 align="center">🛡️ Real-Time Fraud Detection Microservice</h1>
  <p align="center">
    An end-to-end machine learning system for detecting fraudulent transactions — from exploratory analysis to a production-ready, Dockerized REST API.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/XGBoost-E7652E?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge" alt="SHAP">
</p>

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [ML Pipeline & Methodology](#-ml-pipeline--methodology)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Technical Deep Dive](#-technical-deep-dive)
- [Development Journal](#-development-journal)
- [Future Roadmap](#-future-roadmap)

---

## ✨ Highlights

| Feature | Description |
|---|---|
| **End-to-End ML Pipeline** | Data ingestion → preprocessing → training → evaluation → serialization — fully automated |
| **Production REST API** | FastAPI service with single & batch prediction endpoints |
| **XGBoost Classifier** | Tuned gradient boosting model with cost-sensitive learning via `scale_pos_weight` |
| **Threshold Optimization** | Custom decision threshold (0.11) calibrated on PR-AUC to maximize recall without sacrificing precision |
| **Model Explainability** | SHAP-based feature importance analysis for transparent, auditable predictions |
| **Dockerized Deployment** | One-command containerized deployment for consistent cross-environment execution |
| **Structured Logging** | File-based logging system for monitoring and debugging in production |

---

## 🏛️ System Architecture

```
┌──────────────────┐
│  Client Request  │   JSON transaction payload
│  (single/batch)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FastAPI Server  │   Uvicorn ASGI · port 8000
│  /predict        │   Single transaction inference
│  /predict_batch  │   Batch transaction inference
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ FraudPredictor   │   Inference wrapper with threshold logic
│ (predictor.py)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   ML Pipeline    │   StandardScaler (Amount, Time)
│   (sklearn +     │         │
│    XGBoost)      │   XGBClassifier (tuned)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Response       │   { fraud_probability, fraud_prediction }
└──────────────────┘
```

---

## 📁 Project Structure

```
fraud-detection/
│
├── data/
│   └── creditcard.csv              # Kaggle credit card fraud dataset (284,807 transactions)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis & visualization
│   ├── 02_xgb_model.ipynb          # XGBoost model training & threshold tuning
│   └── 03_ensemble_model.ipynb     # Ensemble model experiments (RF, GB, XGB)
│
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI application with /predict & /predict_batch
│   ├── inference/
│   │   └── predictor.py            # FraudPredictor class — loads pipeline, applies threshold
│   ├── pipeline/
│   │   └── training_pipeline.py    # End-to-end training: load → split → preprocess → train → evaluate → save
│   ├── monitoring/
│   │   └── logger.py               # Structured file-based logging utility
│   ├── config/                     # Configuration management (extensible)
│   ├── data/                       # Data processing utilities (extensible)
│   └── models/                     # Model definitions (extensible)
│
├── models/
│   └── fraud_pipeline_v01.pkl      # Serialized sklearn Pipeline (StandardScaler + XGBoost)
│
├── tests/
│   └── test_api.py                 # API integration tests using requests
│
├── reports/
│   └── day{1-11}_notes.md          # Daily development journal & research notes
│
├── logs/                           # Runtime application logs
├── Dockerfile                      # Container configuration (Python 3.10)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md
```

---

## 🔬 ML Pipeline & Methodology

### Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions (492 fraudulent → **0.17% positive class**)
- **Features:** 28 PCA-transformed features (V1–V28), `Time`, `Amount`, and binary `Class` label

### Problem Framing

Credit card fraud detection is fundamentally a **threshold optimization problem** on a heavily imbalanced dataset. Key challenges addressed:

1. **Extreme class imbalance** — 99.83% legitimate vs. 0.17% fraud
2. **Accuracy is misleading** — A naive "all legitimate" classifier scores 99.83% accuracy
3. **Asymmetric misclassification costs** — Missing fraud (FN) >> false alarms (FP)
4. **Non-linear fraud patterns** — Fraudsters exhibit complex behavioral patterns that linear models miss

### Methodology

```
1. EDA & Statistical Analysis
   └── Distribution analysis, class imbalance assessment, correlation study

2. Baseline Modeling (Logistic Regression)
   └── Recall ≈ 70%  |  Precision ≈ 88%  |  PR-AUC ≈ 0.77

3. Resampling Experiments (SMOTE, class weights, undersampling)
   └── Conclusion: Aggressive resampling destroys precision (↓ to 5%) — not viable

4. Threshold Optimization
   └── Lowering threshold from 0.5 → 0.1 improved recall to 81% while maintaining 83% precision

5. Tree-based Ensemble Models
   ├── Random Forest  → Recall ≈ 81.6%  |  Precision ≈ 91.9%  |  PR-AUC ≈ 0.88
   └── Gradient Boosting → Recall ≈ 76.5%  |  Precision ≈ 86.2%  |  PR-AUC ≈ 0.71

6. XGBoost with Hyperparameter Tuning
   └── scale_pos_weight, learning_rate=0.3, max_depth=4, n_estimators=200

7. Model Explainability (SHAP)
   └── V14 is the most influential feature; Amount/Time have minimal impact

8. Production Pipeline (sklearn Pipeline)
   └── StandardScaler (Amount, Time) → XGBClassifier → Serialized with joblib

9. API Development & Dockerization
   └── FastAPI + Uvicorn → Docker container
```

### Preprocessing Pipeline

| Step | Transformer | Features |
|---|---|---|
| Scaling | `StandardScaler` | `Amount`, `Time` |
| Passthrough | — | `V1` – `V28` (already PCA-transformed) |

### Model Configuration

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.3,
    max_depth=4,
    subsample=0.9,
    colsample_bylevel=0.8,
    scale_pos_weight=sqrt(neg/pos),   # Cost-sensitive learning
    random_state=42,
    n_jobs=-1
)
```

### Decision Threshold

The default classification threshold of `0.5` is **not optimal** for imbalanced fraud detection. After threshold analysis:

- **Production threshold:** `0.11`
- Calibrated to maximize **recall** (catching more fraud) while keeping **precision** practically useful

---

## 📊 Model Performance

### Final Model — Tuned XGBoost

| Metric | Score |
|---|---|
| **PR-AUC** | **0.885** |
| **Recall** | **87.8%** |
| **Precision** | **85.1%** |
| **Decision Threshold** | 0.11 |

### Model Comparison Summary

| Model | Recall | Precision | PR-AUC | Notes |
|---|---|---|---|---|
| Logistic Regression | 70% | 88% | 0.77 | Baseline — misses non-linear patterns |
| LR + SMOTE | 95% | 5% | — | Unusable — thousands of false positives |
| Random Forest | 81.6% | 91.9% | 0.88 | Strong with default parameters |
| Gradient Boosting | 76.5% | 86.2% | 0.71 | Requires tuning to unlock potential |
| **XGBoost (tuned)** | **87.8%** | **85.1%** | **0.885** | **Selected for production** |

### Key SHAP Insights

- **V14** is the most important feature — low values strongly increase fraud probability
- **V4, V12, V3** are significant secondary contributors
- **Transaction Amount** has minimal predictive power — fraud detection relies on behavioral patterns captured by PCA components, not raw transaction size

---

## 🔌 API Reference

### Single Prediction

```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Time": 50000,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536346,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

**Response:**
```json
{
  "fraud_probability": 0.0023,
  "fraud_prediction": 0
}
```

### Batch Prediction

```http
POST /predict_batch
Content-Type: application/json
```

**Request Body:** Array of transaction objects (same schema as above)

**Response:**
```json
{
  "probablities": [0.0023, 0.87, 0.001],
  "predictions": [0, 1, 0]
}
```

### Interactive Docs

Once the server is running, visit **http://localhost:8000/docs** for the auto-generated Swagger UI.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- Docker *(optional, for containerized deployment)*

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/ankitshri00132/Real-Time-Fraud-Detection-Microservice.git
cd Real-Time-Fraud-Detection-Microservice

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Place creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into the data/ directory

# 5. Train the model (optional — a pre-trained pipeline is included)
python -m src.pipeline.training_pipeline

# 6. Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Quick Test

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":50000,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62}'
```

Or run the included test script:

```bash
python tests/test_api.py
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t fraud-detection-api .

# Run the container
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api
```

The API will be available at `http://localhost:8000`.

**Dockerfile Summary:**

| Layer | Detail |
|---|---|
| Base Image | `python:3.10` |
| Working Directory | `/app` |
| Dependencies | Installed from `requirements.txt` |
| Entrypoint | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` |
| Exposed Port | `8000` |

---

## 🧠 Technical Deep Dive

### Why Threshold Tuning Over SMOTE?

| Approach | Recall | Precision | False Positives | Verdict |
|---|---|---|---|---|
| SMOTE / class_weight | ~95% | ~5% | Thousands | ❌ Unusable |
| **Threshold tuning (0.11)** | **~87.8%** | **~85.1%** | **Very few** | ✅ Production-ready |

Threshold tuning preserves the model's learned decision boundary while adjusting the operating point — making it far more stable and controllable than data-level resampling.

### Why XGBoost?

- **Handles imbalance natively** via `scale_pos_weight` — no need to modify the dataset
- **Captures non-linear patterns** that linear models (Logistic Regression) miss
- **Regularization built-in** — reduces overfitting on noisy financial data
- **Fast inference** — critical for real-time transaction scoring

### Why PCA Features Matter More Than Amount

SHAP analysis revealed that PCA-transformed features (especially **V14**) are far more predictive than raw `Amount` or `Time`. This is because:

- Fraudsters deliberately keep transaction amounts moderate to evade rule-based systems
- PCA components encode complex behavioral signals (spending patterns, merchant relationships, temporal context)
- The model learns from these hidden behavioral signatures rather than simple heuristics

---

## 📓 Development Journal

This project was built iteratively over 11 days and will be continued till completion. Each day's research, experiments, and learnings are documented in `reports/`:

| Day | Focus Area |
|---|---|
| Day 1 | EDA, class imbalance analysis, problem framing |
| Day 2 | Evaluation metrics deep dive (ROC-AUC vs PR-AUC) |
| Day 3 | Resampling experiments (SMOTE, undersampling, class weights) |
| Day 4 | Threshold optimization as the core strategy |
| Day 5 | Random Forest & Gradient Boosting experiments |
| Day 6 | Hyperparameter tuning & model comparison |
| Day 7 | XGBoost with `scale_pos_weight` & tuning |
| Day 8 | SHAP explainability & business interpretation |
| Day 9 | sklearn Pipeline for reproducible training |
| Day 10 | FastAPI microservice with single & batch endpoints |
| Day 11 | Docker containerization for deployment |

---

## 🗺️ Future Roadmap

-  **Data drift monitoring** — Detect distribution shifts in incoming transactions
-  **A/B threshold testing** — Compare threshold strategies in production
-  **Database integration** — Log predictions to PostgreSQL/MongoDB for audit trails
-  **Authentication & rate limiting** — Secure the API for production environments
- **CI/CD pipeline** — GitHub Actions for automated testing & deployment
-  **Prometheus + Grafana** — Real-time model performance dashboards
-  **Feature store integration** — Centralized feature management for retraining
-  **Multi-threshold strategy** — Auto-block / manual review / allow tiers based on probability ranges

---
