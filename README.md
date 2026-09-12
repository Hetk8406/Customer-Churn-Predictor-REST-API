# Customer Churn Predictor using RFM, XGBoost & FastAPI

This repository contains a complete end-to-end Machine Learning pipeline to predict customer churn for an e-commerce retailer. Using the transaction-level **Online Retail II** dataset, we clean the transactions, engineer customer-level behavioral features using **RFM (Recency, Frequency, Monetary)** analysis, train and evaluate machine learning models (including XGBoost), explain predictions globally and locally using **SHAP**, and expose the model through a production-ready **FastAPI** service and **Streamlit** dashboard.

---

## 📌 Project Overview
Customer churn is one of the most critical metrics in e-commerce. Retaining an existing customer is significantly cheaper than acquiring a new one. This project demonstrates how to turn raw, unlabeled transaction logs into a binary classification problem (Churn vs. Active) and deploy a predictive service to flag at-risk customers.

### 💼 Business Problem
In e-commerce, customers do not "cancel" a subscription; they simply stop purchasing. Without a churn column, we must define churn using business logic. By tracking Recency, Frequency, and Monetary parameters, we can profile buying patterns and identify customers drifting away.

### 📊 Dataset
- **Name**: Online Retail II (UCI Machine Learning Repository)
- **Timeframe**: December 1, 2009 to December 9, 2011 (738 days)
- **Format**: Transaction-level records (1.06M rows)
- **Target**: None (defined custom churn labels via a 90-day inactivity threshold)

### ⚙️ Project Workflow
1. **Initial Audit**: Load workbook sheets and evaluate missing fields, duplicates, and ranges.
2. **Exploratory Data Analysis (EDA)**: Map out monthly revenue spikes, country distribution, product sales, hourly buying patterns, and correlations.
3. **Data Cleaning**: Remove cancellations, duplicates, non-positive quantities/prices, and records lacking a Customer ID.
4. **RFM Feature Engineering**: Compress 779k transaction rows into 5,878 customer rows. Create auxiliary features like Average Order Value, Customer Lifetime, and Product Diversity.
5. **Target Labeling**: Define Churn (1) for customers with >90 days of inactivity.
6. **Model Training**: Establish baselines (Logistic Regression, Decision Tree, Random Forest) and train XGBoost.
7. **SHAP Explanations**: Extract game-theoretic feature importance and dependency plots.
8. **Deployment**: Wrap the champion model in a robust FastAPI application and Streamlit Dashboard.

---

## 🛠️ Technology Stack
- **Core**: Python, Pandas, NumPy, Joblib
- **Machine Learning**: Scikit-learn, XGBoost
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **REST API**: FastAPI, Uvicorn, Pydantic
- **Dashboard**: Streamlit, Plotly

---

## 📂 Project Structure
```text
├── api/
│   ├── app.py                  # FastAPI Application
│   ├── schemas.py              # Pydantic Schemas for Validation
│   ├── utils.py                # Model Loader & Feature Formatter
│   ├── model.joblib            # Trained XGBoost Champion Model
│   └── requirements.txt        # API dependencies
├── Customer Churn Analytics/   # Dashboard UI Screenshots
│   ├── CCA-1.png ... CCA-9.png
├── data/
│   └── online_retail_II.xlsx   # Raw Excel Dataset (if available)
├── customer_churn_predictor.ipynb  # End-to-End Notebook
├── customer_churn_dataset.csv  # Processed Customer-Level Dataset
├── dashboard.py                # Streamlit Analytics Dashboard
├── online_retail_II.csv        # Combined Transaction Dataset
├── INTERVIEW_PREP.md           # 20 Deep-Dive Interview Q&As
└── README.md                   # Project Documentation
```

---

## 🖼️ Dashboard Screenshots

### 1. Executive Overview & Key Performance Indicators
![Overview 1](Customer%20Churn%20Analytics/CCA-1.png)
![Overview 2](Customer%20Churn%20Analytics/CCA-2.png)

### 2. Exploratory Data Analysis (EDA)
![EDA Trends](Customer%20Churn%20Analytics/CCA-3.png)
![EDA Distribution](Customer%20Churn%20Analytics/CCA-4.png)

### 3. RFM Analysis & Correlation Matrix
![RFM Metrics](Customer%20Churn%20Analytics/CCA-5.png)
![Correlation Heatmap](Customer%20Churn%20Analytics/CCA-6.png)

### 4. Model Performance & Feature Importance
![Model Performance & Feature Importance](Customer%20Churn%20Analytics/CCA-7.png)

### 5. Interactive Customer Churn Predictor
![Customer Prediction Form](Customer%20Churn%20Analytics/CCA-8.png)

### 6. Business Insights & Retention Strategy
![Business Strategy & Insights](Customer%20Churn%20Analytics/CCA-9.png)

---

## Notebook Overview
The Jupyter Notebook ([customer_churn_predictor.ipynb](customer_churn_predictor.ipynb)) walks through the research and development pipeline in sequential detail:
- **Sections 1-5**: Loads datasets and audits raw statistics.
- **Section 6**: Investigates purchasing trends, distributions, and weekly/daily schedules.
- **Section 7**: Discards cancellations, invalid values, and drops missing Customer IDs.
- **Section 8**: Generates Recency, Frequency, and Monetary scores, along with 7 auxiliary features.
- **Section 9**: Establishes the 90-day inactivity threshold, creating a balanced 50-50 target variable.
- **Section 10**: Runs baseline model splits and feature scaling.
- **Section 11**: Trains models and plots confusion matrices and ROC/PR curves.
- **Section 12**: Explores global and local tree decisions using SHAP summaries, dependence, and waterfall plots.

---

## 📈 Model Performance
Tree models successfully learn the deterministic inactivity boundary (leakage warning on `Recency` noted during evaluations):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Decision Tree** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Logistic Regression** | 0.9923 | 1.0000 | 0.9847 | 0.9923 | 0.9999 |

*Note: In production deployments, `Recency` must be excluded from modeling features to avoid direct leakage and force models to classify churn based entirely on underlying behaviors like frequency, basket size, and spend intervals.*

---

## 🔌 Usage Instructions

### Running the FastAPI Server
To start the FastAPI service, execute from the root folder:
```bash
uvicorn api.app:app --reload
```

### Running the Streamlit Dashboard
To start the interactive analytics dashboard, execute from the root folder:
```bash
streamlit run dashboard.py
```

### Endpoints
- **GET `/`**: Simple API welcome message.
- **GET `/health`**: Returns model status and server check.
- **POST `/predict`**: Predicts churn.
  - **Payload**:
    ```json
    {
      "Recency": 45.0,
      "Frequency": 5,
      "Monetary": 1250.50,
      "AverageOrderValue": 250.10,
      "AverageBasketSize": 15.2,
      "UniqueProducts": 45,
      "CustomerLifetime": 365.0,
      "PurchaseInterval": 73.0,
      "TransactionsPerMonth": 0.41,
      "ProductDiversity": 0.05
    }
    ```
  - **Response**:
    ```json
    {
      "prediction": 0,
      "probability": 0.125,
      "confidence_score": 0.875
    }
    ```

---

## 🖥️ Dashboard Features
The Streamlit dashboard (`dashboard.py`) provides an interactive interface for stakeholders:
1. **Customer Search**: Query by ID to pull up active/churn predictions.
2. **Probability Gauges**: Ring charts mapping churn risk.
3. **Local Explanations**: Streamlit rendering of SHAP waterfall values to explain the prediction to marketing teams.
4. **Cohort Analysis**: Segment customers into Loyal/Active, At-Risk, or High Spenders.

---

## ⚙️ Installation
1. Clone this repository.
2. Create and activate a virtual environment.
3. Install project dependencies:
   ```bash
   pip install -r api/requirements.txt
   ```
4. Place the dataset `online_retail_II.csv` in the root folder.
5. Run the notebook or start the Streamlit application immediately!

---

## 🔮 Future Improvements
- **Time-Series Cohorts**: Evaluate churn dynamically using rolling temporal windows (e.g. predicting churn in month $T+1$ based on behavior in month $T$).
- **A/B Testing Integration**: Hook the model to promotional triggers to measure retention lifts.
- **Leakage-Free Features**: Retrain models without `Recency` to benchmark accuracy based purely on basket characteristics and buying rhythms.

---

## 🤝 Acknowledgements
- **UCI Machine Learning Repository** for providing the Online Retail II dataset.
- **Lundberg & Lee** for their foundational work on SHAP.

---

## 📄 License
This project is open-source and licensed under the MIT License.
