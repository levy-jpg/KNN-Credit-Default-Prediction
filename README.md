# Credit Card Default Prediction using K-Nearest Neighbours (KNN)

This project explores the prediction of credit card default using a **K-Nearest Neighbours (KNN)** classifier on a real-world financial dataset.  
The focus is on identifying **behavioural patterns** associated with default risk and evaluating the trade-offs involved in building an interpretable, distance-based machine learning model for credit risk assessment.

The project emphasises:
- careful data preprocessing,
- behavioural feature engineering,
- handling class imbalance,
- and evaluation using business-relevant metrics.

---

## 📊 Dataset

The dataset is sourced from the **UCI Machine Learning Repository**:

- **Default of Credit Card Clients** (Yeh & Lien, 2009)
- 30,000 credit card clients
- 23 original features
- Binary target: default in the following month

The dataset includes:
- Demographic attributes (age, gender, education, marital status)
- Credit limit information
- Repayment status over six months
- Monthly bill amounts
- Monthly repayment amounts

> ⚠️ The dataset is publicly available.  
> To keep the repository lightweight, the data file is not included here.  
> Please download it directly from UCI and place it in the `data/` directory if you wish to run the code.

---

## 🎯 Problem Motivation

Credit card default is a major source of loss for financial institutions.  
Even small increases in default rates can significantly impact profitability due to charge-offs, collection costs, and regulatory capital requirements.

Rather than focusing purely on predictive accuracy, this project prioritises:
- **early detection of high-risk clients**, and
- **interpretability**, which is critical in regulated financial environments.

---

## 🧠 Approach Overview

### 1. Data Preparation
- Removed non-informative identifier variables
- Converted categorical features using **one-hot encoding**
- Standardised numerical features using **z-score normalisation**

### 2. Behavioural Feature Engineering
To capture repayment behaviour more effectively, additional features were engineered, including:
- Average and maximum delinquency across months
- Count of late and severely late payments
- Credit utilisation ratios
- Repayment-to-bill ratios
- Billing volatility

After feature engineering, the final dataset contained **38 features**.

---

## 🤖 Model: K-Nearest Neighbours (KNN)

KNN was selected due to:
- its simplicity,
- transparency,
- and suitability for similarity-based reasoning.

Key design choices:
- **5-fold cross-validation** for selecting the number of neighbours
- **F1-score** used as the primary selection metric due to class imbalance
- **Cost-sensitive learning** applied to prioritise detection of defaulters

The optimal number of neighbours was found to be **K = 9**.

---

## 📈 Model Evaluation

The final model was evaluated using:
- Confusion matrix
- Precision, Recall, and F1-score
- Receiver Operating Characteristic (ROC) curve and AUC

Key results:
- Recall prioritised to reduce missed defaulters
- ROC–AUC ≈ **0.73**, indicating good class separability
- Clear trade-off between risk reduction and customer fairness

---

## 📊 Visual Results

### 🔹 Cross-Validation Performance
**K vs F1-score plot**

> 📌 *Insert figure here*  
> `Figures/k_vs_f1.png`

This plot shows how the F1-score varies with the number of neighbours, motivating the selection of **K = 9**.

---

### 🔹 ROC Curve
**ROC curve for the final cost-sensitive KNN model**

> 📌 *Insert figure here*  
> `Figures/roc_curve.png`

The ROC curve demonstrates the model’s ability to distinguish between defaulted and non-defaulted clients across decision thresholds.

---

## ⚠️ Error Analysis

A detailed analysis of misclassifications was conducted:

- **False Negatives** (missed defaulters):
  - Most costly error type in credit risk
  - Often resemble non-defaulters in billing magnitude

- **False Positives** (incorrectly flagged clients):
  - Often high-usage but financially stable clients
  - Highlight the fairness vs risk trade-off

Supporting outputs:
- `Outputs/top_fp_features.csv`
- `Outputs/top_fn_features.csv`

---

## 🧩 Project Structure

