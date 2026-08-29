# 🧾 Invoice Intelligence System

An end-to-end Machine Learning application designed to analyze vendor invoices, identify potentially risky invoices for manual review, and estimate freight costs.

The project combines **data analysis, SQL, machine learning, and Streamlit** into an interactive application.

---

## 🚀 Features

### 🔍 Invoice Risk Detection

Predicts whether a vendor invoice should be:

* ✅ **Normal**
* 🚨 **Flagged for Manual Review**

The classification system analyzes invoice and purchase-related information such as:

* Total invoice quantity
* Total invoice amount
* Freight cost
* Days to payment
* Purchase quantity
* Purchase amount
* Average receiving delay

**Models explored:**

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The trained **Random Forest Classifier** is used by the Streamlit application.

---

### 🚚 Freight Cost Prediction

Predicts the expected freight cost based on:

* Quantity
* Purchase dollar amount

**Regression models explored:**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

The trained regression model is integrated into the application for freight estimation.

---

## 🎯 Business Objective

Vendor invoices can contain discrepancies that may result in:

* Cost leakage
* Overpayments
* Unexpected freight charges
* Purchase-order mismatches
* Audit risk
* Operational inefficiencies

The Invoice Intelligence System provides a data-driven approach for identifying invoices that may require additional attention.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **SQLite / SQL**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

---

## 🤖 Machine Learning Workflow

```text
Raw Invoice & Purchase Data
            ↓
      Data Extraction
            ↓
       SQL Queries
            ↓
    Data Preprocessing
            ↓
    Feature Engineering
            ↓
      Train-Test Split
            ↓
       Model Training
            ↓
     Model Evaluation
            ↓
    Model Serialization
          (.pkl)
            ↓
      Streamlit App
            ↓
     User Prediction
```

---

## 📂 Project Structure

```text
Invoice-Intelligence-System/
│
├── App.py
├── requirements.txt
├── README.md
│
├── invoice_model.pkl
├── invoice_scaler.pkl
├── invoice_columns.pkl
│
├── freight_model.pkl
├── freight_columns.pkl
│
├── 3_Freight_Cost_Prediction.ipynb
├── 4_Invoice_Flagging.ipynb
│
└── screenshots/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Invoice-Intelligence-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run App.py
```

The application will open in your browser.

---

## 📦 Requirements

The main Python dependencies are:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 📊 Application Modules

The Streamlit application contains two primary ML modules:

```text
             Invoice Intelligence System
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
 Invoice Risk Detection      Freight Cost Prediction
          │                         │
 Random Forest Classifier    Random Forest Regressor
          │                         │
          ▼                         ▼
 Normal / Manual Review      Expected Freight Cost
```

---

## 💡 Future Improvements

Potential improvements include:

* Freight anomaly detection
* Invoice-vs-PO mismatch analysis
* Vendor risk scoring
* Historical vendor performance analysis
* Automated invoice ingestion
* PDF invoice extraction using OCR
* Explainable AI for flagged invoices
* Interactive analytics dashboard
* Database integration for real-time predictions
* REST API integration

---

## 📌 Disclaimer

This project is intended as a machine learning and data analytics demonstration. Predictions should not be treated as final financial or audit decisions without appropriate human review.

---

## 👨‍💻 Author

**Aman Trivedi**


If you find this project useful, consider giving the repository a ⭐.
