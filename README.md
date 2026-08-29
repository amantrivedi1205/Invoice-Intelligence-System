# 📄 Invoice Intelligence System

An end-to-end Machine Learning project that analyzes vendor invoices and predicts whether an invoice should be **flagged for manual review**.

The project combines vendor invoice data with purchase-order information, performs feature engineering and statistical analysis, trains multiple classification models, and deploys the final model using **Streamlit**.

---

## 🚀 Project Overview

Organizations process a large number of vendor invoices. Manually reviewing every invoice can be time-consuming, while incorrect invoices may lead to financial leakage or audit risk.

This project builds an **Invoice Intelligence System** that helps identify invoices that may require additional manual verification.

The system analyzes invoice and purchase information such as:

* Invoice amount
* Purchase amount
* Invoice quantity
* Purchase quantity
* Freight cost
* Payment delay
* Receiving delay

Based on these features, the Machine Learning model predicts whether the invoice is:

* ✅ **Normal**
* ⚠️ **Flagged for Manual Review**

---

## 🎯 Business Problem

Vendor invoices may contain discrepancies such as:

* Invoice amount not matching purchase records
* Abnormal receiving delays
* Quantity differences
* Unusual freight charges
* Payment timing differences

Checking every invoice manually can consume significant time.

The goal of this project is to build a Machine Learning-based system that helps prioritize potentially risky invoices for manual review.

---

## 💡 Project Solution

The project follows this workflow:

```text
SQLite Database
       ↓
Data Extraction using SQL
       ↓
Purchase Data Aggregation
       ↓
Vendor Invoice Data
       ↓
LEFT JOIN using PONumber
       ↓
Data Cleaning & Analysis
       ↓
Feature Engineering
       ↓
Invoice Risk Label Creation
       ↓
Train-Test Split
       ↓
StandardScaler
       ↓
Machine Learning Models
       ↓
Model Evaluation
       ↓
Random Forest Classifier
       ↓
Model Serialization using Joblib
       ↓
Streamlit Application
```

---

## 🗂️ Dataset

The project uses an SQLite database named:

```text
inventory.db
```

Important tables used in this project include:

```text
purchases
vendor_invoice
```

The two datasets are connected using:

```text
PONumber
```

---

## 🔗 SQL Data Integration

Purchase-level information is first aggregated using SQL.

Examples of calculated purchase features include:

* Total number of brands
* Total purchased quantity
* Total purchase amount
* Average receiving delay

The aggregated purchase data is then joined with the vendor invoice table using a **LEFT JOIN on `PONumber`**.

Example structure:

```sql
FROM vendor_invoice vi

LEFT JOIN (
    SELECT
        PONumber,
        COUNT(DISTINCT Brand) AS total_brands,
        SUM(Quantity) AS total_item_quantity,
        SUM(Dollars) AS total_item_dollars,
        AVG(
            julianday(ReceivingDate) -
            julianday(PODate)
        ) AS avg_receiving_delay

    FROM purchases

    GROUP BY PONumber

) p

ON vi.PONumber = p.PONumber
```

---

## 🛠️ Feature Engineering

Several useful features are created from invoice and purchase records.

The final Machine Learning model uses the following **7 features**:

| Feature                  | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| `total_invoice_quantity` | Total quantity reported in the vendor invoice             |
| `total_invoice_dollars`  | Total invoice amount                                      |
| `Freight`                | Freight cost charged by the vendor                        |
| `days_to_pay`            | Number of days between invoice date and payment date      |
| `total_item_quantity`    | Total quantity recorded in purchase data                  |
| `total_item_dollars`     | Total purchase amount                                     |
| `avg_receiving_delay`    | Average number of days between PO date and receiving date |

---

## 🚩 Target Variable

The target variable used for classification is:

```text
flag_invoice
```

Two classes are used:

```text
0 → Normal Invoice
1 → Invoice Requires Manual Review
```

The project creates the invoice risk label using two main conditions.

An invoice is flagged when:

```python
abs(total_invoice_dollars - total_item_dollars) > 5
```

or:

```python
avg_receiving_delay > 10
```

The label creation function used in the project is:

```python
def create_invoice_risk_label(row):

    if abs(
        row["total_invoice_dollars"]
        - row["total_item_dollars"]
    ) > 5:

        return 1

    if row["avg_receiving_delay"] > 10:

        return 1

    return 0
```

---

## 📊 Exploratory Data Analysis

Before training the Machine Learning models, the dataset is analyzed using:

* Missing value analysis
* Dataset information
* Descriptive statistics
* Correlation analysis
* Correlation heatmap
* Class distribution
* Statistical testing

A correlation heatmap is used to understand relationships between numerical variables.

Welch's independent **t-test** is also used to compare flagged and normal invoices and identify statistically significant variables.

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and compared:

### 1. Logistic Regression

A simple linear classification model used as a baseline.

### 2. Decision Tree Classifier

A tree-based model capable of capturing non-linear relationships.

### 3. Random Forest Classifier

An ensemble model that combines multiple decision trees.

The final deployed model is:

```text
Random Forest Classifier
```

---

## 🔄 Train-Test Split

The dataset is divided into training and testing datasets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

This means:

```text
80% → Training Data
20% → Testing Data
```

---

## 📏 Feature Scaling

`StandardScaler` is used before training the models.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

The scaler learns the mean and standard deviation only from the training dataset.

The same learned transformation is then applied to testing data and new invoice data.

---

## 📈 Model Evaluation

The models are evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report

Example:

```python
print(
    classification_report(
        Y_test,
        y_pred
    )
)
```

The performances of Logistic Regression, Decision Tree and Random Forest are compared before selecting the final model.

---

## 🏆 Final Model

The final model used for deployment is:

```python
RandomForestClassifier(
    random_state=42
)
```

The model is trained using:

```python
model_3.fit(
    X_train_scaled,
    Y_train
)
```

---

## 💾 Saving the Model

The trained model, scaler and feature column order are saved using `joblib`.

```python
joblib.dump(
    model_3,
    "invoice_model.pkl"
)

joblib.dump(
    scaler,
    "invoice_scaler.pkl"
)

joblib.dump(
    list(X.columns),
    "invoice_columns.pkl"
)
```

Three files are therefore required by the Streamlit application:

```text
invoice_model.pkl
invoice_scaler.pkl
invoice_columns.pkl
```

---

## 🖥️ Streamlit Application

The project includes a Streamlit application where users can enter invoice information and receive a prediction.

The application displays:

* Invoice details input
* Invoice vs purchase amount comparison
* Amount mismatch
* Quantity mismatch
* Invoice risk prediction
* Risk probability
* Basic risk indicators
* Manual review recommendation

The prediction flow is:

```text
User Input
    ↓
Pandas DataFrame
    ↓
Correct Feature Order
    ↓
Saved StandardScaler
    ↓
Random Forest Model
    ↓
Risk Prediction
```

---

## 📁 Project Structure

```text
Invoice-Intelligence-System/
│
├── App.py
│
├── Invoice_Intelligence_System.ipynb
│
├── inventory.db
│
├── invoice_model.pkl
│
├── invoice_scaler.pkl
│
├── invoice_columns.pkl
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Technologies Used

### Programming

* Python

### Data Analysis

* NumPy
* Pandas

### Data Visualization

* Matplotlib
* Seaborn

### Database

* SQLite
* SQL

### Machine Learning

* Scikit-learn

### Statistical Analysis

* SciPy

### Deployment

* Streamlit

### Model Serialization

* Joblib

---

## 📚 Python Libraries

```python
numpy
pandas
matplotlib
seaborn
sqlite3
scikit-learn
scipy
joblib
streamlit
```

---

## ⚡ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd Invoice-Intelligence-System
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib streamlit
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run App.py
```

Streamlit will start the application locally.

---

## 📋 requirements.txt

A simple `requirements.txt` for the project can contain:

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
scipy
```

---

## 🧠 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Python programming
* SQL queries
* SQLite database handling
* Data aggregation
* SQL joins
* Data cleaning
* Exploratory Data Analysis
* Feature engineering
* Statistical testing
* Classification
* Train-test splitting
* Feature scaling
* Logistic Regression
* Decision Trees
* Random Forest
* Model evaluation
* Model serialization
* Streamlit application development

---

## 💼 Business Impact

The system can support invoice-review teams by helping them:

* Prioritize invoices requiring attention
* Detect invoice and purchase-record mismatches
* Identify abnormal receiving delays
* Reduce unnecessary manual checking
* Support invoice auditing workflows

The model is intended as a **decision-support system** rather than a replacement for financial or audit professionals.

---

## ⚠️ Limitations

This project is built as a Machine Learning portfolio project.

The current invoice risk labels are created using predefined business conditions based mainly on invoice amount differences and receiving delays.

For a real-world production system, additional information could be considered, such as:

* Vendor history
* Duplicate invoices
* Tax inconsistencies
* Payment terms
* Purchase-order status
* Vendor risk scores
* Approval history
* Historical fraud cases

---

## 🔮 Future Improvements

Possible improvements include:

* Hyperparameter tuning using GridSearchCV
* Feature importance analysis
* Better handling of imbalanced classes
* Additional vendor-level features
* Automated invoice upload
* Batch invoice prediction
* Model monitoring
* Cloud deployment
* Dashboard analytics
* Integration with accounting or ERP systems

---

## 👨‍💻 Author

**Aman Trivedi**

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐.

Feedback and suggestions are always welcome.
