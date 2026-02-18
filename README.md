# Online Purchase Orders Classification

## Project Overview

This project aims to predict the **risk of default payment** for online purchase orders using machine learning techniques.
The dataset contains **30,000 online purchase orders**, each described by **44 attributes**, covering customer behavior, payment methods, order history, and risk indicators.

The business goal is to help an online trader **identify high-risk customers before order fulfillment**, enabling better risk management and reduced financial loss.

---

## Problem Statement

Each order is classified into one of two categories:

* **`yes`** → High risk of default payment
* **`no`** → Low risk of default payment

This is a **binary classification problem**, where the objective is to learn patterns that distinguish risky customers from reliable ones.

---

## Dataset Description

* **Dataset:** `risk-dataset.txt`
* **Records:** ~30,000
* **Features:** 44
* **Target variable:** `CLASS`
* **Attribute definitions:** `risk-attributes.txt`

Each row represents one online purchase order.

---

## Target Variable

* **`CLASS`**

  * `yes` → High risk
  * `no` → Low risk

This column is used as the label for supervised learning.

---

## Initial Data Cleaning

### Identifier & Non-informative Columns Removed

* **`ORDER_ID`**
  A unique identifier with no predictive value.

* **Redundant columns removed:**

  ```
  ANUMMER_01, ANUMMER_02, ANUMMER_03, ANUMMER_04, ANUMMER_05,
  ANUMMER_06, ANUMMER_07, ANUMMER_08, ANUMMER_09, ANUMMER_10
  ```

---

## Handling Missing Values

Missing values in the dataset were represented using the symbol **`?`**.
The following columns contained missing values:

```
Z_CARD_ART, DATE_LORDER, MAHN_AKT, MAHN_HOECHST,
Z_LAST_NAME, B_BIRTHDATE, TIME_ORDER
```

Instead of applying a single imputation strategy, **each column was analyzed and handled based on business logic and data context**.

---

### 1. `Z_CARD_VALID` – Card Expiry Date

* Represents the **expiration date of the card**
* In raw form, not directly useful for prediction

**Feature Engineering Applied:**

* Identified the maximum `DATE_LORDER` as a reference point
* Computed a new numerical feature:

**`CARD_MONTHS_TO_EXPIRY`**
→ Number of months remaining until card expiration at the time of last order

This transformed a raw date field into a **meaningful numerical risk indicator**.

---

### 2. `Z_CARD_ART` – Type of Card

* Missing values were investigated rather than blindly imputed
* Analysis showed:

  * Missing values occur **only when payment was not made by card**
  * These orders were paid via **check or debit card**

**Imputation Strategy:**

* Missing values replaced with:
  **`not_a_card_payment`**

This column is treated as a **categorical feature**.

---

### 3. `Z_LAST_NAME` – Customer Last Name

* Missing values initially appeared suspicious
* Investigation showed:

  * **All rows with missing last name used `Z_METHODE = check` (14,808 rows)**

**Interpretation:**

* Customers paying by check may not be required to provide a name
* Some records likely originate from legacy or offline workflows

**Imputation Strategy:**

* Missing values replaced with:
  **`not_applicable`**

This column is treated as a **categorical feature**.

---

### 4. `MAHN_AKT` & `MAHN_HOECHST` – Reminder Stages

* These columns track reminder history for unpaid orders

**Root Cause Analysis:**

* 15,032 missing values correspond to **new customers**
* Remaining missing values occur when:

  * Customers paid via debit card
  * Reminder system was not applicable (no online credit risk)

**Imputation Strategy:**

* Missing values replaced with **`-1`**

  * Indicates *“reminder not applicable”*
  * Preserves numeric meaning without fabricating risk signals

---

### 5. `DATE_LORDER` – Last Order Date

* Missing values analyzed
* Found to strongly correlate with **new customers**

**Decision:**

* No imputation applied
* Missing values retained as `NULL`

**Feature Engineering Applied:**

* Created a new numerical feature:

**`DAYS_SINCE_LAST_ORDER`**
→ Number of days since the last order relative to the reference date

---

### 6. `B_BIRTHDATE` – Date of Birth

* Not a mandatory field
* Missing values expected and non-random

**Feature Engineering Applied:**

1. Extracted **Age** (numerical)
2. Derived **Age Group** (categorical)

This allowed retention of demographic signal without relying on raw dates.

---

### 7. `TIME_ORDER` – Order Time

**Feature Engineering Applied:**

1. Extracted **Order Hour** (`order_hour`) → Numerical
2. Derived **Order Part of Day** (`ORDER_PART_OF_DAY`) → Categorical
   (e.g., morning, afternoon, evening, night)

This captures customer behavior patterns related to purchase timing.

---

## Persisting the Cleaned Dataset

After completing data cleaning and feature engineering, the processed dataset was saved in **Parquet format**.

**Why Parquet instead of CSV?**

* Preserves original **data types** (numeric, categorical, datetime)
* Prevents implicit type conversion when reloading data for modeling
* Faster read/write performance and smaller file size
* Ensures consistency between preprocessing and model training stages

This step was essential to guarantee that categorical and numerical features were restored exactly as intended during the modeling phase.

---

## Model Training Strategy

### Train–Test Split

The dataset was split into training and testing sets using a **stratified split** to maintain the original class distribution:

* **Training set:** 80%
* **Test set:** 20%
* Stratification applied on the target variable (`CLASS`)
* Fixed random seed for reproducibility

This ensured both high-risk and low-risk orders were proportionally represented.

---

## Feature Preprocessing Pipeline

A **pipeline-based preprocessing approach** was used to avoid data leakage and ensure consistent transformations during cross-validation and inference.

### Numerical Features

* Missing values imputed using **mean or median** (tuned via GridSearch)
* Features standardized using **StandardScaler**

### Categorical Features

* Missing values imputed using the **most frequent value**
* Encoded using **One-Hot Encoding**
* Unknown categories safely handled during inference

A **ColumnTransformer** was used to apply transformations selectively to numerical and categorical features.

---

## Models Evaluated

The following classification models were evaluated using a unified preprocessing + modeling pipeline:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost Classifier

Each model was wrapped in a pipeline to ensure consistent preprocessing and fair comparison.

---

## Hyperparameter Tuning

* **GridSearchCV** with 3-fold cross-validation
* Parallel execution using all available CPU cores
* Hyperparameters tuned for both:

  * Model complexity
  * Numerical imputation strategy

### Primary Evaluation Metric

**ROC–AUC** was used as the primary metric because:

* It measures how well the model ranks customers by risk
* It is independent of a fixed classification threshold
* It is well-suited for credit and fraud-risk prediction problems

---

## Model Performance Summary

| Model               | Accuracy | Precision | Recall | F1 Score | ROC–AUC   |
| ------------------- | -------- | --------- | ------ | -------- | --------- |
| Logistic Regression | 0.942    | 0.922     | 0.942  | 0.915    | 0.736     |
| Random Forest       | 0.942    | 0.887     | 0.942  | 0.914    | 0.745     |
| Gradient Boosting   | 0.942    | 0.909     | 0.942  | 0.914    | 0.742     |
| XGBoost Classifier  | 0.942    | 0.945     | 0.942  | 0.914    | **0.749** |


Although accuracy was similar across models, **ROC–AUC highlighted meaningful differences** in ranking high-risk customers.

---

## Best Model Selection

The **XGBoost Classifier** achieved the highest ROC–AUC score and was selected as the final model:

* Strong performance on non-linear feature interactions
* Robust handling of mixed feature types
* Best overall risk-ranking capability

---

## Model Persistence

The final trained model, including preprocessing steps, was saved using `joblib`:

```
best_classification_model.pkl
```

Saving the full pipeline ensures:

* No train–inference mismatch
* Reproducibility
* Readiness for deployment or batch scoring

---

### Final Note

This project demonstrates an **end-to-end machine learning workflow**, from domain-aware data cleaning and feature engineering to robust model selection and evaluation, with a strong emphasis on reproducibility and real-world risk modeling.

---

## Reproducibility

The project is structured using **two separate notebooks** to ensure reproducibility and a clear ML workflow.

* **Notebook 1 – Data Preprocessing & Feature Engineering**

  * Cleans raw data and performs feature extraction
  * Saves the processed dataset in **Parquet format** to preserve data types
  * Output stored in:

    ```
    data/cleaned/cleaned_data.csv
    ```

* **Notebook 2 – Model Training & Evaluation**

  * Loads the cleaned Parquet file
  * Performs train–test split, preprocessing pipelines, model training, and evaluation
  * Saves the best model (including preprocessing) as:

    ```
    best_classification_model.pkl
    ```

This separation prevents data leakage, ensures consistent feature definitions, and allows the full pipeline to be reproduced end-to-end.

---



