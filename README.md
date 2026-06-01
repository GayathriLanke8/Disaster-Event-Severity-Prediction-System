# 🌍 Disaster Event Severity Prediction System

## 📌 Project Overview

This project uses **Machine Learning** to predict whether a disaster event is **major** or **non-major** based on historical disaster data.

**Target Variable:** `is_major_disaster`

* **1** → Major Disaster
* **0** → Non-Major Disaster

---

# 🚀 Project Workflow

## 1️⃣ Data Collection & Loading

* Import dataset from CSV / Excel / API / Database
* Load using **Pandas**
* Validate dataset size and column structure

---

## 2️⃣ Initial Data Inspection

Performed:

* `head()`, `tail()`
* `info()`
* `describe()`

Checked:

* Data types
* Missing values
* Unique values
* Data consistency

---

## 3️⃣ Data Cleaning

Tasks completed:

* Handle missing values
* Remove duplicates
* Fix incorrect data types
* Standardize column names

---

## 4️⃣ Exploratory Data Analysis (EDA) 📊

### 🔹 Univariate Analysis

* Distribution plots
* Skewness detection

### 🔹 Bivariate Analysis

* Feature vs Target relationship
* Correlation analysis

### 🔹 Multivariate Analysis

* Heatmaps
* Pairplots
* Interaction effects

**Goal:** Understand patterns and relationships

---

## 5️⃣ Outlier Detection & Treatment 📦

Methods used:

* Boxplots
* IQR / Z-score

Actions:

* Remove
* Cap
* Transform

---

## 6️⃣ Feature Encoding 🔤

Converted categorical data into numerical format using:

* Label Encoding
* One-Hot Encoding

---

## 7️⃣ Feature Scaling ⚖️

Applied:

* Standardization
* Normalization

✅ Fit only on training data to avoid data leakage.

---

## 8️⃣ Train-Test Split ✂️

Dataset split into:

* **80% Training**
* **20% Testing**

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🎯 Goal

Build an accurate ML model for:

* Disaster risk assessment
* Emergency response planning
* Resource allocation
