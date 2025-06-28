## 🧠 Social Sphere Submission – SuperDataScience Collaborative Project

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange?logo=mlflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-blue?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-brightgreen?logo=xgboost)

Welcome! This repository contains my contribution to the **Social Sphere: Student Social-Media Behavior & Relationship Analytics** project — an open-source initiative by the SuperDataScience community.

---

## 🚀 Project Summary

The **Social Sphere** project investigates how social media habits influence students’ relationships, sleep, and mental health. Our classification models aim to predict social media conflict levels using self-reported survey data from over 700 students across 100+ countries.

### 🔍 Objectives:

* Classify students based on **conflict levels** caused by social media usage
* Compare model performance with and without self-perceived features (like Mental Health)
* Identify **key behavioral predictors** using SHAP
* Apply MLOps best practices with **MLflow tracking**

---

## 📊 Key Findings

### ✅ **Binary Classification (High vs Low Conflict)**

| Model             | Precision | Recall | F1-Score |
| ----------------- | --------- | ------ | -------- |
| **Logistic Reg.** | 0.98      | 0.98   | 0.98     |
| **XGBoost**       | 0.99      | 0.99   | 0.99     |
| **CatBoost**      | 0.99      | 1.00   | 0.99     |

* All advanced models significantly outperform the baseline (F1: 0.77).
* **XGBoost and CatBoost** achieve nearly perfect performance.
* **Mental Health**, **Daily Usage**, and **Sleep Hours** are top predictors.

---

### 🧠 **Feature Importance (SHAP Analysis)**

* **Mental Health**: Strongest predictor; lower scores → higher conflict.
* **Daily Usage**: Higher usage hours increase conflict probability.
* **Sleep Hours**: Less sleep correlates with more conflict.
* **Relationship Status (In Relationship)** and **Country** also have substantial impact in some models.

---

### 🔍 **Model Without Mental Health Feature**

| Model    | F1-Score | Key Predictors           |
| -------- | -------- | ------------------------ |
| XGBoost  | 0.96     | Daily Usage, Sleep Hours |
| CatBoost | 0.99     | Country, Sleep Hours     |

> Removing Mental Health only slightly reduces performance, suggesting strong signals in observable behaviors.

---

### 🎯 **Multiclass Classification (Low / Medium / High Conflict)**

| Model                | Accuracy | F1-Weighted |
| -------------------- | -------- | ----------- |
| XGBoost (with MH)    | 0.97     | 0.97        |
| XGBoost (w/o MH)     | 0.91     | 0.91        |
| CatBoost (w/o MH)    | 0.96     | 0.96        |

* The **3-class formulation** allows a more nuanced understanding.
* **Daily Usage, Sleep Hours**, and **Mental Health** consistently rank high in feature importance.
* Removing **Mental Health** reduces performance but keeps the model actionable.

---

## 🧰 Technical Highlights

### 🏗️ Pipeline & Preprocessing

* Modular `ColumnTransformer` for:

  * Binary encoding
  * One-hot for low/high cardinality features
  * Rare category grouping (Platform, Country)
  * Country → Continent mapping
* Scaled numeric features (StandardScaler)
* Full pipeline integration in `sklearn` for reproducibility

### ⚙️ MLOps Integration

* **MLflow** used for:

  * Model versioning and performance tracking
  * SHAP plots and ROC curves logging
  * GridSearch results with all hyperparameters
* Utility functions (`utils.py`) automate experiment steps

You can view the online MLflow dashboard hosted on Dagshub [here](https://dagshub.com/bab-git/SDS-social-sphere.mlflow/#/experiments/2).

### 📉 Feature Selection

* SHAP-guided model trained using only:

  * **Daily Usage**
  * **Sleep Hours**
  * **Country**
* CatBoost achieved **0.99 F1-score** with just these 3 features

---

## 🗂️ Repository Structure

```plaintext
submissions/team-members/bob-hosseini/
├── notebooks/
│   ├── 01_EDA_SocialSphere.ipynb
│   ├── 02_classification.ipynb
├── data/
│   ├── data.csv
│   ├── cleaned_data.csv
│   ├── data_cleaned.pickle
├── src/
│   ├── utils.py
├── mlruns/                         # MLflow tracking directory
├── requirements.txt
├── README.md
```

---

## 📌 Insights for Practice

* **Platform Usage Insight**: TikTok and WhatsApp users showed the highest predicted conflict risk.
* **Cross-Country Analysis**: USA students reported the highest conflict and addiction scores.
* **Bias Mitigation**: Self-reported Mental Health is a top predictor but not essential for good model performance.
* **SHAP & MLOps**: Combining interpretability with experiment tracking enhanced both transparency and productivity.

---

## 🙌 Acknowledgments

Special thanks to the SuperDataScience team and all collaborators in the SDS community. This has been an outstanding hands-on opportunity for learning MLOps, model interpretability, and real-world data challenges.

---

## 📬 Connect with Me

* **GitHub**: [@bab-git](https://github.com/bab-git)
* **LinkedIn**: [Behzad Hosseini](https://www.linkedin.com/in/bhosseini/)
