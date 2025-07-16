# 🧠 Social Sphere Submission – SuperDataScience Collaborative Project

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange?logo=mlflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-blue?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-brightgreen?logo=xgboost)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-orange?logo=catboost)

This repository contains my contribution to the **Social Sphere: Student Social-Media Behavior & Relationship Analytics** project — an open-source initiative by the SuperDataScience community.

---

## 🚀 Project Summary

The **Social Sphere** project investigates how social media habits influence students' relationships, sleep, and mental health using survey data from over 700 students across 100+ countries. We developed machine learning models for both classification (predicting conflict levels) and regression (predicting addiction scores).

### 🎯 Key Objectives
- Classify students by social media conflict levels using behavioral patterns
- Predict addiction scores through regression modeling
- Identify key behavioral predictors using SHAP analysis
- Compare model performance with/without self-reported mental health features
- Implement MLOps best practices with MLflow experiment tracking

---

## 📊 Key Results

### Binary Classification (low vs high amount of conflicts)
| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **CatBoost** | 0.99 | 1.00 | 0.99 |
| **XGBoost** | 0.99 | 0.99 | 0.99 |
| Logistic Regression | 0.98 | 0.98 | 0.98 |
| *Baseline* | *0.77* | *0.77* | *0.77* |

### Regression (Addiction Score Prediction)
| Model | R² Score | RMSE | MAPE |
|-------|----------|------|------|
| **CatBoost** | **0.91** | **0.47** | **4%** |
| Linear Regression | 0.93 | 0.32 | 5% |
| XGBoost | 0.87 | 0.57 | 5% |
| *Baseline* | *0.00* | *1.38* | *25%* |

### 🧠 Key Predictors (SHAP Analysis)
1. **Mental Health Score** - Strongest predictor across all models
2. **Daily Usage Hours** - Consistent predictor of both conflict and addiction
3. **Sleep Hours** - Inversely correlated with negative outcomes
4. **Platform Usage** - TikTok users show highest risk profiles
5. **Country/Region** - Geographic variations in social media impact

---

## 🔍 Model Performance Without Mental Health

Removing self-reported mental health features to test model robustness with only observable behaviors:

- **Classification**: F1-Score drops minimally (0.99 → 0.96-0.99)
- **Regression**: R² remains strong (0.91 → 0.87-0.91)
- **Key Insight**: Observable behaviors (usage patterns, sleep) provide strong predictive signal

---

## 🛠️ Technical Approach

### Data & Methodology
- **Dataset**: 707 students, 13 features, global survey data
- **Preprocessing**: Robust pipeline with feature engineering and encoding
- **Models**: Gradient boosting (CatBoost, XGBoost), linear methods, ensemble approaches
- **Validation**: Rigorous train-test splitting with cross-validation

### MLOps Implementation
- **Experiment Tracking**: MLflow for model versioning and performance monitoring
- **Model Interpretability**: SHAP for feature importance and decision explanations
- **Pipeline Design**: Modular sklearn pipelines for reproducibility
- **Deployment**: Interactive Streamlit application for model inference

📊 **View MLflow Experiments**: [Dagshub Dashboard](https://dagshub.com/bab-git/SDS-social-sphere.mlflow/#/experiments/2)

---

## 💡 Key Insights

- **Behavioral Patterns**: Daily usage and sleep patterns are reliable predictors of social media conflicts
- **Platform Differences**: TikTok and WhatsApp users show higher conflict risk
- **Geographic Trends**: Significant variation in social media impact across countries
- **Model Robustness**: High performance achievable without self-reported mental health data
- **Feature Efficiency**: CatBoost achieved 0.99 F1-score using only 3 key features

---

## 🗂️ Repository Structure

```plaintext
submissions/team-members/bob-hosseini/
├── app/                    # Streamlit application
├── configs/                # Model and app configuration
├── data/                   # Processed datasets
├── notebooks/              # EDA and model development
├── src/                    # Core utilities and functions
└── requirements.txt
```

---

## 🙌 Acknowledgments

Special thanks to the SuperDataScience community for this collaborative learning opportunity in MLOps, model interpretability, and real-world analytics challenges.

**Connect**: [GitHub](https://github.com/bab-git) | [LinkedIn](https://www.linkedin.com/in/bhosseini/) | [Website](https://bob-hosseini-portfolio.web.app/)
