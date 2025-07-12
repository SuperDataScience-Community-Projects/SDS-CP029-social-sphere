# 📱 Social Sphere – A Streamlit App for Understanding Student Social Media Behavior

**Social Sphere** is a machine learning-powered platform that explores how student social media habits relate to academic performance, mental health, sleep, and relationship dynamics. It uses supervised and unsupervised learning models to deliver insightful predictions and segmentations through an interactive web app built with **Streamlit**.

---

## 🚀 Project Highlights

- 🔍 **Classification Task**  
  Predicts the likelihood of **relationship conflict** due to social media usage.
  
- 📈 **Regression Task**  
  Estimates students' **self-reported social media addiction score** on a scale of 1–10.

- 🧠 **Clustering Task**  
  Groups students into **behavioral clusters** to help identify those at risk of digital dependency or social disengagement.

- 🧪 **MLflow & DagsHub Integration**  
  All experiments and model metrics were logged using **MLflow** and visualized through **DagsHub**.

  🔗 [View My Experiments on DagsHub](https://dagshub.com/PATRICK079/SDS-CP029-social-sphere/experiments)

---

## 🧪 Tools & Technologies

- **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **scikit-learn**, **XGBoost**, **CatBoost**, **Random Forest**
- **MLflow**, **DagsHub**, **SHAP**, **Streamlit**
- **Cursor** (AI coding assistant) for rapid development

---

## 🧠 Challenges & Mitigations

### 1. 📉 Small Dataset  
- **Issue**: Limited data made it prone to overfitting.  
- **Solution**: Applied **PCA (Principal Component Analysis)** to reduce dimensionality and noise.

### 2. ⚠️ Overfitting  
- **Issue**: High training accuracy with low generalization.  
- **Solution**: Used PCA, regularization techniques (e.g., Ridge, XGBoost), and cross-validation with adjusted R² scoring.

---

## ✅ Outcomes

- 🔧 Overfitting significantly reduced via PCA and model tuning.
- ✅ Regression and classification models achieved stable performance across folds.
- 📊 Generated explainable results with **SHAP**.
- 🧠 Behavioral clusters revealed hidden patterns in sleep and usage behaviors.
- 💻 Fully functional **Streamlit** app for demo and testing.

---

## 🔍 Key Takeaways

1. **SHAP for Model Interpretation**  
   Allowed clear understanding of feature importance and model behavior.

2. **MLflow Experiment Tracking**  
   Hands-on mastery of **MLflow UI** for logging and comparing multiple model runs.

3. **DagsHub Integration**  
   Provided cloud-hosted collaboration and visualization of experiment tracking.

4. **Coding Acceleration with Cursor AI**  
   Boosted productivity and clean implementation using AI-assisted development.

---

## 📚 Dataset Citation

**Title**: Social Media Addiction vs Relationships  
**Author**: Adil Shamim  
**Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)  
**License**: Educational and academic use only

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.  
All predictions are probabilistic and should **not** be used as a substitute for professional psychological or academic advice.

---

## 📁 Folder Structure

📁 submissions/
    └── team-members/
         └── Patrick-Edosoma/
             ├── model/
             │   └── xgboost_classifier_mod.pkl
             ├── preprocessor/
             │   └── scaler.joblib
             ├── Data/
             │   └── Raw/
             │       └── Students Social Media Addiction.csv
             └── Screenshot 2025-07-09 at 21.59.03.png
📄 app.py
📄 requirements.txt
