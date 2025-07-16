# 📱 Social Sphere – Student Social-Media Behavior & Relationship Analytics

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
- **Solution**: Used PCA and cross-validation with adjusted R² scoring.

---

## ✅ Outcomes

- 🔧 Overfitting significantly reduced via PCA
- ✅ Regression and classification models achieved stable performance across folds.
- 📊 Generated explainable results with **SHAP**.
- 🧠 Behavioral clusters revealed hidden patterns in sleep and usage behaviors.
- 💻 Fully functional **Streamlit** app for demo and testing.

 🔗 [View My App](https://patrick--social-sphere-app-visetttduw5v8fjt7nywcv.streamlit.app)

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

## 📁 Project Structure

```text
submissions/
└── team-members/
    └── Patrick-Edosoma/
        ├── data/                                                # All datasets used in the project
        │   ├── raw/                                            # Original unprocessed dataset
        │   │   └── Students Social Media Addiction.csv
        │   ├── classification_processed_data/                  # Cleaned & encoded data for classification
        │   │   ├── train.csv
        │   │   └── test.csv
        │   └── regression_processed_data/                       # Cleaned & encoded data for regression
        │       ├── train.csv
        │       └── test.csv
        │
        ├── notebook/                                             # All Jupyter notebooks used for exploration & prep
        │   ├── Classification_feature_engineering.ipynb
        │   ├── Clustering.ipynb
        │   ├── Social_Sphere_EDA.ipynb
        │   ├── feature_importance.ipynb
        │   └── regression_feature_engineering.ipynb
        │
        ├── model/                                                   # Trained ML models (.pkl files)
        │   ├── random_regressor_model.pkl
        │   └── xgboost_classifier_mod.pkl
        │
        ├── preprocessor/                                             # Fitted preprocessing artifacts
        │   ├── pca.joblib
        │   ├── random_forest_scaler.pkl
        │   └── scaler.joblib
        │
        ├── src/                                                          # Custom training scripts
        │   ├── classification_trainer.py
        │   └── regression_trainer.py
        │
        ├── .gitignore                                                # Git ignored files
        ├── README.md                                                 # Project overview and documentation
        ├── Screenshot 2025-07-09 at 21.59.03.png                     # App overview image
        ├── app.py                                                    # Streamlit app main file
        └── requirements.txt                                          # Project dependencies
```

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

## 📬 Contact
Patrick Edosoma

Machine Learning Engineer

[Linkedlin](https://www.linkedin.com/in/patrickedosoma/)

## ⭐️ Star This Repo

If you found this project helpful, please star ⭐️ it to show support!

