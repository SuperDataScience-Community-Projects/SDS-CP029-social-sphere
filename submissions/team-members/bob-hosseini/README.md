# 🧠 Social Sphere Submission – SuperDataScience Collaborative Project
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange?logo=mlflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-blue?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-brightgreen?logo=xgboost)

Welcome! This folder contains my individual contribution to the **Social Sphere: Student Social-Media Behavior & Relationship Analytics** project, a collaborative, open-source initiative hosted by the SuperDataScience community.

---

## 🚧 Project Status: Ongoing

The Social Sphere project is currently ongoing. We have successfully completed the classification phase, where we developed and evaluated models to classify students into different conflict levels based on their social media behavior, sleep patterns, and mental health scores. The next phase will focus on regression modeling to further explore and predict the impact of social media usage on various aspects of student life.

---

## 📌 Project Overview

**Social Sphere** is a community-driven machine learning project focused on analyzing and predicting student social media behavior patterns and their impact on mental health, academic performance, and relationship dynamics using classification and clustering techniques.

🔗 Main Project Repository: [Social Sphere on GitHub](https://github.com/SuperDataScience-Community-Projects/SDS-CP029-social-sphere)  
🗓️ Timeline: 5-week collaborative sprint  
🧠 Hosted by: SuperDataScience Community  

---
    
## 📈 Dataset Information

This project analyzes **student social media usage patterns** and their relationships with mental health, academic performance, and interpersonal conflicts across multiple countries and platforms.

* 📁 **Dataset Source**: [Social Media Addiction vs Relationships on Kaggle](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)
* 📊 **Size**: 705 students aged 16-25 from ~100 countries
* 🎯 **Target Variables**: Conflicts over social media, addiction scores
* 📌 **License**: Publicly available for educational use

### 🧾 Feature Overview

| Feature Name                    | Description                                           | Type        |
| ------------------------------- | ----------------------------------------------------- | ----------- |
| `Age`                          | Student age (16-25 years)                            | Numeric     |
| `Gender`                       | Self-reported gender (Male/Female)                   | Categorical |
| `Academic_Level`               | Education level (High School/Undergraduate/Graduate)  | Categorical |
| `Country`                      | Country of residence (~100 countries)                | Categorical |
| `Avg_Daily_Usage_Hours`        | Daily social media usage in hours                    | Numeric     |
| `Most_Used_Platform`           | Primary social media platform                        | Categorical |
| `Affects_Academic_Performance` | Whether usage affects academics (Yes/No)             | Binary      |
| `Sleep_Hours_Per_Night`        | Average nightly sleep duration                       | Numeric     |
| `Mental_Health_Score`          | Self-rated mental health (1-10 scale)               | Numeric     |
| `Relationship_Status`          | Current relationship status                           | Categorical |
| `Conflicts_Over_Social_Media`  | **Target**: Number of conflicts (0-5)               | Numeric     |
| `Addicted_Score`               | **Target**: Addiction level (1-10 scale)            | Numeric     |

---

## 👤 My Role & Contribution

As the **Project Lead** and active participant in this collaborative project, I provided technical guidance and team coordination while contributing to the following key areas:

---

### 🎯 **Project Leadership & Team Management**

* **Technical Leadership**: Served as project lead, providing technical guidance and architectural decisions for the team
* **Team Coordination**: Organized and facilitated weekly meetings to track progress, align on goals, and address blockers
* **Mentorship & Support**: Offered one-on-one technical support to team members, helping them overcome challenges and develop their ML skills
* **Collaboration Framework**: Established best practices for code sharing, documentation, and experiment tracking across the team
* **Knowledge Sharing**: Created reusable utility functions and documentation to enable team members to build upon each other's work

---

### ✅ **Technical Contributions**

---

### ✅ **Comprehensive Exploratory Data Analysis (EDA)**

* **Data Quality Assessment**: Verified data integrity with 705 samples, 0 missing values, and no duplicates
* **Demographic Profiling**: Analyzed distribution across 100+ countries, academic levels, and platforms
* **Target Variable Analysis**: 
  - Created binary conflict classification (threshold=3): 73% low vs 27% high conflict
  - Developed 3-class system for better balance: Low/Medium/High conflicts
* **Feature Engineering**: 
  - Continent mapping for 100+ countries using custom transformer
  - Rare category grouping for high-cardinality features
  - Frequency-based encoding strategies
* **Bias Detection**: Identified under-representation in high school students and single-participant countries

---

### ✅ **Advanced Machine Learning Pipeline Development**

* **Preprocessing Pipeline**: Built modular `ColumnTransformer` with specialized handling for:
  - Binary features (Gender) → Label encoding
  - Low-cardinality categoricals → One-hot encoding  
  - High-cardinality features (Platform) → Rare category grouping + OHE
  - Geographic data (Country) → Continent mapping + OHE
  - Numerical features → StandardScaler
* **Cross-Validation Strategy**: Implemented `StratifiedKFold` with proper time-series aware splitting
* **Model Comparison**: Systematic evaluation of Logistic Regression vs XGBoost with hyperparameter tuning

---

### ✅ **Multi-Class Classification Innovation**

* **Problem Reformulation**: Extended binary classification to 3-class problem:
  - Low Conflicts (0-2): 36% of samples
  - Medium Conflicts (3): 37% of samples  
  - High Conflicts (4-5): 27% of samples
* **Metric Adaptation**: Implemented weighted F1-score, precision, and recall for imbalanced multiclass
* **Target Conversion**: Developed utility functions to convert existing train/test splits without data leakage

---

### ✅ **Experiment Tracking & MLOps Best Practices**

* **MLflow Integration**: 
  - Comprehensive experiment tracking with 15+ models logged
  - Automated hyperparameter logging and model versioning
  - Dataset lineage tracking with train/test split preservation
* **Grid Search Automation**: Built reusable `run_classification_gridsearch_experiment()` function
* **Model Registry**: Systematic model registration with performance metadata
* **Reproducibility**: Fixed random seeds and documented all hyperparameter configurations

---

### ✅ **Model Interpretability & Feature Analysis**

* **SHAP Integration**: 
  - Linear SHAP for Logistic Regression interpretability
  - Tree SHAP for XGBoost feature importance
  - Automated SHAP plot generation and MLflow logging
* **Feature Relevance Analysis**: 
  - Identified Daily Usage, Mental Health, and Sleep Hours as top predictors
  - Analyzed impact of self-reported vs. observable features
  - Platform-specific insights (TikTok, Instagram correlation with conflicts)
* **Bias Mitigation**: Evaluated model performance with/without self-reported mental health features

---

### ✅ **Advanced Visualization & Interactive Analysis**

* **ROC Curve Analysis**: Automated ROC-AUC calculation and visualization for binary classification
* **Interactive Prediction Explorer**: Plotly-based visualization showing:
  - Prediction confidence vs. ground truth
  - Feature values on hover for error analysis
  - True Positive/False Positive identification
* **Confusion Matrix Automation**: Multi-class confusion matrix generation with proper class labeling

---

## 📂 Folder Structure

```plaintext
submissions/team-members/bob-hosseini/
│
├── notebooks/
│   ├── 01_EDA_SocialSphere.ipynb       # Comprehensive EDA with bias detection
│   ├── 02_classification.ipynb         # ML pipeline, multiclass, SHAP analysis
├── data/
│   ├── data.csv                        # Raw social media behavior dataset  
│   ├── cleaned_data.csv                # Processed dataset with outlier removal
│   ├── data_cleaned.pickle             # Serialized clean dataset for modeling
├── mlruns/                             # MLflow experiment tracking artifacts
├── src/
│   ├── utils.py                        # 985 lines of reusable ML utilities
├── requirements.txt                    # Python dependencies
├── README.md                           # This comprehensive documentation
```

> *Note: All notebooks include extensive markdown documentation and are designed for educational clarity and reproducibility.*

---

## 🎯 Key Technical Achievements

### **Machine Learning Performance**
- **Binary Classification**: XGBoost achieved **95% F1-score** (97% precision, 94% recall)
- **Multiclass Classification**: **96% weighted F1-score** across 3 conflict levels
- **Feature Engineering**: Reduced 100+ countries to 7 continents without performance loss
- **Hyperparameter Optimization**: Grid search across 64 parameter combinations per model

### **MLOps & Engineering Excellence**
- **Modular Design**: 985-line `utils.py` with 15+ reusable functions
- **Pipeline Automation**: One-function model training, evaluation, and logging
- **Error Handling**: Robust preprocessing with unknown category handling
- **Documentation**: Comprehensive docstrings and type hints throughout

### **Data Science Insights**
- **Correlation vs. Causation**: Distinguished between self-reported and observable predictors
- **Cross-Cultural Analysis**: Identified platform preferences and conflict patterns by region
- **Class Imbalance**: Successfully handled 73:27 imbalanced binary classification
- **Feature Selection**: SHAP-guided feature importance for model interpretability

---

## 📊 Key Findings & Business Impact

### **Predictive Insights**
* **Daily Usage Hours**: Strongest predictor of social media conflicts (SHAP importance: 0.35)
* **Sleep Patterns**: Significant negative correlation with conflicts (-0.42)
* **Platform Effects**: TikTok users show highest conflict rates, Instagram users most variable
* **Geographic Patterns**: USA students report highest addiction scores, Japan lowest

### **Model Generalizability**
* **Observable Features Only**: Model maintains 91% F1-score without self-reported mental health
* **Cross-Platform**: Consistent performance across different social media platforms
* **Demographic Robustness**: Stable predictions across age groups and academic levels

### **Practical Applications**
* **Early Intervention**: Identify at-risk students based on usage patterns
* **Platform Policy**: Evidence for platform-specific intervention strategies  
* **Academic Support**: Predictive model for academic performance impact

---

## 🛠️ Technical Stack & Skills Demonstrated

### **Machine Learning**
- **Algorithms**: Logistic Regression, XGBoost, Dummy Classifiers
- **Techniques**: Grid Search, Cross-Validation, Feature Engineering, SHAP
- **Metrics**: ROC-AUC, F1-Score, Precision/Recall, Confusion Matrices

### **Data Engineering**
- **Preprocessing**: ColumnTransformer, Custom Transformers, Pipeline Design
- **Feature Engineering**: Rare Category Grouping, Geographic Mapping, Encoding Strategies
- **Data Quality**: Outlier Detection, Missing Value Analysis, Duplicate Handling

### **MLOps & Deployment**
- **Experiment Tracking**: MLflow with automated logging and model registry
- **Reproducibility**: Seed management, environment documentation, version control
- **Visualization**: Matplotlib, Seaborn, Plotly for interactive analysis

### **Software Engineering**
- **Code Organization**: Modular design with comprehensive utility functions
- **Documentation**: Detailed README, docstrings, and notebook annotations
- **Testing**: Robust error handling and edge case management

---

## 🚀 Key Professional Achievements

### **Technical & Project Leadership**
- **End-to-End ML Pipeline**: From raw data to production-ready models with full documentation
- **Team Leadership**: Successfully led a collaborative data science project with multiple contributors
- **Code Quality**: 985 lines of well-documented, reusable utility functions shared across the team
- **Best Practices**: MLflow integration, proper CV, reproducible experiments, and team coordination

### **Problem-Solving Skills**
- **Class Imbalance**: Successfully handled 73:27 imbalanced dataset with appropriate metrics
- **High Cardinality**: Innovative solutions for 100+ country categorical variable
- **Model Interpretability**: SHAP integration for stakeholder communication

### **Business Acumen**
- **Stakeholder Communication**: Clear separation of self-reported vs. observable features
- **Actionable Insights**: Platform-specific recommendations and early intervention strategies
- **Scalable Solutions**: Modular code design enabling easy extension and maintenance

### **Leadership & Communication**
- **Project Management**: Led a multi-contributor open-source data science project from inception to completion
- **Team Development**: Mentored team members and facilitated knowledge transfer across different skill levels
- **Knowledge Sharing**: Created comprehensive documentation and reusable frameworks enabling team collaboration
- **Cross-Functional Impact**: Delivered insights applicable to education, mental health, and technology sectors

---

## 🙌 Acknowledgments

Thanks to the SuperDataScience community and all collaborators who contributed to discussions, code reviews, and project insights. Special appreciation for the open-source nature of this educational initiative.

---

## 📧 Contact & Professional Links

* 🔗 **GitHub Portfolio**: [Professional ML/Data Science Projects](https://github.com/bab-git)
* 💼 **LinkedIn**: [Connect for Data Science Discussions](https://www.linkedin.com/in/bhosseini/)
* 📊 **This Project**: Demonstrates end-to-end ML pipeline development, MLOps practices, and business-focused data science

---

*This project showcases advanced machine learning engineering skills, from exploratory data analysis through production-ready model deployment, with emphasis on interpretability, reproducibility, and business impact.*
