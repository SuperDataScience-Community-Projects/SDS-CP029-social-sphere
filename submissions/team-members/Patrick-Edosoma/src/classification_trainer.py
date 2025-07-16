

## This .py file is responsible for training and evaluating the model, as well as tracking performance metrics using MLflow on DagsHub.

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import mlflow
from dotenv import load_dotenv
load_dotenv()  



username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

# ✅ Set credentials and tracking URI
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("socialsphere_classification")

# -------------------------
# Load Data
# -------------------------
base_dir = "/Users/sot/SDS-CP029-social-sphere/submissions/team-members/Patrick-Edosoma"
data_path = os.path.join(base_dir, "Data", "classification_processed_data")

train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

# -------------------------
# Preprocessing
# -------------------------
X_train = train_df.drop(columns=["conflict_level_in_relationship_over_social_media", "student_id"])
y_train = train_df["conflict_level_in_relationship_over_social_media"]

X_test = test_df.drop(columns=["conflict_level_in_relationship_over_social_media", "student_id"])
y_test = test_df["conflict_level_in_relationship_over_social_media"]

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# -------------------------
# Models to Train
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),  
    "XGBoost": XGBClassifier(eval_metric="logloss"),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# -------------------------
# Run Models & Log to MLflow
# -------------------------
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Fit and predict
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)

        # Cross-validation
        cv_accuracy = cross_val_score(model, X_train_pca, y_train, cv=5, scoring='accuracy').mean()

        # Classification metrics
        report = classification_report(
            y_test, y_pred,
            target_names=["Low Conflict", "High Conflict"],
            output_dict=True
        )

        # Log model with input example
        input_example = pd.DataFrame([X_test_pca[0]], columns=["PC1", "PC2"])
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        # Log metrics
        mlflow.log_metric("cv_accuracy", cv_accuracy)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)

        # Save classification report as text
        text_report = classification_report(y_test, y_pred, target_names=["Low Conflict", "High Conflict"])
        report_path = os.path.join(base_dir, f"{model_name.replace(' ', '_')}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(text_report)
        mlflow.log_artifact(report_path)
        
model_dir = os.path.join(base_dir, "model")
filename = model_name.replace(" ", "_") + ".pkl"
joblib.dump(model, os.path.join(model_dir, filename))

# Create output directory if it doesn't exist
preprocessor_dir = os.path.join(base_dir, "preprocessor")
os.makedirs(preprocessor_dir, exist_ok=True)

# Save scaler and PCA
joblib.dump(scaler, os.path.join(preprocessor_dir, "scaler.joblib"))
joblib.dump(pca, os.path.join(preprocessor_dir, "pca.joblib"))
        