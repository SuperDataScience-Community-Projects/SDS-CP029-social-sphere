import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

# Set credentials and tracking URI
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("socialsphere_regression")

# -------------------------
# Load Data
# -------------------------
base_dir = "/Users/sot/SDS-CP029-social-sphere/submissions/team-members/Patrick-Edosoma"
data_path = os.path.join(base_dir, "Data", "regression_processed_data")

train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

# -------------------------
# Preprocessing
# -------------------------
y_col = "addicted_score"

X_train = train_df.drop(columns=[y_col, "student_id"])
y_train = train_df[y_col]

X_test = test_df.drop(columns=[y_col, "student_id"])
y_test = test_df[y_col]

# Scale only (no PCA)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Models
# -------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(verbose=1),
    "Gradient Boosting": GradientBoostingRegressor(verbose=1),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": XGBRegressor(verbose=1),
    "CatBoost": CatBoostRegressor(verbose=1),
    "K-Nearest Neighbors": KNeighborsRegressor()
}

# -------------------------
# Train & Log Models
# -------------------------
n_train = X_train_scaled.shape[0]
n_features = X_train_scaled.shape[1]

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Fit
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (n_train - 1) / (n_train - n_features - 1)
        cv_r2 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2").mean()

        # Log model
        input_example = pd.DataFrame([X_test_scaled[0]], columns=X_train.columns)
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("adjusted_r2", adj_r2)
        mlflow.log_metric("cv_r2", cv_r2)

        # Save report
        report_text = (
            f"Model: {model_name}\n"
            f"MAE: {mae:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"R2: {r2:.4f}\n"
            f"Adjusted R2: {adj_r2:.4f}\n"
            f"Cross-Validation R2: {cv_r2:.4f}\n"
        )
        report_path = os.path.join(base_dir, f"{model_name.replace(' ', '_')}_regression_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_path)

print("✅ Regression modeling completed and logged to MLflow/DagsHub.")
