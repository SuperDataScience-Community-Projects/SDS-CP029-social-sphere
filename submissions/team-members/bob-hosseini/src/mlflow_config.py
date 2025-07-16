# mlflow_config.py
import os
from dotenv import load_dotenv, find_dotenv
import dagshub
import mlflow

def setup_mlflow(DAGSHUB_REPO):
    """Initialize MLflow tracking with DagsHub"""
    load_dotenv(find_dotenv())
    DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
    # DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
    
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    dagshub.init(repo_owner=DAGSHUB_USER_NAME, repo_name=DAGSHUB_REPO, mlflow=True)
    
    print(f"Current MLflow tracking URI: {mlflow.get_tracking_uri()}")
    return mlflow.get_tracking_uri()