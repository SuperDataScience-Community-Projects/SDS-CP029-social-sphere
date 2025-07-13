
# Core data & plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import unique

# Stats
from scipy.stats import chi2_contingency

# ML evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Experiment tracking
import mlflow
from mlflow.models import infer_signature

# Utilities
import warnings
import inspect


def plot_column_distribution(df, col):
    """Plot the distribution of a numeric column."""
    plt.figure(figsize=(6, 4))
    df[col].hist(bins=20)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_countplot(df, col, skewed=False, angle=45):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col)
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    if skewed:
        plt.xticks(rotation=angle, ha='right')  # Rotate if skewed=True
    plt.tight_layout()
    plt.show()


def plot_cramers_v_heatmap(df, target_col='y', target_mapping=None, figsize=(12, 10), cmap="YlGnBu"):
    """
    Computes pairwise Cramér's V for all categorical columns (including the target)
    and plots a heatmap.

    Parameters:
      df : pandas.DataFrame
          DataFrame containing your data.
      target_col : str, optional
          Name of the target column to be converted to categorical (default is 'y').
      target_mapping : dict, optional
          Mapping for the target column (e.g., {0: "no", 1: "yes"}). If provided,
          the target column is mapped using this dictionary before converting to object.
      figsize : tuple, optional
          Figure size for the heatmap plot.
      cmap : str, optional
          Colormap to be used in the heatmap.

    Returns:
      cramer_matrix : pandas.DataFrame
          DataFrame containing the pairwise Cramér's V values.
    """
    
    # Convert the target column to categorical using the provided mapping if available
    if target_mapping is not None:
        df[target_col] = df[target_col].map(target_mapping).astype("object")
    else:
        df[target_col] = df[target_col].astype("object")
    
    # Optionally create a temporary column (if needed) or use the converted target column directly.
    # Here we assume target_col is now categorical.
    
    # Select all categorical columns (object dtype)
    cat_cols = df.select_dtypes(include="object").columns
    
    # Define function to compute Cramér's V
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(k, r) - 1)))
    
    # Compute the pairwise Cramér’s V matrix
    cramer_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
    for col1 in cat_cols:
        for col2 in cat_cols:
            cramer_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    cramer_matrix = cramer_matrix.astype(float)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cramer_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title("Cramér’s V Heatmap (Categorical Features)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cramer_matrix

def log_class_model_run(
    model,                       # fitted estimator or pipeline
    params: dict,                # dict of hyper-parameters
    train_X, train_y,            # training data
    val_X,   val_y,              # validation data
    model_name: str = "Model",   # readable name for the run
    pos_label: int = 1,          # positive class
    experiment_name: str | None = None,   # optional: switch experiment
):
    """
    Generic MLflow run logger for binary-classification estimators that expose
    predict() and *optionally* predict_proba() / decision_function().
    """
    warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")

    # ------------------------------------------------------------------ #
    # 0) MLflow bookkeeping                                              #
    # ------------------------------------------------------------------ #
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        # ----------------------- Hyper-parameters ---------------------- #
        mlflow.log_params(params)

        # ----------------------- Dataset lineage ---------------------- #
        mlflow.log_input(
            mlflow.data.from_pandas(train_X, name="train_v1"), context="training"
        )
        mlflow.log_input(
            mlflow.data.from_pandas(val_X,   name="val_v1"),   context="validation"
        )

        # ----------------------- Predictions -------------------------- #
        train_preds = model.predict(train_X)
        val_preds   = model.predict(val_X)

        # Ensure predictions are 1D label arrays
        if hasattr(train_preds, "ndim") and train_preds.ndim > 1:
            train_preds = np.argmax(train_preds, axis=1)
        if hasattr(val_preds, "ndim") and val_preds.ndim > 1:
            val_preds = np.argmax(val_preds, axis=1)
       
       
        # --- Probabilities or decision scores (needed for ROC/AUC) ---- #
        if hasattr(model, "predict_proba"):
            train_scores = model.predict_proba(train_X)[:, 1]   # pos class prob
            val_scores   = model.predict_proba(val_X)[:, 1]
        elif hasattr(model, "decision_function"):
            train_scores = model.decision_function(train_X)
            val_scores   = model.decision_function(val_X)
        else:   # fall-back: cannot compute ROC/AUC
            train_scores = val_scores = None

        # ----------------------- Metrics ------------------------------ #
        is_binary = len(set(train_y)) == 2
        avg_type = 'binary' if is_binary else 'macro'

        metrics = {
            "accuracy_train":  accuracy_score(train_y, train_preds),
            "precision_train": precision_score(train_y, train_preds, average=avg_type, pos_label=pos_label if is_binary else None),
            "recall_train":    recall_score(train_y, train_preds,   average=avg_type, pos_label=pos_label if is_binary else None),
            "f1_train":        f1_score(train_y, train_preds,       average=avg_type, pos_label=pos_label if is_binary else None),
            "accuracy_val":    accuracy_score(val_y, val_preds),
            "precision_val":   precision_score(val_y, val_preds,   average=avg_type, pos_label=pos_label if is_binary else None),
            "recall_val":      recall_score(val_y, val_preds,       average=avg_type, pos_label=pos_label if is_binary else None),
            "f1_val":          f1_score(val_y, val_preds,           average=avg_type, pos_label=pos_label if is_binary else None),
        }

        if is_binary and train_scores is not None:
            metrics["roc_auc_train"] = roc_auc_score(train_y, train_scores)
            metrics["roc_auc_val"]   = roc_auc_score(val_y,   val_scores)

        mlflow.log_metrics(metrics)

        # -------------------- Confusion matrices ---------------------- #

        labels = sorted(np.unique(train_y)) 
        for split, y_true, y_pred in [
            ("train", train_y, train_preds),
            ("val",   val_y,   val_preds)
        ]:
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap="Purples")
            ax.set_title(f"Confusion Matrix – {split.capitalize()}")

            mlflow.log_figure(fig, f"plots/conf_matrix_{split}.png")
            plt.close(fig)

        # -------------------- ROC curves (if available) --------------- #
        if train_scores is not None and is_binary:
            for split, y_true, y_score, auc in [
                ("train", train_y, train_scores, metrics["roc_auc_train"]),
                ("val",   val_y,   val_scores,   metrics["roc_auc_val"])
            ]:
                fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC Curve – {split.capitalize()}")
                ax.legend()

                mlflow.log_figure(fig, f"plots/roc_curve_{split}.png")
                plt.close(fig)

        # -------------------- Model itself ---------------------------- #
        # Detect the correct MLflow flavour (simple heuristic)
        if "sklearn" in inspect.getmodule(model).__name__:
            mlflow.sklearn.log_model(
                model,
                name=f"{model_name.lower().replace(' ', '_')}_pipeline",
                input_example=val_X.iloc[:1].copy(),
                signature=infer_signature(train_X, train_preds),
                registered_model_name=model_name
            )
        else:
            # fall-back = log as generic pyfunc
            mlflow.pyfunc.log_model(
                name=f"{model_name.lower().replace(' ', '_')}_model",
                python_model=model,
                registered_model_name=model_name
            )

        # --------------- return metrics for convenience --------------- #
        return metrics


def log_linear_model_run(
    model,
    params: dict,
    train_X, train_y,
    val_X,   val_y,
    model_name: str = "LinearRegressionModel",
    experiment_name: str | None = None,
):
    """
    Logs a linear regression model with MLflow and returns key metrics.
    """
    import mlflow
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from mlflow.models.signature import infer_signature

    # ---------- helper ----------
    def adjusted_r2(r2, n, k):
        return 1 - (1 - r2) * (n - 1) / (n - k - 1)
    # ----------------------------

    mlflow.set_experiment(experiment_name or "default")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)

        mlflow.log_input(mlflow.data.from_pandas(train_X, name="train"), context="training")
        mlflow.log_input(mlflow.data.from_pandas(val_X,   name="val"),   context="validation")

        # Predictions
        train_preds = model.predict(train_X)
        val_preds   = model.predict(val_X)

        # Core scores
        r2_train = r2_score(train_y, train_preds)
        r2_val   = r2_score(val_y,   val_preds)

        mse_train = mean_squared_error(train_y, train_preds)
        mse_val   = mean_squared_error(val_y,   val_preds)

        metrics = {
            "mae_train":        mean_absolute_error(train_y, train_preds),
            "mse_train":        mse_train,
            "rmse_train":       np.sqrt(mse_train),
            "r2_train":         r2_train,
            "adjusted_r2_train": adjusted_r2(r2_train, len(train_y), train_X.shape[1]),

            "mae_val":          mean_absolute_error(val_y, val_preds),
            "mse_val":          mse_val,
            "rmse_val":         np.sqrt(mse_val),
            "r2_val":           r2_val,
            "adjusted_r2_val":  adjusted_r2(r2_val, len(val_y), val_X.shape[1]),
        }

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name=f"{model_name.lower().replace(' ', '_')}_model",
            input_example=val_X.iloc[:1].copy(),
            signature=infer_signature(train_X, train_preds),
            registered_model_name=model_name
        )

    return metrics