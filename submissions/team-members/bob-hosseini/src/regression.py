# Regression-specific utility functions for Social Sphere project
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from mlflow.models.signature import infer_signature
from scipy import stats
import warnings
from utils import run_shap_experiment, get_feature_names
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ========================================================================
# MLflow experiment tracking with cross-validation for regression models
# ========================================================================
def run_regression_experiment(
    name: str,
    estimator,
    X_train, y_train,
    cv,
    scoring,
    dataset,
    hparams,
    registered_model_name: str = "addicted_score_baseline"
):
    """
    Run a regression experiment with cross-validation and MLflow tracking.
    
    Parameters:
    -----------
    name : str
        Name of the experiment run for identification in MLflow UI
    estimator : sklearn estimator or Pipeline
        The regression model or pipeline to evaluate
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series or array-like
        Training target values (continuous)
    cv : sklearn cross-validation object
        Cross-validation strategy
    scoring : dict
        Dictionary of scoring metrics for regression
    dataset : dict
        Dictionary containing MLflow dataset objects
    hparams : dict
        Hyperparameters dictionary to log
    registered_model_name : str
        Name for registering the model in MLflow model registry
    """
    with mlflow.start_run(run_name=name):
        # Log inputs
        mlflow.log_input(dataset["train_ds"], context="training")    
        mlflow.log_input(dataset["test_ds"], context="test")

        # Log hyperparameters
        mlflow.log_param("hyperparameters", hparams)

        # Cross-validate & log CV metrics
        cv_results = cross_validate(
            estimator=estimator,
            X=X_train, y=y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                metric_name = metric.replace("test_", "cv_")
                mlflow.log_metric(metric_name, scores.mean().round(4))
                mlflow.log_metric(f"{metric_name}_std", scores.std().round(4))

        # Re-fit on full train and register model
        estimator.fit(X_train, y_train)

        # Infer signature & input example
        example_input = X_train.iloc[:5]
        preds = estimator.predict(example_input)
        signature = infer_signature(example_input, preds)

        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=example_input
        )


# =========================================================================
# Evaluate regression model performance on test set with comprehensive metrics
# =========================================================================

def log_regression_test_performance(
    model,
    X_test,
    y_test,
    prefix: str = "test",
    round_output: bool = False
):
    """
    Evaluate regression model on test set and log metrics to MLflow.
    
    Parameters:
    -----------
    model : fitted sklearn model or pipeline
        Trained model to evaluate
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series or array-like
        Test target values
    prefix : str
        Prefix for metric names in MLflow
    """
    # Make predictions
    y_pred = model.predict(X_test)
    if round_output:
        y_pred = y_pred.round(0)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric(f"{prefix}_mae", round(mae, 4))
    mlflow.log_metric(f"{prefix}_mse", round(mse, 4))
    mlflow.log_metric(f"{prefix}_rmse", round(rmse, 4))
    mlflow.log_metric(f"{prefix}_r2", round(r2, 4))
    mlflow.log_metric(f"{prefix}_mape", round(mape, 4))
    
    return {
        "mae": mae,
        "mse": mse, 
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "predictions": y_pred
    }


# ========================================================================
# Generate residual analysis plots for regression model diagnostics
# ========================================================================

def create_residual_plots(y_true, y_pred, model_name="Model", figsize=(15, 5)):
    """
    Create comprehensive residual analysis plots for regression model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model for plot titles
    figsize : tuple
        Figure size for the plots
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with residual plots
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Residuals vs Predicted Values
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name}: Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Q-Q Plot for Normality of Residuals
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(f'{model_name}: Q-Q Plot of Residuals')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    errors = y_true - y_pred
    axes[2].hist(errors, bins=30, alpha=0.7, density=True, label='Histogram')
    
    # Add KDE plot
    kde = gaussian_kde(errors)
    x_range = np.linspace(errors.min(), errors.max(), 200)
    axes[2].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Density')
    axes[2].set_title(f'{model_name}: Prediction Error Distribution')
    axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add error statistics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    axes[2].text(0.05, 0.95, f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}', 
                transform=axes[2].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


# ========================================================================
# Create actual vs predicted plots for regression model evaluation
# ========================================================================

def create_prediction_plots(y_true, y_pred, model_name="Model", figsize=(12, 5)):
    """
    Create prediction quality plots for regression model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model for plot titles
    figsize : tuple
        Figure size for the plots
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with prediction plots
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    # 1. Actual vs Predicted scatter plot
    axes.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes.set_xlabel('Actual Values')
    axes.set_ylabel('Predicted Values')
    axes.set_title(f'{model_name}: Actual vs Predicted')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    # Add R² score to the plot
    r2 = r2_score(y_true, y_pred)
    axes.text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # # 2. Error distribution
    # errors = y_true - y_pred
    # axes[1].hist(errors, bins=30, alpha=0.7, density=True, label='Histogram')
    
    # # Add KDE plot
    # kde = gaussian_kde(errors)
    # x_range = np.linspace(errors.min(), errors.max(), 200)
    # axes[1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # axes[1].set_xlabel('Prediction Error')
    # axes[1].set_ylabel('Density')
    # axes[1].set_title(f'{model_name}: Prediction Error Distribution')
    # axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    # axes[1].legend()
    # axes[1].grid(True, alpha=0.3)
    
    # # Add error statistics
    # mae = mean_absolute_error(y_true, y_pred)
    # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # axes[1].text(0.05, 0.95, f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}', 
    #             transform=axes[1].transAxes,
    #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


# ========================================================================
# Analyze target variable distribution and check for skewness/normality
# ========================================================================

def check_target_distribution(y, title="Target Variable Distribution", transform=None):
    from scipy.stats import boxcox
    """
    Analyze the distribution of the target variable and check for skewness.
    
    Parameters:
    -----------
    y : pd.Series or array-like
        Target variable values
    title : str
        Title for the plots
    log_transform : bool
        Whether to show log-transformed distribution as well
        
    Returns:
    --------
    dict
        Dictionary with distribution statistics
    """
    fig, axes = plt.subplots(1, 2 if not transform else 3, figsize=(15, 5))
    
    # Original distribution
    axes[0].hist(y, bins=30, alpha=0.7, density=True)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'{title} - Original')
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics
    skewness = stats.skew(y)
    kurtosis = stats.kurtosis(y)
    axes[0].text(0.05, 0.95, f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}', 
                transform=axes[0].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Q-Q plot
    stats.probplot(y, dist="norm", plot=axes[1])
    axes[1].set_title(f'{title} - Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    # Log transform if requested
    if transform == "log":
        y_log = np.log1p(y)  # log(1+x) to handle zeros
        axes[2].hist(y_log, bins=30, alpha=0.7, density=True)
        axes[2].set_xlabel('Log(1+Value)')
        axes[2].set_ylabel('Density')
        axes[2].set_title(f'{title} - Log Transformed')
        axes[2].grid(True, alpha=0.3)
        
        skewness_log = stats.skew(y_log)
        axes[2].text(0.05, 0.95, f'Skewness: {skewness_log:.3f}', 
                    transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))    
    elif transform == "boxcox":
        y_boxcox, lambda_param = boxcox(y)
        axes[2].hist(y_boxcox, bins=30, alpha=0.7, density=True)
        axes[2].set_xlabel('Box-Cox Transformed')
        axes[2].set_ylabel('Density')
        axes[2].set_title(f'{title} - Box-Cox Transformed')
        axes[2].grid(True, alpha=0.3)
        skewness_boxcox = stats.skew(y_boxcox)
        axes[2].text(0.05, 0.95, f'Skewness: {skewness_boxcox:.3f}', 
                    transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()
    
    stats_dict = {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mean': np.mean(y),
        'std': np.std(y),
        'min': np.min(y),
        'max': np.max(y)
    }
    
    if transform == "log":
        stats_dict['skewness_log'] = skewness_log
    elif transform == "boxcox":
        stats_dict['skewness_boxcox'] = skewness_boxcox
        
    return stats_dict


# ========================================================================
# Visualize linear model coefficients with magnitude and sign analysis
# ========================================================================

def create_coefficient_plot(
    fitted_pipeline, 
    model_name="Linear Model",
    top_n=20,
    figsize=(12, 8)
):
    """
    Create coefficient plots for a fitted linear model.
    
    Parameters:
    -----------
    fitted_pipeline : sklearn Pipeline
        Fitted pipeline with preprocessing and linear regressor
    model_name : str
        Name of the model for plot title
    top_n : int
        Number of top coefficients to display
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with coefficient plots
    """
    # from utils import get_feature_names
    
    # Extract components
    preprocessor = fitted_pipeline.named_steps['preprocessing']
    regressor = fitted_pipeline.named_steps['regressor']
    
    # Handle RoundingRegressor wrapper
    if isinstance(regressor, RoundingRegressor):
        # Get the inner regressor's coefficients
        coefficients = regressor.regressor.coef_
    else:
        # Get coefficients directly from the fitted model
        coefficients = regressor.coef_
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    
    # Create coefficient dataframe for easier handling
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{model_name}: Coefficient Analysis', fontsize=16)
    
    # Plot 1: Top N coefficients (horizontal bar)
    ax1 = axes[0]
    top_coefs = coef_df.head(top_n)
    colors = ['red' if coef < 0 else 'blue' for coef in top_coefs['coefficient']]
    
    # Truncate long feature names
    display_names = [name[:25] + '...' if len(name) > 25 else name 
                    for name in top_coefs['feature']]
    
    bars = ax1.barh(range(len(top_coefs)), top_coefs['coefficient'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_coefs)))
    ax1.set_yticklabels(display_names, fontsize=8)
    ax1.set_xlabel('Coefficient Value')
    ax1.set_title(f'Top {top_n} Coefficients by Magnitude')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add coefficient values on bars
    for i, (bar, coef) in enumerate(zip(bars, top_coefs['coefficient'])):
        ax1.text(coef + (0.01 * np.sign(coef) if coef != 0 else 0.01), i, 
                f'{coef:.3f}', ha='left' if coef >= 0 else 'right', va='center', fontsize=7)
    # =====
    # Plot 2: Positive vs Negative coefficients
    ax2 = axes[1]
    positive_coefs = coefficients[coefficients > 0]
    negative_coefs = coefficients[coefficients < 0]
    zero_coefs = coefficients[np.abs(coefficients) <= 1e-6]
    
    categories = ['Positive', 'Negative', 'Zero/Near-Zero']
    counts = [len(positive_coefs), len(negative_coefs), len(zero_coefs)]
    colors_pie = ['blue', 'red', 'gray']
    
    wedges, texts, autotexts = ax2.pie(counts, labels=categories, colors=colors_pie, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Coefficient Sign Distribution')
            
    plt.tight_layout()
    return fig


# ========================================================================
# Generate feature importance plots for tree-based regression models
# ========================================================================

def create_feature_importance_plot(
    best_estimator,
    model_name="Model",
    figsize=(15, 5)
):
    feature_importance = best_estimator.named_steps['regressor'].feature_importances_
    feature_names = get_feature_names(best_estimator.named_steps['preprocessing'])

    # Create feature importance plot
    fig = plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(20)

    plt.barh(range(len(importance_df)), importance_df['importance'], color='skyblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost: Top 20 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.close(fig)
    # plt.show()
    return fig


# ========================================================================
# Rounding transformer for continuous predictions to rounded integers
# ========================================================================
class RoundingRegressor(BaseEstimator):
    """Regressor wrapper that rounds predictions to integers within specified bounds."""
    
    def __init__(self, regressor, min_value=None, max_value=None):
        self.regressor = regressor
        self.min_value = min_value
        self.max_value = max_value
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self
    
    def predict(self, X):
        # Get raw predictions
        predictions = self.regressor.predict(X)
        
        # Round to nearest integer
        rounded = np.round(predictions).astype(int)
        
        # Apply bounds if specified
        if self.min_value is not None:
            rounded = np.maximum(rounded, self.min_value)
        if self.max_value is not None:
            rounded = np.minimum(rounded, self.max_value)
            
        return rounded
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'regressor': self.regressor,
            'min_value': self.min_value,
            'max_value': self.max_value
        }
        if deep and hasattr(self.regressor, 'get_params'):
            regressor_params = self.regressor.get_params(deep=True)
            for key, value in regressor_params.items():
                params[f'regressor__{key}'] = value
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        regressor_params = {}
        valid_params = {}
        
        for key, value in params.items():
            if key.startswith('regressor__'):
                regressor_params[key[11:]] = value  # Remove 'regressor__' prefix
            elif key in ['regressor', 'min_value', 'max_value']:
                valid_params[key] = value
            else:
                raise ValueError(f"Invalid parameter {key}")
        
        # Set regressor parameters
        if regressor_params and hasattr(self.regressor, 'set_params'):
            self.regressor.set_params(**regressor_params)
        
        # Set wrapper parameters
        for key, value in valid_params.items():
            setattr(self, key, value)
            
        return self


# ========================================================================
# Comprehensive grid search with MLflow tracking and visualization logging
# ========================================================================
def run_regression_gridsearch_experiment(
    name: str,
    pipeline,
    param_grid: dict,
    X_train, y_train, X_test, y_test,
    cv,
    scoring,
    dataset: dict,
    registered_model_name: str,
    verbose: bool = False,
    run_tag: str = "all_features",
    refit_metric: str = "r2",
    round_output: bool = False,
    linear: bool = False,
    shap: bool = True,
    coeff_profile: bool = True
):
    """
    Run GridSearchCV experiment for regression with comprehensive logging.
    
    Parameters:
    -----------
    name : str
        Experiment name
    pipeline : sklearn Pipeline
        Pipeline with preprocessing and regressor
    param_grid : dict
        Parameters to search over
    X_train, y_train : training data
    X_test, y_test : test data
    cv : cross-validation strategy
    scoring : dict of scoring metrics
    dataset : MLflow dataset objects
    registered_model_name : str
        Model registry name
    verbose : bool
        Whether to print progress
    run_tag : str
        Description of feature set and other decision parameters used
    refit_metric : str
        Metric to use for selecting best model
        
    Returns:
    --------
    fitted GridSearchCV object
    """
    from sklearn.model_selection import GridSearchCV
    
    with mlflow.start_run(run_name=name):
        # Log inputs and parameters
        mlflow.log_input(dataset["train_ds"], context="training")    
        mlflow.log_input(dataset["test_ds"], context="test")
        mlflow.log_param("run_tag", run_tag)
        mlflow.log_param("refit_metric", refit_metric)
        mlflow.log_param("param_grid", str(param_grid))
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X_train, y_train)
        
        # 4) Log the best hyperparameters
        best_params = grid_search.best_params_
        mlflow.log_param("hyperparameters", best_params)

       
        for metric in scoring.keys():
            cv_score = grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
            cv_std = grid_search.cv_results_[f'std_test_{metric}'][grid_search.best_index_]
            mlflow.log_metric(f"cv_{metric}", round(cv_score, 4))
            mlflow.log_metric(f"cv_{metric}_std", round(cv_std, 4))
            if verbose:
                print(f"{metric}: {cv_score.round(2)} ± {cv_std.round(2)}")
        
        # Test set evaluation
        best_estimator = grid_search.best_estimator_
        test_results = log_regression_test_performance(
            best_estimator, X_test, y_test, prefix="test"
        )

        # Get the actual model, handling RoundingRegressor wrapper
        model = best_estimator.named_steps['regressor']
        if isinstance(model, RoundingRegressor):
            # If it's a wrapper, get the inner regressor
            inner_model = model.regressor
        else:
            inner_model = model
            
        if isinstance(inner_model, (LinearRegression, Lasso, Ridge)):
            model_type = "linear"
        elif isinstance(inner_model, (XGBRegressor, CatBoostRegressor)):
            model_type = "tree"
        else:
            model_type = "other"
        
        # Create and log visualizations for linear models
        # ===== Residual plots
        print(f"Model type: {model_type}")
        residual_fig = None
        if model_type == "linear":
            residual_fig = create_residual_plots(
                y_test, test_results["predictions"], 
                model_name=name,
                figsize=(15, 5)
            )
            mlflow.log_figure(residual_fig, "residual_analysis.png")
            plt.close(residual_fig)
        
        # ===== Prediction plots
        prediction_fig = None        
        prediction_fig = create_prediction_plots(
            y_test, 
            test_results["predictions"],
            model_name=name,
            figsize=(15, 5)
        )
        mlflow.log_figure(prediction_fig, "prediction_analysis.png")
        plt.close(prediction_fig)


        # ===== inner regressor for SHAP and coefficient profile plots
        regressor = best_estimator.named_steps['regressor']
        if type(regressor).__name__ == 'RoundingRegressor':
            inner_pipeline = Pipeline([
                ('preprocessing', best_estimator.named_steps['preprocessing']),
                ('regressor', best_estimator.named_steps['regressor'].regressor)  # Extract inner regressor
            ])
        else:
            inner_pipeline = best_estimator
            
        # ===== SHAP plot for regression
        fig_sum = None
        if 'baseline' not in name and shap:
            # Check if the model is linear regression or lasso based on the best_estimator            
            # model = best_estimator.named_steps['regressor']
            if model_type == "linear":
                shap_type = "linear"
            else:
                shap_type = "tree"
            feature_perturbation = "interventional"
            if 'catboost' in name.lower():
                feature_perturbation = "tree_path_dependent"
            # print(f"Running SHAP for {name} with {plot_type} and {shap_type}")

            fig_sum = run_shap_experiment(
                best_model=inner_pipeline,
                X_train_full=X_train,
                random_state=42,
                feature_perturbation=feature_perturbation,
                plot_type='violin', # other options: "bar", "dot", "violin"
                shap_type=shap_type,
                model_type="regression",
                figsize=(8, 5)
            )
            mlflow.log_figure(fig_sum, "shap_summary.png")
        
        # Register model
        example_input = X_train.iloc[:5]
        preds = best_estimator.predict(example_input)
        signature = infer_signature(example_input, preds)
        
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=example_input
        )
        # ===== Coefficient profile plot
        coef_fig = None
        if coeff_profile:
            if linear:
                coef_fig = create_coefficient_plot(best_estimator, model_name=name, figsize=(15, 5))
                mlflow.log_figure(coef_fig, "coefficient_profile.png")
                plt.close(coef_fig)
            elif model_type == "tree":
                coef_fig = create_feature_importance_plot(inner_pipeline, model_name=name, figsize=(15, 5))
                mlflow.log_figure(coef_fig, "feature_importance.png")
                plt.close(coef_fig)
    
    # ===== Return figures
    figures = []
    if residual_fig is not None:
        figures.append(residual_fig)
    if prediction_fig is not None:
        figures.append(prediction_fig)
    if coef_fig is not None:
        figures.append(coef_fig)
    if fig_sum is not None:
        figures.append(fig_sum)

    # ===== Print results
    if verbose:
        print("Best parameters:", grid_search.best_params_)
        print("Best R² score (CV):", grid_search.best_score_.round(2))
    
    return best_estimator, test_results, *figures

