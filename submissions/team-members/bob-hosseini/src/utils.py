"""
Utility functions for social media conflict classification project.

This module contains reusable functions for data preprocessing, model training,
evaluation, and visualization used in the classification notebooks.
"""

# Standard library imports
import json
import os

# Third-party imports
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models.signature import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
from scipy import sparse
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ======== Warnings ========
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column"
)
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")


# ====================================================================================================================
# MLFLOW EXPERIMENT MANAGEMENT
# ====================================================================================================================

# ====================================================================================================================
# Create MLflow dataset objects for experiment tracking and data lineage
# ====================================================================================================================
def mlflow_dataset(X_train_full, X_test):
    """
    Create MLflow dataset objects for experiment tracking and data lineage.
    
    This function creates MLflow dataset objects that can be logged with experiments
    to track data lineage, ensure reproducibility, and maintain a clear record of
    which datasets were used for training and testing in each experiment.
    
    Parameters:
    -----------
    X_train_full : pd.DataFrame
        Complete training dataset (features only)
    X_test : pd.DataFrame
        Test dataset (features only)
        
    Returns:
    --------
    dict
        Dictionary containing MLflow dataset objects:
        - 'train_ds': Training dataset object for MLflow logging
        - 'test_ds': Test dataset object for MLflow logging
        
    Examples:
    ---------
    >>> # Create dataset objects for experiment tracking
    >>> datasets = mlflow_dataset(X_train, X_test)
    >>> 
    >>> # Log datasets in an MLflow experiment
    >>> with mlflow.start_run():
    ...     mlflow.log_input(datasets['train_ds'], context="training")
    ...     mlflow.log_input(datasets['test_ds'], context="testing")
    """
    train_ds = mlflow.data.from_pandas(
        df=X_train_full,
        source="../data/data_cleaned.pickle",
        name="social_sphere_train_v1"
    )
    test_ds = mlflow.data.from_pandas(
        df=X_test,
        source="../data/data_cleaned.pickle",
        name="social_sphere_test_v1"
    )
    return {"train_ds": train_ds, "test_ds": test_ds}

# ====================================================================================================================
# Delete all runs from a specific MLflow experiment
# ====================================================================================================================
def quick_delete_experiment(experiment_name):
    import mlflow
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        for _, run in runs.iterrows():
            mlflow.delete_run(run.run_id)
        print(f"Deleted {len(runs)} runs from '{experiment_name}'")
    else:
        print(f"Experiment '{experiment_name}' not found")

# ====================================================================================================================
# DATA PREPROCESSING FUNCTIONS
# ====================================================================================================================

# ====================================================================================================================
# Convert continuous conflict scores to binary classification (High vs Low)
# ====================================================================================================================
def create_binary_conflict(df, target_column='Conflicts', threshold=None, visualize=True):
    """
    Create binary conflict classification (High vs Low) from conflict scores.
    
    This function transforms a continuous conflict score into a binary classification
    problem by applying a threshold. Values above the threshold are classified as 'High'
    conflict, while values at or below are classified as 'Low' conflict.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the conflict scores
    target_column : str, default='Conflicts'
        Name of the column containing conflict scores
    threshold : int or float, optional
        Threshold value for binary classification. If None, uses the median of the data
    visualize : bool, default=True
        Whether to create visualization plots showing the distribution and threshold
        
    Returns:
    --------
    tuple[pd.DataFrame, dict]
        - pd.DataFrame: Copy of input dataframe with added 'Conflict_Binary' column (0=Low, 1=High)
        - dict: Analysis results containing threshold value, class counts, imbalance ratio, and proportions
        
    Examples:
    ---------
    >>> df_binary, results = create_binary_conflict(df, threshold=2.5)
    >>> print(f"Threshold used: {results['threshold']}")
    >>> print(f"Class distribution: {results['counts']}")
    """
    # Make a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Determine threshold
    if threshold is None:
        threshold = df_copy[target_column].median()
    
    # Create binary target variable
    df_copy['Conflict_Binary'] = df_copy[target_column].apply(
        lambda x: 'High' if x >= threshold else 'Low'
    )
    
    # Calculate statistics
    conflict_counts = df_copy['Conflict_Binary'].value_counts()
    imbalance_ratio = conflict_counts.min() / conflict_counts.max() * 100
    
    # Print analysis
    print(f"Binary Conflict Classification:")
    print(f"Threshold: {threshold}")
    print(f"Low Conflict (0-{threshold-1}): {conflict_counts.get('Low', 0)} samples")
    print(f"High Conflict ({threshold}-max): {conflict_counts.get('High', 0)} samples")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}%")
    print(f"Class proportions:")
    print(conflict_counts / len(df_copy))

    # convert to 0 and 1
    df_copy['Conflict_Binary'] = df_copy['Conflict_Binary'].map({'Low': 0, 'High': 1})
    
    # Create visualizations if requested
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Original distribution with threshold line
        axes[0].hist(df_copy[target_column], bins=range(int(df_copy[target_column].min()), 
                                                       int(df_copy[target_column].max()) + 2), 
                    alpha=0.7, edgecolor='black')
        axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold = {threshold}')
        axes[0].set_xlabel('Conflict Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Original Conflict Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Binary distribution
        sns.countplot(data=df_copy, x='Conflict_Binary', ax=axes[1])
        axes[1].set_title(f'Binary Conflict Distribution\n(Threshold: {threshold})\n(0: Low, 1: High)')
        axes[1].set_ylabel('Count')
        
        # Add count labels on bars
        for i, bar in enumerate(axes[1].patches):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    # Prepare results dictionary
    results = {
        'threshold': threshold,
        'counts': conflict_counts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'proportions': (conflict_counts / len(df_copy)).to_dict()
    }
    
    return df_copy, results


# ====================================================================================================================
# FEATURE ENGINEERING FUNCTIONS  
# ====================================================================================================================

# ====================================================================================================================
# One-hot encoding with smallest category as reference (dropped to avoid multicollinearity)
# ====================================================================================================================
def encode_onehot_with_reference(df, column_name, prefix=None):
    """
    Perform one-hot encoding with the smallest category as reference (dropped).
    
    This function creates dummy variables for categorical data while dropping the least
    frequent category to serve as a reference category, which helps avoid multicollinearity
    in linear models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the categorical column
    column_name : str
        Name of the categorical column to encode
    prefix : str, optional
        Prefix for the new dummy column names. If None, uses the column name
        
    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        - pd.DataFrame: Original dataframe with added dummy columns
        - pd.DataFrame: Just the dummy columns that were created
        
    Examples:
    ---------
    >>> df_encoded, dummies = encode_onehot_with_reference(df, 'Gender', prefix='gender')
    >>> print(f"New columns: {dummies.columns.tolist()}")
    """
    counts = df[column_name].value_counts()
    smallest_category = counts.index[-1]  # Get smallest category
    
    print(f"\n Encoding {column_name} with prefix {prefix}")
    print(f"Reference category (dropped): {smallest_category}")
    
    # Create one-hot encoded columns
    dummies = pd.get_dummies(df[column_name], prefix=prefix or column_name, drop_first=False)
    # Drop smallest category as reference
    dummies = dummies.drop(f'{prefix or column_name}_{smallest_category}', axis=1)

    # print the first 5 rows of the encoded variable
    print(dummies.head())

   
    # Concatenate with original dataframe
    df = pd.concat([df, dummies], axis=1)

    # print samples of the other category
    print(f"\nSamples of the other category:")
    mask = df[column_name] == smallest_category
    print(pd.concat([df[mask][[column_name]], dummies[mask]], axis=1).head())

    return df, dummies



# ====================================================================================================================
# Frequency-based encoding for high-cardinality categorical variables
# ====================================================================================================================
def encode_frequency(df, column_name):
    """
    Apply frequency-based encoding to a categorical column.
    
    This method replaces categorical values with their frequency count in the dataset.
    It's particularly useful for high-cardinality categorical variables where one-hot
    encoding would create too many columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the categorical column
    column_name : str
        Name of the categorical column to encode
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added frequency-encoded column named '{column_name}_freq_encoded'
        
    Examples:
    ---------
    >>> df_encoded = encode_frequency(df, 'Country')
    >>> print(df_encoded['Country_freq_encoded'].head())
    """
    freq = df[column_name].value_counts().to_dict()
    df[f'{column_name}_freq_encoded'] = df[column_name].map(freq)

    # print the first 5 rows of the encoding mapping   
    print(f"\n{column_name} frequency encoding (top 5):")
    for country, freq in list(freq.items())[:5]:
        print(f"{country}: {freq}")

    # print the first 5 rows of the encoded variable
    print(f"\n{column_name} frequency encoded variable (top 5):")
    print(df[[f'{column_name}_freq_encoded']].head())

    # plot encoded variable distribution
    plt.figure(figsize=(8, 3))
    sns.countplot(data=df, x=f'{column_name}_freq_encoded')
    plt.title(f'Distribution of {column_name} Encoded Variable')
    plt.ylabel('Count')
    plt.show()

    return df

# ====================================================================================================================
# Dictionary mapping each country to its continent
# ====================================================================================================================
country_to_continent = {
    # Africa
    "Egypt": "Africa", "Morocco": "Africa", "South Africa": "Africa",
    "Nigeria": "Africa", "Kenya": "Africa", "Ghana": "Africa",
    # Asia
    "Bangladesh": "Asia", "India": "Asia", "China": "Asia",
    "Japan": "Asia", "South Korea": "Asia", "Malaysia": "Asia",
    "Thailand": "Asia", "Vietnam": "Asia", "Philippines": "Asia",
    "Indonesia": "Asia", "Taiwan": "Asia", "Hong Kong": "Asia",
    "Singapore": "Asia", "UAE": "Asia", "Israel": "Asia",
    "Turkey": "Asia", "Qatar": "Asia", "Kuwait": "Asia",
    "Bahrain": "Asia", "Oman": "Asia", "Jordan": "Asia",
    "Lebanon": "Asia", "Iraq": "Asia", "Yemen": "Asia",
    "Syria": "Asia", "Afghanistan": "Asia", "Pakistan": "Asia",
    "Nepal": "Asia", "Bhutan": "Asia", "Sri Lanka": "Asia",
    "Maldives": "Asia", "Kazakhstan": "Asia", "Uzbekistan": "Asia",
    "Kyrgyzstan": "Asia", "Tajikistan": "Asia", "Armenia": "Asia",
    "Georgia": "Asia", "Azerbaijan": "Asia", "Cyprus": "Asia",
    # Europe
    "UK": "Europe", "Germany": "Europe", "France": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Denmark": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Switzerland": "Europe", "Austria": "Europe",
    "Portugal": "Europe", "Greece": "Europe", "Ireland": "Europe",
    "Iceland": "Europe", "Finland": "Europe", "Poland": "Europe",
    "Romania": "Europe", "Hungary": "Europe", "Czech Republic": "Europe",
    "Slovakia": "Europe", "Croatia": "Europe", "Serbia": "Europe",
    "Slovenia": "Europe", "Bulgaria": "Europe", "Estonia": "Europe",
    "Latvia": "Europe", "Lithuania": "Europe", "Ukraine": "Europe",
    "Moldova": "Europe", "Belarus": "Europe", "Russia": "Europe",
    "Luxembourg": "Europe", "Monaco": "Europe", "Andorra": "Europe",
    "San Marino": "Europe", "Vatican City": "Europe",
    "Liechtenstein": "Europe", "Montenegro": "Europe", "Albania": "Europe",
    "North Macedonia": "Europe", "Kosovo": "Europe", "Bosnia": "Europe",
    # North America
    "USA": "North America", "Canada": "North America",
    "Mexico": "North America", "Costa Rica": "North America",
    "Panama": "North America", "Jamaica": "North America",
    "Trinidad": "North America", "Bahamas": "North America",
    # South America
    "Brazil": "South America", "Argentina": "South America",
    "Chile": "South America", "Colombia": "South America",
    "Peru": "South America", "Venezuela": "South America",
    "Ecuador": "South America", "Uruguay": "South America",
    "Paraguay": "South America", "Bolivia": "South America",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania"
}

# ====================================================================================================================
# Map individual country to its corresponding continent
# ====================================================================================================================
def cont_map(country):
    """
    Map a country name to its corresponding continent.
    
    This function uses a predefined dictionary to map country names to their
    geographical continents. It's useful for reducing the dimensionality of
    high-cardinality country variables by grouping them into broader categories.
    
    Parameters:
    -----------
    country : str
        Name of the country to map
        
    Returns:
    --------
    str
        Continent name ('Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania')
        or 'Other' if country is not found in the mapping dictionary
        
    Examples:
    ---------
    >>> continent = cont_map("USA")
    >>> print(continent)  # Output: "North America"
    >>> 
    >>> continent = cont_map("UnknownCountry") 
    >>> print(continent)  # Output: "Other"
    """
    return country_to_continent.get(country, "Other")

def map_to_continent(df, visualize=False):
    """
    Add a continent column to the dataframe based on the Country column.
    
    This function applies the continent mapping to all countries in the dataframe
    and optionally creates a visualization showing the distribution of continents.
    This is useful for geographic analysis and feature engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing a 'Country' column
    visualize : bool, default=False
        Whether to create a bar plot visualization of the continent distribution
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added 'Continent' column
        
    Examples:
    ---------
    >>> df_with_continent = map_to_continent(df, visualize=True)
    >>> print(df_with_continent['Continent'].value_counts())
    """
    df["Continent"] = df["Country"].apply(cont_map)
    if visualize:
        plt.figure(figsize=(8, 3))
        sns.countplot(data=df, x='Continent')
        plt.title('Distribution of Continent Variable')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    return df



# ====================================================================================================================
# SKLEARN TRANSFORMERS
# ====================================================================================================================

# ====================================================================================================================
# Group rare categories into "Other" category for dimensionality reduction
# ====================================================================================================================
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups categories whose absolute count in the training data is below `min_count`
    into a single 'Other' category.
    
    This transformer helps reduce the number of categories in high-cardinality categorical
    variables by grouping infrequent categories together, which can improve model
    generalization and reduce overfitting. It's particularly useful for variables like
    country names, job titles, or other categorical features with many unique values.
    
    Parameters:
    -----------
    min_count : int, default=30
        Minimum number of occurrences required for a category to be kept separate.
        Categories with fewer occurrences will be grouped into 'Other'.
        
    Attributes:
    -----------
    frequent_categories_ : set
        Set of categories that appear at least min_count times in the training data.
        This is learned during the fit phase and used for transformations.
        
    Examples:
    ---------
    >>> # Group countries that appear less than 50 times
    >>> grouper = RareCategoryGrouper(min_count=50)
    >>> X_transformed = grouper.fit_transform(X[['Country']])
    >>> 
    >>> # Use in a pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    ...     ('grouper', RareCategoryGrouper(min_count=30)),
    ...     ('encoder', OneHotEncoder())
    ... ])
    """
    
    def __init__(self, min_count=30):
        """
        Initialize the RareCategoryGrouper.
        
        Parameters:
        -----------
        min_count : int, default=30
            Minimum frequency threshold for keeping categories separate
        """
        self.min_count = min_count
        self.frequent_categories_ = set()

    def fit(self, X, y=None):
        """
        Fit the transformer by identifying frequent categories.
        
        Analyzes the input data to determine which categories appear frequently
        enough to be kept separate (>= min_count occurrences).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Input categorical data
        y : array-like, optional
            Target values (ignored, present for API consistency)
            
        Returns:
        --------
        self : RareCategoryGrouper
            Returns self for method chaining
        """
        # Flatten to 1-D array and handle missing values
        arr = np.array(X)
        flat = arr.ravel()
        series = pd.Series(flat).fillna("Missing")
        
        # Count frequencies and identify frequent categories
        counts = series.value_counts()
        self.frequent_categories_ = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        """
        Transform the input by grouping rare categories into 'Other'.
        
        Categories that don't appear in frequent_categories_ (i.e., appeared
        less than min_count times in training data) are replaced with 'Other'.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Input categorical data to transform
            
        Returns:
        --------
        pd.DataFrame of shape (n_samples, 1)
            Transformed data with rare categories grouped as 'Other'
        """
        # Flatten input and handle missing values
        arr = np.array(X)
        flat = arr.ravel()
        series = pd.Series(flat).fillna("Missing")
        
        # Replace rare categories with 'Other'
        out = series.where(series.isin(self.frequent_categories_), "Other")
        
        # Return as 2-D DataFrame for pipeline compatibility
        return out.to_frame()


# ====================================================================================================================
# Transform country names to continents using sklearn transformer interface
# ====================================================================================================================
class CountryToContinentMapper(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer to map country names to their corresponding continents.
    
    This transformer provides a pipeline-compatible way to convert country names
    to continent names using a predefined mapping dictionary. This helps reduce
    the dimensionality of country variables from hundreds of countries to just
    6-7 continents, which can improve model generalization.
    
    Parameters:
    -----------
    mapping : dict
        Dictionary mapping country names (keys) to continent names (values)
        
    Attributes:
    -----------
    mapping : dict
        The country-to-continent mapping dictionary used for transformations
        
    Examples:
    ---------
    >>> # Create mapper with predefined continent dictionary
    >>> mapper = CountryToContinentMapper(country_to_continent)
    >>> X_continents = mapper.fit_transform(X[['Country']])
    >>> 
    >>> # Use in a pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    ...     ('country_mapper', CountryToContinentMapper(country_to_continent)),
    ...     ('encoder', OneHotEncoder())
    ... ])
    """
    
    def __init__(self, mapping):
        """
        Initialize the mapper with a country-to-continent mapping.
        
        Parameters:
        -----------
        mapping : dict
            Dictionary where keys are country names and values are continent names
        """
        self.mapping = mapping

    def fit(self, X, y=None):
        """
        Fit method (no operation needed for this transformer).
        
        This transformer doesn't learn anything from the data, it just applies
        a predefined mapping, so fit() simply returns self.
        
        Parameters:
        -----------
        X : array-like
            Input data (not used, present for API consistency)
        y : array-like, optional
            Target values (not used, present for API consistency)
            
        Returns:
        --------
        self : CountryToContinentMapper
            Returns self for method chaining
        """
        return self

    def transform(self, X):
        """
        Transform country names to continent names.
        
        Applies the country-to-continent mapping to the input data. Countries
        not found in the mapping dictionary are assigned to 'Other' continent.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input data containing country names
            
        Returns:
        --------
        pd.DataFrame of shape (n_samples, 1)
            Transformed data with continent names. Unknown countries become 'Other'.
        """
        # Handle both numpy arrays and DataFrames
        vals = X.values if hasattr(X, 'values') else X
        flat = vals.ravel()  # Flatten to 1D array
        
        # Apply mapping and handle missing countries
        mapped = pd.Series(flat).map(self.mapping)
        return mapped.fillna("Other").to_frame()


# ====================================================================================================================
# UTILITY FUNCTIONS FOR SKLEARN PIPELINES
# ====================================================================================================================

# ====================================================================================================================
# Extract feature names from column transformer for model interpretability
# ====================================================================================================================
def get_feature_names(column_transformer):
    """
    Extract feature names from a fitted ColumnTransformer.
    
    This helper function retrieves the names of features after transformation,
    which is essential for model interpretability and feature importance analysis.
    It handles cases where some transformers may not have the get_feature_names_out
    method, providing fallback strategies for feature name extraction.
    
    Parameters:
    -----------
    column_transformer : sklearn.compose.ColumnTransformer
        A fitted column transformer containing multiple preprocessing steps
        
    Returns:
    --------
    list
        List of feature names after all transformations have been applied.
        Names reflect the output features from the entire preprocessing pipeline.
        
    Examples:
    ---------
    >>> # After fitting a preprocessor
    >>> feature_names = get_feature_names(preprocessor)
    >>> print(f"Number of features after preprocessing: {len(feature_names)}")
    >>> print(f"First 5 features: {feature_names[:5]}")
    >>> 
    >>> # Use with SHAP for feature importance
    >>> X_processed = preprocessor.transform(X)
    >>> feature_names = get_feature_names(preprocessor)
    >>> shap_df = pd.DataFrame(X_processed, columns=feature_names)
    """
    feature_names = []
    
    # Iterate through all transformers in the ColumnTransformer
    for name, transformer, cols in column_transformer.transformers_:
        # Skip dropped columns or remainder transformations
        if transformer == 'drop' or name == 'remainder':
            continue

        # Normalize column specification to a list
        input_cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]

        # For pipelines, get the last step; otherwise use the transformer directly
        tr = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer

        # Try multiple strategies to get feature names
        if hasattr(tr, 'get_feature_names_out'):
            try:
                # Strategy 1: Pass original column names (preferred)
                names = tr.get_feature_names_out(input_cols)
            except Exception:
                try:
                    # Strategy 2: Call without arguments (fallback)
                    names = tr.get_feature_names_out()
                except Exception:
                    # Strategy 3: Use input column names (last resort)
                    names = input_cols
        else:
            # Strategy 4: Transformer has no feature naming method
            names = input_cols

        # Add names to the overall list
        feature_names.extend(names)
    
    return feature_names


# ====================================================================================================================
# Run a basic classification experiment with cross-validation and MLflow tracking
# ====================================================================================================================
def run_classification_experiment(
    name: str,
    estimator,                # e.g. Pipeline([('preproc', preprocessor), ('clf', LogisticRegression(...))])
    X_train, y_train,
    cv,
    scoring,            # dict of scoring metrics
    dataset,
    hparams,
    registered_model_name: str = "conflict_baseline_dummy"
):
    """
    Run a classification experiment with cross-validation and MLflow tracking.
    
    This function performs cross-validation on a classification model and logs
    all relevant metrics, parameters, and the trained model to MLflow for
    experiment tracking and reproducibility. It's designed for rapid experimentation
    with different models and hyperparameters while maintaining full traceability.
    
    Parameters:
    -----------
    name : str
        Name of the experiment run for identification in MLflow UI
    estimator : sklearn estimator or Pipeline
        The classification model or pipeline to evaluate. Should include preprocessing
        and the final classifier (e.g., Pipeline([('preproc', preprocessor), ('clf', classifier)]))
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series or array-like
        Training labels/target values
    cv : sklearn cross-validation object
        Cross-validation strategy (e.g., StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    scoring : dict
        Dictionary of scoring metrics for cross-validation evaluation.
        Keys are metric names, values are sklearn scorer strings
        (e.g., {"accuracy": "accuracy", "f1_score": "f1", "roc_auc": "roc_auc"})
    dataset : dict
        Dictionary containing MLflow dataset objects created by mlflow_dataset()
        Must contain keys 'train_ds' and 'test_ds' for data lineage tracking
    hparams : dict
        Hyperparameters dictionary to log for experiment tracking
    registered_model_name : str, default="conflict_baseline_dummy"
        Name for registering the model in MLflow model registry
        
    Returns:
    --------
    None
        Function logs all results to MLflow but doesn't return values.
        Check MLflow UI for experiment results and model artifacts.
        
    Examples:
    ---------
    >>> # Basic usage with dummy classifier
    >>> from sklearn.dummy import DummyClassifier
    >>> from sklearn.model_selection import StratifiedKFold
    >>> 
    >>> dummy_pipeline = Pipeline([
    ...     ('classifier', DummyClassifier(strategy='most_frequent'))
    ... ])
    >>> cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> scoring = {"accuracy": "accuracy", "f1_score": "f1"}
    >>> datasets = mlflow_dataset(X_train, X_test)
    >>> 
    >>> run_classification_experiment(
    ...     name="baseline_dummy",
    ...     estimator=dummy_pipeline,
    ...     X_train=X_train, y_train=y_train,
    ...     cv=cv, scoring=scoring,
    ...     dataset=datasets, 
    ...     hparams={"strategy": "most_frequent"},
    ...     registered_model_name="conflict_baseline_dummy"
    ... )
    
    Notes:
    ------
    - This function starts an MLflow run automatically
    - All cross-validation metrics are logged with mean scores
    - The final model is fitted on the full training set and registered
    - Model signature and input example are automatically inferred
    - Use this for baseline models and simple experiments
    """
    with mlflow.start_run(run_name=name):
        # Log inputs
        mlflow.log_input(dataset["train_ds"], context="training")    
        mlflow.log_input(dataset["test_ds"], context="test")

        # log strategy
        mlflow.log_param("hyperparameters", hparams)

        # 2) Cross-validate & log CV metrics
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
                mlflow.log_metric(metric.replace("test_", ""), scores.mean().round(2))

        # 3) Re-fit on full train and register model
        estimator.fit(X_train, y_train)

        # 4) Infer signature & input example for better model packaging
        example_input = X_train.iloc[:5]
        preds = estimator.predict(example_input)
        signature = infer_signature(example_input, preds)

        mlflow.sklearn.log_model(
            sk_model=estimator,
            name=name,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=example_input
        )




# ====================================================================================================================
# MODEL INTERPRETABILITY
# ====================================================================================================================

# ====================================================================================================================
# Generate SHAP explanations for model interpretability and feature importance analysis
# ====================================================================================================================
def run_shap_experiment(
    best_model,
    X_train_full,
    random_state = 42,
    feature_perturbation="interventional",
    plot_type="bar",
    shap_type="linear",
    model_type="classification",
    figsize=(10, 6)
):
    """
    Generate SHAP (SHapley Additive exPlanations) plots for model interpretability.
    
    This function creates SHAP explanations to understand feature importance and
    model predictions. It handles both linear and tree-based models, with special
    handling for different model types including CatBoost. SHAP values help explain
    individual predictions and overall model behavior.
    
    Parameters:
    -----------
    best_model : sklearn Pipeline
        Fitted model pipeline containing 'preprocessing' and 'classifier' named steps.
        The pipeline should be trained and ready for inference.
    X_train_full : pd.DataFrame
        Training data for generating SHAP explanations. A random subset of 200 samples
        will be used for computational efficiency.
    random_state : int, default=42
        Random seed for reproducible sampling of training data subset
    feature_perturbation : str, default="interventional"
        SHAP feature perturbation method. Options:
        - "interventional": Replace features with random draws from marginal distribution
        - "correlation_dependent": Account for feature correlations during perturbation
        - "tree_path_dependent": Required for CatBoost models with categorical features
    plot_type : str, default="bar"
        Type of SHAP plot to generate. Options:
        - "bar": Horizontal bar plot showing feature importance
        - "violin": Violin plot showing distribution of SHAP values
        - "dot": Dot plot showing SHAP values for individual samples
    shap_type : str, default="linear"
        Type of SHAP explainer to use. Options:
        - "linear": For linear models (LogisticRegression, LinearSVC, etc.)
        - "tree": For tree-based models (RandomForest, XGBoost, CatBoost, etc.)
    model_type : str, default="classification"
        Type of model to use. Options:
        - "classification": For classification models (LogisticRegression, RandomForest, XGBoost, etc.)
        - "regression": For regression models (LinearRegression, RandomForest, XGBoost, etc.)
    Returns:
    --------
    matplotlib.figure.Figure
        SHAP summary plot figure object that can be displayed or logged to MLflow
        
    Examples:
    ---------
    >>> # For a logistic regression model
    >>> fig = run_shap_experiment(
    ...     best_model=trained_pipeline,
    ...     X_train_full=X_train,
    ...     plot_type="violin",
    ...     shap_type="linear"
    ... )
    >>> 
    >>> # For a tree-based model
    >>> fig = run_shap_experiment(
    ...     best_model=xgb_pipeline,
    ...     X_train_full=X_train,
    ...     plot_type="bar",
    ...     shap_type="tree"
    ... )
    >>> 
    >>> # For CatBoost model (special handling)
    >>> fig = run_shap_experiment(
    ...     best_model=catboost_pipeline,
    ...     X_train_full=X_train,
    ...     feature_perturbation="tree_path_dependent",
    ...     shap_type="tree"
    ... )
    
    Notes:
    ------
    - Uses only 200 random samples from training data for computational efficiency
    - Automatically handles sparse matrices by converting to dense arrays
    - For multiclass problems, uses SHAP values for the positive class (index 1)
    - CatBoost models automatically override plot_type to "bar" for compatibility
    - The returned figure is closed to save memory but can still be logged to MLflow
    """
    # 1. Extract best model and its preprocessing step
    # best_model = grid_search_result.best_estimator_

    # Extract inner regressor if it exists
    if model_type != "classification":
        regressor = best_model.named_steps['regressor']
        if type(regressor).__name__ == 'RoundingRegressor':
            inner_pipeline = Pipeline([
                ('preprocessing', best_model.named_steps['preprocessing']),
                ('regressor', best_model.named_steps['regressor'].regressor)  # Extract inner regressor
            ])
        else:
            inner_pipeline = best_model
        best_model = inner_pipeline

    preprocessor = best_model.named_steps['preprocessing']
    model_step_name = 'classifier' if model_type == 'classification' else 'regressor'
    model   = best_model.named_steps[model_step_name]

    # 2. Prepare data for SHAP
    #    Use a subset of training data (or validation) for faster computation
    if len(X_train_full) > 200:
        X_shap_raw = X_train_full.sample(n=200, random_state=random_state)
    else:
        X_shap_raw = X_train_full
    # Transform to model inputs
    X_shap_proc = preprocessor.transform(X_shap_raw)
    if sparse.issparse(X_shap_proc):
        X_shap_proc = X_shap_proc.toarray()

    # Recover feature names
    feature_names = get_feature_names(preprocessor)
    X_shap_df = pd.DataFrame(X_shap_proc, columns=feature_names)

    model_name = type(model).__name__
    print(f"Model name: {model_name}")
    if shap_type == "linear":    
        # Initialize a SHAP explainer
        explainer = shap.LinearExplainer(
            model=model,
            masker=X_shap_df,
            feature_perturbation=feature_perturbation   # supported: "interventional" or "correlation_dependent"
        )
    else:
        # Check if this is CatBoost and adjust parameters accordingly
        # model_name = type(model).__name__
        
        if 'CatBoost' in model_name:            
            feature_perturbation="tree_path_dependent"  # Required for CatBoost with categorical features                
            plot_type = "bar"            
            explainer = shap.TreeExplainer(
                model=model,                
                feature_perturbation=feature_perturbation
            )   

            print(f"CatBoost SHAP plot type: {plot_type}")
        else:
            # XGBoost, RandomForest, etc.
            print(f"Tree SHAP plot type: {plot_type}")

            explainer = shap.TreeExplainer(
                model=model,
                data=X_shap_df,
                feature_perturbation=feature_perturbation
            )
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_shap_df)
    
    # Handle multiclass SHAP values for plotting
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # Multiclass: use the second class (index 1) or high conflict class
        shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        # Binary classification or single array
        shap_values_plot = shap_values

    # Summary plot
    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values_plot, X_shap_df, plot_type=plot_type, show=False, max_display=15
    )
    plt.title(f"SHAP Summary Plot - model: {model_name}")
    fig_sum = plt.gcf()
    plt.close(fig_sum)
    return fig_sum

# ====================================================================================================================
# MODEL EVALUATION
# ====================================================================================================================

# ====================================================================================================================
# Comprehensive test set evaluation with MLflow logging and visualization
# ====================================================================================================================
def log_test_set_performance(
    model,
    X_test,
    y_test,
    prefix: str = "test"
):
    """
    Evaluate and log model performance on the test set using MLflow.

    Computes classification metrics, generates a confusion matrix plot, and logs
    results to MLflow. Supports both binary and multiclass classification.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model with predict() and predict_proba() methods.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series or array-like
        True test labels.
    prefix : str, default="test"
        Prefix for metric names in MLflow logging.

    Returns:
    --------
    None
        Logs metrics and artifacts to MLflow.

    Examples:
    ---------
    >>> log_test_set_performance(trained_model, X_test, y_test, prefix="test")
    """
    # 1) Predictions & probabilities
    y_pred = model.predict(X_test)
    
    # Determine if this is binary or multiclass
    n_classes = len(np.unique(y_test))
    is_binary = n_classes == 2
    
    # 2) Compute metrics based on problem type
    if is_binary:
        # Binary classification metrics
        proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        metrics = {
            f"{prefix}_accuracy":   accuracy_score(y_test, y_pred),
            f"{prefix}_precision":  precision_score(y_test, y_pred),
            f"{prefix}_recall":     recall_score(y_test, y_pred),
            f"{prefix}_f1_score":   f1_score(y_test, y_pred),
            f"{prefix}_roc_auc":    roc_auc_score(y_test, proba),
        }
    else:
        # Multiclass classification metrics
        metrics = {
            f"{prefix}_accuracy":        accuracy_score(y_test, y_pred),
            f"{prefix}_precision_macro": precision_score(y_test, y_pred, average='macro'),
            f"{prefix}_precision_weighted": precision_score(y_test, y_pred, average='weighted'),
            f"{prefix}_recall_macro":    recall_score(y_test, y_pred, average='macro'),
            f"{prefix}_recall_weighted": recall_score(y_test, y_pred, average='weighted'),
            f"{prefix}_f1_macro":        f1_score(y_test, y_pred, average='macro'),
            f"{prefix}_f1_weighted":     f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Add multiclass ROC-AUC if model supports predict_proba
        try:
            proba = model.predict_proba(X_test)
            metrics[f"{prefix}_roc_auc_ovr"] = roc_auc_score(y_test, proba, multi_class='ovr')
            metrics[f"{prefix}_roc_auc_ovo"] = roc_auc_score(y_test, proba, multi_class='ovo')
        except:
            # Some models might not support predict_proba
            pass
    
    # 3) Log metrics
    for name, val in metrics.items():
        mlflow.log_metric(name, round(val, 4))

    # 4) Classification report
    report = classification_report(y_test, y_pred)
    report_path = f"{prefix}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # 5) Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create labels for confusion matrix
    if is_binary:
        labels = ['Low', 'High']
    else:
        # For 3-class: 0=Low, 1=Medium, 2=High
        labels = ['Low', 'Medium', 'High']
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{prefix.capitalize()} Confusion Matrix")
    plt.tight_layout()
    
    cm_path = f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifact(cm_path)
    
    # Clean up temporary files
    import os
    if os.path.exists(report_path):
        os.remove(report_path)
    if os.path.exists(cm_path):
        os.remove(cm_path)


# ====================================================================================================================
# HYPERPARAMETER OPTIMIZATION
# ====================================================================================================================

# ====================================================================================================================
# Comprehensive grid search experiment with MLflow tracking, SHAP analysis, and visualization
# ====================================================================================================================
def run_classification_gridsearch_experiment(
    name: str,
    pipeline,
    param_grid: dict,
    X_train, y_train, X_test, y_test,
    cv,
    scoring,
    dataset: dict,
    registered_model_name: str,
    verbose: bool = False,
    feature_set: str = "all",
    refit_metric: str = "f1_score"
):
    """
    Conduct a grid search for hyperparameter tuning with MLflow tracking.

    This function optimizes hyperparameters using grid search with cross-validation,
    logs results to MLflow, generates SHAP plots for interpretability, and evaluates
    the best model on the test set.

    Parameters:
    -----------
    name : str
        Experiment run name for MLflow.
    pipeline : sklearn Pipeline
        Model pipeline with preprocessing and classifier steps.
    param_grid : dict
        Hyperparameters grid for search.
    X_train, y_train : pd.DataFrame, array-like
        Training data and labels.
    X_test, y_test : pd.DataFrame, array-like
        Test data and labels.
    cv : sklearn cross-validation object
        Cross-validation strategy.
    scoring : dict
        Scoring metrics for evaluation.
    dataset : dict
        MLflow dataset objects for tracking.
    registered_model_name : str
        Name for model registration in MLflow.
    verbose : bool, default=False
        Print detailed results if True.
    feature_set : str, default="all"
        Description of the feature set used.
    refit_metric : str, default="f1_score"
        Metric to optimize during grid search.

    Returns:
    --------
    tuple
        Best estimator, GridSearchCV object, SHAP plot figure, ROC curve figure.

    Examples:
    ---------
    >>> best_model, grid_search, shap_fig, roc_fig = run_classification_gridsearch_experiment(
    ...     name="logistic_regression_tuned",
    ...     pipeline=pipeline,
    ...     param_grid=param_grid,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     cv=cv, scoring=scoring,
    ...     dataset=datasets,
    ...     registered_model_name="conflict_logreg_model",
    ...     verbose=True,
    ...     refit_metric="f1_score"
    ... )
    """

    # Detect if we're using DagsHub
    def is_dagshub_tracking():
        """Check if current MLflow tracking URI points to DagsHub"""
        tracking_uri = mlflow.get_tracking_uri()
        return 'dagshub.com' in tracking_uri.lower()
    
    grid_search = GridSearchCV(
        estimator=pipeline,        # your Pipeline([... ('classifier', LogisticRegression()) ])
        param_grid=param_grid,
        cv=cv,                         # 5-fold stratified CV
        scoring=scoring,            # primary metric
        n_jobs=-1,                    # parallelize across all cores
        return_train_score=True,
        refit=refit_metric,            # ← what to optimize/return as best_estimator_
        error_score=np.nan,  # treat fold‐errors as NaN rather than crashing
        verbose=1
    )

    print(f"Running grid search for {name}")
    
    with mlflow.start_run(run_name=name):
        # 1) Log dataset inputs
        mlflow.log_input(dataset["train_ds"], context="training")
        mlflow.log_input(dataset["test_ds"],  context="test")

        # 2) Log the grid search hyperparameter space
        mlflow.log_param("grid_search_params", param_grid)
        
        # Log the feature set
        mlflow.log_param("feature_set", feature_set)

        grid_search.fit(X_train, y_train)

        # 4) Log the best hyperparameters
        best_params = grid_search.best_params_
        mlflow.log_param("hyperparameters", best_params)

        # 5) Log CV metrics (mean ± std) for each scoring key
        results = grid_search.cv_results_
        # if scoring is a dict, iterate its keys; else assume single metric
        for metric in scoring:
            mean_score = results[f"mean_test_{metric}"][grid_search.best_index_]
            std_score  = results[f"std_test_{metric}"][grid_search.best_index_]
            mlflow.log_metric(metric, mean_score.round(2))
            mlflow.log_metric(f"{metric}_std", std_score)
            if verbose:
                print(f"{metric}: {mean_score.round(2)} ± {std_score.round(2)}")

        # 6) Log and register the best estimator
        best_estimator = grid_search.best_estimator_
        example_input = X_train.iloc[:5]
        example_preds = best_estimator.predict(example_input)
        signature = infer_signature(example_input, example_preds)
        
        # ===== SHAP plot for binary classification
        n_classes = len(np.unique(y_train))
        is_binary = n_classes == 2
        if is_binary:
            plot_type = "violin"
        else:
            plot_type = "bar"
        # Adding shap to the model
        # print(f"Running SHAP for {name}")
        fig_sum = None
        if 'baseline' not in name:
            if 'logreg' in name:
                shap_type = "linear"
            else:
                shap_type = "tree"
            feature_perturbation = "interventional"
            if 'catboost' in name.lower():
                feature_perturbation = "tree_path_dependent"
            # print(f"Running SHAP for {name} with {plot_type} and {shap_type}")
            fig_sum = run_shap_experiment(
                best_model=best_estimator,
                X_train_full=X_train,
                random_state=42,
                feature_perturbation=feature_perturbation,
                plot_type=plot_type, # other options: "bar", "dot", "violin"
                shap_type=shap_type
            )
            mlflow.log_figure(fig_sum, "shap_summary.png")

        # ===== ROC plot for binary classification
        fig_roc = None
        if is_binary:
            fig_roc = create_and_log_roc_plot_binary(
                model=best_estimator,
                X_test=X_test,
                y_test=y_test,
                prefix="test",
                show_plot=False
            )
            mlflow.log_figure(fig_roc, "roc_curve.png")


        # Log test set performance
        log_test_set_performance(
            model=best_estimator,
            X_test=X_test,
            y_test=y_test,
            prefix="test"
        )

        # 7) Log the best estimator
        if is_dagshub_tracking():
            # DagsHub-compatible logging
            if verbose:
                print("DagsHub tracking detected - using compatible logging parameters")

            mlflow.sklearn.log_model(
                best_estimator,
                artifact_path="model",
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=example_input
            )

        else:
            # Standard MLflow logging with full feature set
            if verbose:
                print("Local/standard MLflow tracking - using full parameter set")

            mlflow.sklearn.log_model(
                sk_model=best_estimator,
                artifact_path="model",
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=example_input
            )


    if verbose:
        # 3. Print summary in the output cell
        print("Best parameters:", grid_search.best_params_)
        print("Best F1 score (CV):", grid_search.best_score_.round(2))

        # 4. (Optional) inspect all CV results
        # cv_df = pd.DataFrame(grid_search.cv_results_)
        # display(cv_df.sort_values("mean_test_f1_score", ascending=False).head(3))
    
    # plot SHAP summary
    # if 'baseline' not in name and fig_sum is not None:
    #     plt.figure(figsize=(10, 6))
    #     plt.title("SHAP Summary Plot")
    #     plt.show()

    return best_estimator, grid_search, fig_sum, fig_roc


# ====================================================================================================================
# MULTI-CLASS CLASSIFICATION
# ====================================================================================================================

# ====================================================================================================================
# Create 3-class conflict classification (Low, Medium, High) from continuous scores
# ====================================================================================================================
def create_multiclass_conflict(df, target_column='Conflicts', low_threshold=1, high_threshold=3, visualize=True):
    """
    Create 3-class conflict classification (Low, Medium, High) from conflict scores.
    
    This function transforms continuous conflict scores into a 3-class classification
    problem using two thresholds. This approach provides more granular classification
    compared to binary classification and can capture intermediate conflict levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing conflict scores
    target_column : str, default='Conflicts'
        Name of the column containing conflict scores
    low_threshold : int, default=1
        Upper bound for Low class (inclusive). Low: 0 to low_threshold
    high_threshold : int, default=3  
        Lower bound for High class (exclusive). High: > high_threshold
        Medium: low_threshold+1 to high_threshold
    visualize : bool, default=True
        Whether to create visualization plots
        
    Returns:
    --------
    tuple[pd.DataFrame, dict]
        - pd.DataFrame: Copy of input dataframe with added 'Conflict_3Class' and 'Conflict_3Class_Numeric' columns
        - dict: Analysis results containing thresholds, class counts, proportions, and class mapping
        
    Examples:
    ---------
    >>> df_multiclass, results = create_multiclass_conflict(df, low_threshold=1, high_threshold=3)
    >>> print(f"Class distribution: {results['counts']}")
    >>> print(f"Class mapping: {results['class_mapping']}")
    """
    # Make a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Create 3-class target variable
    def classify_conflict(score):
        if score <= low_threshold:
            return 'Low'
        elif score <= high_threshold:
            return 'Medium' 
        else:
            return 'High'
    
    df_copy['Conflict_3Class'] = df_copy[target_column].apply(classify_conflict)
    
    # Calculate statistics
    conflict_counts = df_copy['Conflict_3Class'].value_counts()
    total_samples = len(df_copy)
    
    # Print analysis
    print(f"3-Class Conflict Classification:")
    print(f"Low Conflict (0-{low_threshold}): {conflict_counts.get('Low', 0)} samples")
    print(f"Medium Conflict ({low_threshold+1}-{high_threshold}): {conflict_counts.get('Medium', 0)} samples")
    print(f"High Conflict ({high_threshold+1}-max): {conflict_counts.get('High', 0)} samples")
    print(f"Class proportions:")
    proportions = conflict_counts / total_samples
    for class_name, prop in proportions.items():
        print(f"  {class_name}: {prop:.3f} ({prop*100:.1f}%)")
    
    # Convert to numeric labels (0, 1, 2)
    class_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df_copy['Conflict_3Class_Numeric'] = df_copy['Conflict_3Class'].map(class_mapping)
    
    # Create visualizations if requested
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original distribution with threshold lines
        axes[0].hist(df_copy[target_column], bins=range(int(df_copy[target_column].min()), 
                                                       int(df_copy[target_column].max()) + 2), 
                    alpha=0.7, edgecolor='black')
        axes[0].axvline(low_threshold + 0.5, color='red', linestyle='--', linewidth=2,
                       label=f'Low/Medium = {low_threshold + 0.5}')
        axes[0].axvline(high_threshold + 0.5, color='orange', linestyle='--', linewidth=2,
                       label=f'Medium/High = {high_threshold + 0.5}')
        axes[0].set_xlabel('Conflict Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Original Conflict Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 3-class distribution (categorical)
        sns.countplot(data=df_copy, x='Conflict_3Class', ax=axes[1], 
                     order=['Low', 'Medium', 'High'])
        axes[1].set_title('3-Class Conflict Distribution\n(Categorical Labels)')
        axes[1].set_ylabel('Count')
        
        # Add count labels on bars
        for i, bar in enumerate(axes[1].patches):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 3-class distribution (numeric)
        sns.countplot(data=df_copy, x='Conflict_3Class_Numeric', ax=axes[2])
        axes[2].set_title('3-Class Conflict Distribution\n(0=Low, 1=Medium, 2=High)')
        axes[2].set_ylabel('Count')
        axes[2].set_xlabel('Class (Numeric)')
        
        # Add count labels on bars
        for i, bar in enumerate(axes[2].patches):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    # Prepare results dictionary
    results = {
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'counts': conflict_counts.to_dict(),
        'proportions': proportions.to_dict(),
        'class_mapping': class_mapping
    }
    
    return df_copy, results

# ====================================================================================================================
# Convert existing target variables to multi-class format without recreating train/test splits
# ====================================================================================================================
def convert_to_multiclass_target(y_series, original_data, target_column='Conflicts', 
                                low_threshold=1, high_threshold=3):
    """
    Convert existing target variables to 3-class format without recreating train/test splits.
    
    This function is useful when you already have train/test splits and want to convert
    from binary classification to multi-class classification while preserving the
    data partitioning.
    
    Parameters:
    -----------
    y_series : pd.Series or array-like
        The target variable series (can be binary or original conflicts)
    original_data : pd.DataFrame  
        Original dataframe containing the original conflict scores
    target_column : str, default='Conflicts'
        Name of the original conflict column in the dataframe
    low_threshold : int, default=1
        Upper bound for Low class (inclusive)
    high_threshold : int, default=3
        Lower bound for High class (exclusive)
        
    Returns:
    --------
    pd.Series
        3-class target variable with numeric labels (0=Low, 1=Medium, 2=High)
        
    Examples:
    ---------
    >>> y_train_multiclass = convert_to_multiclass_target(
    ...     y_train_binary, original_df, 'Conflicts', low_threshold=1, high_threshold=3
    ... )
    >>> print(y_train_multiclass.value_counts())
    """
    # Get the original conflict scores for the same indices as y_series
    original_conflicts = original_data.loc[y_series.index, target_column]
    
    # Create 3-class labels
    def classify_conflict(score):
        if score <= low_threshold:
            return 0  # Low
        elif score <= high_threshold:
            return 1  # Medium
        else:
            return 2  # High
    
    multiclass_target = original_conflicts.apply(classify_conflict)
    
    # Print distribution
    counts = multiclass_target.value_counts().sort_index()
    print(f"3-Class Distribution:")
    print(f"Low (0): {counts.get(0, 0)} samples")
    print(f"Medium (1): {counts.get(1, 0)} samples") 
    print(f"High (2): {counts.get(2, 0)} samples")
    
    return multiclass_target


# ====================================================================================================================
# VISUALIZATION FUNCTIONS
# ====================================================================================================================

# ====================================================================================================================
# Create and log ROC curve for binary classification models
# ====================================================================================================================
def create_and_log_roc_plot_binary(model, X_test, y_test, prefix="test", show_plot=True, mlflow_log=True):
    """
    Create ROC (Receiver Operating Characteristic) curve for binary classification and log to MLflow.
    
    This function generates an ROC curve plot showing the trade-off between true positive
    rate and false positive rate at various threshold settings. The area under the curve
    (AUC) provides a single metric summarizing model performance across all thresholds.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained binary classification model with predict_proba method
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels (binary: 0 or 1)
    prefix : str, default="test"
        Prefix for the saved plot filename
    show_plot : bool, default=True
        Whether to display the plot in the notebook
        
    Returns:
    --------
    matplotlib.figure.Figure
        ROC curve figure object
        
    Examples:
    ---------
    >>> roc_fig = create_and_log_roc_plot_binary(
    ...     model=trained_model,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     prefix="validation"
    ... )
    """
    from sklearn.metrics import roc_curve, auc
    
    # Get predicted probabilities for positive class
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    fig_roc = plt.gcf()
    
    # Log to MLflow
    roc_path = f"{prefix}_roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    if mlflow_log:
        mlflow.log_artifact(roc_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig_roc)
    
    # Clean up
    if os.path.exists(roc_path):
        os.remove(roc_path)
    
    return fig_roc