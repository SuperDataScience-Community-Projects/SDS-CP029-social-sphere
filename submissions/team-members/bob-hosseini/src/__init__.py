"""
Social Media Conflict Classification Utilities

This package provides modular utilities for social media conflict classification including:
- Data preprocessing and feature engineering
- Classification modeling and experiments  
- MLflow experiment tracking
- Visualization functions

Import all functions for backward compatibility with existing notebooks.
"""

# Import all functions from each module for backward compatibility
from .preprocessing import (
    # Target variable creation
    create_binary_conflict,
    create_multiclass_conflict, 
    convert_to_multiclass_target,
    
    # Feature engineering
    encode_onehot_with_reference,
    encode_frequency,
    
    # Geographical mapping
    cont_map,
    map_to_continent,
    country_to_continent,
    
    # Sklearn transformers
    RareCategoryGrouper,
    CountryToContinentMapper,
    
    # Utility functions
    get_feature_names
)

from .classification import (
    # Basic experiments
    run_classification_experiment,
    
    # Hyperparameter optimization
    run_classification_gridsearch_experiment,
    
    # Model evaluation
    log_test_set_performance,
    
    # Model interpretability
    run_shap_experiment
)

from .mlflow_utils import (
    # Dataset tracking
    mlflow_dataset,
    
    # Experiment management
    setup_mlflow_experiment,
    log_model_metadata,
    log_dataset_info
)

from .visualizations import (
    # Classification visualizations
    create_and_log_roc_plot_binary,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
    plot_classification_metrics_comparison,
    
    # Data exploration
    plot_target_distribution,
    plot_correlation_heatmap
)

# Version info
__version__ = "1.0.0"
__author__ = "Bob Hosseini"

# Define what gets imported with "from src import *"
__all__ = [
    # Preprocessing
    'create_binary_conflict',
    'create_multiclass_conflict', 
    'convert_to_multiclass_target',
    'encode_onehot_with_reference',
    'encode_frequency',
    'cont_map',
    'map_to_continent',
    'country_to_continent',
    'RareCategoryGrouper',
    'CountryToContinentMapper',
    'get_feature_names',
    
    # Classification
    'run_classification_experiment',
    'run_classification_gridsearch_experiment',
    'log_test_set_performance',
    'run_shap_experiment',
    
    # MLflow utilities
    'mlflow_dataset',
    'setup_mlflow_experiment',
    'log_model_metadata',
    'log_dataset_info',
    
    # Visualizations
    'create_and_log_roc_plot_binary',
    'plot_confusion_matrix_heatmap',
    'plot_feature_importance',
    'plot_classification_metrics_comparison',
    'plot_target_distribution',
    'plot_correlation_heatmap'
] 