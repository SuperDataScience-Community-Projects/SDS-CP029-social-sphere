# Configuration Management

This directory manages all configurable parameters for the SocialSphere Analytics application, enabling seamless model updates and environment management.

## 📋 Configuration File: `configs.yaml`

The main configuration file contains structured parameters organized by component:

### Model Configuration
```yaml
models:
  conflicts:
    pyfunc_uri: "runs:/a7f3a1fd156443e58e7554ac1e8b53fa/model"
    type: "classification"
    name: "CatBoost Binary Classifier"
  
  addiction:
    pyfunc_uri: "runs:/594b916daee046ff8f9fa0ed3aed8748/model"
    type: "regression"
    name: "CatBoost Regressor"
```

## 🔄 Model URI Management

### Production Deployment (Recommended)
```yaml
models:
  conflicts:
    pyfunc_uri: "models:/conflict_model/Production"
  addiction:
    pyfunc_uri: "models:/addiction_model/Production"
```

### Version-Specific Deployment
```yaml
models:
  conflicts:
    pyfunc_uri: "models:/conflict_model/3"
  addiction:
    pyfunc_uri: "models:/addiction_model/2"
```

### Development/Testing
```yaml
models:
  conflicts:
    pyfunc_uri: "runs:/NEW_RUN_ID/model"
  addiction:
    pyfunc_uri: "runs:/EXPERIMENT_RUN_ID/model"
```

## 🛠️ Configuration Sections

| Section | Purpose | Key Parameters |
|---------|---------|----------------|
| `models` | ML model references and metadata | URIs, types, descriptions |
| `data` | Dataset paths and URLs | File locations, data sources |
| `mlflow` | Experiment tracking configuration | Tracking URI, experiment IDs |
| `app` | Application settings | UI parameters, themes |
| `shap` | Model explainability settings | SHAP configuration parameters |

## 🚀 Deployment Workflow

### 1. Model Training & Registration
```bash
# Train models in notebooks/
# Models automatically logged to MLflow
# Register best models in MLflow Model Registry
```

### 2. Configuration Update
```yaml
# Update configs.yaml with new model URIs
models:
  conflicts:
    pyfunc_uri: "models:/conflict_model/Production"
```

### 3. Application Restart
```bash
# Restart Streamlit app to load new configuration
streamlit run app/app.py
```

## 🔧 Configuration Loading

The configuration system provides automatic loading and validation:

```python
from config_loader import load_config, reload_config

# Load configuration on startup
config = load_config()

# Reload after configuration changes
config = reload_config()
```

## 📋 Best Practices

### Development
- Use `runs:/` URIs for experiment testing
- Validate configuration changes before deployment
- Keep separate config files for different environments

### Production
- Use MLflow Model Registry with stage-based deployment
- Always test configuration updates in staging environment
- Version control all configuration changes
- Monitor model performance after updates

### Security
- Store sensitive configurations (credentials) in environment variables
- Use relative paths for local file references
- Validate all configuration parameters on load

## 🔍 Troubleshooting

### Common Issues
- **Model Loading Errors**: Verify MLflow tracking URI and model registry access
- **URI Format**: Ensure proper format (`runs:/`, `models:/`, `file:/`)
- **Permissions**: Check MLflow server access and model registry permissions
- **Version Conflicts**: Verify model compatibility with application requirements

### Configuration Validation
The configuration loader automatically validates:
- Required fields presence
- URI format correctness
- Model type consistency
- Parameter data types

This configuration management system enables zero-downtime model updates and maintains consistency across different deployment environments. 