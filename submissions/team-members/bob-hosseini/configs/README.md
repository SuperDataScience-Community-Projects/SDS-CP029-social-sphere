# Configuration Management for SocialSphere Analytics

This directory contains configuration files for the SocialSphere Analytics application.

## Configuration File: `configs.yaml`

The main configuration file contains all configurable parameters for the application, organized into sections:

### Model Configuration
```yaml
models:
  conflicts:
    pyfunc_uri: "runs:/a7f3a1fd156443e58e7554ac1e8b53fa/model"
    sklearn_uri: "runs:/a7f3a1fd156443e58e7554ac1e8b53fa/model"
    type: "classification"
    name: "CatBoost Binary Classifier"
    description: "Predicts social media conflicts (binary classification)"
  
  addiction:
    pyfunc_uri: "runs:/594b916daee046ff8f9fa0ed3aed8748/model"
    sklearn_uri: "runs:/594b916daee046ff8f9fa0ed3aed8748/model"
    type: "regression"
    name: "CatBoost Regressor"
    description: "Predicts addiction score (regression with rounding)"
```

### How to Update Model URIs

1. **After training new models**, update the URIs in `configs.yaml`:
   ```yaml
   models:
     conflicts:
       pyfunc_uri: "runs:/NEW_RUN_ID/model"
       sklearn_uri: "runs:/NEW_RUN_ID/model"
   ```

2. **Using MLflow Model Registry** (recommended):
   ```yaml
   models:
     conflicts:
       pyfunc_uri: "models:/conflict_model/Production"
       sklearn_uri: "models:/conflict_model/Production"
   ```

3. **Using model versions**:
   ```yaml
   models:
     conflicts:
       pyfunc_uri: "models:/conflict_model/1"
       sklearn_uri: "models:/conflict_model/1"
   ```

### Other Configuration Sections

- **Data Configuration**: Dataset paths and URLs
- **MLflow Configuration**: Tracking URI and experiment URLs
- **SHAP Configuration**: Default parameters for SHAP explanations
- **UI Configuration**: Page settings and theme
- **App Settings**: General application parameters

### Configuration Loading

The configuration is automatically loaded when the app starts. To reload configuration after changes:

```python
from config_loader import reload_config
config = reload_config()
```

### Best Practices

1. **Version Control**: Keep the configuration file in version control
2. **Environment-Specific Configs**: Create separate config files for different environments if needed
3. **Model Registry**: Use MLflow Model Registry for production deployments
4. **Validation**: Test configuration changes in a development environment first

### Example: Updating to New Models

```bash
# 1. Train new models using MLflow
# 2. Note the new run IDs or register models in MLflow Model Registry
# 3. Update configs.yaml with new URIs:

models:
  conflicts:
    pyfunc_uri: "runs:/NEW_CONFLICTS_RUN_ID/model"
    sklearn_uri: "runs:/NEW_CONFLICTS_RUN_ID/model"
  addiction:
    pyfunc_uri: "runs:/NEW_ADDICTION_RUN_ID/model"
    sklearn_uri: "runs:/NEW_ADDICTION_RUN_ID/model"

# 4. Restart the Streamlit app
```

This configuration system makes it easy to:
- Update model URIs without changing code
- Switch between different model versions
- Maintain different configurations for different environments
- Keep all settings organized in one place 