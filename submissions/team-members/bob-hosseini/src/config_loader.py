"""
Configuration Loader for SocialSphere Analytics

This module provides utilities to load and manage configuration settings
from YAML files for the SocialSphere Analytics application.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Handles loading and accessing configuration from YAML files"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration loader
        
        Args:
            config_path (str): Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default path relative to src directory
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "configs" / "configs.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
                
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path (str): Dot-separated path to the configuration value (e.g., 'models.conflicts.pyfunc_uri')
            default (Any): Default value if key is not found
            
        Returns:
            Any: Configuration value or default
        """
        if self._config is None:
            return default
        
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a specific model
        
        Args:
            model_name (str): Name of the model ('conflicts' or 'addiction')
            
        Returns:
            Dict[str, Any]: Model configuration dictionary
        """
        return self.get(f'models.{model_name}', {})
    
    def get_all_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all models"""
        return self.get('models', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration"""
        return self.get('mlflow', {})
    
    def get_shap_config(self) -> Dict[str, Any]:
        """Get SHAP configuration"""
        return self.get('shap', {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return self.get('ui', {})
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get general app configuration"""
        return self.get('app', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.get('data', {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary"""
        return self._config or {}


# Global configuration instance
_config_loader = None

def get_config() -> ConfigLoader:
    """Get the global configuration loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def reload_config() -> ConfigLoader:
    """Reload the configuration from file"""
    global _config_loader
    _config_loader = ConfigLoader()
    return _config_loader 