"""
Configuration loader with validation and environment support.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import json
from loguru import logger


class ConfigLoader:
    """Advanced configuration loader with validation and environment override."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config directory relative to project root
            self.config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        else:
            self.config_path = Path(config_path)
        
        self.env_prefix = "RTSD_"  # Real-Time Scene Description prefix
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment overrides."""
        try:
            # Load base configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply environment variable overrides
            config = self._apply_env_overrides(config)
            
            # Validate configuration
            self._validate_config(config)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Configuration loading error: {e}")
            raise
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        
        # Define environment variable mappings
        env_mappings = {
            f"{self.env_prefix}DEBUG": "app.debug",
            f"{self.env_prefix}LOG_LEVEL": "app.log_level",
            f"{self.env_prefix}DEVICE": "models.blip.device",
            f"{self.env_prefix}CAMERA_ID": "video.input.camera_id",
            f"{self.env_prefix}API_HOST": "api.host",
            f"{self.env_prefix}API_PORT": "api.port",
            f"{self.env_prefix}ENABLE_GPU": "performance.gpu.enable_mixed_precision",
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config, config_path, self._convert_env_value(env_value))
                logger.info(f"Override applied: {config_path} = {env_value}")
        
        return config
    
    def _set_nested_value(self, config: Dict, path: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # JSON conversion (for complex types)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Return as string
        return value
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure and values."""
        required_sections = ['app', 'models', 'video', 'performance']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model configurations
        if 'blip' not in config['models']:
            raise ValueError("BLIP model configuration is required")
        
        # Validate video input
        if 'input' not in config['video']:
            raise ValueError("Video input configuration is required")
        
        # Validate camera ID
        camera_id = config['video']['input'].get('camera_id')
        if camera_id is not None and not isinstance(camera_id, int):
            raise ValueError("Camera ID must be an integer")
        
        # Validate resolution
        resolution = config['video']['input'].get('resolution')
        if resolution and (not isinstance(resolution, list) or len(resolution) != 2):
            raise ValueError("Resolution must be a list of two integers [width, height]")
        
        logger.info("Configuration validation passed")
    
    def save_config(self, config: Dict[str, Any], output_path: str = None):
        """Save configuration to YAML file."""
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        config = self.load_config()
        return config.get('models', {}).get(model_name, {})
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration template."""
        return {
            "app": {
                "name": "Real-Time AI Scene Description",
                "version": "2.0.0",
                "debug": False,
                "log_level": "INFO"
            },
            "models": {
                "blip": {
                    "model_name": "Salesforce/blip-image-captioning-base",
                    "device": "auto",
                    "precision": "fp16",
                    "max_new_tokens": 50,
                    "temperature": 0.7
                }
            },
            "video": {
                "input": {
                    "camera_id": 0,
                    "resolution": [640, 480],
                    "fps": 30
                },
                "processing": {
                    "frame_skip": 30,
                    "resize_for_inference": [384, 384]
                }
            },
            "performance": {
                "gpu": {
                    "enable_mixed_precision": True
                },
                "cache": {
                    "enable_result_cache": True
                }
            }
        }
