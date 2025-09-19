"""
Simple configuration loader for AstroNav.
Loads config.yaml and provides default values.
"""
import yaml
from pathlib import Path
from typing import Dict, Any

# Default configuration values
DEFAULT_CONFIG = {
    "general": {
        "use_camera": False,
        "visualize": False,
        "image_path": "src/photos/5star_pairs_center.jpeg"
    },
    "star_processing": {
        "angle_tolerance": 0.1,
        "identification_limit": 40,
        "visualize_hip": True,
        "fov_degrees": 66.3
    },
    "star_detection": {
        "threshold_val": 180,
        "min_area": 15,
        "max_area": 500
    },
    "imu": {
        "enabled": True,
        "calibration_timeout": 30,
        "update_interval": 0.25
    },
    "tracking": {
        "image_tracking": True,
        "imu_tracking": True
    },
    "network": {
        "udp_ip": "127.0.0.1",
        "udp_port": 12345
    },
    "test_mode": {
        "enabled": False,
        "rotation_speed": 1.0,
        "update_interval": 0.5
    }
}

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with fallback to defaults.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in project root.
        
    Returns:
        Dictionary containing configuration values.
    """
    if config_path is None:
        # Look for config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
    
    config_path = Path(config_path)
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if it exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as file:
                file_config = yaml.safe_load(file)
                if file_config:
                    # Merge file config with defaults (file config takes precedence)
                    config = _merge_configs(config, file_config)
                    print(f"Loaded configuration from: {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
    else:
        print(f"No config file found at {config_path}, using defaults.")
    
    return config

def _merge_configs(default: Dict, override: Dict) -> Dict:
    """Recursively merge configuration dictionaries."""
    result = default.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result

def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """
    Get a configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "general.use_camera")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

# Global config instance
_config = None

def get_config() -> Dict[str, Any]:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    config = get_config()
    print(config)