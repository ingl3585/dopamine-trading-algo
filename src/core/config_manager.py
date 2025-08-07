"""
Unified Configuration Management System

This module consolidates configuration management functionality from the previous
Config and ConfigurationManager classes, providing both simple and advanced interfaces
while following clean architecture principles.

Responsibilities:
- Load and validate configurations from multiple sources (files, environment, defaults)
- Provide type-safe configuration access with validation
- Support both simple property access and advanced schema-based validation
- Handle environment-specific configurations
- Support runtime configuration updates
- Manage configuration schemas and validation rules
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Type
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationSchema:
    """Schema definition for configuration validation"""
    key: str
    data_type: Type
    default_value: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    required: bool = False

class IConfigurationProvider:
    """Interface for configuration providers following interface segregation principle"""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        raise NotImplementedError
    
    def get_typed(self, key: str, expected_type: Type, default: Any = None) -> Any:
        """Get configuration value with type checking"""
        raise NotImplementedError

class IConfigurationValidator:
    """Interface for configuration validation"""
    
    def validate(self, key: str, value: Any) -> bool:
        """Validate a configuration value"""
        raise NotImplementedError
    
    def validate_all(self) -> bool:
        """Validate all configuration values"""
        raise NotImplementedError

class ConfigurationManager(IConfigurationProvider, IConfigurationValidator):
    """
    Unified configuration management system following clean architecture principles.
    
    Features:
    - Multiple configuration sources (defaults, files, environment variables)
    - Schema-based validation with type checking
    - Environment-aware configuration loading
    - Runtime configuration updates
    - Simple and advanced access patterns
    - Comprehensive error handling and logging
    """
    
    def __init__(self, config_file: Optional[str] = None, enable_validation: bool = True):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional specific configuration file to load
            enable_validation: Whether to enable schema validation
        """
        self._settings: Dict[str, Any] = {}
        self._schemas: Dict[str, ConfigurationSchema] = {}
        self._config_sources: List[str] = []
        self._validation_errors: List[str] = []
        self._enable_validation = enable_validation
        self._load_timestamp: Optional[datetime] = None
        
        # Configuration state
        self.config_file = config_file
        self.environment = os.getenv('TRADING_ENV', 'development')
        
        # Initialize the configuration system
        self._initialize_schemas()
        self._load_all_configurations()
        
        logger.info(f"Configuration manager initialized for {self.environment} environment")
        logger.info(f"Loaded from sources: {', '.join(self._config_sources)}")
    
    def _initialize_schemas(self) -> None:
        """Initialize configuration schemas for validation"""
        schemas = [
            # TCP Configuration
            ConfigurationSchema('tcp_data_port', int, 5556, 1024, 65535, 
                               description="TCP port for market data", required=True),
            ConfigurationSchema('tcp_signal_port', int, 5557, 1024, 65535,
                               description="TCP port for trade signals", required=True),
            ConfigurationSchema('tcp_host', str, 'localhost',
                               description="TCP host address", required=True),
            ConfigurationSchema('tcp_timeout', int, 30, 1, 300,
                               description="TCP connection timeout in seconds"),
            
            # Trading Configuration
            ConfigurationSchema('trading_interval_seconds', int, 60, 1, 3600,
                               description="Trading decision interval"),
            ConfigurationSchema('min_trade_interval_seconds', int, 300, 60, 3600,
                               description="Minimum time between trades"),
            ConfigurationSchema('max_daily_trades', int, 10, 1, 100,
                               description="Maximum trades per day"),
            ConfigurationSchema('max_hold_time_hours', int, 24, 1, 168,
                               description="Maximum position hold time"),
            
            # Risk Management
            ConfigurationSchema('max_position_size', float, 0.1, 0.01, 1.0,
                               description="Maximum position size as fraction of account"),
            ConfigurationSchema('leverage', float, 50.0, 1.0, 100.0,
                               description="Trading leverage"),
            ConfigurationSchema('kelly_lookback', int, 100, 10, 1000,
                               description="Number of trades for Kelly calculation"),
            
            # Market Data Configuration
            ConfigurationSchema('mnq_point_value', float, 2.0, 0.1, 10.0,
                               description="MNQ futures point value"),
            ConfigurationSchema('mnq_tick_size', float, 0.25, 0.01, 1.0,
                               description="MNQ minimum tick size"),
            ConfigurationSchema('contract_value', float, 2000.0, 100.0, 10000.0,
                               description="Contract value for position sizing"),
            
            # System Configuration
            ConfigurationSchema('model_save_interval', int, 300, 60, 3600,
                               description="Model save interval in seconds"),
            ConfigurationSchema('log_level', str, 'INFO', 
                               allowed_values=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               description="Logging level"),
            ConfigurationSchema('data_directory', str, 'data',
                               description="Data storage directory"),
            ConfigurationSchema('models_directory', str, 'models',
                               description="Model storage directory"),
            ConfigurationSchema('logs_directory', str, 'logs',
                               description="Log storage directory"),
            
            # Bootstrap Configuration
            ConfigurationSchema('min_historical_bars', int, 100, 10, 1000,
                               description="Minimum historical bars for bootstrap"),
            ConfigurationSchema('bootstrap_timeout', int, 300, 30, 600,
                               description="Bootstrap timeout in seconds"),
            ConfigurationSchema('historical_data_timeout', int, 30, 5, 120,
                               description="Historical data request timeout"),
            
            # Emergency Limits
            ConfigurationSchema('emergency_max_margin_usage', float, 0.95, 0.5, 1.0,
                               description="Emergency maximum margin usage"),
            ConfigurationSchema('emergency_max_drawdown', float, 0.20, 0.05, 0.5,
                               description="Emergency maximum drawdown"),
            ConfigurationSchema('close_positions_on_shutdown', bool, False,
                               description="Close positions on system shutdown"),
            
            # Neural Network Configuration
            ConfigurationSchema('batch_size', int, 32, 8, 128,
                               description="Neural network batch size"),
            ConfigurationSchema('learning_rate', float, 0.001, 0.0001, 0.01,
                               description="Neural network learning rate"),
            ConfigurationSchema('memory_buffer_size', int, 20000, 1000, 100000,
                               description="Experience replay buffer size"),
            
            # Risk Configuration
            ConfigurationSchema('var_confidence', float, 0.95, 0.90, 0.99,
                               description="Value at Risk confidence level"),
            ConfigurationSchema('risk_lookback_days', int, 30, 7, 90,
                               description="Risk calculation lookback period"),
            ConfigurationSchema('max_position_risk', float, 0.1, 0.01, 0.5,
                               description="Maximum risk per position"),
            ConfigurationSchema('max_portfolio_risk', float, 0.2, 0.05, 0.8,
                               description="Maximum portfolio risk"),
        ]
        
        # Store schemas for validation
        for schema in schemas:
            self._schemas[schema.key] = schema
    
    def _load_all_configurations(self) -> None:
        """Load configurations from all sources with proper error handling"""
        try:
            # Load base configuration from schemas
            self._settings = self._load_base_config()
            self._config_sources.append('schema_defaults')
            
            # Load environment-specific configuration
            self._load_environment_config()
            
            # Load from specific file if provided
            if self.config_file:
                self._load_config_file(self.config_file)
            
            # Apply environment variable overrides
            self._load_environment_variables()
            
            # Validate all configurations if enabled
            if self._enable_validation:
                self.validate_all()
            
            # Create necessary directories
            self._create_directories()
            
            self._load_timestamp = datetime.now()
            
        except Exception as e:
            logger.error(f"Critical error loading configurations: {e}")
            raise RuntimeError(f"Configuration system initialization failed: {e}") from e
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration with defaults from schemas"""
        base_config = {}
        
        # Set defaults from schemas
        for key, schema in self._schemas.items():
            base_config[key] = schema.default_value
        
        # Add system environment
        base_config['environment'] = self.environment
        
        return base_config
    
    def _load_environment_config(self) -> None:
        """Load environment-specific configuration file"""
        try:
            config_file = f"config/{self.environment}.json"
            
            if os.path.exists(config_file):
                self._load_config_file(config_file)
                self._config_sources.append(f'env_{self.environment}')
                logger.info(f"Loaded {self.environment} environment configuration")
            else:
                logger.info(f"No {self.environment} environment config found, using defaults")
                
        except Exception as e:
            logger.warning(f"Error loading environment config: {e}")
    
    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                
            # Deep merge configuration
            self._deep_merge_config(file_config)
            
            if f'file_{config_file}' not in self._config_sources:
                self._config_sources.append(f'file_{config_file}')
                
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.warning(f"Error loading config file {config_file}: {e}")
    
    def _deep_merge_config(self, new_config: Dict[str, Any]) -> None:
        """Deep merge configuration dictionaries"""
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self._settings and isinstance(self._settings[key], dict):
                # Recursively merge nested dictionaries
                self._deep_merge_dict(self._settings[key], value)
            else:
                self._settings[key] = value
    
    def _deep_merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively merge dictionaries"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables with type conversion"""
        env_mappings = {
            'TRADING_TCP_DATA_PORT': 'tcp_data_port',
            'TRADING_TCP_SIGNAL_PORT': 'tcp_signal_port',
            'TRADING_TCP_HOST': 'tcp_host',
            'TRADING_LOG_LEVEL': 'log_level',
            'TRADING_MAX_POSITION_SIZE': 'max_position_size',
            'TRADING_LEVERAGE': 'leverage',
            'TRADING_INTERVAL_SECONDS': 'trading_interval_seconds',
            'TRADING_CLOSE_ON_SHUTDOWN': 'close_positions_on_shutdown',
            'TRADING_MODEL_SAVE_INTERVAL': 'model_save_interval',
            'TRADING_MAX_DAILY_TRADES': 'max_daily_trades',
            'TRADING_MAX_HOLD_TIME': 'max_hold_time_hours',
            'TRADING_BATCH_SIZE': 'batch_size',
            'TRADING_LEARNING_RATE': 'learning_rate',
            'TRADING_MEMORY_BUFFER_SIZE': 'memory_buffer_size'
        }
        
        env_overrides = 0
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    
                    # Type conversion based on schema
                    if config_key in self._schemas:
                        schema = self._schemas[config_key]
                        value = self._convert_to_type(value, schema.data_type)
                    
                    self._settings[config_key] = value
                    env_overrides += 1
                    logger.debug(f"Applied environment override: {config_key} = {value}")
                    
                except Exception as e:
                    logger.warning(f"Error applying environment variable {env_var}: {e}")
        
        if env_overrides > 0:
            self._config_sources.append(f'env_vars_{env_overrides}')
            logger.info(f"Applied {env_overrides} environment variable overrides")
    
    def _convert_to_type(self, value: str, target_type: Type) -> Any:
        """Convert string value to target type"""
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        else:
            return value
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        directory_keys = ['data_directory', 'models_directory', 'logs_directory']
        
        for key in directory_keys:
            if key in self._settings:
                try:
                    dir_path = Path(self._settings[key])
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Ensured directory exists: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory for {key}: {e}")
    
    # IConfigurationProvider implementation
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        return self._settings.get(key, default)
    
    def get_typed(self, key: str, expected_type: Type, default: Any = None) -> Any:
        """Get configuration value with type checking and conversion"""
        value = self.get(key, default)
        
        if value is not None and not isinstance(value, expected_type):
            try:
                value = self._convert_to_type(str(value), expected_type)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert '{key}' to {expected_type.__name__}: {e}")
                return default
        
        return value
    
    # IConfigurationValidator implementation
    def validate(self, key: str, value: Any) -> bool:
        """Validate a single configuration value"""
        if not self._enable_validation or key not in self._schemas:
            return True
        
        schema = self._schemas[key]
        return self._validate_config_value(key, value, schema)
    
    def validate_all(self) -> bool:
        """Validate all configuration values"""
        if not self._enable_validation:
            return True
        
        self._validation_errors.clear()
        
        for key, schema in self._schemas.items():
            if key in self._settings:
                if not self._validate_config_value(key, self._settings[key], schema):
                    return False
            elif schema.required:
                self._validation_errors.append(f"Required configuration '{key}' is missing")
        
        if self._validation_errors:
            error_msg = f"Configuration validation failed: {', '.join(self._validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("All configurations validated successfully")
        return True
    
    def _validate_config_value(self, key: str, value: Any, schema: ConfigurationSchema) -> bool:
        """Validate a single configuration value against its schema"""
        try:
            # Type validation
            if not isinstance(value, schema.data_type):
                self._validation_errors.append(f"'{key}' must be {schema.data_type.__name__}")
                return False
            
            # Range validation for numeric values
            if schema.min_value is not None and isinstance(value, (int, float)):
                if value < schema.min_value:
                    self._validation_errors.append(f"'{key}' must be >= {schema.min_value}")
                    return False
            
            if schema.max_value is not None and isinstance(value, (int, float)):
                if value > schema.max_value:
                    self._validation_errors.append(f"'{key}' must be <= {schema.max_value}")
                    return False
            
            # Allowed values validation
            if schema.allowed_values is not None:
                if value not in schema.allowed_values:
                    self._validation_errors.append(f"'{key}' must be one of {schema.allowed_values}")
                    return False
            
            return True
            
        except Exception as e:
            self._validation_errors.append(f"Error validating '{key}': {e}")
            return False
    
    # Advanced configuration management methods
    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """Set configuration value with optional validation"""
        try:
            if validate and self._enable_validation:
                if not self.validate(key, value):
                    logger.error(f"Validation failed for '{key}': {self._validation_errors}")
                    return False
            
            old_value = self._settings.get(key)
            self._settings[key] = value
            
            logger.info(f"Configuration updated: {key} = {value} (was: {old_value})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration '{key}': {e}")
            return False
    
    def get_section(self, prefix: str) -> Dict[str, Any]:
        """Get all configuration values with a specific prefix"""
        return {
            key: value for key, value in self._settings.items()
            if key.startswith(prefix)
        }
    
    # Convenience methods for backward compatibility
    def get_tcp_config(self) -> Dict[str, Any]:
        """Get TCP-specific configuration"""
        return {
            'data_port': self.get('tcp_data_port'),
            'signal_port': self.get('tcp_signal_port'),
            'host': self.get('tcp_host'),
            'timeout': self.get('tcp_timeout')
        }
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return {
            'max_position_size': self.get('max_position_size'),
            'kelly_lookback': self.get('kelly_lookback'),
            'leverage': self.get('leverage'),
            'contract_value': self.get('contract_value'),
            'var_confidence': self.get('var_confidence'),
            'risk_lookback_days': self.get('risk_lookback_days'),
            'max_position_risk': self.get('max_position_risk'),
            'max_portfolio_risk': self.get('max_portfolio_risk')
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return {
            'trading_interval_seconds': self.get('trading_interval_seconds'),
            'min_trade_interval_seconds': self.get('min_trade_interval_seconds'),
            'max_daily_trades': self.get('max_daily_trades'),
            'max_hold_time_hours': self.get('max_hold_time_hours'),
            'mnq_point_value': self.get('mnq_point_value'),
            'mnq_tick_size': self.get('mnq_tick_size'),
            'contract_value': self.get('contract_value')
        }
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system-specific configuration"""
        return {
            'model_save_interval': self.get('model_save_interval'),
            'log_level': self.get('log_level'),
            'data_directory': self.get('data_directory'),
            'models_directory': self.get('models_directory'),
            'logs_directory': self.get('logs_directory'),
            'environment': self.get('environment')
        }
    
    def get_learnable_parameters(self) -> List[str]:
        """Return parameters that should be meta-learned rather than hardcoded"""
        return [
            'max_position_size_factor',
            'min_confidence_threshold',
            'risk_per_trade_factor',
            'max_trades_per_hour',
            'stop_preference',
            'target_preference',
            'consecutive_loss_tolerance',
            'position_size_factor',
            'stop_distance_factor',
            'target_distance_factor',
        ]
    
    # System management methods
    def reload_configuration(self, config_file: Optional[str] = None) -> None:
        """Reload configuration from all sources"""
        try:
            logger.info("Reloading configuration...")
            
            if config_file:
                self.config_file = config_file
            
            # Clear current state
            self._settings.clear()
            self._config_sources.clear()
            self._validation_errors.clear()
            
            # Reload all configurations
            self._load_all_configurations()
            
            logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            raise
    
    def save_configuration(self, output_file: str, include_defaults: bool = False) -> None:
        """Save current configuration to file"""
        try:
            config_to_save = {}
            
            for key, value in self._settings.items():
                if include_defaults or key not in self._schemas or value != self._schemas[key].default_value:
                    config_to_save[key] = value
            
            # Add metadata
            config_to_save['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'environment': self.environment,
                'sources': self._config_sources
            }
            
            with open(output_file, 'w') as f:
                json.dump(config_to_save, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary"""
        return {
            'environment': self.environment,
            'sources': self._config_sources,
            'total_settings': len(self._settings),
            'validation_errors': len(self._validation_errors),
            'schemas_defined': len(self._schemas),
            'load_timestamp': self._load_timestamp.isoformat() if self._load_timestamp else None,
            'required_settings': [k for k, v in self._schemas.items() if v.required],
            'modified_from_defaults': [
                k for k, v in self._settings.items() 
                if k in self._schemas and v != self._schemas[k].default_value
            ],
            'validation_enabled': self._enable_validation
        }

# Simple Config class for backward compatibility and simple use cases
class Config(IConfigurationProvider):
    """
    Simple configuration interface for backward compatibility.
    
    This provides the same interface as the original Config class while
    delegating to the more sophisticated ConfigurationManager.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize simple config interface"""
        self._config_manager = ConfigurationManager(config_file, enable_validation=False)
        self.settings = self._config_manager._settings
        
        logger.info(f"Simple config interface initialized: {self._get_config_summary()}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config_manager.get(key, default)
    
    def get_typed(self, key: str, expected_type: Type, default: Any = None) -> Any:
        """Get configuration value with type checking"""
        return self._config_manager.get_typed(key, expected_type, default)
    
    def get_tcp_config(self) -> Dict[str, Any]:
        """Get TCP-specific configuration"""
        return self._config_manager.get_tcp_config()
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self._config_manager.get_risk_config()
    
    def get_learnable_parameters(self) -> List[str]:
        """Return parameters that should be meta-learned"""
        return self._config_manager.get_learnable_parameters()
    
    def _get_config_summary(self) -> str:
        """Get a summary of current configuration for backward compatibility"""
        summary_keys = [
            'environment', 'tcp_data_port', 'tcp_signal_port', 
            'max_position_size', 'leverage'
        ]
        summary = {k: self._config_manager.get(k) for k in summary_keys}
        return str(summary)