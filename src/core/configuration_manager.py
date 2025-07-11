# configuration_manager.py

import logging
import os
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationSchema:
    """Schema for configuration validation"""
    key: str
    data_type: type
    default_value: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    required: bool = False

class ConfigurationManager:
    """
    Centralized configuration management for the trading system.
    
    Responsibilities:
    - Load and validate configurations from multiple sources
    - Handle environment-specific overrides
    - Provide type-safe configuration access
    - Support runtime configuration updates
    - Manage configuration schemas and validation
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.settings = {}
        self.schemas = {}
        self.environment = os.getenv('TRADING_ENV', 'development')
        self.config_sources = []
        
        # Configuration validation
        self.validation_errors = []
        self.load_timestamp = None
        
        # Initialize configuration schemas
        self._initialize_schemas()
        
        # Load configurations
        self._load_all_configurations()
        
        logger.info(f"Configuration manager initialized for {self.environment} environment")
    
    def _initialize_schemas(self):
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
            ConfigurationSchema('max_daily_loss', float, 0.02, 0.001, 0.1,
                               description="Maximum daily loss as fraction of account"),
            ConfigurationSchema('leverage', float, 50.0, 1.0, 100.0,
                               description="Trading leverage"),
            ConfigurationSchema('kelly_lookback', int, 100, 10, 1000,
                               description="Number of trades for Kelly calculation"),
            
            # Market Data
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
            
            # Performance Tuning
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
            self.schemas[schema.key] = schema
    
    def _load_all_configurations(self):
        """Load configurations from all sources"""
        try:
            # Load base configuration
            self.settings = self._load_base_config()
            self.config_sources.append('base_defaults')
            
            # Load environment-specific configuration
            self._load_environment_config()
            
            # Load from specific file if provided
            if self.config_file:
                self._load_config_file(self.config_file)
            
            # Apply environment variable overrides
            self._load_environment_variables()
            
            # Validate all configurations
            self._validate_all_configs()
            
            # Create necessary directories
            self._create_directories()
            
            self.load_timestamp = datetime.now()
            
            logger.info(f"Configuration loaded from sources: {', '.join(self.config_sources)}")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration with defaults"""
        base_config = {}
        
        # Set defaults from schemas
        for key, schema in self.schemas.items():
            base_config[key] = schema.default_value
        
        # Add environment
        base_config['environment'] = self.environment
        
        return base_config
    
    def _load_environment_config(self):
        """Load environment-specific configuration"""
        try:
            config_file = f"config/{self.environment}.json"
            
            if os.path.exists(config_file):
                self._load_config_file(config_file)
                self.config_sources.append(f'env_{self.environment}')
                logger.info(f"Loaded {self.environment} environment configuration")
            else:
                logger.info(f"No {self.environment} environment config found")
                
        except Exception as e:
            logger.warning(f"Error loading environment config: {e}")
    
    def _load_config_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.settings.update(file_config)
                
            if config_file not in self.config_sources:
                self.config_sources.append(f'file_{config_file}')
                
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.warning(f"Error loading config file {config_file}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'TRADING_TCP_DATA_PORT': 'tcp_data_port',
            'TRADING_TCP_SIGNAL_PORT': 'tcp_signal_port',
            'TRADING_TCP_HOST': 'tcp_host',
            'TRADING_LOG_LEVEL': 'log_level',
            'TRADING_MAX_POSITION_SIZE': 'max_position_size',
            'TRADING_MAX_DAILY_LOSS': 'max_daily_loss',
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
                    
                    # Convert to appropriate type based on schema
                    if config_key in self.schemas:
                        schema = self.schemas[config_key]
                        if schema.data_type == int:
                            value = int(value)
                        elif schema.data_type == float:
                            value = float(value)
                        elif schema.data_type == bool:
                            value = value.lower() in ('true', '1', 'yes', 'on')
                    
                    self.settings[config_key] = value
                    env_overrides += 1
                    
                except Exception as e:
                    logger.warning(f"Error applying environment variable {env_var}: {e}")
        
        if env_overrides > 0:
            self.config_sources.append(f'env_vars_{env_overrides}')
            logger.info(f"Applied {env_overrides} environment variable overrides")
    
    def _validate_all_configs(self):
        """Validate all configuration values"""
        self.validation_errors = []
        
        for key, schema in self.schemas.items():
            if key in self.settings:
                self._validate_config_value(key, self.settings[key], schema)
            elif schema.required:
                self.validation_errors.append(f"Required configuration '{key}' is missing")
        
        if self.validation_errors:
            error_msg = f"Configuration validation failed: {', '.join(self.validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("All configurations validated successfully")
    
    def _validate_config_value(self, key: str, value: Any, schema: ConfigurationSchema):
        """Validate a single configuration value"""
        try:
            # Type validation
            if not isinstance(value, schema.data_type):
                self.validation_errors.append(f"'{key}' must be {schema.data_type.__name__}")
                return
            
            # Range validation for numeric values
            if schema.min_value is not None and isinstance(value, (int, float)):
                if value < schema.min_value:
                    self.validation_errors.append(f"'{key}' must be >= {schema.min_value}")
            
            if schema.max_value is not None and isinstance(value, (int, float)):
                if value > schema.max_value:
                    self.validation_errors.append(f"'{key}' must be <= {schema.max_value}")
            
            # Allowed values validation
            if schema.allowed_values is not None:
                if value not in schema.allowed_values:
                    self.validation_errors.append(f"'{key}' must be one of {schema.allowed_values}")
            
        except Exception as e:
            self.validation_errors.append(f"Error validating '{key}': {e}")
    
    def _create_directories(self):
        """Create necessary directories"""
        directory_keys = ['data_directory', 'models_directory', 'logs_directory']
        
        for key in directory_keys:
            if key in self.settings:
                dir_path = Path(self.settings[key])
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {dir_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        return self.settings.get(key, default)
    
    def get_typed(self, key: str, expected_type: type, default: Any = None) -> Any:
        """Get configuration value with type checking"""
        value = self.get(key, default)
        
        if value is not None and not isinstance(value, expected_type):
            try:
                # Attempt type conversion
                if expected_type == bool and isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    value = expected_type(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert '{key}' to {expected_type.__name__}")
                return default
        
        return value
    
    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """Set configuration value with optional validation"""
        try:
            if validate and key in self.schemas:
                # Create temporary copy for validation
                temp_settings = self.settings.copy()
                temp_settings[key] = value
                
                # Validate the new value
                schema = self.schemas[key]
                temp_errors = []
                self._validate_config_value(key, value, schema)
                
                if self.validation_errors:
                    logger.error(f"Validation failed for '{key}': {self.validation_errors}")
                    return False
            
            # Set the value
            old_value = self.settings.get(key)
            self.settings[key] = value
            
            logger.info(f"Configuration updated: {key} = {value} (was: {old_value})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration '{key}': {e}")
            return False
    
    def get_section(self, prefix: str) -> Dict[str, Any]:
        """Get all configuration values with a specific prefix"""
        return {
            key: value for key, value in self.settings.items()
            if key.startswith(prefix)
        }
    
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
            'max_daily_loss': self.get('max_daily_loss'),
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
    
    def reload_configuration(self, config_file: Optional[str] = None):
        """Reload configuration from sources"""
        try:
            logger.info("Reloading configuration...")
            
            if config_file:
                self.config_file = config_file
            
            # Clear current state
            self.settings.clear()
            self.config_sources.clear()
            self.validation_errors.clear()
            
            # Reload all configurations
            self._load_all_configurations()
            
            logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            raise
    
    def save_configuration(self, output_file: str, include_defaults: bool = False):
        """Save current configuration to file"""
        try:
            # Prepare configuration for saving
            config_to_save = {}
            
            for key, value in self.settings.items():
                if include_defaults or key in self.schemas:
                    # Don't save if it's just the default value
                    if not include_defaults and key in self.schemas:
                        if value == self.schemas[key].default_value:
                            continue
                    
                    config_to_save[key] = value
            
            # Add metadata
            config_to_save['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'environment': self.environment,
                'sources': self.config_sources
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(config_to_save, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of configuration state"""
        return {
            'environment': self.environment,
            'sources': self.config_sources,
            'total_settings': len(self.settings),
            'validation_errors': len(self.validation_errors),
            'schemas_defined': len(self.schemas),
            'load_timestamp': self.load_timestamp.isoformat() if self.load_timestamp else None,
            'required_settings': [k for k, v in self.schemas.items() if v.required],
            'modified_from_defaults': [
                k for k, v in self.settings.items() 
                if k in self.schemas and v != self.schemas[k].default_value
            ]
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return results"""
        self._validate_all_configs()
        
        return {
            'is_valid': len(self.validation_errors) == 0,
            'errors': self.validation_errors,
            'warnings': [],  # Could add warnings for deprecated settings
            'validated_settings': len(self.settings),
            'validation_timestamp': datetime.now().isoformat()
        }