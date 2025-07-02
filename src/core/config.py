"""
Configuration Management - Environment-aware settings with validation
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_file: Optional[str] = None):
        # Load base configuration
        self.settings = self._load_base_config()
        
        # Load environment-specific overrides
        self._load_environment_config()
        
        # Load from file if specified
        if config_file:
            self._load_config_file(config_file)
        
        # Apply environment variable overrides
        self._load_environment_variables()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Configuration loaded: {self._get_config_summary()}")
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration with defaults"""
        return {
            # TCP Connection Settings
            'tcp_data_port': 5556,
            'tcp_signal_port': 5557,
            'tcp_host': 'localhost',
            'tcp_timeout': 30,
            
            # System operational settings
            'model_save_interval': 300,  # 5 minutes
            'log_level': 'INFO',
            'data_directory': 'data',
            'models_directory': 'models',
            'logs_directory': 'logs',
            
            # Trading intervals
            'trading_interval_seconds': 60,  # 1 minute
            'min_trade_interval_seconds': 300,  # 5 minutes between trades
            'historical_data_timeout': 30,  # seconds
            
            # Market-specific constants
            'mnq_point_value': 2.0,  # MNQ futures point value
            'mnq_tick_size': 0.25,   # MNQ minimum tick size
            'contract_value': 2000,  # MNQ contract value
            'leverage': 50.0,  # Typical futures leverage
            
            # Position sizing
            'max_position_size': 0.1,  # 10% of account
            'min_position_size': 1.0,  # Minimum 1 contract
            'position_increment': 1.0,  # Round to whole contracts
            'max_daily_loss': 0.02,  # 2% daily loss limit
            
            # Risk management
            'kelly_lookback': 100,  # Number of trades for Kelly calculation
            'max_hold_time_hours': 24,  # Maximum position hold time
            'max_daily_trades': 10,  # Maximum trades per day
            
            # Bootstrap settings
            'min_historical_bars': 100,  # Minimum bars needed for bootstrap
            'bootstrap_timeout': 300,  # Max time to wait for historical data
            
            # Emergency safety limits
            'emergency_max_margin_usage': 0.95,  # Hard stop at 95% margin
            'emergency_max_drawdown': 0.20,  # Hard stop at 20% drawdown
            
            # Shutdown behavior
            'close_positions_on_shutdown': False,
            
            # Environment
            'environment': os.getenv('TRADING_ENV', 'development')
        }
    
    def _load_environment_config(self):
        """Load environment-specific configuration"""
        try:
            env = self.settings['environment']
            config_file = f"config/{env}.json"
            
            if os.path.exists(config_file):
                self._load_config_file(config_file)
                logger.info(f"Loaded {env} environment configuration")
            else:
                logger.info(f"No {env} environment config found, using defaults")
                
        except Exception as e:
            logger.warning(f"Error loading environment config: {e}")
    
    def _load_config_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.settings.update(file_config)
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
            'TRADING_CLOSE_ON_SHUTDOWN': 'close_positions_on_shutdown'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    # Try to convert to appropriate type
                    if config_key in ['tcp_data_port', 'tcp_signal_port', 'trading_interval_seconds']:
                        value = int(value)
                    elif config_key in ['max_position_size', 'max_daily_loss', 'leverage']:
                        value = float(value)
                    elif config_key in ['close_positions_on_shutdown']:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    
                    self.settings[config_key] = value
                    logger.info(f"Applied environment override: {config_key} = {value}")
                    
                except Exception as e:
                    logger.warning(f"Error applying environment variable {env_var}: {e}")
    
    def _validate_config(self):
        """Validate configuration settings"""
        validations = [
            ('tcp_data_port', lambda x: 1024 <= x <= 65535, "TCP data port must be 1024-65535"),
            ('tcp_signal_port', lambda x: 1024 <= x <= 65535, "TCP signal port must be 1024-65535"),
            ('max_position_size', lambda x: 0 < x <= 1, "Max position size must be 0-1"),
            ('max_daily_loss', lambda x: 0 < x <= 1, "Max daily loss must be 0-1"),
            ('leverage', lambda x: x > 0, "Leverage must be positive"),
            ('trading_interval_seconds', lambda x: x >= 1, "Trading interval must be >= 1 second"),
            ('min_trade_interval_seconds', lambda x: x >= 60, "Min trade interval must be >= 60 seconds"),
            ('kelly_lookback', lambda x: x >= 10, "Kelly lookback must be >= 10 trades"),
            ('contract_value', lambda x: x > 0, "Contract value must be positive")
        ]
        
        for key, validator, message in validations:
            if key in self.settings:
                if not validator(self.settings[key]):
                    raise ValueError(f"Configuration validation failed: {message}")
        
        # Ensure directories exist
        for dir_key in ['data_directory', 'models_directory', 'logs_directory']:
            dir_path = Path(self.settings[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_config_summary(self) -> str:
        """Get a summary of current configuration"""
        summary_keys = [
            'environment', 'tcp_data_port', 'tcp_signal_port', 
            'max_position_size', 'max_daily_loss', 'leverage'
        ]
        summary = {k: self.settings.get(k) for k in summary_keys}
        return str(summary)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.settings.get(key, default)
    
    
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
            'contract_value': self.get('contract_value')
        }
    
    def get_learnable_parameters(self):
        """Return parameters that should be meta-learned rather than hardcoded"""
        return [
            'max_daily_loss_factor',      # Learned from account size and performance
            'max_position_size_factor',   # Learned from volatility and account
            'min_confidence_threshold',   # Learned from historical performance
            'risk_per_trade_factor',      # Learned via Kelly criterion
            'max_trades_per_hour',        # Learned from market conditions
            'stop_preference',            # Learned from stop effectiveness
            'target_preference',          # Learned from target effectiveness
            'loss_tolerance_factor',      # Learned from drawdown recovery
            'consecutive_loss_tolerance', # Learned from streak analysis
            'position_size_factor',       # Learned from risk-adjusted returns
            'stop_distance_factor',       # Learned from volatility patterns
            'target_distance_factor',     # Learned from profit optimization
        ]