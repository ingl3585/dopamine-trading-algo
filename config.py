# config.py

class Config:
    def __init__(self):
        # Minimal configuration - most parameters should be meta-learned
        self.settings = {
            # System operational settings (not trading logic)
            'tcp_port': 8080,
            'tcp_host': 'localhost',
            'model_save_interval': 300,  # 5 minutes
            'log_level': 'INFO',
            'data_directory': 'data',
            'models_directory': 'models',
            'logs_directory': 'logs',
            
            # Market-specific constants (not decision parameters)
            'mnq_point_value': 2.0,  # MNQ futures point value
            'mnq_tick_size': 0.25,   # MNQ minimum tick size
            
            # Bootstrap settings
            'min_historical_bars': 1000,  # Minimum bars needed for bootstrap
            'bootstrap_timeout': 300,     # Max time to wait for historical data
            
            # Emergency safety limits (learned parameters override these)
            'emergency_max_margin_usage': 0.95,  # Hard stop at 95% margin
            'emergency_max_drawdown': 0.20,      # Hard stop at 20% drawdown
        }
    
    def get(self, key: str, default=None):
        return self.settings.get(key, default)
    
    def update_setting(self, key: str, value):
        """Allow runtime updates to configuration"""
        self.settings[key] = value
    
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