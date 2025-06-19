# config.py

import json

from pathlib import Path

class Config:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        
        # Default configuration
        self.settings = {
            'max_daily_loss': 500.0,
            'max_position_size': 3,
            'min_confidence': 0.5,
            'risk_per_trade': 0.02,
            'max_trades_per_hour': 10,
            'margin_per_contract': 500.0,
            'point_value': 2.0
        }
        
        self.load()
    
    def get(self, key: str, default=None):
        return self.settings.get(key, default)
    
    def set(self, key: str, value):
        self.settings[key] = value
    
    def save(self):
        Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                saved_settings = json.load(f)
                self.settings.update(saved_settings)
        except FileNotFoundError:
            self.save()  # Create default config file