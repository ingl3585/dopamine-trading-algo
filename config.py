# config.py

class Config:
    def __init__(self):
        self.settings = {
            'max_daily_loss': 500.0,
            'max_position_size': 10,
            'min_confidence': 0.5,
            'risk_per_trade': 0.02,
            'max_trades_per_hour': 10,
            'margin_per_contract': 500.0,
            'point_value': 2.0
        }
    
    def get(self, key: str, default=None):
        return self.settings.get(key, default)