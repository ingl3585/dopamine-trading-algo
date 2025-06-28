# data_processor.py

import time

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MarketData:
    timestamp: float
    price: float
    volume: float
    prices_1m: List[float]
    prices_5m: List[float] 
    prices_15m: List[float]
    volumes_1m: List[float]
    volumes_5m: List[float]
    volumes_15m: List[float]
    # Enhanced account data
    account_balance: float
    buying_power: float
    daily_pnl: float
    unrealized_pnl: float
    net_liquidation: float
    margin_used: float
    available_margin: float
    open_positions: int
    total_position_size: int
    # Computed ratios from TCP bridge
    margin_utilization: float
    buying_power_ratio: float
    daily_pnl_pct: float


class DataProcessor:
    def __init__(self, max_history=1000):
        self.prices_1m = deque(maxlen=max_history)
        self.prices_5m = deque(maxlen=200)
        self.prices_15m = deque(maxlen=100)
        self.volumes_1m = deque(maxlen=max_history)
        self.volumes_5m = deque(maxlen=200)
        self.volumes_15m = deque(maxlen=100)

    def process(self, raw_data: Dict) -> Optional[MarketData]:
        if not self._is_valid_data(raw_data):
            return None

        # Update price/volume buffers
        if 'price_1m' in raw_data and raw_data['price_1m']:
            self.prices_1m.extend(raw_data['price_1m'])
            
        if 'volume_1m' in raw_data and raw_data['volume_1m']:
            self.volumes_1m.extend(raw_data['volume_1m'])
            
        if 'price_5m' in raw_data and raw_data['price_5m']:
            self.prices_5m.extend(raw_data['price_5m'])
            
        if 'volume_5m' in raw_data and raw_data['volume_5m']:
            self.volumes_5m.extend(raw_data['volume_5m'])
            
        if 'price_15m' in raw_data and raw_data['price_15m']:
            self.prices_15m.extend(raw_data['price_15m'])
            
        if 'volume_15m' in raw_data and raw_data['volume_15m']:
            self.volumes_15m.extend(raw_data['volume_15m'])

        if not self.prices_1m:
            return None

        return MarketData(
            timestamp=raw_data.get('timestamp', time.time()),
            price=self.prices_1m[-1],
            volume=self.volumes_1m[-1] if self.volumes_1m else 1000,
            prices_1m=list(self.prices_1m),
            prices_5m=list(self.prices_5m),
            prices_15m=list(self.prices_15m),
            volumes_1m=list(self.volumes_1m),
            volumes_5m=list(self.volumes_5m),
            volumes_15m=list(self.volumes_15m),
            # Enhanced account data from NinjaTrader
            account_balance=raw_data.get('account_balance', 25000),
            buying_power=raw_data.get('buying_power', 25000),
            daily_pnl=raw_data.get('daily_pnl', 0),
            unrealized_pnl=raw_data.get('unrealized_pnl', 0),
            net_liquidation=raw_data.get('net_liquidation', 25000),
            margin_used=raw_data.get('margin_used', 0),
            available_margin=raw_data.get('available_margin', 25000),
            open_positions=raw_data.get('open_positions', 0),
            total_position_size=raw_data.get('total_position_size', 0.0),
            # Computed ratios from TCP bridge
            margin_utilization=raw_data.get('margin_utilization', 0.0),
            buying_power_ratio=raw_data.get('buying_power_ratio', 1.0),
            daily_pnl_pct=raw_data.get('daily_pnl_pct', 0.0)
        )

    def _is_valid_data(self, data: Dict) -> bool:
        return (isinstance(data, dict) and 
                ('price_1m' in data or 'price_5m' in data or 'price_15m' in data))

    def prime_with_historical_data(self, historical_data: Dict):
        """Pre-populates the data buffers with historical data."""
        if 'bars_1m' in historical_data:
            self.prices_1m.extend([bar['close'] for bar in historical_data['bars_1m']])
            self.volumes_1m.extend([bar['volume'] for bar in historical_data['bars_1m']])

        if 'bars_5m' in historical_data:
            self.prices_5m.extend([bar['close'] for bar in historical_data['bars_5m']])
            self.volumes_5m.extend([bar['volume'] for bar in historical_data['bars_5m']])

        if 'bars_15m' in historical_data:
            self.prices_15m.extend([bar['close'] for bar in historical_data['bars_15m']])
            self.volumes_15m.extend([bar['volume'] for bar in historical_data['bars_15m']])