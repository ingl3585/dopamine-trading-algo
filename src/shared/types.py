from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class MarketData:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    
@dataclass
class Signal:
    value: float
    confidence: float
    source: str = "unknown"
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
@dataclass
class TradeDecision:
    action: str  # 'buy', 'sell', 'hold'
    size: float
    confidence: float
    reasoning: Dict
    
@dataclass
class TradeOutcome:
    pnl: float
    duration: float
    success: bool
    context: Dict
    
@dataclass
class AccountInfo:
    cash: float = 25000.0
    buying_power: float = 25000.0
    position_size: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class Order:
    action: str  # 'buy' or 'sell'
    size: float
    price: float = 0.0
    confidence: float = 0.7
    stop_price: float = 0.0
    target_price: float = 0.0
    primary_tool: str = "ai_agent"