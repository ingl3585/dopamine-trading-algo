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
    source: str
    timestamp: datetime
    
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
    cash: float
    buying_power: float
    position_size: float
    unrealized_pnl: float
    realized_pnl: float