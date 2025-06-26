"""
Trading Domain Models - Core business entities
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class TradeStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Trade:
    id: str
    action: TradeAction
    size: float
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: TradeStatus = TradeStatus.PENDING
    
@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    
@dataclass
class Account:
    cash: float
    buying_power: float
    total_value: float
    margin_used: float
    day_trading_buying_power: float