from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

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
class Features:
    # Core market features
    price_momentum: float
    volume_momentum: float
    price_position: float
    volatility: float
    time_of_day: float
    pattern_score: float
    confidence: float
    
    # All five subsystem signals
    dna_signal: float
    micro_signal: float
    temporal_signal: float
    immune_signal: float
    microstructure_signal: float
    dopamine_signal: float
    overall_signal: float
    
    # Enhanced context
    regime_adjusted_signal: float = 0.0
    adaptation_quality: float = 0.0
    smart_money_flow: float = 0.0
    liquidity_depth: float = 0.5
    regime_confidence: float = 0.5
    
    # Multi-timeframe analysis
    trend_1m: float = 0.0
    trend_5m: float = 0.0
    trend_15m: float = 0.0
    trend_1h: float = 0.0
    trend_4h: float = 0.0
    trend_alignment: float = 0.0
    higher_tf_bias: float = 0.0
    volatility_1m: float = 0.02
    volatility_5m: float = 0.02
    volatility_15m: float = 0.02
    volatility_1h: float = 0.02
    volatility_4h: float = 0.02