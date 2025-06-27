# intelligence_types.py

from dataclasses import dataclass

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
    
    # All four subsystem signals
    dna_signal: float
    micro_signal: float
    temporal_signal: float
    immune_signal: float
    microstructure_signal: float
    overall_signal: float
    
    # Enhanced context
    regime_adjusted_signal: float = 0.0
    adaptation_quality: float = 0.0
    smart_money_flow: float = 0.0
    liquidity_depth: float = 0.5
    regime_confidence: float = 0.5